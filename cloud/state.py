"""Per-project state for the cloud orchestrator.

A single JSON file per project under ``~/.herbarium-cloud/<project>.json``
records the pod and network-volume IDs, the hash of the last DwC-A uploaded,
the current pipeline step, and a running cost estimate. The file is the
source of truth across desktop-app restarts: if the user closes the webui
mid-training, the next launch reads the file, finds the pod, and reattaches.

Writes are atomic (tempfile + ``os.replace``) so a crash mid-write cannot
leave a half-written JSON file on disk.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".herbarium-cloud"

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize(project: str) -> str:
    """Return a filename-safe version of ``project``.

    Path separators, spaces, and other shell-unsafe chars collapse to ``_``.
    Empty / dot-only names are rejected — the project name is part of the
    on-disk path so we can't let the user escape ``DEFAULT_ROOT``.
    """
    safe = _SAFE_NAME.sub("_", project.strip())
    if not safe or set(safe) <= {".", "_"}:
        raise ValueError(f"invalid project name: {project!r}")
    return safe


def state_path(project: str, root: Path | None = None) -> Path:
    root = root or DEFAULT_ROOT
    return root / f"{_sanitize(project)}.json"


@dataclass
class ProjectState:
    """All persistent state for one cloud project."""

    project: str

    # Cloud resources currently allocated for this project.
    pod_id: str | None = None
    volume_id: str | None = None
    data_center_id: str | None = None  # e.g. "EU-RO-1"

    # SSH endpoint of the running pod, captured at provision time.
    # None when the pod is stopped or terminated.
    ssh_host: str | None = None
    ssh_port: int | None = None

    # Pipeline progress.
    dwca_sha256: str | None = None  # SHA-256 of last DwC-A uploaded to volume
    current_step: str = ""          # "", "download", "prep", "train", "identify"
    completed_steps: list[str] = field(default_factory=list)

    # Cost tracking. ``pod_started_at`` is wall-clock unix time when the
    # current pod began billing; ``pod_hourly_rate`` is its USD/hr rate.
    # ``accumulated_cost_usd`` is cost from earlier pods (already terminated).
    pod_started_at: float | None = None
    pod_hourly_rate: float | None = None
    accumulated_cost_usd: float = 0.0

    last_updated: float = 0.0

    @classmethod
    def fresh(cls, project: str) -> "ProjectState":
        return cls(project=project)


def load(project: str, root: Path | None = None) -> ProjectState:
    """Load state for ``project``. Returns a fresh state if the file is absent."""
    path = state_path(project, root)
    if not path.exists():
        return ProjectState.fresh(project)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Drop unknown keys so old state files don't crash on schema additions.
    known = {f.name for f in ProjectState.__dataclass_fields__.values()}
    return ProjectState(**{k: v for k, v in data.items() if k in known})


def save(state: ProjectState, root: Path | None = None) -> None:
    """Atomic write of ``state`` to its JSON file. Creates the root dir."""
    state.last_updated = time.time()
    path = state_path(state.project, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile in the same dir so os.replace is a true rename
    # (cross-device renames fail on POSIX, and we want it atomic).
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, prefix=path.name + ".",
    ) as tmp:
        json.dump(asdict(state), tmp, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def list_projects(root: Path | None = None) -> list[str]:
    """Return project names that have a state file under ``root``."""
    root = root or DEFAULT_ROOT
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.json"))


def delete(project: str, root: Path | None = None) -> None:
    """Remove the state file for ``project``. No-op if absent.

    Does NOT touch cloud resources — the orchestrator must terminate the pod
    and delete the volume separately. This only forgets the local pointers.
    """
    path = state_path(project, root)
    path.unlink(missing_ok=True)


def current_run_cost(state: ProjectState, *, now: float | None = None) -> float:
    """USD spent so far across all pods for this project.

    ``accumulated_cost_usd`` (closed pods) + active pod time × hourly rate.
    Returns 0 if no pod has ever run.
    """
    total = state.accumulated_cost_usd
    if state.pod_started_at and state.pod_hourly_rate:
        elapsed_hr = ((now or time.time()) - state.pod_started_at) / 3600.0
        total += max(0.0, elapsed_hr) * state.pod_hourly_rate
    return total


def close_active_pod(state: ProjectState, *, now: float | None = None) -> None:
    """Roll the active pod's accrued cost into ``accumulated_cost_usd``.

    Call when stopping or terminating the pod. After this returns,
    ``pod_id``/``ssh_*``/``pod_started_at``/``pod_hourly_rate`` are cleared.
    """
    if state.pod_started_at and state.pod_hourly_rate:
        elapsed_hr = ((now or time.time()) - state.pod_started_at) / 3600.0
        state.accumulated_cost_usd += max(0.0, elapsed_hr) * state.pod_hourly_rate
    state.pod_id = None
    state.ssh_host = None
    state.ssh_port = None
    state.pod_started_at = None
    state.pod_hourly_rate = None
