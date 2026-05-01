"""Async wrapper around the RunPod REST API.

Covers the slice the orchestrator needs:
    - pods:           create, get, list, start, stop, terminate, wait-until-ready
    - network volumes: create, list, delete

Nothing here knows about the herbarium pipeline; it's a thin httpx wrapper
that returns plain dicts (or small dataclasses where it helps readability).
SSH key injection is not part of the API — RunPod auto-installs whatever
keys the user has registered in their account, so the orchestrator only
needs to surface ``publicIp`` + the port-22 mapping after the pod is up.

API reference: https://rest.runpod.io/v1/openapi.json
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import httpx

DEFAULT_BASE_URL = "https://rest.runpod.io/v1"


class RunPodAPIError(RuntimeError):
    """Raised for any non-2xx response from the RunPod REST API."""

    def __init__(self, status: int, body: str, *, method: str = "", url: str = ""):
        super().__init__(f"RunPod {method} {url} → {status}: {body[:500]}")
        self.status = status
        self.body = body
        self.method = method
        self.url = url


@dataclass(frozen=True)
class PodInfo:
    """Convenience view over a RunPod ``Pod`` object."""

    id: str
    name: str
    desired_status: str          # "RUNNING" | "EXITED" | "TERMINATED"
    public_ip: str | None
    port_mappings: dict[str, int]  # {"22": 12345, ...}
    cost_per_hr: float
    network_volume_id: str | None
    raw: dict[str, Any]

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "PodInfo":
        # portMappings keys come back as strings; values are public ports.
        pm_raw = d.get("portMappings") or {}
        pm = {str(k): int(v) for k, v in pm_raw.items()} if pm_raw else {}
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            desired_status=d.get("desiredStatus", ""),
            public_ip=d.get("publicIp"),
            port_mappings=pm,
            cost_per_hr=float(d.get("costPerHr") or 0.0),
            network_volume_id=d.get("networkVolumeId"),
            raw=d,
        )

    @property
    def ssh_endpoint(self) -> tuple[str, int] | None:
        """``(host, port)`` for SSH, or None if not yet routable."""
        if self.public_ip and "22" in self.port_mappings:
            return self.public_ip, self.port_mappings["22"]
        return None


@dataclass(frozen=True)
class NetworkVolume:
    id: str
    name: str
    size_gb: int
    data_center_id: str
    raw: dict[str, Any]

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "NetworkVolume":
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            size_gb=int(d.get("size") or 0),
            data_center_id=d.get("dataCenterId", ""),
            raw=d,
        )


class RunPodClient:
    """Minimal async REST client for RunPod.

    Use as an async context manager so the underlying httpx client closes::

        async with RunPodClient(api_key) as rp:
            pod = await rp.create_pod(...)
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError("RunPod API key is required")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def __aenter__(self) -> "RunPodClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── HTTP plumbing ─────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        resp = await self._client.request(method, path, json=json, params=params)
        if resp.status_code == 204:
            return None
        if resp.status_code >= 400:
            raise RunPodAPIError(
                resp.status_code, resp.text, method=method, url=str(resp.request.url)
            )
        # Some success endpoints return empty bodies.
        if not resp.content:
            return None
        return resp.json()

    # ── Pods ──────────────────────────────────────────────────────────────

    async def create_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_type_ids: Iterable[str] | None = None,
        gpu_count: int = 1,
        compute_type: Literal["GPU", "CPU"] = "GPU",
        cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE",
        container_disk_gb: int = 40,
        volume_gb: int | None = None,
        volume_mount_path: str = "/workspace",
        network_volume_id: str | None = None,
        data_center_ids: Iterable[str] | None = None,
        ports: Iterable[str] = ("22/tcp",),
        env: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PodInfo:
        """Create a new pod. The returned pod typically lacks ``publicIp``
        immediately — call :meth:`wait_until_ready` to poll until SSH is up.
        """
        body: dict[str, Any] = {
            "name": name,
            "imageName": image_name,
            "computeType": compute_type,
            "cloudType": cloud_type,
            "containerDiskInGb": container_disk_gb,
            "volumeMountPath": volume_mount_path,
            "ports": list(ports),
        }
        if compute_type == "GPU":
            body["gpuCount"] = gpu_count
            if gpu_type_ids:
                body["gpuTypeIds"] = list(gpu_type_ids)
        if volume_gb is not None:
            body["volumeInGb"] = volume_gb
        if network_volume_id:
            body["networkVolumeId"] = network_volume_id
        if data_center_ids:
            body["dataCenterIds"] = list(data_center_ids)
        if env:
            body["env"] = env
        if extra:
            body.update(extra)

        data = await self._request("POST", "/pods", json=body)
        return PodInfo.from_api(data)

    async def get_pod(self, pod_id: str) -> PodInfo:
        data = await self._request("GET", f"/pods/{pod_id}")
        return PodInfo.from_api(data)

    async def list_pods(
        self,
        *,
        desired_status: str | None = None,
        compute_type: str | None = None,
    ) -> list[PodInfo]:
        params: dict[str, Any] = {}
        if desired_status:
            params["desiredStatus"] = desired_status
        if compute_type:
            params["computeType"] = compute_type
        data = await self._request("GET", "/pods", params=params)
        return [PodInfo.from_api(d) for d in (data or [])]

    async def start_pod(self, pod_id: str) -> None:
        await self._request("POST", f"/pods/{pod_id}/start")

    async def stop_pod(self, pod_id: str) -> None:
        await self._request("POST", f"/pods/{pod_id}/stop")

    async def terminate_pod(self, pod_id: str) -> None:
        await self._request("DELETE", f"/pods/{pod_id}")

    async def wait_until_ready(
        self,
        pod_id: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 5.0,
    ) -> PodInfo:
        """Poll ``get_pod`` until SSH (port 22) is exposed, or raise.

        RunPod can take a minute or two to schedule a pod, pull the image,
        start it, and assign a public IP. We treat ``ssh_endpoint != None``
        as "ready" — that's the only signal the orchestrator actually needs.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        last: PodInfo | None = None
        while True:
            last = await self.get_pod(pod_id)
            if last.desired_status == "TERMINATED":
                raise RunPodAPIError(
                    0, f"pod {pod_id} terminated before becoming ready",
                    method="GET", url=f"/pods/{pod_id}",
                )
            if last.ssh_endpoint is not None:
                return last
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"pod {pod_id} not ready after {timeout:.0f}s "
                    f"(status={last.desired_status}, ip={last.public_ip})"
                )
            await asyncio.sleep(poll_interval)

    # ── GPU types ─────────────────────────────────────────────────────────

    async def list_gpu_types(self) -> list[str]:
        """Return the list of GPU type IDs RunPod will accept.

        The REST v1 API has no ``/gputypes`` endpoint, but the valid
        identifiers are baked into the published OpenAPI schema as the
        enum on ``PodCreateInput.gpuTypeIds.items``. We fetch the spec
        and pull that list — the same source the API server validates
        against, so the result is authoritative.

        Pass any string from this list as ``gpu_type_ids`` to
        :meth:`create_pod`. ``GPU_BY_PURPOSE`` in ``orchestrator.py``
        should always be a subset of these.
        """
        # Spec endpoint is unauthenticated; uses the same base_url + httpx
        # client the rest of the methods do.
        resp = await self._client.get("/openapi.json")
        if resp.status_code >= 400:
            raise RunPodAPIError(
                resp.status_code, resp.text,
                method="GET", url=str(resp.request.url),
            )
        spec = resp.json()
        try:
            enum = (spec["components"]["schemas"]["PodCreateInput"]
                        ["properties"]["gpuTypeIds"]["items"]["enum"])
        except (KeyError, TypeError) as e:
            raise RunPodAPIError(
                0, f"unexpected OpenAPI shape: {e!r}",
                method="GET", url="/openapi.json",
            )
        return list(enum)

    # ── Network volumes ───────────────────────────────────────────────────

    async def create_volume(
        self, *, name: str, size_gb: int, data_center_id: str,
    ) -> NetworkVolume:
        data = await self._request(
            "POST", "/networkvolumes",
            json={"name": name, "size": size_gb, "dataCenterId": data_center_id},
        )
        return NetworkVolume.from_api(data)

    async def list_volumes(self) -> list[NetworkVolume]:
        data = await self._request("GET", "/networkvolumes")
        return [NetworkVolume.from_api(d) for d in (data or [])]

    async def delete_volume(self, volume_id: str) -> None:
        await self._request("DELETE", f"/networkvolumes/{volume_id}")
