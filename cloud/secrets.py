"""Cross-platform secret storage for the cloud orchestrator.

Uses the ``keyring`` library, which routes to:
    - Windows Credential Manager
    - macOS Keychain
    - Linux Secret Service / kwallet

Secrets are stored under service name ``"herbarium-cloud"`` so they're
discoverable in the OS credential UI and survive desktop-app upgrades.

Two slots:
    - ``runpod``  → API key (single string)
    - ``r2``      → JSON blob: account_id, access_key_id, secret_access_key, bucket
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import keyring
import keyring.errors

SERVICE_NAME = "herbarium-cloud"
RUNPOD_KEY = "runpod"
R2_KEY = "r2"
WANDB_KEY = "wandb"
GBIF_KEY = "gbif"


# ── RunPod API key ────────────────────────────────────────────────────────

def get_runpod_api_key() -> str | None:
    return keyring.get_password(SERVICE_NAME, RUNPOD_KEY)


def set_runpod_api_key(api_key: str) -> None:
    if not api_key.strip():
        raise ValueError("API key is empty")
    keyring.set_password(SERVICE_NAME, RUNPOD_KEY, api_key.strip())


def delete_runpod_api_key() -> None:
    try:
        keyring.delete_password(SERVICE_NAME, RUNPOD_KEY)
    except keyring.errors.PasswordDeleteError:
        pass


# ── WandB API key ─────────────────────────────────────────────────────────

def get_wandb_api_key() -> str | None:
    return keyring.get_password(SERVICE_NAME, WANDB_KEY)


def set_wandb_api_key(api_key: str) -> None:
    if not api_key.strip():
        raise ValueError("API key is empty")
    keyring.set_password(SERVICE_NAME, WANDB_KEY, api_key.strip())


def delete_wandb_api_key() -> None:
    try:
        keyring.delete_password(SERVICE_NAME, WANDB_KEY)
    except keyring.errors.PasswordDeleteError:
        pass


# ── R2 (Cloudflare) credentials for off-site backup ───────────────────────
# Stored as a single JSON blob because rclone needs all four fields together.

@dataclass(frozen=True)
class R2Credentials:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @property
    def endpoint(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


def get_r2_credentials() -> R2Credentials | None:
    raw = keyring.get_password(SERVICE_NAME, R2_KEY)
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    try:
        return R2Credentials(**d)
    except TypeError:
        # Stored blob has unexpected shape; ignore rather than crash.
        return None


def set_r2_credentials(creds: R2Credentials) -> None:
    keyring.set_password(SERVICE_NAME, R2_KEY, json.dumps(asdict(creds)))


def delete_r2_credentials() -> None:
    try:
        keyring.delete_password(SERVICE_NAME, R2_KEY)
    except keyring.errors.PasswordDeleteError:
        pass


# ── GBIF credentials ──────────────────────────────────────────────────────
# Stored as a JSON blob so username and password travel together.

@dataclass(frozen=True)
class GBIFCredentials:
    username: str
    password: str


def get_gbif_credentials() -> GBIFCredentials | None:
    raw = keyring.get_password(SERVICE_NAME, GBIF_KEY)
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return GBIFCredentials(**d)
    except (json.JSONDecodeError, TypeError):
        return None


def set_gbif_credentials(creds: GBIFCredentials) -> None:
    keyring.set_password(SERVICE_NAME, GBIF_KEY, json.dumps(asdict(creds)))


def delete_gbif_credentials() -> None:
    try:
        keyring.delete_password(SERVICE_NAME, GBIF_KEY)
    except keyring.errors.PasswordDeleteError:
        pass
