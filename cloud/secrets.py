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
RUNPOD_S3_KEY = "runpod_s3"
R2_KEY = "r2"
WANDB_KEY = "wandb"
GBIF_KEY = "gbif"
HF_KEY = "huggingface"


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


# ── RunPod S3 API keys (direct network-volume access, no pod needed) ──────
# A *separate* credential from the REST API key above: RunPod's S3-compatible
# gateway (s3api-<datacenter>.runpod.io) authenticates with its own access-key
# /secret pair, created under Settings → S3 API Keys in the console. Stored as
# a JSON blob so the pair travels together. Account-wide, so no datacenter or
# bucket is baked in — those are per-volume and resolved at call time.

@dataclass(frozen=True)
class RunPodS3Credentials:
    access_key_id: str
    secret_access_key: str

    @staticmethod
    def endpoint_for(data_center_id: str) -> str:
        """S3 gateway URL for a datacenter, e.g. EUR-IS-1 → s3api-eur-is-1."""
        return f"https://s3api-{data_center_id.lower()}.runpod.io"


def get_runpod_s3_credentials() -> RunPodS3Credentials | None:
    raw = keyring.get_password(SERVICE_NAME, RUNPOD_S3_KEY)
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return RunPodS3Credentials(**d)
    except (json.JSONDecodeError, TypeError):
        return None


def set_runpod_s3_credentials(creds: RunPodS3Credentials) -> None:
    if not creds.access_key_id.strip() or not creds.secret_access_key.strip():
        raise ValueError("access key id and secret must both be set")
    keyring.set_password(SERVICE_NAME, RUNPOD_S3_KEY, json.dumps(asdict(creds)))


def delete_runpod_s3_credentials() -> None:
    try:
        keyring.delete_password(SERVICE_NAME, RUNPOD_S3_KEY)
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


# ── Hugging Face write token (publish models to the Hub) ──────────────────
# Stored as a JSON blob {token, username} so the Publish tab's "HF user"
# field can default from here instead of being typed in on every project.
# get/set_hf_token() stay token-only for existing callers; older stores hold
# a bare token string (pre-username), which get_hf_credentials() treats as
# a token with no username rather than failing to parse.

@dataclass(frozen=True)
class HFCredentials:
    token: str
    username: str = ""


def get_hf_credentials() -> HFCredentials | None:
    raw = keyring.get_password(SERVICE_NAME, HF_KEY)
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return HFCredentials(token=d.get("token", ""), username=d.get("username", ""))
    except json.JSONDecodeError:
        return HFCredentials(token=raw, username="")


def get_hf_token() -> str | None:
    creds = get_hf_credentials()
    return creds.token if creds and creds.token else None


def get_hf_username() -> str | None:
    creds = get_hf_credentials()
    return creds.username if creds and creds.username else None


def set_hf_credentials(token: str, username: str = "") -> None:
    if not token.strip():
        raise ValueError("token is empty")
    keyring.set_password(SERVICE_NAME, HF_KEY,
                         json.dumps({"token": token.strip(), "username": username.strip()}))


def set_hf_token(token: str) -> None:
    """Back-compat single-value setter; preserves any username already saved."""
    existing = get_hf_credentials()
    set_hf_credentials(token, existing.username if existing else "")


def delete_hf_token() -> None:
    try:
        keyring.delete_password(SERVICE_NAME, HF_KEY)
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
