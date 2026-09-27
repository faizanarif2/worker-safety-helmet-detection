"""Download the chosen private checkpoint once per server process/configuration."""

from hashlib import file_digest, sha256
from pathlib import Path
from time import sleep

import streamlit as st

from app.config import HF_REPO_ID, MODEL_SHA256, setting


class CheckpointDownloadError(RuntimeError):
    """A public-safe error: never include HTTP responses or credentials."""


def remote_checkpoint() -> Path:
    token = setting("HF_TOKEN")
    if not token:
        raise CheckpointDownloadError("Model access is not configured. The app owner must add the private-model access token.")
    revision = setting("HF_MODEL_REVISION", "main")
    expected_hash = setting("HF_MODEL_SHA256", MODEL_SHA256).lower()
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise CheckpointDownloadError("The model checksum configuration is invalid. Contact the app owner.")
    # Do not put the token into cache keys, UI output, or session state. Its
    # fingerprint invalidates the download cache after credential rotation.
    return Path(download_checkpoint(
        HF_REPO_ID, revision, expected_hash, sha256(token.encode()).hexdigest(), _token=token,
    ))


@st.cache_resource(show_spinner=False, max_entries=1)
def download_checkpoint(repo_id: str, revision: str, expected_hash: str,
                        credential_fingerprint: str, *, _token: str) -> str:
    """Cache successful downloads; failed calls can be retried on the next rerun.

    Hugging Face also maintains a version-aware disk cache across app reruns.
    Its cached file is read without changing or copying the original weights.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise CheckpointDownloadError("The model download dependency is missing. Contact the app owner.") from None

    path = None
    for attempt in range(2):
        try:
            path = Path(hf_hub_download(
                repo_id=repo_id, filename="best.pt", repo_type="model",
                revision=revision, token=_token, etag_timeout=10,
            ))
            break
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403, 404):
                raise CheckpointDownloadError(
                    "The trained model cannot be accessed. The app owner should check the token, repository revision and best.pt upload."
                ) from None
            if attempt == 1:
                raise CheckpointDownloadError(
                    "The model download is temporarily unavailable. Please reload to retry."
                ) from None
            sleep(1)

    try:
        with path.open("rb") as checkpoint:
            actual_hash = file_digest(checkpoint, "sha256").hexdigest()
    except OSError:
        raise CheckpointDownloadError("The downloaded checkpoint could not be read. Please reload to retry.") from None
    if actual_hash != expected_hash:
        raise CheckpointDownloadError(
            "The downloaded model does not match the selected checkpoint. The app owner must check the uploaded file."
        )
    # Preserve the snapshot's best.pt symlink, not its extensionless blob target.
    return str(path.absolute())
