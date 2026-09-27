"""Download the chosen private checkpoint once per server process/configuration."""

from hashlib import file_digest, sha256
import logging
from pathlib import Path
from time import sleep

import streamlit as st

from app.config import HF_REPO_ID, MODEL_SHA256, setting

logger = logging.getLogger(__name__)


def access_failure(exc: Exception) -> tuple[str, str] | None:
    """Classify remote failures without exposing response bodies or credentials."""
    from huggingface_hub.errors import GatedRepoError, RemoteEntryNotFoundError, RevisionNotFoundError

    if isinstance(exc, RemoteEntryNotFoundError):
        return "file_missing", "The trained model file best.pt was not found at the repository root. The app owner should check the uploaded filename and folder."
    if isinstance(exc, RevisionNotFoundError):
        return "revision_missing", "The configured model revision was not found. The app owner should check HF_MODEL_REVISION against the repository branch or commit."
    if isinstance(exc, GatedRepoError):
        return "gated_access", "The trained model cannot be accessed because repository access approval is required for the token's account."
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 401:
        return "http_401", "The trained model cannot be accessed (HTTP 401). Hugging Face did not authorize this request. The app owner should verify the token's account can read this private repository."
    if status == 403:
        return "http_403", "The trained model cannot be accessed (HTTP 403). Hugging Face denied the download. The app owner should check the token's repository read permissions."
    if status == 404:
        return "http_404", "The trained model cannot be accessed (HTTP 404). The requested repository or file was not found or is hidden from this token. The app owner should verify the repository ID and best.pt upload using the token's account."
    return None


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
            failure = access_failure(exc)
            if failure is not None:
                code, message = failure
                # Log only our fixed diagnostic code, never the original exception.
                logger.warning("Checkpoint download failed: %s", code)
                raise CheckpointDownloadError(message) from None
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
