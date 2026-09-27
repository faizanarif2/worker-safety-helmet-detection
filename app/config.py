"""Shared application identity and developer-controlled inference defaults."""

import os
from pathlib import Path

APP_TITLE = "Worker Safety Helmet Detection"
APP_ICON = "👷"

ALLOWED_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 8192

# Fixed inference default. Inference imports this directly;
# it is intentionally not exposed through the UI or session state.
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "faizanarif233/worker-safety-helmet-detection"
# SHA-256 of the user's selected checkpoint, verified locally before deployment.
MODEL_SHA256 = "b6a532cc5c38048a9e120184b6187b8037ae20f090562b97f390059f9aa48ca0"


def setting(name: str, default: str = "") -> str:
    """Read deployment settings without requiring a local secrets file."""
    if name in os.environ:
        return os.environ[name].strip()
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError

    try:
        return str(st.secrets.get(name, default)).strip()
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return default


def model_path() -> Path:
    """Relative overrides resolve from the repository, not the current directory."""
    path = Path(setting("HELMET_MODEL_PATH", "models/best.pt")).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def inference_device() -> str:
    return setting("HELMET_DEVICE", "cpu").lower()
