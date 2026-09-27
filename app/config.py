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


def model_path() -> Path:
    """Relative overrides resolve from the repository, not the current directory."""
    path = Path(os.environ.get("HELMET_MODEL_PATH", "models/best.pt")).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def inference_device() -> str:
    return os.environ.get("HELMET_DEVICE", "cpu").strip().lower()
