"""Shared application identity; model configuration will be added later."""

APP_TITLE = "Worker Safety Helmet Detection"
APP_ICON = "👷"

ALLOWED_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 8192

# Fixed inference default. Step 8 model code should import this directly;
# it is intentionally not exposed through the UI or session state.
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
