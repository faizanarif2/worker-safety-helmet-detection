"""Validate and prepare uploaded images entirely in memory."""

from io import BytesIO
from pathlib import PurePath
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
)


class ImageValidationError(ValueError):
    """An uploaded file cannot be safely used as an image."""


def prepare_image(data: bytes, filename: str) -> Image.Image:
    """Return a verified, upright RGB image independent of the input stream.

    Only JPEG, PNG, and WebP contents are accepted. Animated files use their
    first frame. Transparency is composited onto white. Nothing is saved or
    cached, and decoding is bounded by both pixel count and side length.
    """
    if not data:
        raise ImageValidationError("This file is empty. Choose a JPEG, PNG, or WebP image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"This file exceeds the {MAX_UPLOAD_MB} MB limit. Choose a smaller image."
        )
    if PurePath(filename).suffix.lower().lstrip(".") not in ALLOWED_IMAGE_EXTENSIONS:
        raise ImageValidationError("Unsupported file type. Choose a JPEG, PNG, or WebP image.")

    size_error = (
        f"This image is too large to process. Use at most {MAX_IMAGE_PIXELS:,} "
        f"pixels in total and no more than {MAX_IMAGE_SIDE:,} pixels on either side."
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ImageValidationError(
                        "The file contents are not JPEG, PNG, or WebP. "
                        "Export the image in a supported format; renaming it is not enough."
                    )
                width, height = source.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > MAX_IMAGE_PIXELS
                    or max(width, height) > MAX_IMAGE_SIDE
                ):
                    raise ImageValidationError(size_error)
                source.verify()

            # verify() invalidates the decoder; reopen and fully decode before
            # displaying anything so truncated pixel data also fails here.
            with Image.open(BytesIO(data)) as source:
                source.load()
                upright = ImageOps.exif_transpose(source)
                if "A" in upright.getbands() or "transparency" in upright.info:
                    rgba = upright.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    return Image.alpha_composite(background, rgba).convert("RGB")
                return upright.convert("RGB")
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(size_error) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "This file could not be read as a complete image. It may be corrupted "
            "or contain a different file type. Try another image or export it again."
        ) from exc
