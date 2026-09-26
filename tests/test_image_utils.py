"""Small in-memory images exercise upload validation without dataset access."""

from io import BytesIO

from PIL import Image
import pytest

from app import image_utils
from app.config import MAX_IMAGE_SIDE, MAX_UPLOAD_BYTES
from app.image_utils import ImageValidationError, prepare_image


def encode(image, format="PNG", **options):
    output = BytesIO()
    image.save(output, format=format, **options)
    return output.getvalue()


@pytest.mark.parametrize("format,extension", [("JPEG", "jpg"), ("JPEG", "JPEG"), ("PNG", "png"), ("WEBP", "webp")])
def test_supported_formats_return_independent_rgb_image(format, extension):
    prepared = prepare_image(encode(Image.new("RGB", (8, 6)), format), f"workplace.{extension}")
    assert prepared.mode == "RGB"
    assert prepared.size == (8, 6)
    assert prepared.getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.parametrize("data,name,message", [
    (b"", "empty.png", "empty"),
    (b"not an image", "bad.jpg", "could not be read"),
    (b"<svg></svg>", "image.svg", "Unsupported file type"),
    (b"text", "no-extension", "Unsupported file type"),
])
def test_invalid_uploads_have_actionable_errors(data, name, message):
    with pytest.raises(ImageValidationError, match=message):
        prepare_image(data, name)


def test_renaming_unsupported_contents_does_not_bypass_validation():
    with pytest.raises(ImageValidationError, match="file contents are not"):
        prepare_image(encode(Image.new("RGB", (8, 6)), "BMP"), "renamed.png")


def test_truncated_image_is_rejected():
    data = encode(Image.new("RGB", (32, 32)), "JPEG")
    with pytest.raises(ImageValidationError, match="could not be read"):
        prepare_image(data[:-30], "truncated.jpg")


def test_file_size_limit_precedes_decoding():
    with pytest.raises(ImageValidationError, match="exceeds the 10 MB limit"):
        prepare_image(b"x" * (MAX_UPLOAD_BYTES + 1), "oversized.png")


def test_exact_file_size_limit_is_accepted(monkeypatch):
    data = encode(Image.new("RGB", (8, 6)))
    monkeypatch.setattr(image_utils, "MAX_UPLOAD_BYTES", len(data))
    assert prepare_image(data, "boundary.png").size == (8, 6)


@pytest.mark.parametrize("width,accepted", [(MAX_IMAGE_SIDE, True), (MAX_IMAGE_SIDE + 1, False)])
def test_side_length_boundary(width, accepted):
    data = encode(Image.new("RGB", (width, 1)))
    if accepted:
        assert prepare_image(data, "wide.png").width == width
    else:
        with pytest.raises(ImageValidationError, match="too large"):
            prepare_image(data, "wide.png")


@pytest.mark.parametrize("limit,accepted", [(48, True), (47, False)])
def test_total_pixel_boundary(monkeypatch, limit, accepted):
    monkeypatch.setattr(image_utils, "MAX_IMAGE_PIXELS", limit)
    data = encode(Image.new("RGB", (8, 6)))
    if accepted:
        assert prepare_image(data, "pixels.png").size == (8, 6)
    else:
        with pytest.raises(ImageValidationError, match="too large"):
            prepare_image(data, "pixels.png")


def test_pillow_decompression_warning_becomes_validation_error(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 30)
    with pytest.raises(ImageValidationError, match="too large"):
        prepare_image(encode(Image.new("RGB", (8, 6))), "large.png")


def test_exif_orientation_corrects_dimensions_and_pixels():
    source = Image.new("RGB", (2, 3), "red")
    source.putpixel((0, 0), (0, 0, 255))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    prepared = prepare_image(encode(source, exif=exif), "rotated.png")
    assert prepared.size == (3, 2)
    assert prepared.getpixel((2, 0)) == (0, 0, 255)
    assert prepared.getexif().get(274) is None


@pytest.mark.parametrize("alpha,expected", [(0, (255, 255, 255)), (128, (255, 127, 127)), (255, (255, 0, 0))])
def test_alpha_is_composited_onto_white(alpha, expected):
    prepared = prepare_image(encode(Image.new("RGBA", (2, 2), (255, 0, 0, alpha))), "alpha.png")
    assert prepared.mode == "RGB"
    assert prepared.getpixel((0, 0)) == expected


def test_palette_transparency_is_composited_onto_white():
    source = Image.new("P", (2, 2), 0)
    prepared = prepare_image(encode(source, transparency=0), "palette.png")
    assert prepared.getpixel((0, 0)) == (255, 255, 255)


def test_grayscale_converts_to_rgb():
    prepared = prepare_image(encode(Image.new("L", (2, 2), 100)), "gray.png")
    assert prepared.getpixel((0, 0)) == (100, 100, 100)
