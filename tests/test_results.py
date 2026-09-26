"""Synthetic fixtures stay in tests; no model or training assets are loaded."""

from io import BytesIO

from PIL import Image
import pytest

from app.results import Detection, PredictionResult, annotated_png, detection_counts, detection_rows


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError, match="Confidence"):
        Detection("helmet", confidence)


def test_blank_class_is_rejected():
    with pytest.raises(ValueError, match="class name"):
        Detection("  ", 0.5)


def test_counts_use_names_without_assuming_worker_compliance():
    result = PredictionResult((Detection("person", 1), Detection("person", 0.8),
                               Detection("helmet", 0.9), Detection("other", 0)),
                              Image.new("RGB", (2, 2)))
    assert detection_counts(result) == {"person": 2, "helmet": 1, "other": 1}
    assert "no_helmet" not in detection_counts(result)
    assert detection_rows(result)[1] == {"Detection": 2, "Class": "person", "Confidence": "80.0%"}


def test_empty_result_has_no_rows_or_counts():
    result = PredictionResult((), Image.new("RGB", (2, 2)))
    assert detection_counts(result) == {}
    assert detection_rows(result) == []


def test_download_is_a_lossless_png():
    source = Image.new("RGB", (7, 5), (30, 80, 120))
    with Image.open(BytesIO(annotated_png(source))) as downloaded:
        assert downloaded.format == "PNG"
        assert downloaded.size == source.size
        assert downloaded.tobytes() == source.tobytes()
