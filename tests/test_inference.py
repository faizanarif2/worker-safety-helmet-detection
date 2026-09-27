"""Inference boundary tests use doubles only inside the test suite."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from app.inference import InferenceError, predict
from app.model import LoadedModel, ModelUnavailableError, checkpoint_signature, load_model
from app.results import Detection, detection_counts


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def model_double(empty=False):
    boxes = SimpleNamespace(cls=Tensor([] if empty else [7, 2]),
                            conf=Tensor([] if empty else [0.8, 0.9]),
                            xyxy=Tensor([] if empty else [[1, 2, 15, 16], [20, 20, 40, 40]]))
    model = Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    return LoadedModel(model, "cpu", {7: "helmet", 2: "no_helmet", 9: "person"})


def test_inference_uses_checkpoint_names_and_fixed_confidence():
    resource = model_double()
    original = Image.new("RGB", (64, 64), "white")
    result = predict(original, resource)
    assert detection_counts(result) == {"helmet": 1, "no_helmet": 1}
    assert result.detections[0].xyxy == (1, 2, 15, 16)
    assert result.annotated_image.size == original.size
    assert result.annotated_image.tobytes() != original.tobytes()
    assert original.getpixel((1, 2)) == (255, 255, 255)
    arguments = resource.model.predict.call_args.kwargs
    assert arguments["conf"] == 0.25
    assert arguments["device"] == "cpu"
    assert not any(arguments[key] for key in ("save", "save_txt", "save_crop", "show", "stream"))
    assert resource.model.predictor is None


def test_empty_prediction_preserves_image_colors():
    image = Image.new("RGB", (32, 32), (255, 0, 10))
    result = predict(image, model_double(empty=True))
    assert result.detections == ()
    assert np.array_equal(np.array(result.annotated_image), np.array(image))


def test_inference_failure_clears_cached_predictor_and_hides_internal_details():
    resource = model_double()
    resource.model.predict.side_effect = RuntimeError("private internal path")
    with pytest.raises(InferenceError, match="Try a smaller image") as error:
        predict(Image.new("RGB", (32, 32)), resource)
    assert "private" not in str(error.value)
    assert resource.model.predictor is None


def test_shared_model_calls_are_serialized():
    resource = model_double(empty=True)
    original_call = resource.model.predict
    guard = Lock()

    def assert_locked(**kwargs):
        assert resource.lock.locked()
        assert guard.acquire(blocking=False)
        try:
            return original_call(**kwargs)
        finally:
            guard.release()

    resource.model.predict = assert_locked
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: predict(Image.new("RGB", (32, 32)), resource), range(4)))
    assert len(results) == 4


def test_missing_checkpoint_is_rejected_before_import_or_download(monkeypatch):
    monkeypatch.setenv("HELMET_MODEL_PATH", "models/nonexistent-test.pt")
    with pytest.raises(ModelUnavailableError, match="not connected"):
        checkpoint_signature()
    with pytest.raises(ModelUnavailableError, match="not connected"):
        load_model("models/nonexistent-test.pt", 1, 1, "cpu")


@pytest.mark.parametrize("box", [(0, 0, -1, 10), (0, 0, float("nan"), 10)])
def test_invalid_boxes_are_rejected(box):
    with pytest.raises(ValueError, match="Bounding box"):
        Detection("helmet", 0.5, box)
