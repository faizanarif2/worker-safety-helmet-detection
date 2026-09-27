"""Checkpoint-only loading, cache invalidation and configuration failures."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.model import ModelUnavailableError, checkpoint_signature, load_model


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "chosen.pt"
    path.write_bytes(b"test double, not a checkpoint")
    monkeypatch.setenv("HELMET_MODEL_PATH", str(path))
    monkeypatch.setenv("HELMET_DEVICE", "cpu")
    load_model.clear()
    yield path
    load_model.clear()


def fake_modules(monkeypatch, names=None):
    model = Mock(task="detect", names=names or {8: "person", 3: "helmet", 5: "no_helmet"})
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    return model, factory


def test_cache_reuses_weights_and_invalidates_replaced_file(checkpoint, monkeypatch):
    model, factory = fake_modules(monkeypatch)
    first = load_model(*checkpoint_signature())
    assert load_model(*checkpoint_signature()) is first
    assert factory.call_count == 1
    assert first.names == {8: "person", 3: "helmet", 5: "no_helmet"}
    model.to.assert_called_once_with("cpu")
    checkpoint.write_bytes(b"a replacement test checkpoint with a different length")
    assert load_model(*checkpoint_signature()) is not first
    assert factory.call_count == 2


def test_unexpected_classes_fail_clearly(checkpoint, monkeypatch):
    fake_modules(monkeypatch, {0: "car"})
    with pytest.raises(ModelUnavailableError, match="must contain"):
        load_model(*checkpoint_signature())


def test_unavailable_gpu_does_not_silently_fall_back(checkpoint, monkeypatch):
    _, factory = fake_modules(monkeypatch)
    monkeypatch.setenv("HELMET_DEVICE", "0")
    with pytest.raises(ModelUnavailableError, match="GPU is unavailable"):
        load_model(*checkpoint_signature())
    factory.assert_not_called()


def test_corrupt_checkpoint_has_safe_message(checkpoint, monkeypatch):
    _, factory = fake_modules(monkeypatch)
    factory.side_effect = RuntimeError("internal path and traceback")
    with pytest.raises(ModelUnavailableError, match="could not be loaded") as error:
        load_model(*checkpoint_signature())
    assert "internal path" not in str(error.value)


def test_empty_checkpoint_is_rejected(checkpoint):
    checkpoint.write_bytes(b"")
    with pytest.raises(ModelUnavailableError, match="empty"):
        checkpoint_signature()
