"""Private download, integrity and failure handling without network requests."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import model_download
from app.config import HF_REPO_ID, setting
from app.model import ModelUnavailableError, checkpoint_signature
from app.model_download import CheckpointDownloadError, download_checkpoint, remote_checkpoint


@pytest.fixture
def remote_file(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    path.write_bytes(b"test checkpoint bytes, never loaded as a model")
    monkeypatch.setenv("HF_TOKEN", "test-token-only")
    monkeypatch.setenv("HF_MODEL_SHA256", sha256(path.read_bytes()).hexdigest())
    mocked = Mock(return_value=str(path))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", mocked)
    monkeypatch.setattr(model_download, "sleep", Mock())
    download_checkpoint.clear()
    yield path, mocked
    download_checkpoint.clear()


def test_download_is_verified_and_cached(remote_file):
    path, download = remote_file
    assert remote_checkpoint() == path
    assert remote_checkpoint() == path
    assert download.call_count == 1
    assert download.call_args.kwargs == dict(
        repo_id=HF_REPO_ID, filename="best.pt", repo_type="model",
        revision="main", token="test-token-only", etag_timeout=10,
    )


def test_token_rotation_invalidates_cache(remote_file, monkeypatch):
    _, download = remote_file
    remote_checkpoint()
    monkeypatch.setenv("HF_TOKEN", "rotated-test-token")
    remote_checkpoint()
    assert download.call_count == 2


def test_revision_change_invalidates_cache(remote_file, monkeypatch):
    _, download = remote_file
    remote_checkpoint()
    monkeypatch.setenv("HF_MODEL_REVISION", "a" * 40)
    remote_checkpoint()
    assert download.call_count == 2
    assert download.call_args.kwargs["revision"] == "a" * 40


def test_missing_token_makes_no_network_request(remote_file, monkeypatch):
    _, download = remote_file
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(CheckpointDownloadError, match="access is not configured"):
        remote_checkpoint()
    download.assert_not_called()


def test_secrets_setting_and_environment_precedence(monkeypatch):
    monkeypatch.setattr(st, "secrets", {"HF_TOKEN": "secret-test-value"})
    assert setting("HF_TOKEN") == "secret-test-value"
    monkeypatch.setenv("HF_TOKEN", "environment-test-value")
    assert setting("HF_TOKEN") == "environment-test-value"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_access_failure_is_safe_and_not_retried(remote_file, status):
    _, download = remote_file
    error = RuntimeError("sensitive response test-token-only")
    error.response = SimpleNamespace(status_code=status)
    download.side_effect = error
    with pytest.raises(CheckpointDownloadError, match="cannot be accessed") as failure:
        remote_checkpoint()
    assert "test-token" not in str(failure.value)
    assert download.call_count == 1


def test_transient_failure_retries_then_caches_success(remote_file):
    path, download = remote_file
    download.side_effect = [TimeoutError("connection lost"), str(path)]
    assert remote_checkpoint() == path
    assert remote_checkpoint() == path
    assert download.call_count == 2


def test_failed_download_can_recover_on_next_rerun(remote_file):
    path, download = remote_file
    download.side_effect = TimeoutError("connection lost")
    with pytest.raises(CheckpointDownloadError, match="temporarily unavailable"):
        remote_checkpoint()
    assert download.call_count == 2
    download.side_effect = None
    assert remote_checkpoint() == path


def test_different_weights_fail_closed(remote_file):
    path, _ = remote_file
    path.write_bytes(b"different model")
    with pytest.raises(CheckpointDownloadError, match="does not match"):
        remote_checkpoint()


def test_invalid_checksum_does_not_download(remote_file, monkeypatch):
    _, download = remote_file
    monkeypatch.setenv("HF_MODEL_SHA256", "invalid")
    with pytest.raises(CheckpointDownloadError, match="checksum configuration"):
        remote_checkpoint()
    download.assert_not_called()


def test_remote_source_flows_into_model_signature(remote_file, monkeypatch):
    path, _ = remote_file
    monkeypatch.setenv("HELMET_MODEL_SOURCE", "huggingface")
    assert checkpoint_signature()[0] == str(path)


def test_download_error_does_not_fall_back_to_local_weights(monkeypatch):
    monkeypatch.setenv("HELMET_MODEL_SOURCE", "huggingface")
    with pytest.raises(ModelUnavailableError, match="access is not configured"):
        checkpoint_signature()


def test_invalid_source_is_rejected(monkeypatch):
    monkeypatch.setenv("HELMET_MODEL_SOURCE", "typo")
    with pytest.raises(ModelUnavailableError, match="source configuration"):
        checkpoint_signature()


def test_cloud_startup_without_token_is_usable_but_cannot_detect(monkeypatch):
    monkeypatch.setenv("HELMET_MODEL_SOURCE", "huggingface")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py")).run(timeout=15)
    assert not app.exception
    assert app.button(key="detect_button").disabled
    assert app.get("download_button")[0].proto.disabled
    assert any("Model access is not configured" in info.value for info in app.info)
