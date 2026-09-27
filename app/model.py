"""Load the selected local or downloaded checkpoint and cache its resource."""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import streamlit as st

from app.config import inference_device, model_path, setting
from app.model_download import CheckpointDownloadError, remote_checkpoint


class ModelUnavailableError(RuntimeError):
    """A safe message suitable for the public interface."""


@dataclass
class LoadedModel:
    model: Any
    device: str
    names: dict[int, str]
    lock: Any = field(default_factory=Lock)


def checkpoint_signature() -> tuple[str, int, int, str]:
    source = setting("HELMET_MODEL_SOURCE", "local").lower()
    if source == "huggingface":
        try:
            path = remote_checkpoint()
        except CheckpointDownloadError as exc:
            raise ModelUnavailableError(str(exc)) from None
    elif source == "local":
        path = model_path().resolve()
    else:
        raise ModelUnavailableError("The model source configuration is invalid. Contact the app owner.")
    try:
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise ModelUnavailableError("Model not connected. The trained checkpoint is unavailable.")
        stat = path.stat()
        if not stat.st_size:
            raise ModelUnavailableError("The trained checkpoint is empty. Replace it with the selected model.")
        return str(path), stat.st_size, stat.st_mtime_ns, inference_device()
    except OSError as exc:
        raise ModelUnavailableError("The trained checkpoint could not be accessed.") from exc


@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(path: str, size: int, modified_ns: int, device: str) -> LoadedModel:
    """File metadata is part of the cache key, invalidating replaced checkpoints."""
    # Check immediately before YOLO(): a missing filename must never trigger
    # Ultralytics' automatic download of a similarly named pretrained model.
    if not Path(path).is_file():
        raise ModelUnavailableError("Model not connected. The trained checkpoint is unavailable.")
    try:
        import torch
        from ultralytics import YOLO

        if device != "cpu":
            if not device.isdigit() or not torch.cuda.is_available() or int(device) >= torch.cuda.device_count():
                raise ModelUnavailableError("The configured GPU is unavailable. Configure CPU inference or a valid CUDA device.")
        model = YOLO(path, task="detect")
        if model.task != "detect":
            raise ModelUnavailableError("The checkpoint must be an object-detection model.")
        names = {int(key): str(value) for key, value in model.names.items()}
        if set(names.values()) != {"person", "helmet", "no_helmet"} or len(names) != 3:
            raise ModelUnavailableError("The checkpoint must contain person, helmet and no_helmet classes.")
        model.to("cpu" if device == "cpu" else f"cuda:{device}")
        return LoadedModel(model, device, names)
    except ModelUnavailableError:
        raise
    except ImportError as exc:
        raise ModelUnavailableError("Model dependencies are missing. Install the application requirements and restart.") from exc
    except Exception as exc:
        raise ModelUnavailableError("The trained model could not be loaded. Check the checkpoint and installed dependencies, then restart.") from exc
