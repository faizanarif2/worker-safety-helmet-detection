"""Streamlit regression checks with no checkpoint, inference, or network."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import DEFAULT_CONFIDENCE_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_with_detection_unavailable():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=15)
    assert not app.exception
    assert app.button(key="detect_button").disabled
    assert not app.slider
    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.25
    assert app.get("download_button")[0].proto.disabled
    assert any("Model not connected" in info.value for info in app.info)
    assert not app.dataframe


def test_upload_replace_invalid_and_remove_clear_the_preview():
    # AppTest does not expose a file-uploader driver. Substitute only its return
    # value; run the real validation, preview and dashboard on every rerun.
    app = AppTest.from_string('''
from io import BytesIO
from unittest.mock import patch
import streamlit as st
from PIL import Image
from app.ui import render_homepage
state = st.session_state["upload_state"]
uploaded = None
if state != "removed":
    data = BytesIO()
    if state == "valid":
        Image.new("RGB", (12, 8)).save(data, format="PNG")
    else:
        data.write(b"broken image")
    data.name = "workplace.png"
    data.size = len(data.getvalue())
    uploaded = data
with patch("app.ui.st.file_uploader", return_value=uploaded):
    render_homepage()
''')
    for state in ("valid", "invalid", "valid", "removed"):
        app.session_state["upload_state"] = state
        app.run(timeout=15)
        assert not app.exception
        assert app.button(key="detect_button").disabled
        assert app.get("download_button")[0].proto.disabled
        assert len(app.get("image")) == (1 if state == "valid" else 0)
        assert len(app.error) == (1 if state == "invalid" else 0)


@pytest.mark.parametrize("state", ["empty", "complete", "loading", "error"])
def test_result_states_and_stale_result_suppression(state):
    app = AppTest.from_string('''
import streamlit as st
from PIL import Image
from app.results import Detection, PredictionResult, render_results
state = st.session_state["test_state"]
detections = () if state == "empty" else (Detection("helmet", 0.9),)
result = PredictionResult(detections, Image.new("RGB", (2, 2)))
render_results(result, loading=state == "loading",
               error_message="Try another image." if state == "error" else None)
''')
    app.session_state["test_state"] = state
    app.run(timeout=15)
    assert not app.exception
    assert app.get("download_button")[0].proto.disabled == (state in {"loading", "error"})
    assert len(app.dataframe) == (1 if state == "complete" else 0)
    if state == "empty":
        assert any("No objects were detected" in info.value for info in app.info)
    elif state == "error":
        assert "Try another image" in app.error[0].value
    elif state == "loading":
        assert app.status[0].state == "running"
