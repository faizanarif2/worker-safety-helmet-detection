"""Tests never read a developer's real deployment credentials or secrets."""

import pytest
import streamlit as st


@pytest.fixture(autouse=True)
def isolated_deployment_settings(monkeypatch):
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("HELMET_MODEL_SOURCE", "local")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_MODEL_REVISION", raising=False)
    monkeypatch.delenv("HF_MODEL_SHA256", raising=False)
