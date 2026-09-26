"""Presentation for the image-upload and detection dashboard preview."""

from base64 import b64encode
from pathlib import Path
import re

from PIL import Image
import streamlit as st

from app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
)
from app.image_utils import ImageValidationError, prepare_image
from app.results import render_prediction_status, render_results

ASSET_DIR = Path(__file__).parent


def _render_html(markup: str) -> None:
    """Embed our static SVG artwork as images, compatible with st.html."""
    def embed_svg(match: re.Match[str]) -> str:
        svg = match.group().replace('stroke="currentColor"', 'stroke="#8a9b89"')
        if 'xmlns=' not in svg:
            svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        encoded = b64encode(svg.encode("utf-8")).decode("ascii")
        return f'<img class="vector-art" src="data:image/svg+xml;base64,{encoded}" alt="">'

    st.html(re.sub(r"<svg\b.*?</svg>", embed_svg, markup, flags=re.DOTALL))


def render_header() -> None:
    _render_html('''
        <header class="site-header">
          <div class="brand">
            <span class="brand-icon" aria-hidden="true">
              <svg viewBox="0 0 32 32" fill="none"><path d="M6 21a10 10 0 0 1 20 0M4 22h24v4H4zM13 19V8h6v11"
                stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/></svg>
            </span>
            <div><strong>Worker Safety<span class="brand-dot">.</span></strong>
              <span class="brand-subtitle">HELMET DETECTION</span></div>
          </div>
        </header>
    ''')
    illustration = (ASSET_DIR / "assets" / "helmet.svg").read_text(encoding="utf-8")
    _render_html(f'''
        <section class="hero" aria-labelledby="hero-title">
          <div class="hero-copy">
            <div class="eyebrow"><span></span> A CLOSER LOOK AT WORKPLACE SAFETY</div>
            <h1 id="hero-title">Safety starts with<br><em>better visibility.</em></h1>
            <p>A fresh perspective on workplace safety through computer vision.
              Start with an image. Explore the details.</p>
            <div class="hero-foot"><span class="hero-line"></span> Worker Safety Helmet Detection using YOLOv8</div>
          </div>
          <div class="hero-art">{illustration}</div>
        </section>
        <div class="availability" role="status">
          <span class="status-dot" aria-hidden="true"></span>
          <p><strong>Image preview is ready.</strong> Model integration pending — detection is not available yet.</p>
          <span class="preview-tag">INTERFACE PREVIEW</span>
        </div>
        <div class="workspace-heading" id="workspace">
          <div><span class="section-kicker">YOUR WORKSPACE</span><h2>Every detail starts with an image.</h2></div>
          <span class="workspace-note">Upload. Preview. Explore.</span>
        </div>
    ''')


def render_upload_panel() -> Image.Image | None:
    _render_html('''<div class="panel-heading"><span class="step-number">01</span>
      <div><h3>Upload an image</h3><p>Bring your workplace into focus.</p></div></div>''')
    uploaded_file = st.file_uploader(
        "Choose a workplace image",
        type=list(ALLOWED_IMAGE_EXTENSIONS),
        accept_multiple_files=False,
        max_upload_size=MAX_UPLOAD_MB,
        key="workplace_image",
        help=(
            f"JPEG/JPG, PNG, or WebP. Maximum {MAX_UPLOAD_MB} MB, "
            f"{MAX_IMAGE_PIXELS:,} pixels total, and {MAX_IMAGE_SIDE:,} pixels "
            "on either side. Animated images use the first frame."
        ),
    )
    _render_html(f'''<div class="file-specs"><span>JPG / PNG / WEBP</span>
        <span>UP TO {MAX_UPLOAD_MB} MB</span></div>''')

    image = None
    if uploaded_file is not None:
        st.text(f"File: {uploaded_file.name}")
        try:
            if uploaded_file.size > MAX_UPLOAD_BYTES:
                raise ImageValidationError(
                    f"This file exceeds the {MAX_UPLOAD_MB} MB limit. Choose a smaller image."
                )
            with st.spinner("Checking your image…"):
                image = prepare_image(uploaded_file.getvalue(), uploaded_file.name)
        except ImageValidationError as exc:
            st.error(str(exc))

    _render_html('''<div class="upload-guide">
        <div class="guide-title">A little clarity goes a long way.</div>
        <p>Choose a well-lit image where people and their headwear are clearly visible.</p>
        <div class="privacy-note"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z" stroke="currentColor" stroke-width="1.5"/>
          <path d="m8 12 3 3 5-6" stroke="currentColor" stroke-width="1.5"/>
        </svg><span>Images stay in memory. Never saved to disk.</span></div>
      </div>''')
    st.caption("Choose another file to replace your image, or select × beside its name to remove it.")
    return image


def render_preview_panel(image: Image.Image | None) -> None:
    _render_html('''<div class="panel-heading"><span class="step-number">02</span>
      <div><h3>Original image</h3><p>Your image, before any detection.</p></div></div>''')
    if image is None:
        _render_html('''<div class="empty-preview">
            <div class="image-symbol" aria-hidden="true">
              <svg viewBox="0 0 80 80" fill="none">
                <rect x="12" y="14" width="56" height="52" rx="8" stroke="currentColor" stroke-width="2"/>
                <circle cx="29" cy="31" r="5" stroke="currentColor" stroke-width="2"/>
                <path d="m13 55 17-15 12 11 10-9 15 14" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
              </svg>
            </div><h4>A new perspective awaits.</h4>
            <p>Upload a valid image to see it here.</p>
            <span class="empty-label">ORIGINAL IMAGE PREVIEW</span>
          </div>''')
    else:
        st.image(image, width="stretch")
        st.caption(f"{image.width:,} × {image.height:,} pixels · RGB · Ready to preview")
        st.caption("Orientation corrected where needed. Transparency displayed on white.")


def render_homepage() -> None:
    st.html(ASSET_DIR / "styles.css")
    render_header()
    with st.container(key="image_workspace"):
        upload_column, preview_column = st.columns([1, 1], gap="large")
        with upload_column:
            with st.container(border=True, key="upload_panel"):
                image = render_upload_panel()
        with preview_column:
            with st.container(border=True, key="preview_panel"):
                render_preview_panel(image)

    render_detection_dashboard(image_available=image is not None)

    _render_html('''<footer class="site-footer">
        <span>Built for a safer tomorrow<span class="brand-dot">.</span></span>
        <span>Powered by Streamlit</span>
      </footer>''')


def render_detection_dashboard(image_available: bool) -> None:
    """Show controls and explicit unavailable states until model integration."""
    with st.container(border=True, key="detection_panel"):
        _render_html('''<div class="panel-heading"><span class="step-number">03</span>
          <div><h3>Helmet detection</h3><p>Upload an image, then detect and review the results.</p></div></div>''')
        action_column, status_column = st.columns([1, 1], gap="large")
        with action_column:
            st.button(
                "Detect Helmets",
                key="detect_button",
                type="primary",
                disabled=True,
                width="stretch",
                help="Detection is unavailable until the trained model is connected.",
            )
        with status_column:
            _render_html('''<div class="prediction-status-heading"><h4>Prediction status</h4>
              <span class="pending-badge">UNAVAILABLE</span></div>''')
            render_prediction_status()
            if image_available:
                st.caption("Image validated. Detection will be available once the trained model is connected.")
            else:
                st.caption("Upload a valid image to prepare the workspace. Model integration is still pending.")

    render_results()
