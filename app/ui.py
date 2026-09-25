"""Homepage presentation for the initial application preview."""

import streamlit as st

from app.config import APP_TITLE


def render_homepage() -> None:
    st.caption("WORKPLACE SAFETY  /  COMPUTER VISION")
    st.title(APP_TITLE)
    st.markdown("### A clearer view of workplace safety.")
    st.write(
        "A university AI/ML internship project exploring how YOLOv8 can "
        "help identify people, safety helmets, and visible uncovered heads "
        "in construction and workplace images."
    )

    st.divider()

    with st.container(border=True):
        st.subheader("Application preview")
        st.write(
            "This is the initial application homepage. The completed "
            "experience will let you upload a workplace image and review "
            "model detections with bounding boxes and confidence scores."
        )
        st.info(
            "Model integration pending. Image upload and detection are "
            "not available in this preview. No predictions are being generated.",
            icon="ℹ️",
        )

    st.caption("Built with Streamlit · Worker Safety Helmet Detection using YOLOv8")
