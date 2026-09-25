"""Streamlit entrypoint for the Worker Safety Helmet Detection application."""

import streamlit as st

from app.config import APP_ICON, APP_TITLE
from app.ui import render_homepage


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
    )
    render_homepage()


if __name__ == "__main__":
    main()
