"""Model-independent result presentation. No inference or example predictions."""

from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from math import isfinite

from PIL import Image
import streamlit as st


@dataclass(frozen=True)
class Detection:
    """One detection, using the model's class name and a confidence in [0, 1]."""

    class_name: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError("A detection must have a class name.")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be a finite number between 0 and 1.")


@dataclass(frozen=True)
class PredictionResult:
    """A completed prediction supplied by future inference code.

    An empty tuple means inference completed with no detections. None at the
    rendering boundary means no completed result. The caller must discard a
    result when its source image or internal inference configuration changes.
    """

    detections: tuple[Detection, ...]
    annotated_image: Image.Image


TARGET_CLASSES = (
    ("person", "Detected people", "people-card"),
    ("helmet", "Detected helmets", "helmet-card"),
    ("no_helmet", "Detected uncovered heads", "uncovered-card"),
)


def detection_counts(result: PredictionResult) -> dict[str, int]:
    """Count objects by name; never infer a person's compliance."""
    return dict(Counter(detection.class_name for detection in result.detections))


def detection_rows(result: PredictionResult) -> list[dict[str, str | int]]:
    return [
        {"Detection": index, "Class": detection.class_name,
         "Confidence": f"{detection.confidence:.1%}"}
        for index, detection in enumerate(result.detections, start=1)
    ]


def annotated_png(image: Image.Image) -> bytes:
    """Encode the supplied annotation in memory, without writing a file."""
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def render_prediction_status(
    result: PredictionResult | None = None,
    *,
    loading: bool = False,
    error_message: str | None = None,
) -> None:
    """Render status; error_message must be user-facing, not a raw traceback."""
    if error_message is not None:
        st.error(f"Prediction failed. {error_message}")
    elif loading:
        st.status("Running detection… Please wait.", state="running")
    elif result is None:
        st.info("Model not connected. No predictions have been run.", icon="ℹ️")
    elif not result.detections:
        st.info("Prediction complete. No objects were detected in this image.")
    else:
        st.success("Prediction complete. Review the image and detection details below.")


def render_results(
    result: PredictionResult | None = None,
    *,
    loading: bool = False,
    error_message: str | None = None,
) -> None:
    """Render real supplied data, or explicit unavailable/loading/error states.

    Loading and errors suppress any previous result, including its download.
    The running Step 4 application calls this with no result. Test fixtures
    belong only in tests, never in this module or the application entrypoint.
    """
    active_result = None if loading or error_message is not None else result
    if result is not None or loading or error_message is not None:
        render_prediction_status(result, loading=loading, error_message=error_message)

    counts = detection_counts(active_result) if active_result is not None else None
    pending = "Prediction failed" if error_message is not None else "Prediction in progress" if loading else "Awaiting prediction"
    cards = []
    for class_name, label, css_class in TARGET_CLASSES:
        value = str(counts.get(class_name, 0)) if counts is not None else "—"
        accessible_label = f"{value} detected" if counts is not None else "Count unavailable"
        description = "Detected objects" if counts is not None else pending
        cards.append(f'''<div class="summary-card {css_class}"><dt><span class="class-dot" aria-hidden="true"></span>{label}</dt>
          <dd><span aria-label="{accessible_label}">{value}</span><small>{description}</small></dd></div>''')
    note = "Completed prediction" if active_result is not None else "Results are unavailable until a prediction completes."
    st.html(f'''<section class="detection-summary" aria-labelledby="summary-title">
        <div class="results-heading"><div><span class="section-kicker">DETECTION SUMMARY</span>
          <h2 id="summary-title">A place for every finding.</h2></div><span class="summary-note">{note}</span></div>
        <dl class="summary-grid">{"".join(cards)}</dl>
        <p class="count-explanation">Counts represent detected objects, not a per-worker compliance assessment.</p>
      </section>''')
    if counts is not None:
        other_count = sum(value for name, value in counts.items() if name not in {item[0] for item in TARGET_CLASSES})
        if other_count:
            st.info(f"{other_count} detections have other class names. They appear in the details below.")

    with st.container(border=True, key="result_panel"):
        st.html('''<div class="panel-heading"><span class="step-number">04</span>
          <div><h3>Annotated result</h3><p>Review and download the prediction image.</p></div></div>''')
        if active_result is None:
            title = "Result unavailable" if error_message is not None else "Preparing your result" if loading else "No prediction yet"
            message = (
                "Resolve the prediction error and try again." if error_message is not None
                else "The annotated image will appear when detection finishes." if loading
                else "The trained model must be connected before an annotated image can be generated."
            )
            st.html(f'''<div class="result-placeholder"><h4>{title}</h4><p>{message}</p></div>''')
            st.download_button(
                "Download annotated image", data=b"", file_name="helmet-detection.png",
                mime="image/png", disabled=True, key="download_result", width="stretch",
                help="Available after a completed prediction produces an image.",
            )
        else:
            st.image(active_result.annotated_image, width="stretch")
            if not active_result.detections:
                st.caption("No bounding boxes were returned. This does not establish whether a workplace is safe.")
            try:
                png_data = annotated_png(active_result.annotated_image)
            except (OSError, ValueError):
                st.error("The image could not be prepared for download. Run detection again to retry.")
            else:
                st.download_button(
                    "Download annotated image", data=png_data, file_name="helmet-detection.png",
                    mime="image/png", key="download_result", width="stretch", on_click="ignore",
                )

    with st.container(border=True, key="details_panel"):
        st.html('''<div class="panel-heading"><span class="step-number">05</span>
          <div><h3>Detection details</h3><p>Individual class labels and confidence scores.</p></div></div>''')
        if active_result is None:
            st.caption("Detection details are unavailable until a prediction completes.")
        elif not active_result.detections:
            st.info("No detections to list. You can try another clear, well-lit image.")
        else:
            st.dataframe(detection_rows(active_result), hide_index=True, width="stretch")
            st.caption("Confidence is the model's score for each detection, not a guarantee of correctness.")
