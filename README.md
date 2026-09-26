# worker-safety-helmet-detection
This project develops an object-detection system for workplace and construction-site safety monitoring using YOLOv8. The goal is to identify workers and determine whether visible workers are wearing safety helmets. The project covers the complete CV workflow, including manual annotation, model training, and a simple web-based demonstration.

## Streamlit application

The current application is an **interface preview**. Image uploads and validation
work; detection and annotated-image downloads remain disabled until the selected
trained checkpoint is integrated. No predictions are simulated and no model is
downloaded. Counts will represent detected objects, not an automatic assessment
of each worker's compliance.


## Live Demo

The application is deployed on Streamlit Community Cloud.

**Live App:** https://worker-safety-helmet-detection.streamlit.app/

Current status: The application interface is deployed and publicly accessible.
YOLOv8 model integration is pending.

Users can currently explore the interface and upload images.
Real helmet detection will be enabled after the trained model is connected.

## Deployment

- Platform: Streamlit Community Cloud
- Source: GitHub repository
- Entry point: `streamlit_app.py`
- Model: Custom-trained YOLOv8 (integration pending)

### Local Windows setup

Tested with Python 3.13. In PowerShell, from your existing project folder:

```powershell
Set-Location D:\worker-safety-helmet-detection
# Create the environment only if it does not already exist:
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://localhost:8501 and keep the terminal running. Press Ctrl+C to stop.
If the site cannot be reached, check that the command is still running and use
the Local URL printed in the terminal. Environment activation is not required.

### Using the preview

Upload a JPG/JPEG, PNG or WebP image to see its original-image preview. Replace
the file to change the image, or remove it using the uploader's × control.
Files are limited to 10 MB, 20 million pixels and 8,192 pixels on either side.
Corrupt and unsupported files show a validation error. EXIF orientation is
corrected, transparency is displayed on white, and animated images use their
first frame. Uploaded images are processed in memory, without permanent storage.

The interface has no confidence control. The internal inference default is
`DEFAULT_CONFIDENCE_THRESHOLD = 0.25` in `app/config.py`; Step 8 inference must
import this value. Individual detection confidence scores will still appear
when real results are connected. Project and experiment sections are omitted
from the public interface by design.

### Development checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover image formats, invalid and oversized files, dimension boundaries,
orientation, transparency, result formatting, PNG encoding, upload replacement
and removal, missing-model startup, and loading/error/empty result states.
Synthetic fixtures are confined to tests; tests do not read training datasets,
download weights or run inference. Pytest is a development dependency only.

For manual verification:

1. Upload a supported image and check its preview and filename.
2. Replace it with a corrupt image and check that an error replaces the preview.
3. Upload a valid image again, then remove it and check that the preview clears.
4. Confirm Detect Helmets and Download annotated image stay disabled, counts
   show unavailable placeholders, and no confidence slider appears.
5. Resize to a narrow mobile window: panels and summary cards should stack
   without horizontal page scrolling. Use Tab to reach the uploader and check
   its visible focus indicator and label.

Model integration and public deployment remain separate future milestones.
