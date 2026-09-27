# worker-safety-helmet-detection
This project develops an object-detection system for workplace and construction-site safety monitoring using YOLOv8. The goal is to identify workers and determine whether visible workers are wearing safety helmets. The project covers the complete CV workflow, including manual annotation, model training, and a simple web-based demonstration.

## Streamlit application

The application runs real YOLOv8 detection using your selected
`models/best.pt` checkpoint locally, or the same checkpoint downloaded from your
private Hugging Face repository when configured for deployment. Upload an image, select **Detect Helmets**, review
class counts and confidence scores, and download the annotated PNG. No
predictions are simulated and no replacement weights are downloaded. Counts
represent detected objects, not an automatic assessment of each worker's compliance.
Without a usable checkpoint, uploads still work and detection stays unavailable.


## Live Demo

The application is deployed on Streamlit Community Cloud.

**Live App:** https://worker-safety-helmet-detection.streamlit.app/

Current status: The application interface is deployed and publicly accessible.
Private model download support is implemented. Cloud secrets, the owner-managed
Git push and public inference verification are still pending (Step 9).

Users can currently explore the interface and upload images.
Real helmet detection will be enabled after the trained model is connected.

## Deployment

- Platform: Streamlit Community Cloud
- Source: GitHub repository
- Entry point: `streamlit_app.py`
- Model: Custom-trained YOLOv8 (local or authenticated Hugging Face download)

Follow [the deployment guide](docs/deployment.md) for exact token permissions,
Streamlit Secrets, troubleshooting and the public verification checklist.

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

### Using the application

Upload a JPG/JPEG, PNG or WebP image to see its original-image preview. Replace
the file to change the image, or remove it using the uploader's × control.
Files are limited to 10 MB, 20 million pixels and 8,192 pixels on either side.
Corrupt and unsupported files show a validation error. EXIF orientation is
corrected, transparency is displayed on white, and animated images use their
first frame. Uploaded images are processed in memory, without permanent storage.

The interface has no confidence control. The internal inference default is
`DEFAULT_CONFIDENCE_THRESHOLD = 0.25` in `app/config.py`; inference imports
this value directly. Individual detection confidence scores appear with real
results. Project and experiment sections are omitted
from the public interface by design.

### Local model configuration

Keep the selected checkpoint at `models/best.pt` (excluded from Git). Class IDs
come from the checkpoint: its three class names must be `person`, `helmet`, and
`no_helmet`. No numeric ordering is assumed. The model resource is cached; a
lock serializes access across sessions. Images and results are held in each
visitor's session, and replacing or removing an image clears its result.
The predictor is released after each call so the shared resource retains model
weights without retaining the last visitor's image. Bounding boxes are drawn
in memory with Pillow from the model's actual coordinates and scores.

Local integration was checked with the supplied checkpoint on CPU using the
repository's existing annotation screenshot. Model loading, real inference,
PNG encoding and the Streamlit detection/removal flow passed. This screenshot
is a functional smoke test, not a validation-set accuracy measurement. Evaluate
detection quality separately on original workplace images. GPU inference has
not been verified on this machine.

CPU inference is the default. Optional environment overrides, set before launch:

```powershell
$env:HELMET_MODEL_PATH = "models/best.pt"
$env:HELMET_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

An optional CUDA GPU uses `HELMET_DEVICE="0"` and requires a compatible GPU and
CUDA-enabled PyTorch installation. An unavailable configured GPU produces a
clear error. Relative model paths resolve from this repository. After changing
configuration or replacing weights, restart Streamlit. Never commit checkpoints.
The local checkpoint is not automatically available on Streamlit Community Cloud.

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
4. With the checkpoint installed, upload a valid image and click Detect Helmets.
   Check the annotated image, separate class counts and individual confidence
   scores. Download the PNG and verify it matches the displayed annotation.
   Replace or remove the input and confirm old results and downloads clear.
   Without a checkpoint, detection and downloads must remain disabled. No
   confidence slider should appear in either case.
5. Resize to a narrow mobile window: panels and summary cards should stack
   without horizontal page scrolling. Use Tab to reach the uploader and check
   its visible focus indicator and label.

### Limitations and future improvements

The detector can miss small or occluded heads and produce incorrect labels.
When `helmet` and `no_helmet` boxes overlap with intersection-over-union of at
least 0.5, post-processing keeps the higher-confidence label. Person boxes are
excluded from this rule. Counts, details and annotations all use the filtered
detections. The developer-controlled threshold is `HEAD_CONFLICT_IOU_THRESHOLD`
in `app/config.py`. This heuristic removes conflicting labels but does not
correct classification errors; tightly overlapping different heads may also
be merged, and poorly aligned duplicate boxes may remain.
Counts describe separate detected objects; they do not establish workplace
safety or automatically associate a helmet with a particular person. Training
metrics are not fabricated or displayed as live prediction guarantees.

Public CPU speed, resource usage and multi-user behavior must be verified after
deployment. Future improvements include evaluating more representative workplace
photos, expanding difficult examples in the dataset, and investigating ONNX only
if measured hosting limits require it. The checkpoint and training remain unchanged.
