# Deploy the trained detector on Streamlit Community Cloud

Public app: https://worker-safety-helmet-detection.streamlit.app/

Private model repository:
https://huggingface.co/faizanarif233/worker-safety-helmet-detection

## Current status

The code supports private checkpoint downloads. Local inference was verified
in Step 8. Public inference is **not yet verified**: the owner must save the
token in Streamlit Secrets, review and push these code changes, and then test
the deployed app. A successful local test does not confirm the cloud build,
private repository access or the hosting service's resource limits.

## 1. Confirm the checkpoint upload

In Hugging Face, open **Files and versions**. The repository root must contain
`best.pt`. Upload the same checkpoint used locally, without renaming or exporting
it. Keep the Google Drive backup. Do not put the checkpoint in the GitHub app repo.

The application verifies the downloaded bytes against the selected local file's
SHA-256 before loading them:

```text
b6a532cc5c38048a9e120184b6187b8037ae20f090562b97f390059f9aa48ca0
```

The file is 22,522,538 bytes. A checksum mismatch disables detection rather than
loading different weights. For a deliberately selected replacement in the future,
test it locally first, then update `MODEL_SHA256` in `app/config.py` or the
`HF_MODEL_SHA256` deployment setting.

## 2. Create a token for this app

Open https://huggingface.co/settings/tokens and create a **fine-grained** token
named `streamlit-helmet-detector`. Grant read access to the contents of the
selected `faizanarif233/worker-safety-helmet-detection` repository. No write or
inference-provider permissions are needed: the model runs on Streamlit's CPU.

Copy the token directly into the Streamlit Secrets editor in the next step.
Never put it in chat, code, screenshots, a commit or a shell command.
See [Hugging Face's token documentation](https://huggingface.co/docs/hub/security-tokens).

## 3. Save Streamlit Secrets

In your Streamlit Community Cloud workspace, open the existing app's **Settings**
and its **Secrets** editor. Paste the following and replace only the token value:

```toml
HELMET_MODEL_SOURCE = "huggingface"
HELMET_DEVICE = "cpu"
HF_MODEL_REVISION = "main"
HF_TOKEN = "REPLACE_WITH_YOUR_READ_ONLY_TOKEN"
```

These are top-level settings, without a section header. Keep any unrelated
existing secrets. Save, and reboot the app after deploying the code. The same
template is in `.streamlit/secrets.toml.example`; that template contains no secret.

For stronger version pinning, replace `main` with the full commit SHA of the
checkpoint upload shown in the Hugging Face repository history. The checksum
still enforces the exact model when `main` is used.

See [Streamlit Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## 4. Review and deploy through VS Code

Review the changes, then commit and push them yourself. Suggested commit message:

```text
Configure private model downloads for Streamlit deployment
```

Keep the existing entrypoint `streamlit_app.py` and use Python 3.13, matching
the local test environment. The root `requirements.txt` selects CPU PyTorch
wheels on Linux so CUDA libraries are not installed. Root `packages.txt` adds
OpenCV's Linux runtime libraries. Streamlit installs these during the build.
Review build logs if the app does not start.

After the deployed branch updates, reboot the app if it has not restarted.
Do not upload `best.pt` or your actual `.streamlit/secrets.toml` to GitHub.
The app's public URL stays the same.

## Download and model caching

`HELMET_MODEL_SOURCE` defaults to `local`, preserving local development with
`models/best.pt`. Setting it to `huggingface` explicitly enables the private
download; failures never silently fall back to local or pretrained weights.

The Hugging Face client maintains a version-aware disk cache. A Streamlit
resource cache prevents download checks and checksum work on each UI rerun.
The loaded model is cached separately, with serialized inference across visitors.
No user images are sent to Hugging Face. Public visitors do not need an account
or token. Hosted disk caches can be lost when the hosting container is replaced;
the app will download the selected file again when needed.

Transient download errors receive one application retry; failed calls are not
cached, so reloading can try again. Credential, upload and checksum failures
show safe messages and disable detection. Fix the setting/upload and reboot.

See [Hugging Face download caching](https://huggingface.co/docs/huggingface_hub/guides/download).

## Public verification before declaring Step 9 complete

1. Open the public URL in a signed-out/private browser. Confirm **Model ready**.
2. Upload an original workplace photo, click **Detect Helmets**, and check real
   boxes, the three counts and individual confidence scores. The fixed internal
   threshold remains `0.25`; there is no threshold slider.
3. Download the annotated PNG and open it to verify the displayed result matches.
4. Replace and remove the image. Previous counts, details and downloads must clear.
5. Try a corrupt file and a large file. Errors should be clear and stale results
   should not remain. A valid image with zero detections must show zero counts.
6. Test phone-width layout and keyboard access, including results and download.
7. Check first-load and warm inference times, memory consumption and restarts in
   cloud logs/metrics while using representative photos. Local Windows speed and
   memory do not establish cloud performance. Test simultaneous sessions too.

If resources prove insufficient, measure the problem before considering an ONNX
export. Any format change requires comparing real predictions with the selected
PyTorch checkpoint. No model-format conversion is included in this change.

## Troubleshooting

- **Model access is not configured:** add `HF_TOKEN` in Streamlit Secrets.
- **Model cannot be accessed (HTTP 401):** the download was not authorized.
  Verify the token belongs to an account that can read this private repository.
- **Model cannot be accessed (HTTP 403):** the download was denied; verify the
  token's repository read permissions. Gated repositories instead show an
  access-approval message.
- **Model cannot be accessed (HTTP 404):** check the exact repository ID. Private
  resources can also appear missing when the token lacks access.
- **best.pt was not found:** check **Files and versions** in Hugging Face. The
  file must be named exactly `best.pt` at the repository root, not inside `models/`.
- **Configured model revision was not found:** check `HF_MODEL_REVISION`; use
  the actual branch name or an existing commit SHA.
- The server logs contain a safe diagnostic code (`http_401`, `http_403`,
  `http_404`, `file_missing`, `revision_missing`, or `gated_access`). These messages
  do not log the original exception, HTTP response or token. The old generic
  access message could not distinguish these failures; deploy the updated code
  to see the specific message. Local success with the default `local` source
  does not test Hugging Face access.
- **Checksum mismatch:** upload the exact local checkpoint; do not disable the
  check simply to accept an unexpected file.
- **Download temporarily unavailable:** reload once; check service/network status
  and app logs if it persists. Never share logs containing secrets.
- **Missing OpenCV library:** confirm `packages.txt` reached the deployed branch
  and the app rebuilt successfully.
- **Still says model unavailable after fixing settings:** reboot the app so the
  process reads the new configuration and clears old caches.
