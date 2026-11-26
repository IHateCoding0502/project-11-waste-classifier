"""Shared local image input, session predictions and the classification page."""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from starter.dashboard_evidence import material_guidance
from starter.inference import InvalidPrediction, decode_upload, infer
from starter.student_core import CLASSES, load_trained_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "waste_cnn.pth"
SELECTED_SHA256 = "825083ce60c04c0566b736a2089ece09f9a640ba0b72b55ce3b1fadf6ef077c3"
LOG = logging.getLogger(__name__)
MATERIAL_COLORS = dict(zip(CLASSES, ("#bc8047", "#12aa91", "#8394aa", "#5491ca", "#a279c8", "#b99440")))


class PredictionError(Exception):
    """A message suitable for displaying in the dashboard."""


def checkpoint_token() -> int | None:
    try:
        return CHECKPOINT.stat().st_mtime_ns if CHECKPOINT.is_file() else None
    except OSError:
        LOG.exception("Cannot inspect the CNN checkpoint")
        return None


def get_runtime_device() -> str:
    if st.session_state.get("inference_device") == "CPU" or st.session_state.get("cnn_cpu_fallback"):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_data(max_entries=4, show_spinner=False)
def _is_selected_checkpoint(path: str, modified_ns: int) -> bool:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == SELECTED_SHA256


def cnn_status() -> str:
    token = checkpoint_token()
    if token is None:
        return "Checkpoint missing"
    state = st.session_state.get("cnn_load_status", {}).get((token, get_runtime_device()))
    if state is not None:
        return state
    try:
        return "Available" if _is_selected_checkpoint(str(CHECKPOINT), token) else "Unverified"
    except OSError:
        LOG.exception("Cannot read the CNN checkpoint")
        return "Unavailable"


@st.cache_resource(max_entries=4, show_spinner=False)
def get_model(path: str, modified_ns: int, device: str):
    """Cache separately by checkpoint version and execution device."""
    return load_trained_model(path, device=device)


def get_session_stats() -> tuple[int, float | None]:
    count = st.session_state.get("cnn_runs", 0)
    elapsed_ms = st.session_state.get("cnn_total_ms", 0.0)
    return count, elapsed_ms / count if count else None


def _run_cnn(image: Image.Image, token: int, device: str) -> dict:
    started = time.perf_counter()
    status = st.session_state.setdefault("cnn_load_status", {})
    key = (token, device)
    try:
        model = get_model(str(CHECKPOINT), token, device)
        status[key] = "Available"
    except torch.OutOfMemoryError:
        status[key] = "Unavailable"
        raise
    except FileNotFoundError as exc:
        raise PredictionError("The CNN checkpoint is missing. Restore models/waste_cnn.pth from the project.") from exc
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            EOFError, pickle.UnpicklingError) as exc:
        LOG.exception("Cannot load the CNN checkpoint")
        status[key] = "Unavailable"
        raise PredictionError(
            "The checkpoint could not be loaded. Restore the project checkpoint with its original "
            "architecture, preprocessing settings and six-class order."
        ) from exc
    try:
        result = infer(model, image)
        result["pipeline_seconds"] = time.perf_counter() - started
        return result
    except torch.OutOfMemoryError:
        raise
    except InvalidPrediction as exc:
        LOG.exception("Invalid CNN output")
        status[key] = "Unavailable"
        raise PredictionError("The model returned invalid class scores. Check the six-class checkpoint.") from exc
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        LOG.exception("CNN inference failed on %s", device)
        raise PredictionError("The CNN could not process this image. Try a different photo or select CPU in settings.") from exc


def predict(image: Image.Image, digest: str) -> dict:
    """Run a real prediction or reuse its small result within this session."""
    token = checkpoint_token()
    if token is None:
        raise PredictionError("The CNN checkpoint is missing. Restore models/waste_cnn.pth from the project.")
    device = get_runtime_device()
    history = st.session_state.setdefault("cnn_results", {})
    key = (digest, token, device)
    if key in history:
        return history[key]
    try:
        result = _run_cnn(image, token, device)
    except torch.OutOfMemoryError as exc:
        if device != "cuda":
            raise PredictionError("There is insufficient memory for this prediction. Close unused applications and retry.") from exc
        LOG.warning("GPU memory exhausted. Switching this session to CPU.")
        result = None
    if result is None:
        st.session_state["cnn_cpu_fallback"] = True
        key = (digest, token, "cpu")
        if key in history:
            return history[key]
        try:
            result = _run_cnn(image, token, "cpu")
        except torch.OutOfMemoryError as exc:
            raise PredictionError("The GPU and CPU both ran out of memory. Close unused applications and retry.") from exc
        result["fallback"] = "GPU memory was full. This prediction ran on CPU."
    if len(history) >= 16:
        del history[next(iter(history))]
    history[key] = result
    st.session_state["cnn_runs"] = st.session_state.get("cnn_runs", 0) + 1
    st.session_state["cnn_total_ms"] = st.session_state.get("cnn_total_ms", 0.0) + 1000 * result["seconds"]
    return result


@dataclass
class SelectedImage:
    image: Image.Image
    preview: Image.Image
    digest: str
    name: str
    format: str
    original_size: tuple[int, int]
    byte_count: int


def _read_image(raw: bytes, name: str) -> SelectedImage:
    upload = decode_upload(raw)
    preview = upload.image
    width, height = preview.size
    if max(width, height) > 640:
        scale = 640 / max(width, height)
        preview = preview.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                                 Image.Resampling.BILINEAR)
    return SelectedImage(upload.image, preview, hashlib.sha256(raw).hexdigest(), name,
                         upload.format, upload.original_size, len(raw))


def _forget_image(namespace: str) -> None:
    key = namespace + "_input_version"
    st.session_state[key] = st.session_state.get(key, 0) + 1
    st.session_state.pop("eco_image", None)
    st.session_state.pop("eco_image_token", None)
    st.session_state.pop(namespace + "_shown", None)


def _choose_source(mode: str, prefix: str):
    """Return an identity, display name and lazy byte reader for a selected file."""
    if mode == "Example":
        samples = sorted((ROOT / "data" / "sample").glob("*/*.png"))
        path = st.selectbox("Generated example", samples, index=None, format_func=lambda p: p.stem,
                            placeholder="Choose an image", key=prefix + "_sample")
        st.caption("These small generated patterns exercise the pipeline. Use a photo to test a real item.")
        if path is None:
            return None
        return (mode, path.name, path.stat().st_mtime_ns), path.name, path.read_bytes
    if mode == "Camera":
        source = st.camera_input("Take a photo", key=prefix + "_camera")
        st.caption("Allow camera access in your browser, or choose Upload to use an existing photo.")
    else:
        source = st.file_uploader("Choose a waste photo", type=["jpg", "jpeg", "png"],
                                  max_upload_size=10, key=prefix + "_upload")
        st.caption("JPG or PNG · Up to 10 MiB, 20 megapixels and 8,192 px per side.")
    if source is None:
        return None
    file_id = getattr(source, "file_id", None)
    if file_id is None:
        file_id = hashlib.sha256(source.getvalue()).hexdigest()
    return (mode, file_id), source.name, source.getvalue


def image_input(namespace: str) -> SelectedImage | None:
    """Validate uploads once per file, keeping only one decoded image per session."""
    version = st.session_state.get(namespace + "_input_version", 0)
    prefix = f"{namespace}_input_{version}"
    mode = st.radio("Image source", ["Upload", "Camera", "Example", "Last image"],
                    horizontal=True, key=prefix + "_source")
    selected = st.session_state.get("eco_image")
    if mode != "Last image":
        try:
            choice = _choose_source(mode, prefix)
            if choice is None:
                return None
            token, name, read_bytes = choice
            if selected is None or token != st.session_state.get("eco_image_token"):
                selected = _read_image(read_bytes(), name)
                st.session_state["eco_image"] = selected
                st.session_state["eco_image_token"] = token
        except ValueError as exc:
            st.error(str(exc))
            return None
        except (MemoryError, OSError):
            LOG.exception("Image could not be prepared")
            st.error("This image could not be prepared. Choose another JPG or PNG.")
            return None
    if selected is None:
        st.info("Choose an image in either workspace first.")
        return None
    with st.container(key=namespace + "_photo"):
        st.image(selected.preview, caption=selected.name, width="stretch",
                 alt="Selected image for waste classification")
    width, height = selected.original_size
    st.caption(f"{selected.format} · Original {width:,} × {height:,} px · "
               f"Oriented {selected.image.width:,} × {selected.image.height:,} px · {selected.byte_count / 1024:,.1f} KiB")
    st.button("Remove image", key=prefix + "_remove", on_click=_forget_image, args=(namespace,))
    return selected


def render_scores(result: dict) -> None:
    ranked = sorted(result["scores"].items(), key=lambda item: item[1], reverse=True)
    for label, value in ranked:
        st.progress(value, text=f"{label.title()} · {value:.1%}")


def _render_result(result: dict) -> None:
    label = result["label"]
    st.markdown(f'<div style="border-left:5px solid {MATERIAL_COLORS[label]};padding:.1rem 1rem">'
                f'<small>Predicted material</small><h2 style="margin:0;padding:.2rem 0">{label.title()}</h2></div>',
                unsafe_allow_html=True)
    first, second = st.columns(2)
    first.metric("Model confidence", f'{result["confidence"]:.1%}')
    second.metric("Inference time", f'{result["seconds"] * 1000:.2f} ms')
    st.caption(f'{result["device"].upper()} · Forward pass and softmax. Image processing and model loading are excluded.')
    if result.get("fallback"):
        st.info(result["fallback"])
    if result["confidence"] < 0.5:
        st.warning("The leading score is below 50%. Check the material or try a brighter photo with one item in view.")
    ranked = sorted(result["scores"], key=result["scores"].get, reverse=True)
    st.markdown("**Top three**  " + " · ".join(f"{i}. {name.title()}" for i, name in enumerate(ranked[:3], 1)))
    render_scores(result)
    st.caption("Softmax scores are uncalibrated. Every image receives a label, including unfamiliar objects.")
    material_guidance(label)


def render_prediction(model, image: Image.Image) -> None:
    """Legacy rendering entry point for callers supplying an already loaded CNN."""
    _render_result(infer(model, image))


def classify_page() -> None:
    st.title("Waste Classification")
    st.write("Photograph one item in good light to identify the material and review its sorting guidance.")
    left, right = st.columns([1, 1], gap="medium")
    with left, st.container(border=True):
        st.subheader("Image input")
        selected = image_input("classify")
        token = checkpoint_token()
        if token is None:
            st.warning("The CNN checkpoint is missing. Restore models/waste_cnn.pth to enable classification.")
        clicked = st.button("Classify Waste", type="primary", icon=":material/center_focus_strong:",
                            width="stretch", disabled=selected is None or token is None)
    with right, st.container(border=True):
        st.subheader("Prediction")
        if selected is None:
            st.session_state.pop("classify_shown", None)
            st.info("Add a photo and select Classify Waste to see its material, scores and disposal guidance.")
            return
        if clicked:
            try:
                with st.spinner("Classifying the image…"):
                    predict(selected.image, selected.digest)
                st.session_state["classify_shown"] = (selected.digest, token, get_runtime_device())
            except PredictionError as exc:
                st.session_state.pop("classify_shown", None)
                st.error(str(exc))
                return
        key = (selected.digest, token, get_runtime_device())
        result = st.session_state.get("cnn_results", {}).get(key)
        if st.session_state.get("classify_shown") == key and result is not None:
            _render_result(result)
        else:
            st.caption("Your image is ready. Select Classify Waste to run the CNN.")
