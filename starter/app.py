"""Streamlit demo for the Project 11 PyTorch waste classifier.

Start with: ``streamlit run starter/app.py`` after training the model with
``python scripts/run_experiments.py`` on the full TrashNet dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.student_core import CLASSES, class_description, load_trained_model, predict_proba  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "waste_cnn.pth"

st.set_page_config(page_title="Visual Waste Sorting Assistant", layout="wide")
st.title("Visual Waste Sorting Assistant")
st.caption("Midterm AI Core · Custom CNN implemented and trained with PyTorch")

@st.cache_resource(show_spinner="Loading PyTorch CNN ...")
def get_model(model_path: str, model_mtime: float):
    # Include modification time in the cache key so a refreshed checkpoint is loaded.
    del model_mtime
    return load_trained_model(model_path)

with st.sidebar:
    st.header("TrashNet labels")
    st.write("The output classes follow this fixed order:")
    for cls in CLASSES:
        st.caption(f"**{cls}** — {class_description(cls)}")
    st.divider()
    st.caption("Use a genuine held-out TrashNet test image where possible. Synthetic images are for pipeline smoke tests only.")

if not MODEL_PATH.is_file():
    st.error(f"No PyTorch checkpoint found at `{MODEL_PATH.relative_to(PROJECT_ROOT)}`.")
    st.code("python scripts/prepare_dataset.py --confirm\npython scripts/run_experiments.py")
    st.info("The full dataset and a trained PyTorch checkpoint are not shipped in the repository. Prepare TrashNet and run the experiment script to create the checkpoint.")
    st.stop()

try:
    model = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except (OSError, ValueError, KeyError, RuntimeError) as exc:
    st.error(f"Could not load the PyTorch checkpoint: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload a waste photo", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload a JPG or PNG image to classify it.")
else:
    try:
        left, right = st.columns(2)
        with left:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
        probs = predict_proba(model, uploaded)
        best = max(probs, key=probs.get)
        with right:
            st.subheader(f"Prediction: {best}")
            st.metric("Model confidence", f"{probs[best]:.1%}")
            st.caption(class_description(best))
            st.bar_chart(probs, horizontal=True)
            if probs[best] < 0.50:
                st.warning("Low confidence: the model is uncertain. Treat this prediction with caution.")
            st.caption("Predictions are limited to the six TrashNet categories and are not disposal-safety advice.")
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not process this image. Please upload a valid JPG or PNG. Details: {exc}")
