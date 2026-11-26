"""Local EcoVision dashboard, available through either Streamlit entry point."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.dashboard_classify import (
    classify_page, cnn_status, get_runtime_device, get_session_stats,
    render_prediction as render_prediction,
)
from starter.dashboard_compare import compare_page
from starter.dashboard_evidence import about_page, analytics_page, guide_page
from starter.student_core import CLASSES

PAGES = {
    "Home / Dashboard": "space_dashboard",
    "Waste Classification": "center_focus_strong",
    "Model Comparison": "compare",
    "Model Analytics": "monitoring",
    "Waste Sorting Guide": "recycling",
    "About Project": "info",
}
EXAMPLES = {
    "cardboard": "Corrugated boxes", "glass": "Bottles & jars",
    "metal": "Cans & lids", "paper": "Printed sheets",
    "plastic": "Bottles & tubs", "trash": "Mixed wrappers",
}


@st.cache_data(show_spinner=False, max_entries=4)
def asset_content(name: str, version: int) -> str:
    path = ROOT / "assets" / name
    if name.endswith(".css"):
        return path.read_text(encoding="utf-8-sig")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def asset(name: str) -> str:
    path = ROOT / "assets" / name
    return asset_content(name, path.stat().st_mtime_ns) if path.is_file() else ""


def navigate(destination: str):
    st.session_state["navigation"] = destination


def model_status() -> tuple[str, str]:
    return cnn_status(), st.session_state.get("qwen_status", "Not checked")


def home_page():
    photos = asset("material-samples.webp")
    if photos:
        st.html(f'<style>:root {{ --ev-atlas: url("data:image/webp;base64,{photos}"); }}</style>')

    with st.container(key="home_hero"):
        left, right = st.columns([1.35, 1])
        with left:
            st.html('<h1 class="ev-hero-title">EcoVision AI</h1>'
                    '<p class="ev-subtitle">Smart Waste Classification Powered by Deep Learning</p>'
                    '<p class="ev-intro">Photograph a waste item, inspect the model’s prediction '
                    'and find guidance for sorting it.</p>')
            st.button("Start classifying", icon=":material/arrow_forward:", type="primary",
                      key="start_classifying", on_click=navigate, args=("Waste Classification",))
        with right:
            if photos:
                st.html('<div class="ev-prints" aria-hidden="true">'
                        '<figure class="ev-print back"><div class="ev-thumb sample-0"></div>'
                        '<figcaption>Cardboard</figcaption></figure>'
                        '<figure class="ev-print front"><div class="ev-thumb sample-1"></div>'
                        '<figcaption>Glass</figcaption></figure></div>')

    st.subheader("Session overview")
    count, avg_ms = get_session_stats()
    cnn, qwen = model_status()
    models = int(cnn == "Available") + int(qwen == "Available")
    with st.container(key="home_metrics"):
        for col, (label, value, note) in zip(st.columns(4), [
            ("Material classes", str(len(CLASSES)), "TrashNet labels"),
            ("Available models", str(models), f"CNN: {cnn.lower()}"),
            ("CNN predictions", str(count), "In this session"),
            ("Mean CNN latency", f"{avg_ms:.1f} ms" if avg_ms is not None else "—", "Forward pass + softmax"),
        ]):
            with col, st.container(border=True):
                st.metric(label, value)
                st.caption(note if label != "Available models" else f"{note} · Qwen: {qwen.lower()}")

    st.subheader("Material library")
    tiles = "".join(
        f'<li><div class="ev-thumb sample-{index}" role="img" aria-label="TrashNet {label} example"></div>'
        f'<div class="ev-tile-text"><strong>{label.title()}</strong><small>{EXAMPLES[label]}</small></div></li>'
        for index, label in enumerate(CLASSES)
    )
    st.html(f'<ul class="ev-library">{tiles}</ul>')
    st.button("Open sorting guide", icon=":material/menu_book:", key="open_guide",
              on_click=navigate, args=("Waste Sorting Guide",))


def main():
    st.set_page_config(page_title="EcoVision AI — Intelligent Waste Classification",
                       page_icon="♻️", layout="wide")
    css = asset("streamlit.css")
    if css:
        st.html(f"<style>{css}</style>")
    with st.sidebar:
        st.html('<div class="ev-brand"><span class="ev-mark" aria-hidden="true">♻</span>'
                '<div><strong>EcoVision AI</strong><small>Waste sorting assistant</small></div></div>')
        page = st.radio("Workspace", list(PAGES), key="navigation",
                        format_func=lambda name: f":material/{PAGES[name]}:  {name}")
        st.divider()
        with st.expander("App settings", icon=":material/tune:"):
            st.selectbox("Inference device", ["Auto", "CPU"], key="inference_device",
                         on_change=lambda: st.session_state.pop("cnn_cpu_fallback", None),
                         help="Auto uses CUDA when the PyTorch installation supports it.")
            st.caption("Appearance follows Streamlit’s theme. Choose light, dark or system "
                       "in the app menu (⋮) under Settings.")
    pages = {
        "Home / Dashboard": home_page, "Waste Classification": classify_page,
        "Model Comparison": compare_page, "Model Analytics": analytics_page,
        "Waste Sorting Guide": guide_page, "About Project": about_page,
    }
    pages[page]()
    cnn, qwen = model_status()
    with st.sidebar, st.container(border=True, key="runtime_status"):
        st.markdown("**Model status**")
        st.caption("Selected: CNN + Qwen" if page == "Model Comparison" else "Selected: Custom CNN")
        st.text(f"CNN · {cnn}\nQwen · {qwen}")
        device = get_runtime_device()
        st.caption("Inference device · " + ("CUDA GPU" if device == "cuda" else "CPU"))


if __name__ == "__main__":
    main()
