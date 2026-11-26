"""Interactive, opt-in comparison of the local CNN and Qwen vision model."""
from __future__ import annotations

import streamlit as st

from starter.dashboard_classify import (
    PredictionError, checkpoint_token, get_runtime_device, image_input, predict, render_scores,
)
from starter.ollama_service import DEFAULT_MODEL, OllamaError, classify_qwen, model_available


def _check_status():
    try:
        status = "Available" if model_available() else "Model missing"
        detail = "Model installed locally." if status == "Available" else "The local server has no qwen2.5vl:3b model."
    except OllamaError as exc:
        status, detail = exc.status, str(exc)
    st.session_state.update(qwen_status=status, qwen_status_detail=detail)


def _request_comparison():
    st.session_state.compare_requested = True


def compare_page():
    st.title("Model Comparison")
    st.write("Compare the CNN and Qwen on the same photo. Both models run on this computer.")
    with st.expander("Local model connection and timeout"):
        st.caption(f"Ollama model · {DEFAULT_MODEL} · 127.0.0.1:11434")
        st.button("Check Qwen availability", on_click=_check_status)
        st.write(st.session_state.get("qwen_status_detail", "Availability has not been checked."))
        timeout = st.number_input("Qwen timeout (seconds)", min_value=5, max_value=180, value=60, step=5)

    with st.container(border=True):
        selected = image_input("compare")
    if selected is None:
        return

    entry = st.session_state.get("comparison", {})
    if entry.get("digest") != selected.digest:
        entry = {"digest": selected.digest}
    version = checkpoint_token()
    model_key = (version, get_runtime_device())
    cnn = entry.get("cnn") if entry.get("model_key") == model_key else None
    qwen = entry.get("qwen")
    if st.session_state.pop("compare_requested", False):
        if cnn is None:
            with st.spinner("Running the CNN…"):
                try:
                    cnn = predict(selected.image, selected.digest)
                    entry.update(cnn=cnn, cnn_error=None)
                except PredictionError as exc:
                    entry.update(cnn=None, cnn_error=str(exc))
                entry["model_key"] = (version, get_runtime_device())
        if qwen is None or qwen.label is None:
            with st.spinner("Waiting for local Qwen…"):
                try:
                    qwen = classify_qwen(selected.image, int(timeout))
                    entry.update(qwen=qwen, qwen_error=None)
                    st.session_state.update(qwen_status="Available", qwen_status_detail="The model responded locally.")
                except OllamaError as exc:
                    entry.update(qwen=None, qwen_error=str(exc))
                    st.session_state.update(qwen_status=exc.status, qwen_status_detail=str(exc))
        st.session_state.comparison = entry

    complete = cnn is not None and qwen is not None and qwen.label is not None
    st.button("Compare models", type="primary", disabled=complete, key="compare_run", on_click=_request_comparison)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Custom CNN")
        if cnn is not None:
            st.metric("CNN prediction", cnn["label"].title())
            st.caption(f'{cnn["device"]} · {cnn["seconds"] * 1000:,.2f} ms for forward pass and softmax')
            render_scores(cnn)
            st.caption("Scores are uncalibrated softmax outputs.")
        elif entry.get("cnn_error") and entry.get("model_key") == model_key:
            st.error(entry["cnn_error"])
        elif version is None:
            st.warning("CNN checkpoint missing. Restore models/waste_cnn.pth to enable this model.")
        else:
            st.caption("Checkpoint present · awaiting comparison")
    with right, st.container(border=True):
        st.subheader("Qwen2.5-VL · 3B")
        if qwen is not None:
            st.metric("Qwen prediction", qwen.label.title() if qwen.label else "Unrecognized answer")
            st.caption(f"Local Ollama · {qwen.seconds * 1000:,.2f} ms for the full request")
            if qwen.label is None:
                st.warning("The answer did not identify a single supported class. You can retry the comparison.")
            st.text("Model response")
            st.code(qwen.response, language=None)
        elif entry.get("qwen_error"):
            st.warning(entry["qwen_error"])
        else:
            st.caption(st.session_state.get("qwen_status", "Not checked") + " · awaiting comparison")
        st.write("Confidence: Not Available")
        st.caption("Qwen returns a text label. The local interface provides no class probabilities.")

    if complete:
        if cnn["label"] == qwen.label:
            st.success(f'Both models chose {qwen.label.title()}.')
        else:
            st.info(f'The models differ. CNN chose {cnn["label"].title()} and Qwen chose {qwen.label.title()}.')
        cnn_pipeline = cnn.get("pipeline_seconds", 0.0)
        if cnn_pipeline > 0:
            st.write(f"From decoded image to answer: CNN {cnn_pipeline * 1000:,.1f} ms · "
                     f"Qwen {qwen.seconds * 1000:,.1f} ms "
                     f"(Qwen/CNN time ratio: {qwen.seconds / cnn_pipeline:,.1f}×).")
            st.caption("These are the recorded calls for this image. Each includes input preparation "
                       "and any model loading on that call. Reopening a result reuses its original timing.")
        st.caption(
            f'Timing on this image: CNN {cnn["seconds"] * 1000:,.2f} ms · Qwen {qwen.seconds * 1000:,.2f} ms. '
            "Qwen includes image encoding, local transport and any model loading. CNN timing covers its forward "
            "pass and softmax. These timings measure different parts of the workflow."
        )
