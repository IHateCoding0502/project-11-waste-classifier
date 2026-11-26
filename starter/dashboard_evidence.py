"""Research evidence and sorting guidance for the local EcoVision dashboard."""
from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path

import streamlit as st

from starter.student_core import CLASSES

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger(__name__)
SCENARIOS = {
    "standard": "Original test images",
    "complex": "Complex or deformed items",
    "low_light_blur_synthetic": "Synthetic lighting and blur",
    "low_light_blur_real": "Real-camera lighting and blur",
    "campus_novel": "New campus photographs",
}
GALLERIES = {
    "correct": "Correct predictions",
    "incorrect": "Incorrect predictions",
    "high_confidence_errors": "Most confident errors",
    "confusion": "Confusion matrix",
    "per_class": "Per-class scores",
    "confidence": "Confidence distribution",
}


@st.cache_data(show_spinner=False, max_entries=32)
def _read_record(path: str, modified_ns: int, size: int):
    # The modification time and size invalidate cached content after re-evaluation.
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        if path.endswith(".csv"):
            return list(csv.DictReader(stream))
        return json.load(stream)


def _artifact(relative: str):
    path = ROOT / relative
    try:
        stat = path.stat()
        return _read_record(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, csv.Error):
        LOG.info("Could not read evaluation artifact %s", relative, exc_info=True)
        return None


def _unavailable(relative: str):
    st.info(f"Records unavailable. Restore `{relative}` from the project to display this section.")


def _percentage(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def _matrix(record: dict):
    matrix = record.get("confusion_matrix")
    if isinstance(matrix, list) and len(matrix) == len(CLASSES) and all(
        isinstance(row, list) and len(row) == len(CLASSES)
        and all(isinstance(count, int) and count >= 0 for count in row) for row in matrix
    ):
        return matrix
    return None


def _run_record(selected: dict) -> tuple[str, dict]:
    name = selected.get("selected", "")
    if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name):
        return "", {}
    return name, _artifact(f"results/run_{name}.json") or {}


def material_guidance(label: str):
    """Show the same project guidance in both the classifier and the guide."""
    guidance = _artifact("data/disposal_guidance.json")
    item = guidance.get(label) if isinstance(guidance, dict) else None
    if not item:
        _unavailable("data/disposal_guidance.json")
        return
    st.subheader(f"Sorting {label}")
    st.write(item["description"])
    st.markdown(f"**Common examples** · {item['examples']}")
    with st.container(border=True):
        st.markdown("**Before disposal**")
        st.write(item["preparation"])
        st.write(item["general_advice"])
        st.caption(item["warnings"])
    with st.expander("Local collection guidance and sources"):
        st.write(item["local_guidance"])
        for source in item["sources"]:
            st.markdown(f"[{source['title']}]({source['url']})")
        st.caption(f"Project guidance reviewed {item['reviewed']}.")


def guide_page():
    st.title("Waste Sorting Guide")
    st.write("Choose a material to see how to prepare it and what to check with your collector.")
    label = st.selectbox("Material", CLASSES, format_func=str.title, key="guide_material")
    material_guidance(label)
    selected = _artifact("results/selected.json") or {}
    matrix = _matrix(selected.get("test", {}))
    if matrix is None:
        _unavailable("results/selected.json")
        return
    index = CLASSES.index(label)
    row = matrix[index]
    mistakes = sorted(
        ((count, CLASSES[j]) for j, count in enumerate(row) if j != index and count),
        reverse=True,
    )
    if mistakes:
        confused = ", ".join(f"{name} ({count})" for count, name in mistakes[:2])
        st.info(f"Of {sum(row)} test photos of {label}, the CNN most often mistook it for {confused}. "
                "Check the actual item before choosing a bin.")


def _test_metrics(selected: dict):
    record = selected.get("test", {})
    st.subheader("Held-out test performance")
    if not record:
        _unavailable("results/selected.json")
        return
    cols = st.columns(4)
    for column, (title, field) in zip(cols, [
        ("Accuracy", "accuracy"), ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"),
    ]):
        column.metric(title, _percentage(record.get(field)), border=True)
    matrix = _matrix(record)
    if matrix:
        correct = sum(row[i] for i, row in enumerate(matrix))
        st.caption(f"{correct:,} correct out of {sum(map(sum, matrix)):,} held-out images. "
                   "Macro scores give equal weight to each of the six materials.")
    bounds = selected.get("test_ci95", {}).get("macro_f1", [])
    if len(bounds) == 2:
        st.caption(f"Macro F1 95% bootstrap interval: {_percentage(bounds[0])}–{_percentage(bounds[1])}. "
                   "It measures test-image sampling uncertainty for this trained model.")


def _test_detail(selected: dict):
    record = selected.get("test", {})
    matrix = _matrix(record)
    if matrix:
        st.subheader("Confusion matrix")
        labels = [label.title() for label in CLASSES]
        rows = [{"True material": labels[i], "Predicted material": labels[j], "Photos": n}
                for i, row in enumerate(matrix) for j, n in enumerate(row)]
        st.vega_lite_chart(rows, {
            "height": 230,
            "encoding": {
                "x": {"field": "Predicted material", "type": "ordinal", "sort": labels,
                      "axis": {"labelAngle": -25}},
                "y": {"field": "True material", "type": "ordinal", "sort": labels},
                "tooltip": [{"field": field} for field in ("True material", "Predicted material", "Photos")],
            },
            "layer": [
                {"mark": "rect", "encoding": {"color": {
                    "field": "Photos", "type": "quantitative", "legend": None,
                    "scale": {"range": ["#d9f8eb", "#065f46"]}}}},
                {"mark": "text", "encoding": {
                    "text": {"field": "Photos", "type": "quantitative"},
                    "color": {"condition": {"test": f"datum.Photos > {max(map(max, matrix)) * 0.52}",
                                            "value": "#ffffff"}, "value": "#102f27"}}},
            ],
        }, width="stretch")
        st.caption("Read across a row to see where a material went. The diagonal counts correct predictions.")
    elif record:
        _unavailable("results/selected.json")
    per_class = record.get("per_class", {})
    if all(label in per_class for label in CLASSES):
        st.subheader("Performance by material")
        st.dataframe([
            {"Material": label.title(), "Precision": _percentage(per_class[label].get("precision")),
             "Recall": _percentage(per_class[label].get("recall")),
             "F1": _percentage(per_class[label].get("f1")), "Photos": per_class[label].get("support")}
            for label in CLASSES
        ], hide_index=True, width="stretch")
    if record and not per_class:
        _unavailable("results/selected.json")
    baselines = _artifact("results/baselines.json")
    with st.expander("Compare against simple baselines"):
        if baselines:
            rows = []
            for name, metrics in baselines.items():
                test = metrics.get("test", {})
                rows.append({"Method": name.replace("_", " ").title(),
                             "Accuracy": _percentage(test.get("accuracy")), "Macro F1": _percentage(test.get("macro_f1"))})
            if record:
                rows.append({"Method": "Selected CNN", "Accuracy": _percentage(record.get("accuracy")),
                             "Macro F1": _percentage(record.get("macro_f1"))})
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            _unavailable("results/baselines.json")


def _training(selected: dict):
    name, run = _run_record(selected)
    if not name:
        _unavailable("results/selected.json")
        return
    history = run.get("history", {})
    st.subheader("Training curves")
    if run:
        st.caption(f"Configuration {name} · {run.get('epochs_completed', '—')} completed epochs · "
                   f"lowest validation loss at epoch {run.get('best_epoch', '—')}.")
    for column, key, title in zip(st.columns(2), ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy (%)")):
        with column:
            train = history.get(key, [])
            validation = history.get(f"val_{key}", [])
            if train and len(train) == len(validation):
                scale = 100 if key == "accuracy" else 1
                st.markdown(f"**{title}**")
                st.line_chart({
                    "Epoch": list(range(1, len(train) + 1)),
                    "Training": [x * scale for x in train], "Validation": [x * scale for x in validation],
                }, x="Epoch", y=["Training", "Validation"], y_label=title, height=240,
                    color=["#078265", "#3977bd"])
            else:
                _unavailable(f"results/run_{name}.json")
    records = _artifact("results/summary.csv")
    if records:
        st.subheader("Validation experiments")
        st.caption("Among runs with dropout above zero, the highest validation Macro F1 selected the configuration.")
        st.dataframe([{"Run": r["model"], "Parameters": int(r["params"]), "Best epoch": int(r["best_epoch"]),
                       "Validation F1": _percentage(float(r["val_macro_f1"])), "Selected": r["model"] == name}
                      for r in records if r.get("framework") == "pytorch"], hide_index=True, width="stretch")
    else:
        _unavailable("results/summary.csv")


def _preview_manifest(scenario: str):
    manifest = _artifact("data/stress_tests/manifest.csv")
    if not manifest:
        _unavailable("data/stress_tests/manifest.csv")
        return
    images = [r for r in manifest if r.get("condition") == scenario]
    if not images:
        st.info("The manifest has no photos for this scenario.")
        return
    index = st.selectbox("Evaluated photo", range(len(images)),
                         format_func=lambda i: f"{images[i]['true_label'].title()} · {Path(images[i]['path']).name}")
    image_path = (ROOT / images[index]["path"]).resolve()
    if image_path.is_relative_to((ROOT / "data").resolve()) and image_path.is_file():
        st.image(str(image_path), width=300, caption=f"Recorded material · {images[index]['true_label']}")
    else:
        st.info("This source photo is outside the compact submission package. The saved galleries are available.")


def _robustness():
    metrics = _artifact("results/stress/metrics.json") or {}
    scenarios = metrics.get("scenarios")
    if not scenarios:
        _unavailable("results/stress/metrics.json")
        return
    st.subheader("Robustness and latency")
    table = []
    for name, title in SCENARIOS.items():
        record = scenarios.get(name, {})
        latency = record.get("mean_seconds")
        table.append({"Evaluation set": title, "Images": record.get("n_samples"), "Accuracy": _percentage(record.get("accuracy")),
                      "Macro F1": _percentage(record.get("macro_f1")),
                      "Mean latency (ms)": round(latency * 1000, 2) if latency is not None else None})
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Recorded device: {metrics.get('device', 'unreported')}. {metrics.get('latency_protocol', '')}.")
    st.caption("The complex subset overlaps the original test set. Four synthetic variants share each source photo.")
    choice = st.selectbox("Inspect evaluation set", list(SCENARIOS), format_func=SCENARIOS.get, key="scenario")
    record = scenarios.get(choice)
    if not record:
        _unavailable("results/stress/metrics.json")
        return
    if not record.get("n_samples"):
        st.info("Awaiting genuine photographs and independent labels. "
                "The collection instructions are in `data/stress_tests/COLLECTION_CHECKLIST.md`.")
        return
    transforms = record.get("by_transform")
    if transforms:
        st.markdown("**Synthetic results by transformation**")
        st.dataframe([{"Transformation": name.replace("_", " ").title(), "Images": item["n_samples"],
                       "Accuracy": _percentage(item["accuracy"]), "Macro F1": _percentage(item["macro_f1"])}
                      for name, item in transforms.items()], width="stretch", hide_index=True)
    gallery = st.selectbox("Evaluation evidence", ["Choose a view", *GALLERIES, "source"],
                           format_func=lambda key: GALLERIES.get(key, "Original evaluated photo" if key == "source" else key),
                           key="evaluation_evidence")
    if gallery == "source":
        _preview_manifest(choice)
    elif gallery in GALLERIES:
        relative = f"results/stress/{choice}_{gallery}.png"
        if (ROOT / relative).is_file():
            st.image(str(ROOT / relative), caption=GALLERIES[gallery], width="stretch")
        else:
            _unavailable(relative)
    with st.expander("Recorded evaluation details"):
        st.json(record, expanded=False)


def analytics_page():
    st.title("Model Analytics")
    st.write("Explore the saved TrashNet experiments for the team's trained CNN.")
    selected = _artifact("results/selected.json") or {}
    _test_metrics(selected)
    sections = ("Test results", "Training", "Robustness")
    view = st.radio("Analytics view", sections, horizontal=True, key="analytics_view")
    if view == "Training":
        _training(selected)
    elif view == "Robustness":
        _robustness()
    else:
        _test_detail(selected)


def about_page():
    st.title("About Project")
    st.write("Project 11 · Visual Waste Sorting Assistant. EcoVision uses a locally trained convolutional "
             "neural network to identify the material in a photo and explain how to prepare it for collection.")
    st.subheader("From photo to sorting advice")
    st.write("Pillow validates and orients the photo. The evaluation transform converts it to RGB, applies a "
             "bilinear resize to 64 × 64, and scales pixels to 0–1. PyTorch produces six logits. Softmax "
             "turns them into scores, and the leading material selects the matching disposal guide.")
    st.caption("Class index order: " + " · ".join(f"{i} {label}" for i, label in enumerate(CLASSES)))
    with st.container(border=True):
        st.markdown("**CNN architecture**")
        st.write("Four Conv → BatchNorm → ReLU → MaxPool blocks use 16, 32, 64 and 128 channels. Global average "
                 "pooling feeds a dense layer with 128 units and a six-output classifier. The selected network "
                 "uses dropout of 0.3 in its head and has 114,966 trainable parameters.")
        image = ROOT / "reports/architecture.png"
        if image.is_file() and st.checkbox("Show the architecture diagram"):
            st.image(str(image), width="stretch")
    st.subheader("Dataset and evaluation")
    dataset = _artifact("results/dataset_summary.json") or {}
    counts = dataset.get("class_counts", {})
    if all(split in counts for split in ("train", "validation", "test")):
        cols = st.columns(3)
        for col, split in zip(cols, ("train", "validation", "test")):
            col.metric(split.title() + " images", f"{sum(counts[split].values()):,}", border=True)
        st.caption(f"TrashNet · image-level split · seed {dataset.get('seed', 'unreported')}.")
    else:
        _unavailable("results/dataset_summary.json")
    st.write("The team trained with Adam and class-weighted cross-entropy. Validation loss determined the epoch, "
             "and validation Macro F1 determined the final run among configurations meeting the dropout requirement. "
             "The test set was held out for evaluation.")
    st.subheader("Where the classifier struggles")
    st.markdown("- Glass, metal and plastic can look similar at 64 × 64 pixels.\n"
                "- The CNN assigns every photo to one of its six classes. High confidence can accompany a mistake.\n"
                "- Photos were split by image. The same physical object may appear in more than one split.\n"
                "- Lighting changes affect results. Use the Robustness view in Model Analytics to check real-camera coverage.")
    with st.expander("Saved CNN–Qwen comparison"):
        rows = _artifact("results/qwen_comparison_summary.csv")
        if rows:
            st.dataframe([{"Model": r["model"], "Correct": int(r["correct"]), "Images": int(r["images"]),
                           "Accuracy": _percentage(float(r["accuracy"])), "Mean time (ms)": round(float(r["mean_seconds"]) * 1000, 2)}
                          for r in rows], hide_index=True, width="stretch")
            st.caption("Saved comparison on a small subset. Its timing protocol differs from the stress benchmark.")
        else:
            _unavailable("results/qwen_comparison_summary.csv")
    st.caption("Built with Streamlit, PyTorch, Pillow and NumPy. Optional comparison uses Qwen2.5-VL:3B via local Ollama.")
