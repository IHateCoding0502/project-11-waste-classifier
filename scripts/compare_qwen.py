"""Compare the trained CNN with Ollama Qwen2.5-VL on the same test images.

Run from the project root:
  python scripts/compare_qwen.py --limit 30
  python scripts/compare_qwen.py --limit 0          # all test images
  python scripts/compare_qwen.py --limit 30 --seed 42  # random (reproducible) sample
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import ollama
except ImportError:
    print("Missing package 'ollama'. Install it with: python -m pip install ollama")
    raise SystemExit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing matplotlib. Install it with: python -m pip install matplotlib")
    raise SystemExit(1)

try:
    from PIL import Image
except ImportError:
    print("Missing Pillow. Install it with: python -m pip install pillow")
    raise SystemExit(1)

try:
    from starter.student_core import CLASSES, load_trained_model, predict_proba
except ImportError as exc:
    raise SystemExit(
        "Could not import CLASSES, load_trained_model, predict_proba from "
        "starter/student_core.py. Check that your local student_core.py defines them.\n"
        f"Original error: {exc}"
    )

from starter.ollama_service import (
    DEFAULT_MODEL, JSON_HINT, LABEL_SCHEMA, PROMPT, QWEN_MAX_SIDE, STRUCTURED_OPTIONS, encode_image,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNCERTAIN = "uncertain"

# Attempt 1: structured output (answer is forced to be one of the 6 labels and
#            generation ends right after the closing brace).
# Attempt 2: plain text with a little noise + repeat penalty + early stop.
QWEN_ATTEMPTS: list[dict[str, Any]] = [
    {
        "format": LABEL_SCHEMA,
        "options": STRUCTURED_OPTIONS,
    },
    {
        "format": None,
        "options": {
            "temperature": 0.2, "top_k": 20, "num_predict": 8, "repeat_penalty": 1.3,
            "seed": 1, "num_ctx": 2048, "stop": ["\n", "."],
        },
    },
]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def collect_images(test_dir: Path, limit: int, seed: int | None) -> list[tuple[Path, str]]:
    """Collect a class-balanced sample. `limit` means total images (<=0 = all).

    Without --seed the sample is deterministic (sorted filenames). With --seed
    the files of each class are shuffled reproducibly first.
    """
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    rng = random.Random(seed) if seed is not None else None
    by_class: dict[str, list[Path]] = {}
    for class_name in CLASSES:
        class_dir = test_dir / class_name
        if not class_dir.is_dir():
            print(f"Warning: class folder not found: {class_dir}")
            by_class[class_name] = []
            continue
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if rng is not None:
            rng.shuffle(files)
        by_class[class_name] = files

    available = sum(len(items) for items in by_class.values())
    if available == 0:
        raise RuntimeError(f"No images found under {test_dir}. Expected one folder per class.")
    target = available if limit <= 0 else min(limit, available)

    # Round-robin selection keeps the sample balanced across classes.
    selected: list[tuple[Path, str]] = []
    offsets = {name: 0 for name in CLASSES}
    while len(selected) < target:
        added_this_round = False
        for class_name in CLASSES:
            images = by_class[class_name]
            index = offsets[class_name]
            if index < len(images) and len(selected) < target:
                selected.append((images[index], class_name))
                offsets[class_name] += 1
                added_this_round = True
        if not added_this_round:
            break
    return selected


# ----------------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------------
def _confidence_from(value: Any) -> float | None:
    """Turn a number or a probability vector into a single confidence value."""
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            while len(value) == 1 and isinstance(value[0], (list, tuple)):
                value = value[0]
            return max(float(x) for x in value)
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_cnn_result(result: Any) -> tuple[str, float | None]:
    """Accept common predict_proba return formats and normalize to label/confidence."""
    if isinstance(result, tuple) and len(result) >= 2:
        first, second = result[0], result[1]
        # (label, confidence)  or  (probability vector, label)
        if isinstance(first, str):
            label = first.strip().lower()
            return (label if label in CLASSES else UNCERTAIN), _confidence_from(second)
        if isinstance(second, str):
            label = second.strip().lower()
            return (label if label in CLASSES else UNCERTAIN), _confidence_from(first)
        result = first

    # Dict format: {class_name: probability}
    if isinstance(result, dict):
        probs = {str(k).lower(): float(v) for k, v in result.items() if str(k).lower() in CLASSES}
        if probs:
            label = max(probs, key=probs.get)
            return label, probs[label]

    # Probability vector / list / NumPy array / tensor.
    try:
        values = result.tolist() if hasattr(result, "tolist") else list(result)
        while len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = list(values[0])
        if len(values) == len(CLASSES):
            numeric = [float(x) for x in values]
            best = max(range(len(numeric)), key=numeric.__getitem__)
            return CLASSES[best], numeric[best]
    except (TypeError, ValueError, IndexError):
        pass
    return UNCERTAIN, None


def predict_cnn(model: Any, image_path: Path) -> tuple[str, float | None, float, str]:
    start = time.perf_counter()
    try:
        result = predict_proba(model, image_path)
        label, confidence = normalize_cnn_result(result)
        error = "" if label != UNCERTAIN else "Unrecognized predict_proba output format"
        return label, confidence, time.perf_counter() - start, error
    except Exception as exc:
        return UNCERTAIN, None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------
# Qwen (Ollama)
# ----------------------------------------------------------------------------
def load_image_bytes(image_path: Path, max_side: int = QWEN_MAX_SIDE) -> bytes:
    """Load, convert to RGB, shrink and re-encode as JPEG bytes."""
    with Image.open(image_path) as img:
        return encode_image(img, max_side)


def parse_label(text: str) -> str:
    """First class name that appears as a whole word; works for plain text and JSON."""
    matches = re.findall(r"\b(" + "|".join(re.escape(c) for c in CLASSES) + r")\b", text.lower())
    return matches[0] if matches else UNCERTAIN


def predict_qwen(image_path: Path, model_name: str) -> tuple[str, float | None, float, str]:
    start = time.perf_counter()
    try:
        image_bytes = load_image_bytes(image_path)
    except Exception as exc:
        return UNCERTAIN, None, time.perf_counter() - start, f"Image load error: {type(exc).__name__}: {exc}"

    errors: list[str] = []
    for attempt_no, attempt in enumerate(QWEN_ATTEMPTS, start=1):
        try:
            prompt = PROMPT + (JSON_HINT if attempt["format"] else "")
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
                format=attempt["format"],
                options=attempt["options"],
                keep_alive="10m",
            )
            text = response["message"]["content"]
            label = parse_label(text)
            if label != UNCERTAIN:
                note = f"recovered on attempt {attempt_no} ({' | '.join(errors)})" if errors else ""
                return label, None, time.perf_counter() - start, note
            errors.append(f"attempt {attempt_no} unexpected response: {text[:80]!r}")
        except Exception as exc:
            errors.append(f"attempt {attempt_no} {type(exc).__name__}: {exc}")
    return UNCERTAIN, None, time.perf_counter() - start, " | ".join(errors)


def ollama_model_installed(model_name: str) -> bool | None:
    """True/False if we can tell, None if the listing format is not understood."""
    try:
        listing = ollama.list()
        models = listing["models"] if isinstance(listing, dict) else listing.models
        names: set[str] = set()
        for m in models:
            if isinstance(m, dict):
                name = m.get("model") or m.get("name")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.add(str(name))
    except Exception:
        return None
    if not names:
        return None
    candidates = {model_name}
    if ":" not in model_name:
        candidates.add(f"{model_name}:latest")
    return bool(candidates & names)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics_for(rows: list[dict[str, Any]], pred_key: str, time_key: str) -> dict[str, Any]:
    """All metrics count uncertain/error predictions as wrong (never as missing).

    - precision: tp / (tp + fp)           (only real class predictions can be fp)
    - recall   : tp / all images of class  (an uncertain answer is a false negative)
    - macro-F1 : mean over classes that actually appear in the sample
    """
    valid = [r for r in rows if r[pred_key] in CLASSES]
    correct = sum(r["true_label"] == r[pred_key] for r in valid)

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for cls in CLASSES:
        tp = sum(r["true_label"] == cls and r[pred_key] == cls for r in rows)
        fp = sum(r["true_label"] != cls and r[pred_key] == cls for r in rows)
        fn = sum(r["true_label"] == cls and r[pred_key] != cls for r in rows)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support > 0:
            f1_values.append(f1)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    times = [float(r[time_key]) for r in rows
             if r[time_key] is not None and math.isfinite(float(r[time_key]))]
    return {
        "model": "CNN" if pred_key == "cnn_prediction" else "Ollama Qwen",
        "total_images": len(rows),
        "valid_predictions": len(valid),
        "errors_or_uncertain": len(rows) - len(valid),
        "correct": correct,
        "accuracy_valid_predictions": correct / len(valid) if valid else None,
        "accuracy_over_all_images": correct / len(rows) if rows else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "mean_seconds": sum(times) / len(times) if times else None,
        "median_seconds": statistics.median(times) if times else None,
        "per_class": per_class,
    }


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def save_plots(rows: list[dict[str, Any]], cnn_metrics: dict[str, Any],
               qwen_metrics: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["CNN", "Ollama Qwen"]

    # 1) Accuracy
    accuracy_values = [cnn_metrics["accuracy_over_all_images"] or 0,
                       qwen_metrics["accuracy_over_all_images"] or 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, [x * 100 for x in accuracy_values])
    ax.set_title("Model Accuracy (errors counted as incorrect)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=160)
    plt.close(fig)

    # 2) Recall by class
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(CLASSES)))
    width = 0.38
    cnn_recalls = [cnn_metrics["per_class"][c]["recall"] * 100 for c in CLASSES]
    qwen_recalls = [qwen_metrics["per_class"][c]["recall"] * 100 for c in CLASSES]
    ax.bar([i - width / 2 for i in x], cnn_recalls, width, label="CNN")
    ax.bar([i + width / 2 for i in x], qwen_recalls, width, label="Ollama Qwen")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=20)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Recall by Waste Class (errors counted as misses)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "recall_by_class.png", dpi=160)
    plt.close(fig)

    # 3) Time
    mean_times = [cnn_metrics["mean_seconds"] or 0, qwen_metrics["mean_seconds"] or 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, mean_times)
    ax.set_title("Mean Prediction Time per Image")
    ax.set_ylabel("Seconds (lower is faster)")
    ax.bar_label(bars, fmt="%.2f s")
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_time_comparison.png", dpi=160)
    plt.close(fig)

    # 4) Confusion matrices (extra "uncertain" column shows failed predictions)
    columns = list(CLASSES) + [UNCERTAIN]
    for model_label, pred_key, filename in [
        ("CNN", "cnn_prediction", "cnn_confusion_matrix.png"),
        ("Ollama Qwen", "qwen_prediction", "qwen_confusion_matrix.png"),
    ]:
        matrix = [[0 for _ in columns] for _ in CLASSES]
        for row in rows:
            true = row["true_label"]
            pred = row[pred_key] if row[pred_key] in CLASSES else UNCERTAIN
            if true in CLASSES:
                matrix[CLASSES.index(true)][columns.index(pred)] += 1
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, interpolation="nearest")
        ax.set_title(f"{model_label} Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=35, ha="right")
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(columns)):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def fmt_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def fmt_num(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Project 11 CNN and Ollama vision model.")
    parser.add_argument("--limit", type=int, default=30, help="Total test images; 30 by default; use 0 for all")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data" / "test")
    parser.add_argument("--cnn-model", type=Path, default=ROOT / "models" / "waste_cnn.pth")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "comparison")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shuffle each class with this seed before sampling (default: sorted order)")
    args = parser.parse_args()

    print(f"Project root : {ROOT}")
    print(f"Test folder  : {args.test_dir}")
    print(f"CNN weights  : {args.cnn_model}")
    print(f"Ollama model : {args.model}")
    print(f"Image limit  : {'all' if args.limit <= 0 else args.limit}")
    print(f"Sampling     : {'sorted' if args.seed is None else f'random (seed={args.seed})'}")

    images = collect_images(args.test_dir, args.limit, args.seed)
    print(f"Selected {len(images)} images. Class distribution: {dict(Counter(label for _, label in images))}")

    if not args.cnn_model.exists():
        raise SystemExit(
            f"CNN weights file not found: {args.cnn_model}\n"
            "Update --cnn-model to the path of your trained model weights."
        )
    try:
        model = load_trained_model(args.cnn_model)
    except Exception as exc:
        raise SystemExit(f"Could not load CNN model: {type(exc).__name__}: {exc}")

    try:
        ollama.list()
    except Exception as exc:
        raise SystemExit(
            "Cannot connect to Ollama. Open the Ollama app and check that it is running.\n"
            f"Details: {type(exc).__name__}: {exc}"
        )
    if ollama_model_installed(args.model) is False:
        raise SystemExit(
            f"Ollama model '{args.model}' is not installed. Run:\n  ollama pull {args.model}"
        )

    # Warm-up: the first call loads the model / initializes lazily and would
    # otherwise distort the timing. These results are not recorded.
    print("Warming up CNN and Ollama model (not recorded)...")
    predict_cnn(model, images[0][0])
    predict_qwen(images[0][0], args.model)

    rows: list[dict[str, Any]] = []
    for index, (image_path, true_label) in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {true_label}: {image_path.name}")
        cnn_pred, cnn_conf, cnn_secs, cnn_error = predict_cnn(model, image_path)
        qwen_pred, qwen_conf, qwen_secs, qwen_error = predict_qwen(image_path, args.model)
        row = {
            "image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
            "filename": image_path.name,
            "true_label": true_label,
            "cnn_prediction": cnn_pred,
            "cnn_confidence": cnn_conf,
            "cnn_seconds": round(cnn_secs, 4),
            "cnn_error": cnn_error,
            "qwen_prediction": qwen_pred,
            "qwen_confidence": qwen_conf,
            "qwen_seconds": round(qwen_secs, 4),
            "qwen_error": qwen_error,
        }
        rows.append(row)
        print(f"  True={true_label} | CNN={cnn_pred} ({cnn_secs:.2f}s) | Qwen={qwen_pred} ({qwen_secs:.2f}s)")
        if cnn_error:
            print(f"  CNN note: {cnn_error}")
        if qwen_error:
            print(f"  Qwen note: {qwen_error}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.out_dir / "comparison_results.csv"
    with details_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cnn_metrics = metrics_for(rows, "cnn_prediction", "cnn_seconds")
    qwen_metrics = metrics_for(rows, "qwen_prediction", "qwen_seconds")
    summary = {
        "test_dir": str(args.test_dir),
        "cnn_weights": str(args.cnn_model),
        "ollama_model": args.model,
        "sample_size": len(rows),
        "sampling_seed": args.seed,
        "class_distribution": dict(Counter(r["true_label"] for r in rows)),
        "qwen_recovered_after_retry": sum(r["qwen_error"].startswith("recovered") for r in rows),
        "cnn": cnn_metrics,
        "ollama_qwen": qwen_metrics,
    }
    summary_path = args.out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Flat CSV summary is convenient for spreadsheets.
    flat_path = args.out_dir / "comparison_summary.csv"
    flat_fields = ["model", "total_images", "valid_predictions", "errors_or_uncertain", "correct",
                   "accuracy_valid_predictions", "accuracy_over_all_images", "macro_f1",
                   "mean_seconds", "median_seconds"]
    with flat_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for metrics in (cnn_metrics, qwen_metrics):
            writer.writerow({key: metrics.get(key) for key in flat_fields})

    save_plots(rows, cnn_metrics, qwen_metrics, args.out_dir)

    print("\n=== SUMMARY ===")
    for metrics in (cnn_metrics, qwen_metrics):
        print(
            f"{metrics['model']}: "
            f"accuracy(all images)={fmt_pct(metrics['accuracy_over_all_images'])}, "
            f"accuracy(valid only)={fmt_pct(metrics['accuracy_valid_predictions'])}, "
            f"macro-F1={fmt_num(metrics['macro_f1'], '.3f')}, "
            f"errors/uncertain={metrics['errors_or_uncertain']}/{metrics['total_images']}, "
            f"mean time={fmt_num(metrics['mean_seconds'], '.2f')}s, "
            f"median time={fmt_num(metrics['median_seconds'], '.2f')}s"
        )
    print(f"Qwen answers recovered after retry: {summary['qwen_recovered_after_retry']}")
    print(f"\nSaved detailed CSV : {details_path}")
    print(f"Saved summary CSV  : {flat_path}")
    print(f"Saved summary JSON : {summary_path}")
    print(f"Saved charts in    : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())