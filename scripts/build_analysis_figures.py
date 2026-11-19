#!/usr/bin/env python3
"""Create project analysis figures from saved runs and the current dataset.

Outputs (by default under results/figures/):
  - val_macro_f1_all_configs.png: validation-only CNN ablation comparison.
  - class_distribution.png: valid image counts by class and split.
  - sample_grid.png: one deterministic example from each class.
  - misclassified_examples.png: highest-confidence errors on the test split.

The validation ablation chart is deliberately separate from
``summary_macro_f1.png``, which compares the selected model with test baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.student_core import CLASSES, TrashNetDataset, load_trained_model  # noqa: E402


def load_run_records(results_dir: Path) -> tuple[list[dict], str | None]:
    """Load completed PyTorch CNN runs; never read test scores for this plot."""
    records = []
    for path in sorted(results_dir.glob("run_*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        validation = run.get("validation") or {}
        config = run.get("config") or {}
        if run.get("framework") != "pytorch" or not run.get("name"):
            continue
        if "macro_f1" not in validation:
            continue
        records.append({
            "name": str(run["name"]),
            "macro_f1": float(validation["macro_f1"]),
            "dropout": float(config.get("dropout", 0.0)),
            "augmentation": str(config.get("augmentation", "unknown")),
            "filters": config.get("filters", []),
            "learning_rate": float(config.get("learning_rate", 0.0)),
            "best_epoch": run.get("best_epoch"),
        })
    selected = None
    selected_path = results_dir / "selected.json"
    if selected_path.is_file():
        try:
            selected = json.loads(selected_path.read_text(encoding="utf-8")).get("selected")
        except (OSError, json.JSONDecodeError):
            pass
    return records, selected


def plot_validation_macro_f1(results_dir: Path, out_dir: Path) -> Path:
    records, selected = load_run_records(results_dir)
    if not records:
        raise FileNotFoundError(f"No completed run_*.json files found in {results_dir}")
    records.sort(key=lambda item: item["macro_f1"])
    names = [item["name"] for item in records]
    values = [item["macro_f1"] for item in records]
    colors = ["#168C74" if name == selected else "#5B8DB8" for name in names]
    fig_height = max(5.0, 0.48 * len(records) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    bars = ax.barh(range(len(records)), values, color=colors, edgecolor="white")
    labels = []
    for item in records:
        filters = item["filters"]
        depth = len(filters) if isinstance(filters, list) else "?"
        labels.append(
            f"{item['name']}  |  drop={item['dropout']:g}, aug={item['augmentation']}, "
            f"depth={depth}, lr={item['learning_rate']:g}"
        )
    ax.set_yticks(range(len(records)), labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Validation macro-F1")
    ax.set_title("Validation macro-F1 by CNN configuration\nAblation results only; not test-baseline comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.012, 0.96), bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=9)
    if selected in names:
        ax.text(0.99, 0.015, f"Selected (validation rule): {selected}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color="#126B59")
    fig.tight_layout()
    path = out_dir / "val_macro_f1_all_configs.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def valid_images_by_class(directory: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for class_name in CLASSES:
        class_dir = directory / class_name
        files = []
        if class_dir.is_dir():
            for path in sorted(class_dir.iterdir(), key=lambda p: p.name.casefold()):
                if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        files.append(path)
                    except (OSError, ValueError, SyntaxError):
                        continue
        found[class_name] = files
    return found


def plot_class_distribution(data_dir: Path, out_dir: Path) -> Path:
    split_dirs = [(name, data_dir / name) for name in ("train", "validation", "test")]
    split_dirs = [(name, path) for name, path in split_dirs if path.is_dir()]
    if not split_dirs and (data_dir / "sample").is_dir():
        split_dirs = [("sample (synthetic)", data_dir / "sample")]
    if not split_dirs:
        raise FileNotFoundError(f"No train/validation/test or sample folders found under {data_dir}")

    counts = {split: valid_images_by_class(directory) for split, directory in split_dirs}
    x = np.arange(len(CLASSES))
    width = 0.82 / len(split_dirs)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    palette = ["#326A9B", "#E3A535", "#2D9B7D"]
    for i, (split, by_class) in enumerate(counts.items()):
        values = [len(by_class[name]) for name in CLASSES]
        bars = ax.bar(x + (i - (len(split_dirs) - 1) / 2) * width, values,
                      width=width, label=split, color=palette[i % len(palette)])
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, CLASSES, rotation=20, ha="right")
    ax.set_ylabel("Number of valid images")
    ax.set_title("Class distribution by dataset split")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = out_dir / "class_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sample_grid(data_dir: Path, out_dir: Path, split: str = "train") -> Path:
    directory = data_dir / split
    title_suffix = split
    if not directory.is_dir() and (data_dir / "sample").is_dir():
        directory = data_dir / "sample"
        title_suffix = "synthetic smoke-test sample"
    if not directory.is_dir():
        raise FileNotFoundError(f"Could not find {split} or sample images under {data_dir}")

    by_class = valid_images_by_class(directory)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, class_name in zip(axes.flat, CLASSES):
        files = by_class[class_name]
        ax.axis("off")
        ax.set_title(class_name)
        if files:
            with Image.open(files[0]) as image:
                ax.imshow(image.convert("RGB"))
        else:
            ax.text(0.5, 0.5, "No valid image", ha="center", va="center")
    fig.suptitle(f"One example per class — {title_suffix}", fontsize=14)
    fig.tight_layout()
    path = out_dir / "sample_grid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_misclassified(data_dir: Path, model_path: Path, out_dir: Path,
                       device: str | None = None, limit: int = 12) -> Path | None:
    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        print(f"[skip] Misclassified examples: test split not found at {test_dir}")
        return None
    if not model_path.is_file():
        print(f"[skip] Misclassified examples: checkpoint not found at {model_path}")
        return None
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_trained_model(model_path, device=device_obj)
    dataset = TrashNetDataset(test_dir, model.image_size, augmentation="none")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    mistakes = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device_obj)), dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            for i, (true_idx, pred_idx) in enumerate(zip(labels.tolist(), predictions.tolist())):
                if true_idx != pred_idx:
                    sample_path, _ = dataset.samples[offset + i]
                    confidence = float(probabilities[i, pred_idx])
                    mistakes.append((confidence, sample_path, true_idx, pred_idx))
            offset += len(labels)
    if not mistakes:
        print("[info] No misclassified images found on the test split.")
        return None

    # Show the highest-confidence errors first to surface potentially risky mistakes.
    mistakes.sort(key=lambda item: item[0], reverse=True)
    mistakes = mistakes[:max(1, limit)]
    cols = 4
    rows = (len(mistakes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.7 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (confidence, path, true_idx, pred_idx) in zip(axes.flat, mistakes):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(
            f"True: {CLASSES[true_idx]}\nPred: {CLASSES[pred_idx]} ({confidence:.1%})",
            color="#9B2C2C", fontsize=10,
        )
    fig.suptitle(f"Misclassified test images — top {len(mistakes)} by wrong-class confidence", fontsize=14)
    fig.tight_layout()
    path_out = out_dir / "misclassified_examples.png"
    fig.savefig(path_out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "waste_cnn.pth")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto (default: auto)")
    parser.add_argument("--misclassified-limit", type=int, default=12)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = None if args.device in (None, "auto") else args.device

    outputs = [
        plot_validation_macro_f1(args.results_dir, args.out_dir),
        plot_class_distribution(args.data_dir, args.out_dir),
        plot_sample_grid(args.data_dir, args.out_dir),
    ]
    misclassified = plot_misclassified(args.data_dir, args.model, args.out_dir,
                                        device=device, limit=args.misclassified_limit)
    if misclassified is not None:
        outputs.append(misclassified)
    print("Created analysis figures:")
    for path in outputs:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
