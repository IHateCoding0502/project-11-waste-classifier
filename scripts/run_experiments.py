#!/usr/bin/env python3
"""Reproducible PyTorch Midterm experiments for Project 11.

The ten CNN configurations share the same fixed split and seed. Every run is
selected by validation loss at the epoch level; final configuration selection
uses validation macro-F1. The held-out test set is evaluated only once for the
selected CNN and the required baselines, avoiding ablation-driven test leakage.

Full run: python scripts/prepare_dataset.py --confirm && python scripts/run_experiments.py
Smoke test: python scripts/run_experiments.py --quick
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}.\n"
        "From the project root, install dependencies with:\n"
        "  python -m pip install -r requirements.txt\n"
        "Make sure the same virtual environment is active for installation and execution."
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("p11_student_core", ROOT / "starter" / "student_core.py")
sc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sc
assert _spec.loader is not None
_spec.loader.exec_module(sc)
CLASSES = sc.CLASSES
N_CLS = len(CLASSES)

BASE = dict(filters=(16, 32, 64, 128), dropout=0.3, augmentation="geometric", learning_rate=2e-3)
EXPERIMENTS: dict[str, dict] = {
    "base": {},
    "no_reg": {"augmentation": "none", "dropout": 0.0},
    "aug_none": {"augmentation": "none"},
    "aug_full": {"augmentation": "full"},
    "drop_0.0": {"dropout": 0.0},
    "drop_0.5": {"dropout": 0.5},
    "depth_2": {"filters": (32, 64)},
    "depth_3": {"filters": (16, 32, 64)},
    "lr_2e-4": {"learning_rate": 2e-4},
    "lr_1e-2": {"learning_rate": 1e-2},
}
ABLATION_AXES = {
    "augmentation": ["aug_none", "base", "aug_full"],
    "dropout": ["drop_0.0", "base", "drop_0.5"],
    "depth": ["depth_2", "depth_3", "base"],
    "learning_rate": ["lr_2e-4", "base", "lr_1e-2"],
}
LABELS = {
    "base": "base (4 blocks, p=0.3, geometric, lr=2e-3)",
    "no_reg": "no augmentation + no dropout", "aug_none": "no augmentation",
    "aug_full": "geometric + photometric augmentation", "drop_0.0": "dropout 0.0",
    "drop_0.5": "dropout 0.5", "depth_2": "2 convolutional blocks",
    "depth_3": "3 convolutional blocks", "lr_2e-4": "learning rate 2e-4",
    "lr_1e-2": "learning rate 1e-2",
}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return np.bincount(y_true * N_CLS + y_pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)


def metrics_from_cm(cm: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(N_CLS), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(N_CLS), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(N_CLS), where=denom > 0)
    total = int(cm.sum())
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "support": int(cm[i].sum())}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(), "n_samples": total,
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile bootstrap interval over test images, not over random seeds."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"accuracy": [0.0, 0.0], "macro_f1": [0.0, 0.0], "method": "percentile bootstrap over test images"}
    acc, f1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics_from_cm(confusion_matrix(y_true[idx], y_pred[idx]))
        acc.append(m["accuracy"])
        f1.append(m["macro_f1"])
    return {"accuracy": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "macro_f1": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
            "method": "95% percentile bootstrap over test images; does not capture seed/training variability",
            "resamples": n_boot, "seed": seed}


def colour_histogram(x: np.ndarray, bins: int = 16) -> np.ndarray:
    feats = np.zeros((len(x), 3 * bins), dtype=np.float64)
    for c in range(3):
        indices = (x[..., c].reshape(len(x), -1).astype(np.int32) * bins) // 256
        for i in range(len(x)):
            feats[i, c * bins:(c + 1) * bins] = np.bincount(indices[i], minlength=bins)
        sums = feats[:, c * bins:(c + 1) * bins].sum(axis=1, keepdims=True)
        feats[:, c * bins:(c + 1) * bins] /= np.maximum(sums, 1.0)
    return feats


def _distance(features: np.ndarray, centroids: np.ndarray, kind: str) -> np.ndarray:
    dif = features[:, None, :] - centroids[None, :, :]
    if kind == "l1":
        return np.abs(dif).sum(axis=-1)
    return (dif ** 2 / (features[:, None, :] + centroids[None, :, :] + 1e-9)).sum(axis=-1)


def run_baselines(xs: dict, ys: dict, *, include_test: bool = True) -> dict:
    """Fit on train, tune distance by validation, and optionally report test."""
    out: dict[str, Any] = {}
    majority = int(np.bincount(ys["train"], minlength=N_CLS).argmax())
    out["majority_class"] = {"predicts": CLASSES[majority], "validation": metrics_from_cm(
        confusion_matrix(ys["validation"], np.full_like(ys["validation"], majority)))}
    if include_test:
        out["majority_class"]["test"] = metrics_from_cm(
            confusion_matrix(ys["test"], np.full_like(ys["test"], majority)))
    features = {split: colour_histogram(images) for split, images in xs.items()}
    centroids = np.stack([features["train"][ys["train"] == k].mean(axis=0)
                          if (ys["train"] == k).any() else np.zeros(features["train"].shape[1])
                          for k in range(N_CLS)])
    candidates = []
    for kind in ("l1", "chi2"):
        pred = _distance(features["validation"], centroids, kind).argmin(axis=1)
        candidates.append((metrics_from_cm(confusion_matrix(ys["validation"], pred))["macro_f1"], kind))
    _, best_kind = max(candidates)
    out["colour_histogram_centroid"] = {"distance": best_kind, "bins": 16,
                                          "validation": metrics_from_cm(confusion_matrix(
                                              ys["validation"], _distance(features["validation"], centroids, best_kind).argmin(axis=1)))}
    if include_test:
        out["colour_histogram_centroid"]["test"] = metrics_from_cm(confusion_matrix(
            ys["test"], _distance(features["test"], centroids, best_kind).argmin(axis=1)))
    return out


def _data_fingerprint(dirs: dict[str, Path], image_size: int) -> str:
    h = hashlib.sha256(f"image_size={image_size};class-order={CLASSES}".encode())
    for split in ("train", "validation", "test"):
        root = dirs[split]
        h.update(split.encode())
        for cls in CLASSES:
            folder = root / cls
            if not folder.is_dir():
                h.update(f"MISSING:{folder}".encode())
                continue
            for p in sorted(folder.iterdir()):
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                    h.update(f"{p.relative_to(root)}:{content_hash}".encode())
    return h.hexdigest()


def _load_arrays(dirs: dict[str, Path], size: tuple[int, int]):
    xs, ys, counts, skipped = {}, {}, {}, {}
    for split, directory in dirs.items():
        images, onehot, paths = sc.load_image_folder(directory, size)
        xs[split], ys[split] = images, onehot.argmax(axis=1).astype(np.int64)
        counts[split] = {cls: int((ys[split] == i).sum()) for i, cls in enumerate(CLASSES)}
        skipped[split] = int(len(list(directory.rglob("._*"))))
        print(f"{split}: {len(paths)} valid images; classes={counts[split]}")
    return xs, ys, counts, skipped


def _predict_labels(model: torch.nn.Module, images: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            batch = torch.from_numpy(images[start:start + 64].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            logits = model(batch.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.int64)


def _val_metrics(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, image_size: int) -> tuple[dict, np.ndarray]:
    device = next(model.parameters()).device
    pred = _predict_labels(model, x, image_size, device)
    return metrics_from_cm(confusion_matrix(y, pred)), pred


def select_final_model(results: dict[str, dict]) -> str:
    """Choose only among dropout-regularized models to meet the rubric.

    Zero-dropout configurations remain useful ablations, but the submitted CNN
    architecture is required to include dropout regularization.
    """
    candidates = {
        name: record for name, record in results.items()
        if float(record.get("config", {}).get("dropout", 0.0)) > 0.0
    }
    if not candidates:
        raise ValueError(
            "Cannot select a Midterm model: no completed configuration has dropout > 0. "
            "Include at least one regularized configuration (for example, base or aug_none)."
        )
    return max(candidates, key=lambda name: (candidates[name]["validation"]["macro_f1"],
                                             -candidates[name]["best_epoch"]))


def _overfitting_stats(hist: dict) -> dict:
    val_loss = np.asarray(hist["val_loss"])
    best_epoch = int(val_loss.argmin()) + 1
    tr_acc, va_acc = np.asarray(hist["accuracy"]), np.asarray(hist["val_accuracy"])
    return {"best_epoch": best_epoch, "min_val_loss": float(val_loss.min()),
            "final_val_loss": float(val_loss[-1]), "val_loss_rise_after_best": float(val_loss[-1] - val_loss.min()),
            "final_train_acc": float(tr_acc[-1]), "final_val_acc": float(va_acc[-1]),
            "final_gap_train_minus_val_acc": float(tr_acc[-1] - va_acc[-1]),
            "gap_at_best_epoch": float(tr_acc[best_epoch - 1] - va_acc[best_epoch - 1])}


def _normalise_config(cfg: dict) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}


def run_one(name: str, args, dirs: dict[str, Path], xs: dict, ys: dict, fingerprint: str, out_dir: Path) -> dict:
    cfg = {**BASE, **EXPERIMENTS[name]}
    result_path = out_dir / f"run_{name}.json"
    model_path = out_dir / "models" / f"{name}.pth"
    cache_key = {"fingerprint": fingerprint, "seed": args.seed, "epochs": args.epochs,
                 "batch_size": args.batch_size, "image_size": args.image_size, "config": _normalise_config(cfg)}
    if not args.force and result_path.is_file() and model_path.is_file():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("framework") == "pytorch" and cached.get("cache_key") == cache_key:
                print(f"[cache] {name}: valid PyTorch checkpoint/result match")
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    model = sc.build_model(input_shape=(args.image_size, args.image_size, 3), seed=args.seed, **cfg)
    params = sc.count_parameters(model)
    print(f"\n=== {name}: {cfg} ({params:,} trainable parameters) ===", flush=True)
    start = time.time()
    history = sc.train(model, dirs["train"], dirs["validation"], epochs=args.epochs,
                       batch_size=args.batch_size, seed=args.seed, verbose=1,
                       checkpoint_path=model_path, patience=args.patience, restore_best=True,
                       num_workers=args.num_workers, device=args.device)
    elapsed = time.time() - start
    val_metrics, _ = _val_metrics(model, xs["validation"], ys["validation"], args.image_size)
    hist = {k: [float(v) for v in values] for k, values in history.history.items()}
    res = {"framework": "pytorch", "name": name, "config": _normalise_config(cfg),
           "image_size": [args.image_size, args.image_size], "params": params,
           "epochs_requested": args.epochs, "epochs_completed": len(hist["loss"]),
           "best_epoch": history.best_epoch, "best_val_loss": history.best_val_loss,
           "seconds": elapsed, "history": hist, "validation": val_metrics,
           "overfitting": _overfitting_stats(hist), "cache_key": cache_key,
           "test": None, "note": "Test metrics intentionally omitted for non-selected ablation configurations."}
    result_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"--> val acc={val_metrics['accuracy']:.3f} val macro-F1={val_metrics['macro_f1']:.3f}; best epoch={history.best_epoch} ({elapsed:.1f}s)", flush=True)
    return res


def _plot_curves(ax_loss, ax_acc, res: dict, label: str, mark_best=True):
    hist = res["history"]
    epochs = np.arange(1, len(hist["loss"]) + 1)
    (line,) = ax_loss.plot(epochs, hist["val_loss"], label=f"{label} (val)")
    ax_loss.plot(epochs, hist["loss"], "--", color=line.get_color(), alpha=0.65, label=f"{label} (train)")
    ax_acc.plot(epochs, hist["val_accuracy"], color=line.get_color(), label=f"{label} (val)")
    ax_acc.plot(epochs, hist["accuracy"], "--", color=line.get_color(), alpha=0.65, label=f"{label} (train)")
    if mark_best:
        best = int(res["best_epoch"])
        ax_loss.axvline(best, color=line.get_color(), ls=":", alpha=0.65)
        ax_acc.axvline(best, color=line.get_color(), ls=":", alpha=0.65)


def make_figures(results: dict, baselines: dict, out_dir: Path, selected: str, selected_test: dict | None):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        _plot_curves(a1, a2, res, name)
        a1.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
        a2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
        for ax in (a1, a2): ax.grid(alpha=0.25)
        a2.legend(fontsize=7, loc="best")
        fig.suptitle(f"PyTorch training curves: {name} (dotted = best val-loss epoch)", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"curves_{name}.png", dpi=140)
        plt.close(fig)
    for axis, names in ABLATION_AXES.items():
        names = [name for name in names if name in results]
        if len(names) < 2: continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4))
        for name in names: _plot_curves(a1, a2, results[name], LABELS[name])
        a1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
        a2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
        for ax in (a1, a2): ax.grid(alpha=0.25)
        a2.legend(fontsize=7, loc="best")
        fig.suptitle(f"One-factor-at-a-time ablation: {axis}", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"ablation_{axis}.png", dpi=140)
        plt.close(fig)
    if "base" in results and "no_reg" in results:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4))
        _plot_curves(a1, a2, results["base"], "base: p=0.3 + geometric")
        _plot_curves(a1, a2, results["no_reg"], "no_reg: p=0 + no augmentation")
        a1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
        a2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
        for ax in (a1, a2): ax.grid(alpha=0.25)
        a2.legend(fontsize=7, loc="best")
        fig.suptitle("Overfitting comparison (training vs validation)")
        fig.tight_layout()
        fig.savefig(fig_dir / "overfitting_mitigation.png", dpi=140)
        plt.close(fig)
    # Crucially, this chart only compares the final selected CNN and baselines.
    if selected_test is not None:
        rows = [("majority class", baselines["majority_class"]["test"]["macro_f1"]),
                ("colour-hist centroid", baselines["colour_histogram_centroid"]["test"]["macro_f1"]),
                (f"selected CNN: {selected}", selected_test["macro_f1"])]
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        names = [r[0] for r in rows]
        values = [r[1] for r in rows]
        ax.barh(names[::-1], values[::-1])
        for i, value in enumerate(values[::-1]): ax.text(value + 0.01, i, f"{value:.3f}", va="center")
        ax.set(xlabel="Macro-F1 on held-out test split", xlim=(0, 1), title="Final test comparison (selected CNN + baselines only)")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_dir / "summary_macro_f1.png", dpi=150)
        plt.close(fig)
        cm = np.asarray(selected_test["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=range(N_CLS), yticks=range(N_CLS), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted", ylabel="True", title=f"Selected PyTorch CNN: {selected} (test confusion matrix)")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(N_CLS):
            for j in range(N_CLS): ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.045)
        fig.tight_layout()
        fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)


def write_tables(results: dict, baselines: dict, out_dir: Path, selected: str, selected_test: dict | None, ci: dict | None):
    rows: list[dict] = []
    for name in ("majority_class", "colour_histogram_centroid"):
        va, te = baselines[name]["validation"], baselines[name].get("test", {})
        rows.append({"model": name, "framework": "baseline", "params": 0, "best_epoch": "",
                     "val_accuracy": va["accuracy"], "val_macro_f1": va["macro_f1"],
                     "test_accuracy": te.get("accuracy", ""), "test_macro_precision": te.get("macro_precision", ""),
                     "test_macro_recall": te.get("macro_recall", ""), "test_macro_f1": te.get("macro_f1", ""),
                     "test_macro_f1_ci95": "", "selected": False})
    for name, res in results.items():
        te = selected_test if name == selected and selected_test is not None else {}
        ci_text = ""
        if name == selected and ci:
            lo, hi = ci["macro_f1"]
            ci_text = f"[{lo:.3f}, {hi:.3f}]"
        va = res["validation"]
        rows.append({"model": name, "framework": "pytorch", "params": res["params"], "best_epoch": res["best_epoch"],
                     "val_accuracy": va["accuracy"], "val_macro_f1": va["macro_f1"],
                     "test_accuracy": te.get("accuracy", ""), "test_macro_precision": te.get("macro_precision", ""),
                     "test_macro_recall": te.get("macro_recall", ""), "test_macro_f1": te.get("macro_f1", ""),
                     "test_macro_f1_ci95": ci_text, "selected": name == selected})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# PyTorch experiment summary", "",
             "The final model is selected by validation macro-F1 among configurations with dropout > 0, as required by the Midterm rubric. Zero-dropout variants remain ablations. Test metrics are reported only for the selected CNN and the two required baselines.", "",
             "| Model | Framework | Params | Best epoch | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 | Test macro-F1 95% CI |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        def fmt(value): return "-" if value == "" else (f"{value:.3f}" if isinstance(value, float) else str(value))
        model = r["model"] + (" **(selected)**" if r["selected"] else "")
        lines.append(f"| {model} | {r['framework']} | {r['params']:,} | {r['best_epoch'] or '-'} | {fmt(r['val_accuracy'])} | {fmt(r['val_macro_f1'])} | {fmt(r['test_accuracy'])} | {fmt(r['test_macro_f1'])} | {r['test_macro_f1_ci95'] or '-'} |")
    lines.extend(["", "## Interpretation notes", "",
                  "- Training epoch selection uses minimum validation loss.",
                  "- Configuration selection uses validation macro-F1 among dropout-regularized configurations (dropout > 0); test metrics do not influence selection.",
                  "- CI is a percentile bootstrap over individual test images and does not measure seed-to-seed training variability.",
                  "- Per-configuration ablation conclusions must be based on validation results and learning curves, not hidden test results."])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default="", help="Comma-separated experiment names; default runs all ten")
    parser.add_argument("--force", action="store_true", help="Ignore cached results")
    parser.add_argument("--quick", action="store_true", help="2-epoch infrastructure smoke test on synthetic data/sample only")
    parser.add_argument("--patience", type=int, default=None, help="Optional early-stopping patience; default trains full epoch budget")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto (default: auto)")
    args = parser.parse_args(argv)
    if args.quick:
        args.epochs = 2
        args.out = ROOT / "results_quick"
        dirs = {split: args.data_dir / "sample" for split in ("train", "validation", "test")}
        names = ["base", "aug_none"]
    else:
        dirs = {split: args.data_dir / split for split in ("train", "validation", "test")}
        missing = [str(path) for path in dirs.values() if not path.is_dir()]
        if missing:
            raise SystemExit("Missing dataset splits: " + ", ".join(missing) + ". Run python scripts/prepare_dataset.py --confirm first.")
        names = [x.strip() for x in args.only.split(",") if x.strip()] or list(EXPERIMENTS)
    invalid = [name for name in names if name not in EXPERIMENTS]
    if invalid: raise SystemExit(f"Unknown experiment(s) {invalid}; choose from {list(EXPERIMENTS)}")
    args.device = args.device if args.device and args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available; use --device cpu or auto.")
    args.out.mkdir(parents=True, exist_ok=True)
    size = (args.image_size, args.image_size)
    xs, ys, class_counts, skipped = _load_arrays(dirs, size)
    if args.quick:
        data_mode = "synthetic_sample_smoke_only_not_a_performance_evaluation"
        fingerprint = hashlib.sha256((str(args.seed) + str(args.image_size) + "sample-smoke").encode()).hexdigest()
    else:
        data_mode = "full_trashnet"
        fingerprint = _data_fingerprint(dirs, args.image_size)
    (args.out / "dataset_summary.json").write_text(json.dumps({"data_mode": data_mode, "seed": args.seed,
        "image_size": size, "class_order": CLASSES, "class_counts": class_counts,
        "skipped_metadata_candidate_counts": skipped, "fingerprint": fingerprint}, indent=2), encoding="utf-8")
    # Validation-only baseline selection happens before any held-out test evaluation.
    baselines = run_baselines(xs, ys, include_test=False)
    results = {}
    for name in names:
        results[name] = run_one(name, args, dirs, xs, ys, fingerprint, args.out)
    # Include same-fingerprint cached runs when the complete experiment grid is available.
    for name in EXPERIMENTS:
        rp = args.out / f"run_{name}.json"
        mp = args.out / "models" / f"{name}.pth"
        expected_cfg = _normalise_config({**BASE, **EXPERIMENTS[name]})
        expected_cache_key = {"fingerprint": fingerprint, "seed": args.seed, "epochs": args.epochs,
                              "batch_size": args.batch_size, "image_size": args.image_size,
                              "config": expected_cfg}
        if not args.force and name not in results and rp.is_file() and mp.is_file():
            try:
                cached = json.loads(rp.read_text(encoding="utf-8"))
                if cached.get("framework") == "pytorch" and cached.get("cache_key") == expected_cache_key:
                    results[name] = cached
            except (OSError, json.JSONDecodeError):
                pass
    results = {name: results[name] for name in EXPERIMENTS if name in results}
    if not results:
        raise SystemExit("No experiment results were produced.")
    try:
        selected = select_final_model(results)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selected_path = args.out / "models" / f"{selected}.pth"
    selected_model = sc.load_trained_model(selected_path, device=args.device)
    test_metrics = None
    ci = None
    if not args.quick:
        pred_test = _predict_labels(selected_model, xs["test"], args.image_size, torch.device(args.device))
        test_metrics = metrics_from_cm(confusion_matrix(ys["test"], pred_test))
        ci = bootstrap_ci(ys["test"], pred_test, n_boot=1000, seed=args.seed)
        results[selected]["test"] = test_metrics
        results[selected]["test_ci95"] = ci
        results[selected]["test_prediction_rule"] = "evaluated once after validation-only model selection"
        (args.out / f"run_{selected}.json").write_text(json.dumps(results[selected], indent=2), encoding="utf-8")
        baselines = run_baselines(xs, ys, include_test=True)
        # Save the selected PyTorch checkpoint in the one canonical demo location.
        sc._save_checkpoint(ROOT / "models" / "waste_cnn.pth", selected_model, epoch=results[selected]["best_epoch"],
                           val_loss=results[selected]["best_val_loss"], config=results[selected]["config"])
    else:
        results[selected]["test"] = None
        results[selected]["test_ci95"] = None
    if not args.quick:
        # Remove any historical test field from cached results that are not selected.
        for run_name, run_result in results.items():
            if run_name == selected:
                continue
            run_result.pop("test", None)
            run_result.pop("test_ci95", None)
            run_result.pop("test_prediction_rule", None)
            (args.out / f"run_{run_name}.json").write_text(json.dumps(run_result, indent=2), encoding="utf-8")
    (args.out / "baselines.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    selected_record = {"selected": selected,
                       "criterion": "highest validation macro-F1 among completed configurations with dropout > 0; epoch selected by validation loss",
                       "selection_constraint": "dropout > 0 to satisfy the Midterm CNN architecture requirement; no_reg and drop_0.0 remain ablations only",
                       "framework": "pytorch", "data_mode": data_mode, "config": results[selected]["config"],
                       "best_epoch": results[selected]["best_epoch"], "validation": results[selected]["validation"],
                       "test": test_metrics, "test_ci95": ci,
                       "note": "Quick mode uses synthetic sample images only and is not a valid performance evaluation." if args.quick else None}
    (args.out / "selected.json").write_text(json.dumps(selected_record, indent=2), encoding="utf-8")
    make_figures(results, baselines, args.out, selected, test_metrics)
    write_tables(results, baselines, args.out, selected, test_metrics, ci)
    (args.out / "experiment_metadata.json").write_text(json.dumps({"framework": "pytorch", "data_mode": data_mode,
        "seed": args.seed, "device": args.device, "epochs": args.epochs, "batch_size": args.batch_size,
        "image_size": args.image_size, "fingerprint": fingerprint, "experiments_run_this_command": names,
        "selected": selected, "selection_constraint": "dropout > 0", "test_used_for_selection": False}, indent=2), encoding="utf-8")
    print(f"\nSelected by validation macro-F1: {selected}; data mode={data_mode}; output={args.out}")
    print((args.out / "summary.md").read_text(encoding="utf-8"))
    if args.quick:
        print("WARNING: these are synthetic-sample smoke-test results, not TrashNet test performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
