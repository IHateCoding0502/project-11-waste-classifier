"""Offline unit and smoke tests for the implemented PyTorch AI Core."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

sc = _load_module("project_11_waste_classifier_student_core", ROOT / "starter" / "student_core.py")
exp = _load_module("project_11_run_experiments", ROOT / "scripts" / "run_experiments.py")
prep = _load_module("project_11_prepare_dataset", ROOT / "scripts" / "prepare_dataset.py")
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SAMPLE_DIR = ROOT / "data" / "sample"


def test_classes_and_descriptions_match_trashnet():
    assert sc.CLASSES == CLASSES
    assert set(sc.CLASS_TO_IDX) == set(CLASSES)
    assert all(sc.class_description(name) for name in CLASSES)


def test_license_documents_pinned_source_and_checksum():
    text = (ROOT / "data" / "LICENSE.md").read_text(encoding="utf-8").lower()
    assert "github.com/garythung/trashnet" in text
    assert "mit" in text and "sha256" in text and "checksum" in text


def test_sample_data_has_each_class_and_stays_small():
    image_files = []
    for cls in CLASSES:
        files = [p for p in (SAMPLE_DIR / cls).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        assert files, f"missing sample for {cls}"
        image_files.extend(files)
    assert len(image_files) <= 20
    assert any("hard" in p.name for p in image_files)
    for path in image_files:
        with Image.open(path) as image:
            image.verify()


def test_full_dataset_is_not_bundled_in_submission_tree():
    # Local datasets are allowed; tracked submission files must exclude them.
    ignored = (ROOT / '.gitignore').read_text(encoding='utf-8-sig')
    assert all(f'data/{split}/' in ignored for split in ('train','validation','test'))
    if not (ROOT / '.git').is_dir():
        # A course archive intentionally has no Git metadata. Its content is
        # checked independently by scripts/package_submission.py.
        return
    import subprocess
    tracked = subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    assert not any(p.startswith(('data/train/','data/validation/','data/test/')) for p in tracked)
    assert 'data/dataset-resized.zip' not in tracked


def test_active_pipeline_is_pytorch_not_keras():
    source = (ROOT / "starter" / "student_core.py").read_text(encoding="utf-8").lower()
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "import tensorflow" not in source
    assert "tensorflow==" not in requirements and "keras" not in requirements
    assert "torch" in source and "torch" in requirements


def test_model_architecture_and_parameter_count():
    model = sc.build_model((64, 64, 3), seed=7)
    assert isinstance(model, torch.nn.Module)
    assert sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()) == 4
    assert sum(isinstance(m, torch.nn.MaxPool2d) for m in model.modules()) == 4
    assert sum(isinstance(m, torch.nn.Dropout) for m in model.modules()) == 2
    assert isinstance(model.gap, torch.nn.AdaptiveAvgPool2d)
    assert sc.count_parameters(model) > 100_000
    with torch.inference_mode():
        logits = model(torch.rand(2, 3, 64, 64))
    assert logits.shape == (2, len(CLASSES))
    loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    assert torch.isfinite(loss)


def test_model_configuration_validation():
    for kwargs in ({"dropout": 1.0}, {"filters": ()}, {"num_classes": 1}, {"augmentation": "bogus"}, {"learning_rate": 0}):
        with pytest.raises(ValueError):
            sc.build_model((32, 32, 3), **kwargs)
    shallow = sc.build_model((32, 32, 3), filters=(8, 16), dropout=0.5, augmentation="none")
    assert sum(isinstance(m, torch.nn.Conv2d) for m in shallow.modules()) == 2
    assert [m.p for m in shallow.modules() if isinstance(m, torch.nn.Dropout)] == [0.5, 0.5]


def test_loader_excludes_hidden_metadata_and_invalid_images(tmp_path: Path):
    cls_dir = tmp_path / "cardboard"
    cls_dir.mkdir()
    good = next((SAMPLE_DIR / "cardboard").glob("*.png"))
    (cls_dir / "box.png").write_bytes(good.read_bytes())
    (cls_dir / "._box.png").write_bytes(good.read_bytes())
    (cls_dir / ".hidden.png").write_bytes(good.read_bytes())
    (cls_dir / "broken.jpg").write_bytes(b"not an image")
    ds = sc.TrashNetDataset(tmp_path, (32, 32), augmentation="none")
    assert len(ds) == 1
    assert ds.samples[0][0].name == "box.png"
    assert "broken.jpg" in ds.skipped[0]
    batch, label = ds[0]
    assert batch.shape == (3, 32, 32) and batch.dtype == torch.float32
    assert float(batch.min()) >= 0 and float(batch.max()) <= 1
    assert label == 0


def test_training_step_best_checkpoint_roundtrip_and_predictions(tmp_path: Path):
    model = sc.build_model((32, 32, 3), filters=(4, 8), dropout=0.1,
                           augmentation="none", learning_rate=1e-3, seed=11)
    checkpoint = tmp_path / "smoke_model.pth"
    history = sc.train(model, SAMPLE_DIR, SAMPLE_DIR, epochs=1, batch_size=4, seed=11,
                       verbose=0, device="cpu", checkpoint_path=checkpoint, reduce_lr=False)
    assert len(history.history["loss"]) == len(history.history["val_loss"]) == 1
    assert all(np.isfinite(history.history[k]).all() for k in ("loss", "accuracy", "val_loss", "val_accuracy"))
    assert checkpoint.is_file() and history.best_epoch == 1
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["classes"] == list(CLASSES)
    assert saved["preprocessing"]["color_mode"] == "RGB"
    assert saved["preprocessing"]["resize_hw"] == [32, 32]
    loaded = sc.load_trained_model(checkpoint, device="cpu")
    sample = next((SAMPLE_DIR / "cardboard").glob("*.png"))
    label, confidence = sc.predict(loaded, sample)
    probs = sc.predict_proba(loaded, sample)
    assert label in CLASSES and 0 <= confidence <= 1
    assert set(probs) == set(CLASSES) and sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    with sample.open("rb") as fh:
        assert sc.predict(loaded, fh)[0] in CLASSES
    with pytest.raises(ValueError):
        sc.predict(None, sample)


def test_evaluation_metrics_match_hand_computation():
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
    metrics = exp.metrics_from_cm(exp.confusion_matrix(y_true, y_pred))
    assert metrics["accuracy"] == pytest.approx(11 / 12)
    assert metrics["per_class"]["glass"]["f1"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["metal"]["f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx((4 + 2 / 3 + 0.8) / 6)


def test_final_model_selection_requires_dropout_regularization():
    fake_results = {
        "no_reg": {"config": {"dropout": 0.0}, "validation": {"macro_f1": 0.95}, "best_epoch": 2},
        "drop_0.0": {"config": {"dropout": 0.0}, "validation": {"macro_f1": 0.90}, "best_epoch": 2},
        "base": {"config": {"dropout": 0.3}, "validation": {"macro_f1": 0.70}, "best_epoch": 3},
        "aug_none": {"config": {"dropout": 0.3}, "validation": {"macro_f1": 0.75}, "best_epoch": 4},
    }
    assert exp.select_final_model(fake_results) == "aug_none"
    with pytest.raises(ValueError, match="dropout > 0"):
        exp.select_final_model({"no_reg": fake_results["no_reg"]})


def test_baselines_run_without_using_test_for_selection():
    x, y_onehot, _ = sc.load_image_folder(SAMPLE_DIR, (32, 32))
    y = y_onehot.argmax(1)
    data = {"train": x, "validation": x, "test": x}
    labels = {"train": y, "validation": y, "test": y}
    results = exp.run_baselines(data, labels, include_test=False)
    assert set(results) == {"majority_class", "colour_histogram_centroid"}
    assert "validation" in results["majority_class"] and "test" not in results["majority_class"]
    assert 0 <= results["colour_histogram_centroid"]["validation"]["macro_f1"] <= 1


def test_balanced_class_weights_use_only_supplied_training_labels():
    weights = sc._balanced_weights([0, 0, 1], len(CLASSES)).numpy()
    assert weights[0] == pytest.approx(0.75)
    assert weights[1] == pytest.approx(1.5)
    assert np.all(weights[2:] == 0.0)


def test_train_pipeline_uses_augmentation_only_for_training(monkeypatch, tmp_path: Path):
    original = sc.TrashNetDataset
    seen = []
    class TrackingDataset(original):
        def __init__(self, root, image_size=(64, 64), augmentation="none", validate_images=True):
            seen.append((Path(root), augmentation))
            super().__init__(root, image_size, augmentation, validate_images)
    monkeypatch.setattr(sc, "TrashNetDataset", TrackingDataset)
    model = sc.build_model((32, 32, 3), filters=(4, 8), dropout=0.1, augmentation="full", seed=13)
    sc.train(model, SAMPLE_DIR, SAMPLE_DIR, epochs=1, batch_size=4, verbose=0, device="cpu", reduce_lr=False)
    assert seen == [(SAMPLE_DIR, "full"), (SAMPLE_DIR, "none")]


def test_prepare_dataset_dry_run_has_no_download(capsys):
    assert prep.main([]) == 0
    output = capsys.readouterr().out
    assert "Dry run" in output and "Would download" in output


def test_prepare_dataset_excludes_metadata_and_corrupt_images(tmp_path: Path):
    archive = tmp_path / "tiny.zip"
    sample_images = {cls: next((SAMPLE_DIR / cls).glob("*.png")).read_bytes() for cls in CLASSES}
    with zipfile.ZipFile(archive, "w") as zf:
        for cls, content in sample_images.items():
            for i in range(10):
                zf.writestr(f"dataset-resized/{cls}/{cls}_{i:02}.png", content)
            zf.writestr(f"dataset-resized/{cls}/._{cls}_metadata.jpg", content)
            zf.writestr(f"dataset-resized/{cls}/{cls}_broken.jpg", b"not an image")
    out = tmp_path / "out"
    manifest = prep.extract_and_split(archive, out, seed=123)
    assert len(manifest["excluded_hidden_or_metadata"]) == len(CLASSES)
    assert len(manifest["excluded_invalid_images"]) == len(CLASSES)
    for split in ("train", "validation", "test"):
        files = [p for p in (out / split).rglob("*") if p.is_file()]
        assert files and all(not p.name.startswith(".") for p in files)
        assert not any("broken" in p.name or p.name.startswith("._") for p in files)
    saved = json.loads((out / "split_manifest.json").read_text())
    assert saved["seed"] == 123 and saved["counts"]["train"]["cardboard"] == 7
    out2 = tmp_path / "out_repeat"
    repeated = prep.extract_and_split(archive, out2, seed=123)
    assert repeated["splits"] == manifest["splits"]


def test_streamlit_app_uses_pytorch_checkpoint_and_syntax_compiles():
    import py_compile
    app_source = (ROOT / "starter" / "app.py").read_text(encoding="utf-8")
    assert 'waste_cnn.pth' in app_source
    assert "load_trained_model" in app_source and "render_prediction" in app_source
    assert ".keras" not in app_source and "tensorflow" not in app_source.lower()
    py_compile.compile(str(ROOT / "starter" / "app.py"), doraise=True)
