"""Inference safety, session accounting and real camera-widget input handling."""
import io
import os
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from starter import dashboard_classify as ui
from starter.inference import InvalidPrediction, infer
from starter.student_core import CLASSES, build_model, image_to_tensor


def test_predecoded_rgb_uses_evaluation_tensor_without_reencoding(monkeypatch):
    image = Image.new("RGB", (43, 19), (121, 37, 232))
    image.putpixel((12, 9), (5, 122, 61))
    file = io.BytesIO()
    image.save(file, "PNG")
    expected = image_to_tensor(file, (64, 64))
    torch.testing.assert_close(image_to_tensor(image, (64, 64)), expected, rtol=0, atol=0)
    model = build_model(augmentation="none")
    model.eval()
    with torch.inference_mode():
        expected_scores = model(expected.unsqueeze(0)).softmax(1)[0].tolist()
    monkeypatch.setattr(Image.Image, "save", Mock(side_effect=AssertionError("redundant image encode")))
    result = infer(model, image)
    assert list(result["scores"]) == list(CLASSES)
    assert list(result["scores"].values()) == pytest.approx(expected_scores)
    assert result["label"] == CLASSES[expected_scores.index(max(expected_scores))]


@pytest.mark.parametrize("bad", [torch.zeros(1, 5), torch.full((1, 6), float("nan")),
                                 torch.ones(1, 6, dtype=torch.long), {"logits": None}])
def test_invalid_scores_are_rejected(monkeypatch, bad):
    model = build_model(augmentation="none")
    monkeypatch.setattr(model, "forward", lambda _: bad)
    with pytest.raises(InvalidPrediction):
        infer(model, Image.new("RGB", (32, 32)))


def test_cache_stats_change_only_on_new_successful_cnn_runs(monkeypatch, tmp_path):
    path = tmp_path / "model.pth"
    path.touch()
    monkeypatch.setattr(ui, "CHECKPOINT", path)
    monkeypatch.setattr(ui.st, "session_state", {"inference_device": "CPU"})
    load = Mock(return_value=object())
    run = Mock(return_value={"label": "glass", "seconds": 0.012})
    monkeypatch.setattr(ui, "load_trained_model", load)
    monkeypatch.setattr(ui, "infer", run)
    ui.get_model.clear()
    image = Image.new("RGB", (32, 32))
    assert ui.get_session_stats() == (0, None)
    assert ui.cnn_status() == "Unverified"
    ui.predict(image, "a")
    ui.predict(image, "a")
    ui.predict(image, "b")
    assert load.call_count == 1 and run.call_count == 2
    assert ui.get_session_stats() == (2, 12.0)
    assert ui.cnn_status() == "Available"
    timestamp = path.stat().st_mtime_ns + 2_000_000_000
    os.utime(path, ns=(timestamp, timestamp))
    assert ui.cnn_status() == "Unverified"
    ui.predict(image, "a")
    assert load.call_count == 2 and run.call_count == 3
    assert ui.get_session_stats() == (3, 12.0)
    ui.get_model.clear()


def test_missing_or_incompatible_checkpoint_gives_public_error(monkeypatch, tmp_path):
    path = tmp_path / "model.pth"
    monkeypatch.setattr(ui, "CHECKPOINT", path)
    monkeypatch.setattr(ui.st, "session_state", {"inference_device": "CPU"})
    image = Image.new("RGB", (16, 16))
    with pytest.raises(ui.PredictionError, match="missing"):
        ui.predict(image, "image")
    assert ui.cnn_status() == "Checkpoint missing"
    torch.save({"format": "project11-pytorch-v1", "classes": list(reversed(CLASSES))}, path)
    with pytest.raises(ui.PredictionError, match="six-class order") as error:
        ui.predict(image, "image")
    assert str(tmp_path) not in str(error.value)
    assert ui.cnn_status() == "Unavailable"
    assert ui.get_session_stats() == (0, None)


def test_gpu_memory_error_retries_once_on_cpu(monkeypatch):
    monkeypatch.setattr(ui.st, "session_state", {})
    monkeypatch.setattr(ui, "checkpoint_token", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    run = Mock(side_effect=[torch.OutOfMemoryError("CUDA OOM"), {"seconds": 0.005}])
    monkeypatch.setattr(ui, "_run_cnn", run)
    image = Image.new("RGB", (16, 16))
    result = ui.predict(image, "photo")
    assert [c.args[-1] for c in run.call_args_list] == ["cuda", "cpu"]
    assert ui.get_runtime_device() == "cpu" and "CPU" in result["fallback"]
    assert ui.predict(image, "photo") is result
    assert ui.get_session_stats() == (1, 5.0)


def test_camera_input_uses_the_validated_upload_path(monkeypatch):
    raw = io.BytesIO()
    Image.new("RGB", (36, 21), "green").save(raw, "JPEG")
    captured = Mock(name="capture")
    captured.name = "camera.jpg"
    captured.getvalue.return_value = raw.getvalue()
    captured.file_id = "capture-1"
    monkeypatch.setattr(ui.st, "camera_input", lambda *args, **kwargs: captured)
    app = AppTest.from_string(
        "from starter.dashboard_classify import image_input\nimage_input('camera_test')"
    ).run()
    app.radio[0].set_value("Camera").run()
    assert not app.exception
    assert any("Original 36 × 21" in caption.value for caption in app.caption)
    app.run()
    assert captured.getvalue.call_count == 1
    next(button for button in app.button if button.label == "Remove image").click().run()
    assert not app.exception and app.radio[0].value == "Upload"
    assert "eco_image" not in app.session_state
