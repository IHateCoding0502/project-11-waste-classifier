"""Local, deterministic tests for the optional model service and comparison UI."""
import base64
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starter import dashboard_compare as ui
from starter import ollama_service as service


@pytest.mark.parametrize(("answer", "expected"), [
    ('{"label": "plastic"}', "plastic"),
    (" Paper\n", "paper"),
    ("possibly metal or paper", None),
    ('{"label":"battery"}', None),
    ('["paper"]', None),
])
def test_qwen_label_requires_one_supported_answer(answer, expected):
    assert service.response_label(answer) == expected


def test_local_chat_contract(monkeypatch):
    image = Image.new("RGB", (670, 335))

    def fake_request(path, timeout, payload):
        assert path == "/api/chat" and timeout == 10
        assert payload["model"] == "qwen2.5vl:3b" and payload["stream"] is False
        assert tuple(payload["format"]["properties"]["label"]["enum"]) == service.CLASSES
        encoded = payload["messages"][0]["images"][0]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as thumbnail:
            assert thumbnail.format == "JPEG" and thumbnail.size == (448, 224)
        return {"done": True, "message": {"content": '{"label":"glass"}'}}

    monkeypatch.setattr(service, "_request", fake_request)
    result = service.classify_qwen(image, 10)
    assert (result.label, result.response) == ("glass", '{"label":"glass"}')
    assert result.seconds >= 0 and image.size == (670, 335)


@pytest.mark.parametrize(("status", "payload", "error"), [
    (404, b"", "Model missing"),
    (500, b"internal diagnostic", "Unavailable"),
    (200, b"not json", "Unavailable"),
    (200, b'{"message":{}}', "Unavailable"),
    (200, b'{"done":false,"message":{"content":"metal"}}', "Unavailable"),
])
def test_bad_service_responses_are_safe(monkeypatch, status, payload, error):
    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("127.0.0.1", 11434)

        def request(self, method, path, body, headers):
            assert method == "POST" and path == "/api/chat"

        def getresponse(self):
            return SimpleNamespace(status=status, read=lambda size: payload)

        def close(self):
            pass

    monkeypatch.setattr(service.http.client, "HTTPConnection", Connection)
    with pytest.raises(service.OllamaError) as caught:
        service.classify_qwen(Image.new("RGB", (10, 10)))
    assert caught.value.status == error
    assert "internal diagnostic" not in str(caught.value)


@pytest.mark.parametrize(("failure", "status"), [
    (ConnectionRefusedError, "Offline"), (TimeoutError, "Unavailable"),
])
def test_status_failure_without_inference(monkeypatch, failure, status):
    class Connection:
        def __init__(self, host, port, timeout):
            pass

        def request(self, method, path, body, headers):
            assert method == "GET" and path == "/api/tags" and body is None
            raise failure("private local diagnostic")

        def close(self):
            pass

    monkeypatch.setattr(service.http.client, "HTTPConnection", Connection)
    with pytest.raises(service.OllamaError) as caught:
        service.model_available()
    assert caught.value.status == status and "private" not in str(caught.value)


def test_ui_requires_click_recovers_offline_and_invalidates_results(monkeypatch):
    selected = SimpleNamespace(image=Image.new("RGB", (5, 5)), digest="image-one")
    calls = {"cnn": 0, "qwen": 0}
    token = [100]
    monkeypatch.setattr(ui, "image_input", lambda namespace: selected)
    monkeypatch.setattr(ui, "checkpoint_token", lambda: token[0])
    monkeypatch.setattr(ui, "get_runtime_device", lambda: "cpu")

    def fake_predict(image, digest):
        calls["cnn"] += 1
        return {"label": "glass", "confidence": 1.0, "seconds": 0.003, "device": "cpu",
                "scores": {label: float(label == "glass") for label in service.CLASSES}}

    def fake_qwen(image, timeout):
        calls["qwen"] += 1
        if calls["qwen"] == 1:
            raise service.OllamaError("Local service is offline", "Offline")
        return service.QwenPrediction("glass", '{"label":"glass"}', 0.8)

    monkeypatch.setattr(ui, "predict", fake_predict)
    monkeypatch.setattr(ui, "classify_qwen", fake_qwen)
    app = AppTest.from_string("from starter.dashboard_compare import compare_page\ncompare_page()").run(timeout=30)
    assert not app.exception and calls == {"cnn": 0, "qwen": 0}
    app.button(key="compare_run").click().run()
    assert not app.exception and calls == {"cnn": 1, "qwen": 1}
    assert app.session_state["qwen_status"] == "Offline"
    assert app.metric[0].value == "Glass" and "offline" in app.warning[0].value.lower()
    app.number_input[0].set_value(75).run()
    assert calls == {"cnn": 1, "qwen": 1}
    app.button(key="compare_run").click().run()
    assert not app.exception and calls == {"cnn": 1, "qwen": 2}
    assert app.session_state["qwen_status"] == "Available" and app.button(key="compare_run").disabled
    assert "Both models chose Glass" in app.success[0].value
    assert json.loads(app.code[0].value) == {"label": "glass"}
    token[0] += 1
    app.run()
    assert [item.label for item in app.metric] == ["Qwen prediction"]
    app.button(key="compare_run").click().run()
    assert calls == {"cnn": 2, "qwen": 2}
    selected.digest = "another-image"
    app.run()
    assert not app.exception and not app.metric and not app.code
