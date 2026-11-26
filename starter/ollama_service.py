"""Optional loopback-only Ollama comparison, independent of Streamlit and its reruns."""
from __future__ import annotations

import base64
import http.client
import io
import json
import logging
import time
from dataclasses import dataclass

from PIL import Image

from starter.student_core import CLASSES


DEFAULT_MODEL = "qwen2.5vl:3b"
QWEN_MAX_SIDE = 448
PROMPT = (
    "Classify this waste image into exactly ONE category: "
    + ", ".join(CLASSES)
    + ". Reply with only one category name."
)
JSON_HINT = ' Respond as JSON: {"label": "<category>"}'
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(CLASSES)}},
    "required": ["label"],
}
STRUCTURED_OPTIONS = {"temperature": 0, "num_predict": 30, "repeat_penalty": 1.1, "num_ctx": 2048}
_LOG = logging.getLogger(__name__)
_MAX_RESPONSE = 1024 * 1024


class OllamaError(RuntimeError):
    """User-facing error with a short connection status for the sidebar."""

    def __init__(self, message: str, status: str = "Unavailable"):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class QwenPrediction:
    label: str | None
    response: str
    seconds: float


def encode_image(image: Image.Image, max_side: int = QWEN_MAX_SIDE) -> bytes:
    """Use the project's benchmark JPEG recipe without mutating the source."""
    if image.mode == "RGB" and max(image.size) <= max_side:
        thumbnail = image
    else:
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((max_side, max_side))
    stream = io.BytesIO()
    thumbnail.save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def _request(path: str, timeout: float, payload: dict | None = None) -> dict:
    # Direct HTTPConnection ignores proxy/OLLAMA_HOST settings and never redirects.
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=timeout)
    try:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        connection.request("POST" if body else "GET", path, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status == 404 and path == "/api/chat":
            raise OllamaError("The local Ollama server could not find qwen2.5vl:3b.", "Model missing")
        if response.status != 200:
            _LOG.warning("Local Ollama %s returned HTTP %d", path, response.status)
            raise OllamaError("Ollama could not complete this request. Check its local service log.")
        data = response.read(_MAX_RESPONSE + 1)
        if len(data) > _MAX_RESPONSE:
            raise OllamaError("Ollama returned an unusually large response.")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise OllamaError("Ollama returned an unreadable response.")
        return result
    except TimeoutError as exc:
        _LOG.warning("Local Ollama request timed out on %s", path)
        raise OllamaError("Ollama timed out. Increase the timeout or try again once the model is ready.") from exc
    except (OSError, http.client.HTTPException) as exc:
        _LOG.info("Local Ollama connection failed on %s (%s)", path, type(exc).__name__)
        raise OllamaError("Could not reach Ollama on this computer. Open the local Ollama app and try again.", "Offline") from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        _LOG.warning("Invalid JSON from local Ollama on %s", path)
        raise OllamaError("Ollama returned an unreadable response.") from exc
    finally:
        connection.close()


def model_available() -> bool:
    """Inspect installed tags only. Listing does not load a model."""
    models = _request("/api/tags", timeout=2).get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama returned an unreadable model list.")
    return any(isinstance(item, dict) and DEFAULT_MODEL in (item.get("name"), item.get("model"))
               for item in models)


def response_label(text: str) -> str | None:
    """Accept an exact class answer or structured label; ambiguous prose stays unlabeled."""
    answer = text.strip().lower()
    if answer in CLASSES:
        return answer
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    label = data.get("label") if isinstance(data, dict) else None
    if isinstance(label, str) and label.strip().lower() in CLASSES:
        return label.strip().lower()
    return None


def classify_qwen(image: Image.Image, timeout: int = 60) -> QwenPrediction:
    if not 1 <= timeout <= 180:
        raise ValueError("Choose a timeout between 1 and 180 seconds.")
    start = time.perf_counter()
    try:
        image_b64 = base64.b64encode(encode_image(image)).decode("ascii")
    except (MemoryError, OSError, ValueError) as exc:
        _LOG.warning("Could not prepare image for local Qwen (%s)", type(exc).__name__)
        raise OllamaError("Could not prepare this photo for Qwen. Try a smaller image.") from exc
    reply = _request("/api/chat", timeout, {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": PROMPT + JSON_HINT, "images": [image_b64]}],
        "format": LABEL_SCHEMA,
        "options": STRUCTURED_OPTIONS,
        "stream": False,
        "keep_alive": "2m",
    })
    elapsed = time.perf_counter() - start
    message = reply.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip() or reply.get("done") is False:
        raise OllamaError("Ollama returned no complete model answer. Please try again.")
    return QwenPrediction(response_label(content), content[:2000], elapsed)
