"""Validated uploads and shared, timed inference for the app and evaluation."""
from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from starter.student_core import CLASSES, image_to_tensor

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_SIDE = 8192


@dataclass
class Upload:
    image: Image.Image
    format: str
    original_size: tuple[int, int]


def decode_upload(data: bytes) -> Upload:
    """Check bytes before decoding; composite transparency on white for display."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Upload must contain an image and be at most 10 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                size = probe.size
                if fmt not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG image contents are supported.")
                if max(size) > MAX_SIDE or size[0] * size[1] > MAX_PIXELS:
                    raise ValueError("Image exceeds 8192 pixels per side or 20 megapixels.")
                probe.verify()
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                image = ImageOps.exif_transpose(source)
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba)
                return Upload(image.convert("RGB"), fmt, size)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid, corrupted, or unsafe image. Please use a valid JPEG/PNG.") from exc


def infer(model, image: Image.Image) -> dict:
    """Measure batch-one forward + softmax, excluding decode/resize/transfer.

    Preprocessing is exactly the training RGB/bilinear/[0,1] transform.
    EXIF and alpha handling are explicitly upload-only adaptations.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    device = next(model.parameters()).device
    tensor = image_to_tensor(buffer, model.image_size).unsqueeze(0).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        if logits.shape != (1, 6) or not torch.isfinite(logits).all():
            raise ValueError("Checkpoint must produce exactly six finite logits.")
        scores = torch.softmax(logits, dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    values = scores[0].cpu().tolist()
    label = CLASSES[max(range(6), key=values.__getitem__)]
    return {"label": label, "confidence": max(values),
            "scores": dict(zip(CLASSES, values)), "seconds": elapsed,
            "device": str(device)}
