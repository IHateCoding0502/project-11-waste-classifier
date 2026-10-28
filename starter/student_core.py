"""PyTorch AI core for Project 11 - Visual Waste Sorting Assistant.

Custom CNN only: no pretrained model, third-party solution weights, or hosted API.
Public functions keep the original starter entry points (build_model, train,
predict) while adding probability prediction and checkpoint loading.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASSES: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
CLASS_DESCRIPTIONS: dict[str, str] = {
    "cardboard": "Flattened boxes and packaging paperboard.",
    "glass": "Bottles and jars (clear or colored glass).",
    "metal": "Cans and other small metal containers.",
    "paper": "Loose paper, flyers, and non-cardboard paper waste.",
    "plastic": "Bottles, containers, and rigid plastic packaging.",
    "trash": "Items that do not belong to the other five benchmark classes.",
}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
AUGMENTATION_MODES = ("none", "flip", "geometric", "full")


def class_description(class_name: str) -> str:
    return CLASS_DESCRIPTIONS.get(class_name, "Unknown class.")


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed random number generators for repeatable runs where supported."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _is_image_candidate(path: Path) -> bool:
    # AppleDouble files such as ._cardboard9.jpg are not real images.
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in _IMAGE_SUFFIXES


def _open_rgb(source: Any, image_size: tuple[int, int]) -> Image.Image:
    """Open a path or file-like object, return RGB PIL image resized to (W,H)."""
    if hasattr(source, "seek"):
        source.seek(0)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image = image.resize((int(image_size[1]), int(image_size[0])), Image.Resampling.BILINEAR)
        return image.copy()


def image_to_tensor(source: Any, image_size: tuple[int, int]) -> torch.Tensor:
    """Deterministic inference transform: RGB, resize, float32 in [0,1], CHW."""
    image = _open_rgb(source, image_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _apply_augmentation(image: Image.Image, mode: str) -> Image.Image:
    """PIL-only random augmentation, called only by the training Dataset."""
    if mode not in AUGMENTATION_MODES:
        raise ValueError(f"augmentation must be one of {AUGMENTATION_MODES}, got {mode!r}")
    if mode == "none":
        return image
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if mode in {"flip", "geometric", "full"} and random.random() < 0.25:
        image = ImageOps.flip(image)
    if mode in {"geometric", "full"}:
        angle = random.uniform(-28.8, 28.8)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        # Mild zoom through a random crop followed by resizing to original size.
        if random.random() < 0.7:
            w, h = image.size
            scale = random.uniform(0.85, 1.0)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            left = random.randint(0, w - nw)
            top = random.randint(0, h - nh)
            image = image.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.BILINEAR)
    if mode == "full":
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
    return image


class TrashNetDataset(Dataset):
    """Dataset for ``root/<class>/*`` with stable TrashNet label ordering."""
    def __init__(self, root: Path | str, image_size: tuple[int, int] = (64, 64),
                 augmentation: str = "none", validate_images: bool = True):
        self.root = Path(root)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.augmentation = augmentation
        if augmentation not in AUGMENTATION_MODES:
            raise ValueError(f"augmentation must be one of {AUGMENTATION_MODES}")
        self.samples: list[tuple[Path, int]] = []
        self.skipped: list[str] = []
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir(), key=lambda p: p.name.casefold()):
                if not _is_image_candidate(path):
                    continue
                if validate_images:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                    except (OSError, ValueError, SyntaxError):
                        self.skipped.append(str(path))
                        continue
                self.samples.append((path, class_idx))
        if not self.samples:
            raise FileNotFoundError(f"No valid JPG/PNG images found in {self.root}/<class>/")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR
            )
        image = _apply_augmentation(image, self.augmentation)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, label


def load_image_folder(directory: Path | str, image_size: tuple[int, int] = (128, 128)):
    """Load an image folder into ``(uint8 NHWC images, one-hot labels, paths)``.

    This helper preserves the previous project's utility contract for metrics
    and baseline scripts. Hidden, AppleDouble, non-image and corrupt files are
    omitted before they can affect counts.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[str] = []
    dataset = TrashNetDataset(directory, image_size, augmentation="none", validate_images=True)
    # Reopen each source with deterministic preprocessing; keeping uint8 helps
    # the colour-histogram baseline operate on the original colour scale.
    for path, label in dataset.samples:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB").resize(
                    (int(image_size[1]), int(image_size[0])), Image.Resampling.BILINEAR
                )
            xs.append(np.asarray(image, dtype=np.uint8))
            ys.append(label)
            paths.append(str(path))
        except (OSError, ValueError, SyntaxError):
            continue
    if not xs:
        raise FileNotFoundError(f"No valid images found under {directory}/<class>/")
    x = np.stack(xs)
    y = np.eye(len(CLASSES), dtype=np.float32)[np.asarray(ys, dtype=np.int64)]
    return x, y, paths


class WasteCNN(nn.Module):
    """Compact CNN built from scratch. ``forward`` returns class logits."""
    def __init__(self, input_shape=(64, 64, 3), num_classes=len(CLASSES),
                 filters=(16, 32, 64, 128), dense_units=128, dropout=0.3,
                 augmentation="geometric", learning_rate=2e-3):
        super().__init__()
        height, width, channels = tuple(input_shape)
        if height < 16 or width < 16 or channels != 3:
            raise ValueError("input_shape must be (H,W,3) with H,W >= 16 for four pooling blocks")
        if num_classes != len(CLASSES):
            raise ValueError(f"Project 11 is fixed to {len(CLASSES)} TrashNet classes in canonical order")
        if not filters or any(int(f) < 1 for f in filters):
            raise ValueError("filters must contain positive channel counts")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if augmentation not in AUGMENTATION_MODES:
            raise ValueError(f"augmentation must be one of {AUGMENTATION_MODES}")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.input_shape_hwc = (int(height), int(width), int(channels))
        self.image_size = (int(height), int(width))
        self.num_classes = int(num_classes)
        self.filters = tuple(int(f) for f in filters)
        self.dense_units = int(dense_units)
        self.dropout_p = float(dropout)
        self.augmentation = augmentation
        self.learning_rate = float(learning_rate)

        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in self.filters:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            blocks.extend([conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(in_channels, self.dense_units)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.dense_units, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = self.gap(x).flatten(1)
        x = self.dropout1(x)
        x = torch.relu(self.fc(x))
        x = self.dropout2(x)
        return self.classifier(x)

    def config_dict(self) -> dict[str, Any]:
        return {
            "input_shape": list(self.input_shape_hwc),
            "num_classes": self.num_classes,
            "filters": list(self.filters),
            "dense_units": self.dense_units,
            "dropout": self.dropout_p,
            "augmentation": self.augmentation,
            "learning_rate": self.learning_rate,
        }


def build_model(
    input_shape: tuple[int, int, int] = (64, 64, 3),
    num_classes: int = len(CLASSES),
    *, filters: tuple[int, ...] = (16, 32, 64, 128),
    dense_units: int = 128, dropout: float = 0.3,
    augmentation: str = "geometric", learning_rate: float = 2e-3, seed: int = 42,
) -> WasteCNN:
    """Build an untrained CNN; input_shape is supplied as (height,width,channels).

    Output is logits for ``CrossEntropyLoss``. Softmax is applied only in
    ``predict_proba`` / reporting, not inside this model.
    """
    set_seed(seed)
    return WasteCNN(input_shape, num_classes, filters, dense_units, dropout, augmentation, learning_rate)


@dataclass
class TrainingHistory:
    history: dict[str, list[float]]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str | None = None


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _balanced_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    if present.any():
        weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def _save_checkpoint(path: Path, model: WasteCNN, *, epoch: int, val_loss: float, config: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "project11-pytorch-v1",
        "classes": list(CLASSES),
        "preprocessing": {
            "color_mode": "RGB",
            "resize_hw": list(model.image_size),
            "tensor_layout": "NCHW",
            "dtype": "float32",
            "scale": "pixel_values_divided_by_255_to_[0,1]",
            "normalization_mean": None,
            "normalization_std": None,
        },
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "experiment_config": config or model.config_dict(),
    }, path)


def train(
    model: WasteCNN, train_dir: Path | str, validation_dir: Path | str, *,
    epochs: int = 10, batch_size: int = 32, class_weight: Any = "balanced",
    patience: int | None = None, restore_best: bool = True, reduce_lr: bool = True,
    seed: int = 42, verbose: int = 1, checkpoint_path: Path | str | None = None,
    num_workers: int = 0, device: str | torch.device | None = None, **kwargs: Any,
) -> TrainingHistory:
    """Train with weighted CrossEntropy, Adam, ReduceLROnPlateau and best val loss.

    Returns ``TrainingHistory`` with Keras-like ``history`` keys to ease
    reporting, plus a 1-based best epoch. Class weights are estimated from the
    training split only.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if kwargs:
        raise TypeError(f"Unsupported train() keyword argument(s): {', '.join(sorted(kwargs))}")
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_data = TrashNetDataset(train_dir, model.image_size, augmentation=model.augmentation)
    val_data = TrashNetDataset(validation_dir, model.image_size, augmentation="none")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              worker_init_fn=_seed_worker, generator=generator, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            worker_init_fn=_seed_worker, pin_memory=device.type == "cuda")
    if class_weight == "balanced":
        weights = _balanced_weights([y for _, y in train_data.samples], model.num_classes).to(device)
    elif class_weight is None:
        weights = None
    elif isinstance(class_weight, dict):
        weights = torch.tensor([float(class_weight.get(i, 1.0)) for i in range(model.num_classes)], dtype=torch.float32, device=device)
    else:
        weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
        if len(weights) != model.num_classes:
            raise ValueError("class_weight sequence length must match num_classes")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5) if reduce_lr else None
    model = model.to(device)
    hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "learning_rate": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = 1
    epochs_without_improvement = 0
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            train_loss_sum += float(loss.detach()) * n
            train_correct += int((logits.detach().argmax(1) == labels).sum())
            train_total += n
        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                n = labels.size(0)
                val_loss_sum += float(loss) * n
                val_correct += int((logits.argmax(1) == labels).sum())
                val_total += n
        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        hist["loss"].append(float(train_loss))
        hist["accuracy"].append(float(train_acc))
        hist["val_loss"].append(float(val_loss))
        hist["val_accuracy"].append(float(val_acc))
        hist["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            if ckpt_path is not None:
                _save_checkpoint(ckpt_path, model, epoch=epoch, val_loss=val_loss)
        else:
            epochs_without_improvement += 1
        if verbose:
            print(f"epoch {epoch:02d}/{epochs} - loss {train_loss:.4f} acc {train_acc:.3f} "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} lr {optimizer.param_groups[0]['lr']:.2g}", flush=True)
        if patience is not None and epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}; best validation loss was epoch {best_epoch}.")
            break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    # Return model to CPU-neutral mode and retain the actual model object.
    model.eval()
    history = TrainingHistory(hist, best_epoch, best_loss, str(ckpt_path) if ckpt_path else None)
    return history


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def predict_proba(model: Any, image_path: Any) -> dict[str, float]:
    """Return softmax probabilities in canonical class order for one image."""
    if model is None:
        raise ValueError("predict_proba() needs a model; got None")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module loaded with load_trained_model()")
    size = getattr(model, "image_size", (64, 64))
    tensor = image_to_tensor(image_path, size).unsqueeze(0).to(_model_device(model))
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    if was_training:
        model.train()
    return {cls: float(probs[i]) for i, cls in enumerate(CLASSES[: len(probs)])}


def predict(model: Any, image_path: Any) -> tuple[str, float]:
    """Return ``(predicted_class, confidence)`` for one image."""
    probs = predict_proba(model, image_path)
    label = max(probs, key=probs.get)
    return label, probs[label]


def load_trained_model(path: Path | str, *, device: str | torch.device | None = None) -> WasteCNN:
    """Load a checkpoint produced by this PyTorch implementation."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with earlier PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("format") != "project11-pytorch-v1":
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}; retrain using scripts/run_experiments.py")
    if tuple(checkpoint.get("classes", ())) != CLASSES:
        raise ValueError("Checkpoint class mapping does not match the canonical TrashNet order")
    preprocessing = checkpoint.get("preprocessing", {})
    if preprocessing and preprocessing.get("color_mode") != "RGB":
        raise ValueError("Unsupported checkpoint preprocessing color mode")
    config = dict(checkpoint["model_config"])
    config["input_shape"] = tuple(config["input_shape"])
    config["filters"] = tuple(config["filters"])
    model = WasteCNN(**config)
    resize_hw = tuple(preprocessing.get("resize_hw", model.image_size))
    if resize_hw != model.image_size:
        raise ValueError("Checkpoint preprocessing resize does not match its model configuration")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
