#!/usr/bin/env python3
"""Download, validate, and reproducibly split the pinned TrashNet archive.

Run ``python scripts/prepare_dataset.py --confirm`` to download the source
archive and create git-ignored data/train, data/validation, data/test folders.
Hidden/AppleDouble files and invalid images are removed from the candidate list
*before* the split. A split manifest is written to data/split_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

REPO_URL = "https://github.com/garythung/trashnet"
PINNED_COMMIT = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
DATASET_URL = f"https://raw.githubusercontent.com/garythung/trashnet/{PINNED_COMMIT}/data/dataset-resized.zip"
PINNED_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SPLIT = {"train": 0.70, "validation": 0.13, "test": 0.17}
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def download(dest: Path) -> None:
    print(f"Source: {REPO_URL} (pinned commit {PINNED_COMMIT})")
    print(f"Downloading {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response, dest.open("wb") as target:
        shutil.copyfileobj(response, target)


def report_checksum(path: Path, expected_sha256: str | None) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"sha256({path.name}) = {digest}")
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise SystemExit(
            "Checksum mismatch: the downloaded archive does not match "
            f"--sha256 {expected_sha256!r}. Stop and verify the pinned source."
        )
    return digest


def _valid_image_bytes(content: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError):
        return False


def extract_and_split(zip_path: Path, out_dir: Path, seed: int) -> dict:
    """Validate first; then split each class deterministically and write manifest."""
    rng = random.Random(seed)
    manifest = {
        "source_url": DATASET_URL,
        "source_repo": REPO_URL,
        "pinned_commit": PINNED_COMMIT,
        "seed": seed,
        "split_ratio": SPLIT,
        "classes": list(CLASSES),
        "excluded_hidden_or_metadata": [],
        "excluded_invalid_images": [],
        "splits": {name: {} for name in SPLIT},
    }
    candidates: dict[str, list[tuple[str, bytes]]] = {name: [] for name in CLASSES}
    with zipfile.ZipFile(zip_path) as zf:
        for raw_name in sorted(zf.namelist()):
            entry = PurePosixPath(raw_name)
            if raw_name.endswith("/") or len(entry.parts) < 2:
                continue
            class_name = entry.parts[-2]
            filename = entry.name
            if class_name not in candidates or Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # Do not let AppleDouble resource forks affect counts/splitting.
            if filename.startswith(".") or filename.startswith("._") or any(p.startswith(".") for p in entry.parts):
                manifest["excluded_hidden_or_metadata"].append(raw_name)
                continue
            content = zf.read(raw_name)
            if not _valid_image_bytes(content):
                manifest["excluded_invalid_images"].append(raw_name)
                continue
            candidates[class_name].append((raw_name, content))

    if not any(candidates.values()):
        raise SystemExit("No valid images found for the six TrashNet classes in the archive.")

    # Recreate split folders to prevent stale files contaminating a repeat run.
    for split_name in SPLIT:
        target = out_dir / split_name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        records = sorted(candidates[cls], key=lambda item: item[0])
        rng.shuffle(records)
        n = len(records)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["validation"])
        partitions = {
            "train": records[:n_train],
            "validation": records[n_train:n_train + n_val],
            "test": records[n_train + n_val:],
        }
        for split_name, items in partitions.items():
            class_dir = out_dir / split_name / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            manifest["splits"][split_name][cls] = []
            for raw_name, content in items:
                out_path = class_dir / PurePosixPath(raw_name).name
                out_path.write_bytes(content)
                manifest["splits"][split_name][cls].append({"archive_path": raw_name, "filename": out_path.name})
        print(f"{cls}: {n} valid images -> " + ", ".join(f"{k}={len(v)}" for k, v in partitions.items()))

    manifest["counts"] = {
        split: {cls: len(files) for cls, files in classes.items()}
        for split, classes in manifest["splits"].items()
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Excluded metadata: {len(manifest['excluded_hidden_or_metadata'])}; invalid images: {len(manifest['excluded_invalid_images'])}")
    print(f"Wrote split manifest: {out_dir / 'split_manifest.json'}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Download the dataset and create splits.")
    parser.add_argument("--sha256", default=PINNED_SHA256, help="Expected SHA-256; pass empty string to skip verification.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--keep-zip", action="store_true", help="Keep data/dataset-resized.zip after extraction.")
    parser.add_argument("--archive", type=Path, help="Use a local archive (useful for offline preparation/testing).")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="Output data directory.")
    args = parser.parse_args(argv)
    archive = args.archive or (args.out_dir / "dataset-resized.zip")
    if not args.confirm and args.archive is None:
        print("Dry run only; no network or filesystem changes.")
        print(f"Would download {DATASET_URL}, verify SHA-256, exclude metadata/invalid images, and split 70/13/17 with seed={args.seed}.")
        print("Re-run with --confirm to proceed.")
        return 0
    if args.archive is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        download(archive)
    elif not archive.is_file():
        raise SystemExit(f"Local archive does not exist: {archive}")
    digest = report_checksum(archive, args.sha256 or None)
    manifest = extract_and_split(archive, args.out_dir, args.seed)
    manifest["archive_sha256"] = digest
    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.archive is None and not args.keep_zip:
        archive.unlink(missing_ok=True)
    print("Done. data/train, data/validation, and data/test are git-ignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
