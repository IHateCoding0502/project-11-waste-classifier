# Project 11 data

`sample/` contains tiny synthetic PNG images for offline infrastructure tests only. These images are not real TrashNet photographs and must not be used to report model performance.

The optional full dataset is downloaded from the pinned TrashNet commit documented in `LICENSE.md`. Run `python scripts/prepare_dataset.py --confirm` to verify the archive checksum, validate each candidate image, exclude hidden/AppleDouble metadata files **before splitting**, and create `train/`, `validation/`, `test/`, plus `split_manifest.json`. The split is deterministic with default seed 42 and proportions 70/13/17 per class. These generated files are excluded from version control and must not be included in the submission archive.
