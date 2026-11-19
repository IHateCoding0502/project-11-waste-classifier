# Visual Waste Sorting Assistant — Final Milestone

Experimental six-class image classifier, using the original from-scratch PyTorch CNN and the selected **aug_none** checkpoint. No model was replaced or retrained during this upgrade.

## Run locally (Windows PowerShell)

Python 3.12 is the tested interpreter. The existing `.venv` can be used immediately:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For a new environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Linux/macOS use `python3.12 -m venv .venv`, activate it and run the same Python module commands. Upgrade pip first (the original pip 25.0.1 failed to resolve Pillow here; pip 26.3 resolved it). The pinned requirements use the tested package release versions. For CUDA install the corresponding PyTorch build from the official PyTorch index; CPU is supported automatically. `requirements-dev.txt` adds the test runner and historical presentation tooling.

## Product

Four pages: Waste Classifier, Model Information, Known Limitations & Failure Modes, Robustness Testing. Upload JPEG/PNG, inspect the preview, then select Predict. Six horizontal scores, timing and source-linked material guidance are shown. Limits: 10 MiB, 20 megapixels, 8192 pixels per side. EXIF orientation, grayscale and transparent PNG are supported; invalid files receive an error.

The Eco-Tech interface uses a forest-green sidebar, responsive cards and a dedicated classification workspace. Styles are maintained in `assets/streamlit.css`; the native theme is in `.streamlit/config.toml`. Prediction results persist while reviewing guidance and clear when the uploaded image changes. UI verification and desktop/mobile screenshots are recorded in `reports/UI_REDESIGN_QA.md`.

Softmax scores are **not calibrated probabilities**. Unknown objects still receive one of the six labels. The model is an educational prototype; it does not determine hazardous-waste safety or guarantee collector acceptance. Guidance distinguishes general recycling preparation from Vietnam reference context; campus location and local collection arrangements are unconfirmed.

## Original model and results

Actual source: `starter/student_core.py:WasteCNN`. RGB/bilinear 64×64, float32 CHW divided by 255; **no mean/std normalization**. Label indices: cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5. Four bias-free Conv/BN/ReLU/Pool blocks (16,32,64,128), global average pool, dropout .3, dense 128, ReLU, dropout .3, dense six logits. **114,966** trainable parameters. Adam .002, weighted cross-entropy, batch 32, 25 epochs, seed 42; epoch chosen by minimum validation loss.

The selected aug_none has validation Macro-F1 .720. no_reg achieved .815 but had zero dropout and was excluded by the rubric. These numbers do not establish that no_reg overfits. Preserve this selection trade-off.

| Original test model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline | 23.4% | 6.3% |
| Color-histogram centroid | 38.4% | 35.3% |
| Selected CNN | 76.8% | 75.7% |

334/435 correct; original Macro-F1 bootstrap 95% CI 70.9%–79.9%. Single-seed and possible physical-object leakage limit conclusions. The preliminary 30-image Qwen comparison (CNN 25/30, Qwen 20/30) is exploratory and does not prove general CNN superiority.

## Checkpoint and dataset

Local selected checkpoint: `models/waste_cnn.pth`, SHA-256 `825083ce60c04c0566b736a2089ece09f9a640ba0b72b55ce3b1fadf6ef077c3`. Its size is about 473 KB, so Git LFS is unnecessary. This file is explicitly allowed by `.gitignore` for a reproducible submission; other training checkpoints remain ignored. Do not load untrusted checkpoints. The loader checks metadata, class mapping and strict state-dict compatibility.

A fresh clone only has the checkpoint if the team includes it in its commit/release. No hosted checkpoint URL is invented. To regenerate from the pinned TrashNet source:

```powershell
python scripts/prepare_dataset.py --confirm
python scripts/run_experiments.py --epochs 25 --seed 42
```

This downloads and prepares data, then runs the original experiments; new hardware/training runs need not reproduce scores bit-for-bit. See `data/LICENSE.md` for pinned source, upstream MIT attribution and dataset licensing notes. The existing 2,527 local images and all historical experiments are preserved; full datasets remain ignored. The selected checkpoint and documented evaluation artifacts are included for Git publication.

## Reproduce evaluation

```powershell
python scripts/validate_checkpoint.py --image data/test/paper/paper100.jpg
python scripts/collect_dataset_info.py
python scripts/evaluate_stress_tests.py --prepare --device cpu
```

The last command creates controlled variants outside the original splits, combines the separate challenge manifests, and evaluates using the same selected checkpoint. Without `--prepare`, evaluate an existing manifest; `--manifest` and `--out` support separately frozen challenge sets. Missing images, invalid labels, duplicate rows and invalid camera object IDs fail explicitly.

New CPU evaluation (see `results/stress/metrics.json` and `scenario_comparison.csv`):

| Condition | n | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Standard held-out TrashNet | 435 | 76.78% | 75.75% |
| Complex descriptive TrashNet subset | 12 | 83.33% | 82.78% |
| Synthetic low-light/blur aggregate | 1,740 | 48.62% | 48.40% |
| Real-camera low-light/blur | 0 | Pending | Pending |
| Novel campus photographs | 0 | Pending | Pending |

The complex subset was chosen visually before new predictions and overlaps standard data. It supports no general superiority/robustness claim and does not cover every required deformation/background. Synthetic tests apply brightness ×.30, contrast ×.35, Gaussian radius 2 px, and horizontal motion kernel 9 px at original resolution, then model preprocessing. Four variants share each source image; aggregate bootstrap clusters by source. They are not real-camera or campus photographs.

Outputs include all six scores per image, checksums, environment, source-cluster uncertainty intervals where supported, per-transform results, confusion matrices, per-class precision/recall/F1, confidence distributions, galleries and latency charts. Timing is warmed-up batch-one forward + softmax, excluding decode/resize/transfers, with CUDA synchronization. See the collection protocol in `data/stress_tests/COLLECTION_CHECKLIST.md`. The team must supply genuine camera/campus photographs and independent labels; these missing evaluations are not passed tests.

## Report and defense

`reports/report.pdf` uses the installed official IEEEtran conference class. Editable files: `reports/report.tex`, `reports/references.bib`; report and architecture generation source: `scripts/build_final_report.py`. The generator reads actual saved evaluations and original ablations. To rebuild after evaluation:

```powershell
python scripts/build_final_report.py
cd reports
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

MiKTeX or TeX Live with IEEEtran, amsmath, graphicx and hyperref is required. No author identities or affiliations are invented: add actual team details before submission. `ORAL_DEFENSE.md` has technical questions and answers; `DEMO_SCRIPT.md` gives the 5–7 minute sequence; `FINAL_DELIVERY_CHECKLIST.md` records outstanding inputs. Source URLs are verified and included in the app and bibliography.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite covers model/checkpoint shape and mapping, CPU/CUDA (when available), training checkpoint roundtrip, training/evaluation transforms, valid/corrupt/unsupported/oversized uploads, grayscale/alpha/EXIF, aspect ratios, guidance schema, manifest validation, Streamlit page rendering and upload-to-prediction/error rendering. Synthetic smoke images are not benchmark evidence. See `reports/QA.md` for executed checks and environment limitations.

## Repository

```text
app.py                       canonical Streamlit entry point
starter/student_core.py      original model/training/checkpoint API
starter/inference.py         upload validation and shared timed inference
starter/app.py               four-page interface; legacy entry point works
models/waste_cnn.pth          selected original checkpoint
data/disposal_guidance.json  six materials, warnings, local scope and URLs
data/stress_tests/           manifests, protocol, separate photo directories
scripts/                     original experiments + audit/evaluation/report utilities
results/                     preserved original runs + separate stress results
reports/                     IEEE PDF, LaTeX, bibliography, audit and QA
tests/                       original and final-product tests
ORAL_DEFENSE.md
DEMO_SCRIPT.md
FINAL_DELIVERY_CHECKLIST.md
```

Existing analysis-figure work and Midterm presentation sources are preserved. To regenerate original analysis figures: `python scripts/build_analysis_figures.py`. The optional `--quick` experiment uses overlapping tiny synthetic data and must never be cited as held-out performance.
