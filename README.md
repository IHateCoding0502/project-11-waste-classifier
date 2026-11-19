# Visual Waste Sorting Assistant

Experimental six-class image classifier, using the original from-scratch PyTorch CNN and the selected **aug_none** checkpoint. No model was replaced or retrained during this upgrade.

## Run locally (Windows PowerShell)

Python **3.11** is the AIN submission target. Install dependencies once:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the application with one command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Linux/macOS use `python3.11 -m venv .venv`, activate it and run the same Python module commands. Upgrade pip before installation. NumPy 2.3.5 is pinned for Python 3.11; Python 3.12 and newer retain NumPy 2.5.4 from the original environment. Other direct dependencies remain pinned. CPU inference is supported automatically; CUDA depends on the installed PyTorch build. `requirements-dev.txt` adds tests and historical presentation tooling. Once dependencies and the supplied team-trained checkpoint are installed, the application runs offline and does not call an LLM API.

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

The selected team-trained checkpoint is committed, so a fresh clone can run inference without downloading the dataset or any model weights. To reproduce training from the pinned TrashNet source, first activate the environment (`.\.venv\Scripts\Activate.ps1` on Windows):

```powershell
python scripts/prepare_dataset.py --confirm
python scripts/run_experiments.py --epochs 25 --seed 42
```

This downloads and prepares data, then runs the original experiments; new hardware/training runs need not reproduce scores bit-for-bit. See `data/LICENSE.md` for pinned source, upstream MIT attribution and dataset licensing notes. The existing 2,527 local images and all historical experiments are preserved; full datasets remain ignored. The selected checkpoint and documented evaluation artifacts are included for Git publication.

## Reproduce evaluation

Activate the installed environment first. Checkpoint validation works with the small supplied sample; full benchmark evaluation additionally requires the prepared dataset:

```powershell
python scripts/validate_checkpoint.py --image data/sample/paper/paper_01.png
python scripts/build_analysis_figures.py
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

The [submission instructions](submission/README.md) map the guide to the repository and explain how to build the required ZIP. GitHub contains the runnable starter project; the ZIP contains one root named `61FIT3AIN-TUT02_G05_P11` with `slides/`, `code/`, and `report/`. Do not rename or duplicate the starter tree merely to mirror the ZIP layout.

The guide requires an English Word report based on `AIN_Final_Report_Template.docx`, a 15–20-page main body, IEEE citations, and Appendices A–C. The locally prepared course report uses exactly 15 main-body pages, satisfying the template's additional maximum. The report, member metadata, and submission ZIP are local artifacts excluded from Git; they are not downloadable from a fresh clone. Prepare them using the official template and commands in the submission instructions. The separately requested APA 7 edition is a local alternative and is not the IEEE report required by this guide. The earlier IEEE PDF below is supplementary and is excluded from the ZIP.

The required ZIP has one matching root folder containing `slides/`, `code/` and `report/`. `python scripts/package_submission.py` validates group metadata, agreed integer contributions totaling 100, original midterm-slide confirmation, report pagination and archive exclusions. `--review` creates an explicitly incomplete local review ZIP when required confirmations are pending. See `submission/SUBMISSION_STATUS.md`. Never regenerate slides and present them as the previously shown midterm deck. Only one representative submits through the instructor's link before the oral examination deadline.

`reports/report.pdf` uses the installed official IEEEtran conference class. Editable files: `reports/report.tex`, `reports/references.bib`; report and architecture generation source: `scripts/build_final_report.py`. The generator reads actual saved evaluations and original ablations. To rebuild after evaluation:

```powershell
python scripts/build_final_report.py
cd reports
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

MiKTeX or TeX Live with IEEEtran, amsmath, graphicx and hyperref is required for this supplementary PDF. Its author placeholders are not a completed submission cover. `ORAL_DEFENSE.md` has technical questions and answers; `DEMO_SCRIPT.md` gives the 5–7 minute sequence; `FINAL_DELIVERY_CHECKLIST.md` records outstanding inputs. Source URLs are included in the app and bibliography.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/check_submission_repo.py
```

Independent Python 3.11.16 CPU installation passed 38 tests with one CUDA test skipped; see [submission QA](reports/AIN_SUBMISSION_QA.md). Repository and packaging checks added later are recorded separately. A clean checkout needs neither the full dataset nor local Word reports to run the product tests.

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
