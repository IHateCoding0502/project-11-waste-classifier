# Executed quality assurance

## Passed in the existing project environment

- `python -m pytest tests -q --basetemp .qa_temp_6`: **39 passed in 20.23 seconds**. Full output: test_results.txt.
- Coverage includes original CNN/configuration, a one-epoch temporary training roundtrip, checkpoint mapping, CPU and CUDA inference, six finite logits, training-equivalent preprocessing, JPEG/PNG/grayscale/palette/alpha/EXIF uploads, corrupt/unsupported/truncated/oversized files, aspect ratios, guidance schema, manifest validation, four Streamlit pages and upload/preview/prediction/error rendering.
- `python scripts/validate_checkpoint.py --image data/test/paper/paper100.jpg`: CPU and CUDA both load the selected original checkpoint, report 114,966 parameters, shape [1,6] and predict paper. CUDA is available on this machine.
- `python scripts/evaluate_stress_tests.py --prepare --device cpu`: executed 435 standard, 12 complex subset and 1,740 transformed images. Metrics and all six scores are saved under results/stress. Standard rerun matches the original 334/435 benchmark exactly. Synthetic and real-camera conditions are separate.
- Actual Streamlit server started at http://127.0.0.1:8501; HTTP `/_stcore/health` returns `ok`. AppTest also executes the product Python code; health alone is not used as proof of prediction correctness.
- `python -m pip check` in the existing environment: no broken requirements found.
- Python compileall and `git diff --check`: passed (Git reports only line-ending conversion notices).
- Official installed IEEEtran 1.8b conference class, pdflatex + BibTeX + two further pdflatex passes: report.pdf is five letter-size pages. Final log has no undefined references/citations or overfull boxes. All pages were rendered with pdftoppm and visually inspected; text, equations, figures, tables and references are readable with no clipping or overlap.
- Academic references were verified against arXiv, JMLR and PMLR; disposal references against US EPA, Washington Ecology and the Vietnam Environment Agency. General foreign preparation guidance is not presented as Vietnamese collection acceptance.

## Fresh installation: not yet verified

A separate `.venv-qa` was created without inheriting installed project packages. The first sandbox install failed DNS resolution. An authorized network retry found torch/numpy but pip 25.0.1 could not resolve Pillow. Upgrading this new environment to pip 26.3 resolved the package metadata, including Pillow 13.0.0. The subsequent 124.1 MB PyTorch wheel download stalled before yielding file bytes. A retry with timeout 15 seconds and retries 0 was also attempted; it again stalled at the PyTorch wheel, so both stalled installation attempts were stopped. Therefore independent fresh-environment installation and tests are **not reported as passed**. The working app and 39 passing tests above use the original project environment.

To repeat on a working package network:

```powershell
py -3.12 -m venv .venv-qa
.\.venv-qa\Scripts\python.exe -m pip install --upgrade pip
.\.venv-qa\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-qa\Scripts\python.exe -m pytest tests -q
```

## Not executed: missing genuine images

Real-camera low-light/blur and novel campus sets have zero supplied samples. Their directories, empty manifests, checklist and evaluator are implemented; no performance or pass claim is made. The 12-image complex subset overlaps the standard set, is too small for an interval and does not cover every requested deformation or clutter condition. Broader real photo collection remains required. Campus location/collector arrangements and team author details also need confirmation before submission.

## Environment issues resolved during QA

Streamlit AppTest requires an internal socket. Sandbox socket restrictions stalled its event loop before app execution; the successful full test run used authorized execution outside that restriction. Oversized byte-parameter test identifiers initially exceeded the Windows environment-variable limit; explicit short test IDs fixed that test harness issue. No model result was altered to make a test pass.
