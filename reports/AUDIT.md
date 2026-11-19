# Technical audit

- Existing research/training: starter/student_core.py and scripts/run_experiments.py. Dataset preparation: scripts/prepare_dataset.py. Existing analysis, presentation, Qwen comparisons and experiment outputs are preserved.
- Actual selected checkpoint: models/waste_cnn.pth, 472,595 bytes, SHA-256 825083ce60c04c0566b736a2089ece09f9a640ba0b72b55ce3b1fadf6ef077c3. Original selected metadata identifies aug_none, dropout .3, validation-selected epoch 24.
- Architecture: WasteCNN in starter/student_core.py; four bias-free convolutions, BatchNorm/ReLU/pooling, adaptive global average pooling, Dropout, Linear(128,128), ReLU, Dropout, Linear(128,6). Verified 114,966 trainable parameters.
- Training preprocessing: RGB conversion, PIL bilinear resize to 64×64, float32 division by 255, CHW tensor. No mean/std normalization. The checkpoint metadata matches the code.
- Canonical mapping: cardboard 0, glass 1, metal 2, paper 3, plastic 4, trash 5.
- Checkpoint loaded strictly on CPU and CUDA. A real paper100.jpg sample produces six finite logits and predicts paper. No retraining or architecture change was needed.
- Data exists locally: train 1,766 / validation 326 / test 435, total 2,527. Split manifest and saved fingerprint exist. Full datasets remain ignored in Git; original files are unchanged by stress generation.
- Original selected, ablation, baseline, learning-curve and Qwen results exist in results/. The new stress evaluation is written separately under results/stress/.
- Environment: Python 3.12.14, torch 2.14.1+cu132, NumPy 2.5.4, Pillow 13.0.0, Streamlit 1.66.1, Matplotlib 3.11.2. CUDA is available. See audit.json for machine-readable evidence and parameter groups.
- Git remote: https://github.com/IHateCoding0502/project-11-waste-classifier.git. No push or commit was performed. README.md and scripts/build_analysis_figures.py had user changes before this task; the existing analysis script was preserved and README content updated for the final milestone.
- Missing inputs: genuine campus photographs, real-camera low-light/blur captures, broader complex/deformed challenge images, confirmed campus/collector location and actual team author details.
