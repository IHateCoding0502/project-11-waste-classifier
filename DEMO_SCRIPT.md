# Product demonstration — 5 to 7 minutes

Prepare the original checkpoint and start `python -m streamlit run app.py` before the session. Keep the report and results/stress/scenario_comparison.csv available.

1. **0:00–0:40 — Introduce the product.** Open Waste Classifier. Explain the six supported classes and educational prototype scope.
2. **0:40–1:30 — Standard image.** Upload `data/test/paper/paper100.jpg`, inspect preview/format/dimensions and click Predict. Explain the predicted class, timing and all six confidence bars. Scores sum to one but are not calibrated correctness probabilities.
3. **1:30–2:10 — Guidance.** Show preparation, warnings, local collection scope and authoritative source links. Material identification does not guarantee acceptance.
4. **2:10–2:55 — Complex item.** Upload `data/test/metal/metal122.jpg` (crumpled foil). State that it is an actual TrashNet example, not a new campus photograph. Discuss observed appearance without claiming a proven cause for any error.
5. **2:55–3:45 — Controlled darkness/blur.** Upload the corresponding files in `data/stress_tests/low_light_blur_synthetic/`, for example `paper100_dark.png` and `paper100_motion.png`. Show the robustness page and distinguish transformation sensitivity from real-camera performance.
6. **3:45–4:30 — Genuinely new campus photo.** Upload a photograph collected by the team using the collection checklist and independent ground truth. If none has been collected, show the explicit pending state and say this requirement is not yet evaluated; do not substitute a generated or TrashNet image.
7. **4:30–5:20 — Inspect a mistake.** Open Known Limitations & Failure Modes and the highest-confidence error gallery. Explain how even a 99% score can be wrong and why a 50% threshold does not solve unknown detection.
8. **5:20–6:15 — Defend evaluation.** Show original benchmark and the new standard rerun (334/435), constrained aug_none selection versus no_reg, and the synthetic results. Complex samples overlap standard; real-camera and campus results remain separate.
9. **6:15–6:45 — Close with limits.** Mention 64×64 detail loss, visually similar materials, six-label taxonomy, single seed and domain shift. Point to the IEEE report, reproducible manifests and independent campus collection protocol.

## Live demonstrations requiring team photographs

- Novel campus waste: actual unseen camera image, preferably five distinct objects per class in the final evaluation set.
- Real low-light, motion-blurred and defocused captures: separate camera objects and metadata; synthetic transforms are not substitutes.
- Full complex challenge: crumpled paper, crushed cans, clearly deformed bottles, occlusion and cluttered backgrounds. The supplied 12-image descriptive subset is limited.

Record labels before opening predictions, keep these photographs out of training/validation, and use unique physical_object_id values. The app and report intentionally display missing evaluations until genuine images are supplied.
