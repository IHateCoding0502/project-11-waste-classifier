# Real photograph collection and evaluation

The available complex set is a **12-image descriptive subset of held-out TrashNet**, selected visually before running the new predictions. It overlaps the standard evaluation and is not an independent real-world challenge. It covers printed/curled/overlapping paper, flattened board, crumpled foil, cropped/labeled glass and metal, ridged plastic, and crumpled residual items. It does not establish robustness for every requested deformation or complex background. Collect crushed cans, crumpled paper, clearly deformed bottles, occluded objects and cluttered scenes to extend it.

## Collection checklist

- [ ] Confirm campus and collector location; no campus location is assumed in disposal guidance.
- [ ] Aim for at least five distinct physical objects per class (30 total), across backgrounds, lighting and angles.
- [ ] Keep one evaluation photograph per physical object. Use separate objects for real low-light/blur and novel campus sets.
- [ ] Capture actual low-light, low-contrast, motion blur and defocus photographs; do not substitute generated variants.
- [ ] Capture genuinely new campus images; do not copy TrashNet or use training/validation objects.
- [ ] Obtain ground truth independently of predictions; use two reviewers if possible. Exclude ambiguous multi-material items or record them separately without forcing labels.
- [ ] Avoid people, personal details and restricted areas. Remove unnecessary identifying EXIF metadata before sharing.
- [ ] Record path, true_label, condition, physical_object_id, lighting, blur_level, capture_device, notes, transform, source_path and sha256 (optional; evaluator computes actual hashes).
- [ ] Labels must be lowercase: cardboard, glass, metal, paper, plastic, trash. Conditions: complex, low_light_blur_real, campus_novel.
- [ ] Place photos under the corresponding data/stress_tests directories. Paths in CSV are relative to the repository root.
- [ ] Add rows to complex_manifest.csv, camera_manifest.csv or campus_manifest.csv using manifest_template.csv headers. Do not put example/fake rows in a real manifest.
- [ ] Freeze labels before inference; do not tune or train using these photographs.

Run `python scripts/evaluate_stress_tests.py --prepare --device cpu` to rebuild the combined manifest and evaluate all available photographs. The empty real-camera and campus manifests are intentional pending inputs, not passed tests. Record sample counts and missing classes; small convenience samples do not support population-level claims.

## Controlled transformations

Generated outside original splits, applied at source resolution, saved losslessly as PNG:
brightness ×0.30; contrast ×0.35; Gaussian blur radius 2 pixels; horizontal motion box kernel 9 pixels, edge padding, rounded uint8. Four variants per held-out source image. The aggregate confidence interval resamples source-image groups, not individual variants. Results are paired sensitivity measurements, not independent additional objects. The original TrashNet object IDs are unknown: filename IDs track images only, not validated physical identities.
