# PyTorch experiment summary

The final model is selected by validation macro-F1 among configurations with dropout > 0, as required by the Midterm rubric. Zero-dropout variants remain ablations. Test metrics are reported only for the selected CNN and the two required baselines.

| Model | Framework | Params | Best epoch | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 | Test macro-F1 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| majority_class | baseline | 0 | - | 0.236 | 0.064 | 0.234 | 0.063 | - |
| colour_histogram_centroid | baseline | 0 | - | 0.411 | 0.385 | 0.384 | 0.353 | - |
| base | pytorch | 114,966 | 24 | 0.681 | 0.641 | - | - | - |
| no_reg | pytorch | 114,966 | 24 | 0.831 | 0.815 | - | - | - |
| aug_none **(selected)** | pytorch | 114,966 | 24 | 0.739 | 0.720 | 0.768 | 0.757 | [0.709, 0.799] |
| aug_full | pytorch | 114,966 | 24 | 0.592 | 0.573 | - | - | - |
| drop_0.0 | pytorch | 114,966 | 24 | 0.748 | 0.717 | - | - | - |
| drop_0.5 | pytorch | 114,966 | 23 | 0.595 | 0.583 | - | - | - |
| depth_2 | pytorch | 28,582 | 25 | 0.601 | 0.576 | - | - | - |
| depth_3 | pytorch | 32,790 | 25 | 0.650 | 0.618 | - | - | - |
| lr_2e-4 | pytorch | 114,966 | 24 | 0.617 | 0.584 | - | - | - |
| lr_1e-2 | pytorch | 114,966 | 24 | 0.696 | 0.665 | - | - | - |

## Interpretation notes

- Training epoch selection uses minimum validation loss.
- Configuration selection uses validation macro-F1 among dropout-regularized configurations (dropout > 0); test metrics do not influence selection.
- CI is a percentile bootstrap over individual test images and does not measure seed-to-seed training variability.
- Per-configuration ablation conclusions must be based on validation results and learning curves, not hidden test results.
