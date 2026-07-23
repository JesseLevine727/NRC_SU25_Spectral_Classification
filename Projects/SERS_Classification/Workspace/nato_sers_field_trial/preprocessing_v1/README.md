# NATO SERS frozen preprocessing v1

This bundle freezes six auditable representations for the 598-spectrum strict core and the aligned 500-spectrum quality subset. All source spectra remain unchanged.

## Frozen selections

- `minimal_minmax`
- `arpls_minmax`
- `derivative_1`

`minimal_minmax` is the no-baseline reconstructive control. The baseline-corrected and derivative selections were chosen from role-specific nested inner-validation Pareto fronts.

## Nested selection evidence

| Representation | Target core | Target quality | Composite corruption | Instrument leakage increment | Sensor leakage increment | Same-master cross-instrument distance | Frozen |
|---|---:|---:|---:|---:|---:|---:|---|
| minimal_minmax | 0.697 | 0.739 | 0.664 | 0.671 | 0.442 | 0.760 | yes |
| robust_minmax | 0.696 | 0.730 | 0.685 | 0.682 | 0.442 | 0.768 | no |
| asls_minmax | 0.714 | 0.755 | 0.654 | 0.661 | 0.446 | 0.700 | no |
| arpls_minmax | 0.707 | 0.742 | 0.653 | 0.611 | 0.446 | 0.675 | yes |
| derivative_1 | 0.701 | 0.739 | 0.550 | 0.640 | 0.433 | 0.684 | yes |
| derivative_2 | 0.646 | 0.711 | 0.452 | 0.527 | 0.409 | 0.707 | no |

Higher target/corruption scores are better. Lower target-adjusted domain leakage and same-master distance are better. These are fixed PCA/logistic screening baselines, not VAE results.

## Artifact detection

- Conservative candidate spike points in observed core: 6
- Spectra with numeric maximum plateaus: 0
- Candidate spike locations are preserved in `spike_mask`; repaired values never overwrite `raw_common_grid`.
- Synthetic injection recall and precision are in `artifact_detection_summary.json`.

## Files

- `candidate_spectra_core.npz` and `candidate_spectra_quality.npz`: raw, despiked, masks, baselines, and six representations;
- `frozen_model_inputs_core.npz` and `frozen_model_inputs_quality.npz`: only the selected arrays authorized as downstream model inputs;
- `core_preprocessing_manifest.csv` and `quality_preprocessing_manifest.csv`: provenance, flags, folds, and scaling metadata;
- `benchmark_fold_metrics.csv`: all outer and nested-inner results;
- `selection_objectives.csv` and `frozen_selection.json`: explicit selection evidence;
- `dataset_version.json` and `artifact_hashes.json`: configuration, software, input hashes, and bundle hashes.

## Rebuild and validate

```bash
.venv/bin/python scripts/freeze_nato_sers_preprocessing.py
.venv/bin/python scripts/validate_nato_sers_preprocessing_freeze.py
```

The selected arrays are fixed inputs for subsequent AE, denoising-AE, VAE, disentangled-VAE, and Siamese-hybrid comparisons. Model selection must not alter these preprocessing choices using outer-test results.
