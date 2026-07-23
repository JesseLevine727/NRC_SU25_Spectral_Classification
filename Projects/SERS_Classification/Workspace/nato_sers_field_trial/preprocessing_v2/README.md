# NATO SERS preprocessing v2

This is the closed, validated preprocessing bundle for downstream AE, denoising-AE, VAE, disentangled-VAE, and classifier experiments.
It contains 598 attributable SERS spectra on a 400--1800 cm^-1 axis at 1 cm^-1 spacing. The 500-row quality cohort and 98-row field-quality stress cohort are disjoint and exhaust the core.

## Frozen model inputs

- `minimal_minmax`
- `arpls_minmax`
- `derivative_1`

`minimal_minmax` and `arpls_minmax` are reconstructive inputs on [0,1]. `derivative_1` is the signed, row-L2-normalized poster/Siamese discriminative control. No general smoothing and no spectral alignment are applied (`none`).

Use `final_model_inputs_core.npz` for the primary 598-row experiment. Use `final_model_inputs_quality.npz` only as the prespecified 500-row sensitivity analysis. `final_model_inputs_field_quality_stress.npz` is a confirmatory 98-row stress cohort and must not be used to tune preprocessing.

## Evidence and provenance

- `DECISION_REGISTRY.md`: human-readable decision summary.
- `predeclared_protocol.json`: candidate grid, gates, and the pre-benchmark peak-gate amendment.
- `final_selection.json`: complete smoothing selection record.
- `alignment_decision.json`: complete alignment decision record.
- `benchmark_fold_metrics.csv`: 495 nested and stress evaluations.
- `smoothing_preservation_*.csv`: spectrum- and instrument-level fidelity evidence.
- `alignment_*.csv`: named-standard and paired-lag evidence.
- `*_manifest.csv`: observation provenance and cohort membership.
- `*_split_assignments.csv` and `nested_group_cv_assignments.csv`: master-sample-grouped frozen splits.
- `candidate_spectra_*.npz`: all nine audited candidates; these are evidence archives, not permission to reopen selection.
- `artifact_hashes.json`: SHA-256 catalog for this bundle.
- `v1_control_hashes.json`: immutable preprocessing-v1 snapshot.

The candidate archives also retain common-grid raw spectra, despiked spectra, spike and saturation masks, the arPLS baseline, and all candidate representations. No source data were modified.

## Rebuild and validate

From the repository root:

```bash
.venv/bin/python scripts/finalize_nato_sers_preprocessing_v2.py
.venv/bin/python scripts/validate_nato_sers_preprocessing_v2.py
```

The scientific rationale and exact downstream contract are in [`docs/NATO_SERS_PREPROCESSING_FINAL_V2.md`](../../../docs/NATO_SERS_PREPROCESSING_FINAL_V2.md).
