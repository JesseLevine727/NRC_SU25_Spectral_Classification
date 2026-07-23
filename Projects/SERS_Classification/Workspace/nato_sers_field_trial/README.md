# NATO SERS field-trial derived dataset

This directory is generated from the read-only March 2024 archive at
`../../../../2026July21/NATO SERS Data`.

The complete audit, field semantics, preprocessing recommendations, split
protocols, and SERS-VAE roadmap are in
[`docs/NATO_SERS_FIELD_TRIAL_AUDIT.md`](../../docs/NATO_SERS_FIELD_TRIAL_AUDIT.md).

Rebuild from the repository root:

```bash
.venv/bin/python scripts/build_nato_sers_field_trial.py
.venv/bin/python scripts/analyze_nato_sers_preprocessing.py
.venv/bin/python scripts/make_nato_sers_splits.py
.venv/bin/python scripts/freeze_nato_sers_preprocessing.py
.venv/bin/python scripts/validate_nato_sers_preprocessing_freeze.py
.venv/bin/python scripts/finalize_nato_sers_preprocessing_v2.py
.venv/bin/python scripts/validate_nato_sers_preprocessing_v2.py
```

`sers_core_manifest.csv` is the main 598-observation manifest, and
`sers_qc_pass_manifest.csv` is the conservative 500-observation sensitivity
subset. The validated [`preprocessing_v1`](preprocessing_v1/README.md) bundle
freezes `minimal_minmax`, `arpls_minmax`, and `derivative_1` for the subsequent
model comparisons. It remains the immutable control.

The closed
[`preprocessing_v2`](preprocessing_v2/README.md) bundle is the authorized
downstream dataset. It retains `minimal_minmax`, `arpls_minmax`, and
`derivative_1`; the bounded study rejected global or conditional smoothing
and additional spectral alignment. The full rationale, exact protocol, and
results are in
[`docs/NATO_SERS_PREPROCESSING_FINAL_V2.md`](../../docs/NATO_SERS_PREPROCESSING_FINAL_V2.md).
