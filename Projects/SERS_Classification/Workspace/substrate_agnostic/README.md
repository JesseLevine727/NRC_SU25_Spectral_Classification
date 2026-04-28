# Substrate-Agnostic SERS Outputs

This directory contains generated outputs for the current substrate-agnostic SERS classification story. Raw spectra, cleaned CSVs, and legacy notebooks remain one level up in `Workspace/` so older exploratory notebooks are not broken.

## Layout

- `current/best_siamese_triplet/`
  Best current canonical-label Siamese/triplet run: `derivative_1`, triplet loss, margin `0.2`, substrate-balanced prototypes.
- `diagnostics/agnp_failure/`
  AgNP failure deep dive: average spectra, PCA, UMAP, t-SNE, prototype distances, confusion matrices, and raw-file audit.
- `diagnostics/geometry_analysis/`
  Quantitative derivative-input vs Siamese-embedding geometry analysis across all held-out substrates and classes. Includes all-fold PCA, UMAP, and t-SNE projection coordinate tables under `projections/`.
- `sweeps/siamese_feature_loss_sweep/`
  Feature/loss/prototype sweep outputs and summary CSV.
- `classical_baselines/`
  Non-deep-learning leave-one-substrate-out baseline results and confusion matrices.
- `archive/comparison_runs/`
  Older or secondary Siamese comparison runs retained for traceability but not treated as the current best model.

## Current Interpretation

The current best model is substrate-agnostic in formulation: it predicts chemical identity and evaluates by holding out whole substrates. The remaining failure is held-out `AgNP`, specifically `4np` on `AgNP` being mapped to `benzenethiol`. The diagnostics show this is a learned embedding/prototype geometry problem rather than a missing-file issue.
