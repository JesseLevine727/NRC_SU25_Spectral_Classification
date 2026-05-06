# Grouped Metal Substrate Analysis

This folder contains the corrected substrate-family analysis where `AgNP` is grouped with `Ag` and `AuNP` is grouped with `Au`.

## Result Change

| Analysis | Mean accuracy | Mean balanced accuracy | Mean macro F1 | Weakest fold |
|---|---:|---:|---:|---|
| old six substrate labels, best Siamese | 0.854 | 0.854 | 0.686 | `AgNP` at 0.380 |
| grouped substrate families, best Siamese | 0.975 | 0.975 | 0.895 | `Ag` at 0.920 |
| grouped substrate families, raw-spectrum Siamese | 0.440 | 0.440 | 0.399 | `Au` at 0.000 |
| grouped substrate families, best classical baseline | 0.987 | 0.987 | 0.987 | `Ag` at 0.960 |

The corrected grouped result removes the old `AgNP`-specific collapse as the main story. The remaining Siamese errors are concentrated in the held-out silver-family fold: 6/25 `4np` spectra are predicted as `benzenethiol`.

## Current Best Siamese Run

- Feature: `derivative_1`
- Model: Conv1D Siamese encoder
- Loss: triplet loss
- Prototype mode: `row_mean`
- Device: CUDA/GPU
- Folds: leave-one-substrate-family-out over `Ag`, `Au`, `PICO`, `pSERS`

| Held-out substrate family | Test labels | Accuracy |
|---|---|---:|
| Ag | `4np,benzenethiol,pyridine` | 0.920 |
| Au | `benzenethiol,pyridine` | 0.980 |
| PICO | `4np,benzenethiol,pyridine` | 1.000 |
| pSERS | `4np,benzenethiol,pyridine` | 1.000 |

## Files

- `current/best_siamese_triplet/results.csv`
- `current/best_siamese_triplet/confusions/*.csv`
- `classical_baselines/results.csv`
- `classical_baselines/confusions/*.csv`
- `sweeps/siamese_feature_loss_sweep/summary.csv`
- `diagnostics/geometry_analysis/geometry_analysis.md`
- `diagnostics/geometry_analysis/silhouette_*.png`
- `diagnostics/geometry_analysis/projections/*.png`
  PCA, UMAP, and t-SNE clustering plots. Each image has a chemical-colored view and a substrate-family-colored view.
- `archive/comparison_runs/raw_siamese_triplet_row_mean/results.csv`
