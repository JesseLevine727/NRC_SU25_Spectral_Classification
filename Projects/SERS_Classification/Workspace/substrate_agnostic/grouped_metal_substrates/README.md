# Grouped Metal Substrate Analysis

This folder contains the corrected substrate-family analysis where `AgNP` is grouped with `Ag` and `AuNP` is grouped with `Au`.

## Result Change

| Analysis | Mean accuracy | Mean balanced accuracy | Mean macro F1 | Weakest fold |
|---|---:|---:|---:|---|
| old six substrate labels, best Siamese | 0.854 | 0.854 | 0.686 | `AgNP` at 0.380 |
| grouped substrate families, best Siamese | 0.975 | 0.975 | 0.895 | `Ag` at 0.920 |
| grouped substrate families, K-shot Siamese (`K=5`) | 0.873 | 0.873 | 0.850* | `pSERS` at 0.333 |
| grouped substrate families, raw-spectrum Siamese | 0.440 | 0.440 | 0.399 | `Au` at 0.000 |
| grouped substrate families, best classical baseline | 0.987 | 0.987 | 0.987 | `Ag` at 0.960 |

The corrected grouped result removes the old `AgNP`-specific collapse as the main story. The remaining Siamese errors are concentrated in the held-out silver-family fold: 6/25 `4np` spectra are predicted as `benzenethiol`.

`*` The K-shot value shown here is mean true-label macro F1, computed over the chemical labels actually present in each held-out fold. The raw CSV also keeps standard macro F1, which can include predicted absent labels.

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
- `kshot_siamese/`
  Formal K-shot Siamese evaluation. `K` is sampled per held-in chemical-substrate-family cell, then the model is tested on the held-out substrate family. Includes `summary_by_k.csv`, `detailed_results.csv`, confusion matrices for every K/seed/fold, and K=5 clustering/geometry diagnostics.
- `classical_baselines/results.csv`
- `classical_baselines/confusions/*.csv`
- `sweeps/siamese_feature_loss_sweep/summary.csv`
- `diagnostics/geometry_analysis/geometry_analysis.md`
- `diagnostics/geometry_analysis/silhouette_*.png`
- `diagnostics/geometry_analysis/projections/*.png`
  PCA, UMAP, and t-SNE clustering plots. Each image has a chemical-colored view and a substrate-family-colored view.
- `archive/comparison_runs/raw_siamese_triplet_row_mean/results.csv`

## Formal K-Shot Check

The abstract title used "few-shot" because the original submitted work trained on a small number of spectra per chemical-substrate pair. The current substrate-agnostic model is best described as Siamese metric learning on a small dataset unless it is evaluated under an explicit K-shot protocol.

The K-shot protocol implemented in `scripts/sers_kshot_substrate_agnostic.py` samples `K` support spectra per held-in chemical-substrate-family cell, trains the same Conv1D Siamese encoder with triplet loss on CUDA, and tests all known chemicals in the held-out substrate family. Five seeds were run for `K=1,3,5,10,25`.

| K | Mean accuracy | Mean true-label macro F1 | Std true-label macro F1 | Worst fold accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.913 | 0.905 | 0.140 | 0.653 |
| 3 | 0.840 | 0.819 | 0.260 | 0.000 |
| 5 | 0.873 | 0.850 | 0.218 | 0.333 |
| 10 | 0.874 | 0.853 | 0.200 | 0.560 |
| 25 | 0.930 | 0.924 | 0.138 | 0.600 |

Interpretation: the method can work with few support spectra, but the formal K-shot result is high-variance and not as stable as the full-data substrate-agnostic run. The current poster should say the work began as few-shot chemical-substrate pair learning and now tests substrate-agnostic transfer; it should not imply that the final substrate-agnostic result is a fully robust few-shot result.
