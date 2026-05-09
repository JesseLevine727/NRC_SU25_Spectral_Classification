# Substrate-Agnostic SERS Outputs

This directory contains generated outputs for the substrate-agnostic SERS classification work. Raw spectra, cleaned CSVs, and legacy notebooks remain one level up in `Workspace/` so older exploratory notebooks are not broken.

## Current Interpretation

The corrected substrate-family analysis groups `AgNP -> Ag` and `AuNP -> Au`. Under this interpretation, the earlier held-out `AgNP` failure is not the right primary conclusion because `Ag` and `AgNP` are the same substrate family. The corrected leave-one-substrate-family-out folds are `Ag`, `Au`, `PICO`, and `pSERS`.

The grouped analysis is much stronger than the original six-substrate-label analysis:

| Analysis | Mean accuracy | Mean balanced accuracy | Mean macro F1 | Weakest fold |
|---|---:|---:|---:|---|
| old six substrate labels, best Siamese | 0.854 | 0.854 | 0.686 | `AgNP` at 0.380 |
| grouped substrate families, best Siamese | 0.975 | 0.975 | 0.895 | `Ag` at 0.920 |
| grouped substrate families, K-shot Siamese (`K=5`) | 0.873 | 0.873 | 0.850* | `pSERS` at 0.333 |
| grouped substrate families, raw-spectrum Siamese | 0.440 | 0.440 | 0.399 | `Au` at 0.000 |
| grouped substrate families, best classical baseline | 0.987 | 0.987 | 0.987 | `Ag` at 0.960 |

The strongest current Siamese run after regrouping is `derivative_1` + triplet loss + row-mean chemical prototypes. Its only material errors are in the held-out `Ag` family: 6/25 `4np` spectra are predicted as `benzenethiol`. The grouped `PICO` and `pSERS` folds are perfect, and the grouped `Au` fold has 1/50 error.

`*` The K-shot value is mean true-label macro F1, computed over labels actually present in each held-out fold.

## Corrected Dataset Matrix

Current canonical coverage after `bt -> benzenethiol` and metal-substrate grouping:

| Chemical | Ag | Au | PICO | pSERS | Current total | Notes |
|---|---:|---:|---:|---:|---:|---|
| `4np` | 25 | 0 | 25 | 25 | 75 | `4np` does not respond on Au, so Au is not a useful target cell. |
| `benzenethiol` | 25 | 25 | 25 | 25 | 100 | Complete across four substrate families. |
| `pyridine` | 25 | 25 | 25 | 25 | 100 | Complete across four substrate families. |
| `n,n-dimethylformamide` | 0 | 228 | 0 | 0 | 228 | Present only on Au, so not currently substrate-agnostic. |

Minimum target matrix:

| Chemical | Ag | Au | PICO | pSERS | Target total | Priority |
|---|---:|---:|---:|---:|---:|---|
| `4np` | 25+ | N/A | 25+ | 25+ | 75+ | Already covers the valid responding families. |
| `benzenethiol` | 25+ | 25+ | 25+ | 25+ | 100+ | Already complete; add independent repeats if time allows. |
| `pyridine` | 25+ | 25+ | 25+ | 25+ | 100+ | Already complete; add independent repeats if time allows. |
| additional chemical, e.g. `n,n-dimethylformamide` if it responds | 25+ | 25+ | 25+ | 25+ | 100+ | Highest-value expansion is another complete four-family chemical. |

If time is limited, the most useful expansion is not more row count from existing maps. It is one or more additional chemicals measured on all four substrate families, with independent preparations/maps where possible.

## Layout

- `grouped_metal_substrates/current/best_siamese_triplet/`
  Current corrected best Siamese/triplet run: grouped substrate families, `derivative_1`, triplet loss, row-mean prototypes.
- `grouped_metal_substrates/classical_baselines/`
  Corrected grouped-substrate classical leave-one-substrate-family-out baselines.
- `grouped_metal_substrates/diagnostics/geometry_analysis/`
  Corrected grouped-substrate derivative-input vs Siamese-embedding geometry analysis with PCA, UMAP, t-SNE scatter plots, prototype distances, and silhouette visuals.
- `grouped_metal_substrates/kshot_siamese/`
  Formal K-shot grouped-substrate Siamese evaluation. `K` is sampled per held-in chemical-substrate-family cell and tested on the held-out substrate family. Includes five-seed results for `K=1,3,5,10,25` and K=5 PCA/UMAP/t-SNE geometry diagnostics.
- `grouped_metal_substrates/sweeps/siamese_feature_loss_sweep/`
  Corrected grouped-substrate Siamese feature/loss/prototype sweep outputs and summary CSV.
- `grouped_metal_substrates/archive/comparison_runs/raw_siamese_triplet_row_mean/`
  Corrected grouped-substrate raw-spectra Siamese comparison. It performs poorly, confirming the derivative preprocessing is still necessary.
- `current/`, `classical_baselines/`, `diagnostics/`, `sweeps/`
  Historical six-substrate-label outputs retained for traceability. These should be treated as superseded by the grouped-substrate-family results for scientific interpretation.

## Corrected Geometry

Grouped geometry still shows that the Siamese embedding organizes spectra more by chemical identity than by substrate family:

| Space | Mean chemical-label silhouette | Mean substrate-family silhouette | Label minus substrate |
|---|---:|---:|---:|
| derivative input | 0.304 | 0.101 | 0.203 |
| Siamese embedding | 0.797 | -0.040 | 0.837 |
| K=5 Siamese embedding | 0.748 | -0.042 | 0.789 |

Useful grouped visual summaries:

- `grouped_metal_substrates/diagnostics/geometry_analysis/silhouette_scores_by_fold.png`
- `grouped_metal_substrates/diagnostics/geometry_analysis/silhouette_label_minus_substrate_by_fold.png`
- `grouped_metal_substrates/diagnostics/geometry_analysis/silhouette_sample_distributions.png`
- `grouped_metal_substrates/diagnostics/geometry_analysis/silhouette_by_class_and_substrate.png`
- `grouped_metal_substrates/diagnostics/geometry_analysis/projections/*.png`
