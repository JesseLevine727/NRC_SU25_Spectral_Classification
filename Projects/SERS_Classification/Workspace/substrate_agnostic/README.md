# Substrate-Agnostic SERS Outputs

This directory contains generated outputs for the current substrate-agnostic SERS classification story. Raw spectra, cleaned CSVs, and legacy notebooks remain one level up in `Workspace/` so older exploratory notebooks are not broken.

## Layout

- `current/best_siamese_triplet/`
  Best current canonical-label Siamese/triplet run: `derivative_1`, triplet loss, margin `0.2`, substrate-balanced prototypes.
- `diagnostics/agnp_failure/`
  AgNP failure deep dive: average spectra, PCA, UMAP, t-SNE, prototype distances, confusion matrices, and raw-file audit.
- `diagnostics/geometry_analysis/`
  Quantitative derivative-input vs Siamese-embedding geometry analysis across all held-out substrates and classes. Includes all-fold PCA, UMAP, and t-SNE projection coordinate tables under `projections/`, plus silhouette visualizations showing chemical-label clustering versus substrate clustering.
- `sweeps/siamese_feature_loss_sweep/`
  Feature/loss/prototype sweep outputs and summary CSV.
- `classical_baselines/`
  Non-deep-learning leave-one-substrate-out baseline results and confusion matrices.
- `archive/comparison_runs/`
  Older or secondary Siamese comparison runs retained for traceability but not treated as the current best model.

## Current Interpretation

The current best model is substrate-agnostic in formulation: it predicts chemical identity and evaluates by holding out whole substrates. The remaining failure is held-out `AgNP`, specifically `4np` on `AgNP` being mapped to `benzenethiol`. The diagnostics show this is a learned embedding/prototype geometry problem rather than a missing-file issue.

## Dataset Matrix

Current canonical three-chemical coverage used for substrate-agnostic evaluation:

| Chemical | Ag | AgNP | Au | AuNP | PICO | pSERS | Current total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4np` | 0 | 25 | 0 | 0 | 25 | 25 | 75 |
| `benzenethiol` | 25 | 0 | 25 | 0 | 25 | 25 | 100 |
| `pyridine` | 0 | 25 | 0 | 25 | 25 | 25 | 100 |

Target minimum matrix for a stronger substrate-agnostic claim:

| Chemical | Ag | AgNP | Au | AuNP | PICO | pSERS | Target total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4np` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |
| `benzenethiol` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |
| `pyridine` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |

Higher-value target if time allows: `2-3` independent preparations/maps per chemical-substrate pair, with `25` spectra per preparation. Independent preparations are more useful than many additional correlated spectra from one existing map.

The all-class geometry analysis shows that the Siamese embedding generally improves chemical organization:

| Space | Mean chemical-label silhouette | Mean substrate silhouette | Label minus substrate |
|---|---:|---:|---:|
| derivative input | 0.304 | 0.054 | 0.250 |
| Siamese embedding | 0.791 | -0.246 | 1.037 |

Useful visual summaries:

- `diagnostics/geometry_analysis/silhouette_scores_by_fold.png`
- `diagnostics/geometry_analysis/silhouette_label_minus_substrate_by_fold.png`
- `diagnostics/geometry_analysis/silhouette_sample_distributions.png`
- `diagnostics/geometry_analysis/silhouette_by_class_and_substrate.png`
