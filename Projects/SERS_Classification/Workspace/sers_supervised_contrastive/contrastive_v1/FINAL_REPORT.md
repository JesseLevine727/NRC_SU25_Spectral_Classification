# Supervised-contrastive NATO SERS experiment

## Terminal decision

**Successor promotion: NOT SUPPORTED.**

This experiment tests domain-robust classification and abstention. It does not claim physical denoising or chemical/nuisance disentanglement.

## Locked outer results

| Model | Strict BA | Quality BA | Field-stress BA |
|---|---:|---:|---:|
| Classical | 0.685 ± 0.066 | 0.745 ± 0.086 | 0.478 ± 0.180 |
| Siamese | 0.632 ± 0.071 | 0.677 ± 0.049 | 0.370 ± 0.141 |
| Contrastive successor | 0.701 ± 0.060 | 0.731 ± 0.058 | 0.430 ± 0.097 |

Uncertainty is a 95% interval over five outer master-group folds after averaging repeated seeds. Spectral rows are not treated as independent uncertainty units.

## Promotion gates

- No material strict/quality degradation: **True**.
- Material held-domain and/or stress advantage: **False**.
- Beats the Siamese control in at least two of three seed directions: **True** (3/3).
- Primary held-instrument plus new-sample advantage over classical: 0.008 across 20 supported domain/cohort pairs.
- Primary held-instrument plus new-sample advantage over the historical Siamese: 0.033.
- Field-stress selective-accuracy advantage at 80% requested coverage: -0.030 across 5 outer folds.
- Full-coverage field-stress balanced-accuracy difference (secondary): -0.048.

## Selected model

Stage 1 selected `derivative_1` with supervised-contrastive weight 0.5 and pair-margin weight 0.25. Stage 2 selected `legacy` with 64 embedding dimensions for the global domain evaluation.

## Representation and Siamese-control diagnostics

- Successor different-minus-same-analyte distance margin: 0.409; historical Siamese: 0.315.
- Successor embedding effective rank: 6.96; historical Siamese: 3.12.
- Historical Siamese leave-one-master-out analyte probe balanced accuracy: 0.568 where supported.
- Historical Siamese cross-fitted correctness-confidence ECE10: 0.106. This is correctness calibration from nearest-prototype distance, not multiclass probability calibration.
- Historical Siamese encoder parameters: 719,584; selected successor total parameters: 361,383.

## Frozen preprocessing sensitivity

Each representation below uses all five outer folds: when a representation was selected it contributes the full-model row; otherwise it contributes the registered sensitivity row. This prevents partial-fold aggregation.

| Representation | Strict BA | Quality BA | Stress BA |
|---|---:|---:|---:|
| `arpls_minmax` | 0.707 ± 0.058 | 0.724 ± 0.071 | 0.417 ± 0.135 |
| `minimal_minmax` | 0.703 ± 0.064 | 0.733 ± 0.063 | 0.449 ± 0.144 |
| `derivative_1` | 0.702 ± 0.061 | 0.723 ± 0.088 | 0.455 ± 0.152 |

## Direct answers

- **Which classical model won?** Strict: `pca_logistic` on `arpls_minmax` with `{"C": 0.1, "pca_components": 32, "pca_whiten": true}`. Quality/stress-development: `pca_logistic` on `derivative_1` with `{"C": 0.1, "pca_components": 32, "pca_whiten": true}`.
- **Does this dataset support preferring the tested deep model?** No. The registered evidence does not justify replacing the classical champion with this successor.
- **Did supervised contrastive learning improve the Siamese control?** Yes under the registered two-of-three seed rule.
- **Did the gain transfer?** Primary domain difference versus classical 0.008; field-stress selective difference at 80% coverage -0.030.
- **Publication interpretation:** The publishable result is a leakage-safe grouped benchmark and mechanistic negative/ablation study, not a claimed invariant or disentangled representation.
- **What crossed data are still needed?** More independent master samples for every analyte; the same physical samples measured across every instrument and sensor family; balanced sensor-family support; raw vendor spectra before proprietary baseline removal; reference Raman/SERS spectra for chemicals and blank substrates; and controlled concentration, substrate-lot, acquisition-time, and environmental replicates.

## OOD and attribution evidence

- `class_mahalanobis`: field-stress AUROC 0.763, AUPRC 0.382 across folds and seeds.
- `energy`: field-stress AUROC 0.685, AUPRC 0.325 across folds and seeds.
- `one_minus_max_probability`: field-stress AUROC 0.721, AUPRC 0.361 across folds and seeds.
- Development-selected rejection scores by outer fold: `calibrated_max_probability` in 5 fold(s).
- Same-master cross-instrument attribution Jaccard: 0.168.
- Same-analyte, different-master attribution Jaccard: 0.205.

## Negative controls

- `master_group_analyte_label_permutation`: mean balanced accuracy 0.160.
- `randomized_domain_relationships`: mean balanced accuracy 0.718.

## Interpretation

The full model is promoted only if every registered gate above is satisfied. Otherwise, the result remains a controlled comparison showing which objective or domain relationship helped, without reinterpreting a failure as invariance or disentanglement. Held-sensor values with few supported classes must be read together with their support counts.

The held-domain models never train on their test instrument or sensor, and held-domain outcomes never select the configuration. However, the global architecture was chosen by master-group CV across the archive's available domain identities, matching the classical benchmark. This is locked leave-one-domain-out transfer, not a substitute for a genuinely external instrument acquisition.

## Main figure

- `figures/model_comparison.pdf` and 600-DPI PNG: locked model comparison, objective ablation, and field-stress rejection.
- `figures/training_diagnostics.pdf` and 600-DPI PNG: nested-development convergence, collapse geometry, chemistry-versus-instrument probes, and locked risk–coverage curves.
- `successor_confusion_matrices.json`: pooled, seed-specific, and fold/seed-specific chemical confusions.
- `successor_failure_cases.csv`: every unsupported or incorrect full-successor outer prediction.
- `rejection_decisions_at_80.csv`: accepted and rejected examples at the locked 80% coverage endpoint.
- `siamese_control_diagnostics.csv` and `siamese_control_failures.csv`: reconstructed geometry, collapse, grouped-probe, correctness-calibration, and failure evidence for the immutable historical control.
