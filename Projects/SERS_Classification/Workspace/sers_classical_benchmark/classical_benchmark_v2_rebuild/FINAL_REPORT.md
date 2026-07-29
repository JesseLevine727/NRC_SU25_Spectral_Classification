# NATO SERS classical benchmark v2

## Status

This bundle is the locked classical foundation for the supervised-contrastive CNN experiment. It evaluates chemical classification and selective prediction; it does not claim physical chemical/nuisance disentanglement.

## Selected configurations

- Strict-core champion: `pca_logistic` on `arpls_minmax` with `{"C": 0.1, "pca_components": 32, "pca_whiten": true}`.
- Quality-pass champion: `pca_logistic` on `derivative_1` with `{"C": 0.1, "pca_components": 32, "pca_whiten": true}`.
- Selection used 20 nested inner folds and did not use outer, field-stress, or held-domain outcomes.

## Master-group outer performance

| Evaluation | Balanced accuracy, mean ± 95% CI half-width |
|---|---:|
| Strict core | 0.685 ± 0.066 |
| Quality pass | 0.745 ± 0.086 |
| Field-quality stress | 0.478 ± 0.180 |

The uncertainty unit is the outer master-sample fold (n=5), not an individual spectrum.

## Data adequacy

Strict-core balanced accuracy changed from 0.623 using about 16.6 training master samples to 0.687 using about 55.2. The learning curve is the evidence used to judge whether added independent samples are likely to matter more than model capacity.

## Held-domain evaluation

| Subset | Protocol | Domain | Mean supported-class BA | 95% CI half-width | Held domains |
|---|---|---|---:|---:|---:|
| quality_pass | domain_and_sample | instrument | 0.613 | 0.182 | 10 |
| quality_pass | domain_only | instrument | 0.678 | 0.150 | 10 |
| quality_pass | domain_and_sample | sensor_family | 0.557 | 4.356 | 2 |
| quality_pass | domain_only | sensor_family | 0.498 | 0.418 | 4 |
| strict_core | domain_and_sample | instrument | 0.618 | 0.201 | 10 |
| strict_core | domain_only | instrument | 0.629 | 0.129 | 10 |
| strict_core | domain_and_sample | sensor_family | 0.591 | 0.983 | 3 |
| strict_core | domain_only | sensor_family | 0.413 | 0.230 | 4 |

Sensor-family confidence intervals can be extremely wide because only a few held sensor families retain supported analytes. These intervals must not be read as precision estimates from hundreds of independent spectra.

## Calibration, abstention, and field stress

The mean field-stress OOD AUROC from one minus calibrated maximum probability was 0.678; mean AUPRC was 0.313. These values measure stress detection, not chemical classification.
- field_quality_stress: mean NLL 2.293 before and 2.171 after temperature scaling.
- quality_pass: mean NLL 0.966 before and 0.825 after temperature scaling.
- strict_core: mean NLL 1.628 before and 0.857 after temperature scaling.

## Negative control

Master-group label permutation produced mean outer balanced accuracy 0.128 (maximum 0.194), compared with seven-class chance of 0.143.

## Decision for the next stage

The selected classical configurations, fold predictions, calibration temperatures, held-domain results, and learning curves are now the immutable comparison bar. The supervised-contrastive CNN must be trained on these same master-group partitions and is useful only if it improves the current Siamese control and offers a reproducible held-domain and/or field-stress selective advantage without materially degrading strict/quality performance.

## Figures

- `figures/classical_benchmark_summary.pdf`: grouped outer performance and master-sample learning curves.
- `figures/selective_domain_summary.pdf`: abstention and held-domain results.

Outer and learning-curve error bars are 95% intervals over master-group folds. Held-domain points show individual domains with mean and observed range because sensor-family counts are too small for stable visual intervals. Vector PDF and 600-DPI PNG exports use a colorblind-safe palette and redundant markers.
