# NATO SERS classical benchmark protocol v2

Date declared: 2026-07-29  
Status: frozen before expanded classical candidate execution  
Machine-readable protocol:
[`configs/sers_classical_benchmark_v2.json`](../configs/sers_classical_benchmark_v2.json)

## Purpose

This stage establishes the strongest leakage-safe classical classification
and selective-prediction reference before the supervised-contrastive CNN is
developed. It tests chemical classification under new master samples,
field-quality stress, held instruments, and held sensor families. It makes no
claim that any representation is a physical chemical/nuisance decomposition.

The effective independent unit is `master_sample_id`, not a spectral row.
Every preprocessing transform, dimensionality reduction, scaling operation,
model fit, hyperparameter choice, and calibration fit is therefore confined
to development master groups.

## Immutable inputs

The only authorized input is the validated NATO preprocessing-v2 bundle.
The representations are:

1. `arpls_minmax`, the primary baseline-corrected intensity view;
2. `minimal_minmax`, the mandatory peak-preserving sensitivity view;
3. `derivative_1`, the frozen discriminative control.

The 598-row strict core is primary. The 500 quality-pass rows form a
prespecified sensitivity/development cohort. The complementary 98 rows are
sealed field-stress confirmation data and may not select a representation,
algorithm, hyperparameter, temperature, or rejection threshold.

## Nested selection

The existing five master-group folds are retained. Within each outer
development partition, the other four group folds act as inner validation
folds. Candidate ranking uses mean balanced accuracy, then macro-F1, then the
declared candidate order. The outer test fold is opened only after the
candidate and temperature have been selected.

The bounded grid compares PCA-logistic regression, elastic-net multinomial
logistic regression, shrinkage LDA, PLS-DA, linear SVM, and RBF SVM. Exact
values are frozen in the machine-readable protocol.

## Calibration and abstention

For each selected outer model, a single softmax temperature is fit from
cross-fitted scores generated only by the inner development folds. The final
model is refit on the complete outer development partition, and the locked
temperature is applied to its outer-test scores.

Selective classification is summarized at fixed retained coverages. Coverage
selection uses confidence rank and never changes the predicted class. Field
stress is evaluated by models trained only on quality-pass development rows.

## Domain evaluation

The preprocessing-v2 `domain_only` and `domain_and_sample` partitions are
used unchanged for instrument and sensor-family holdouts. Candidate choice is
locked before these outcomes are calculated. Unsupported test classes remain
in prediction records but are excluded from supported-class balanced
accuracy and macro-F1.

## Data-adequacy and controls

Learning curves subsample target-stratified master groups within each outer
development partition at four declared fractions and three declared seeds.
A master-group label-permutation control is evaluated over the five outer
folds. These analyses determine whether additional independent samples are
more likely to help than additional model capacity.

## Decision gate for deep learning

The classical champion is the mandatory bar for the supervised-contrastive
CNN. The CNN can be promoted only through repeated, grouped evidence of a
meaningful advantage on held-domain and/or field-stress selective
classification without a material strict/quality-pass loss. A deep model is
not promoted merely because it has a lower instrument probe or a visually
appealing embedding.
