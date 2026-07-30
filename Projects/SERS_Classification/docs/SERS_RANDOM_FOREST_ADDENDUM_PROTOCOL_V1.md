# NATO SERS random-forest addendum protocol v1

## Purpose

This addendum answers a bounded question that was not included in the sealed
classical benchmark: how well does a conventional random forest classify the
NATO SERS analytes when it is subjected to the same master-sample grouping,
preprocessing candidates, field-stress cohort, and held-domain tests?

The existing classical and neural bundles remain immutable. Random forest is
an addendum rather than a retroactive candidate in the earlier selection.

## Leakage boundary

- The independent unit is `master_sample_id`, not a spectral row.
- Every inner and outer partition uses the frozen five-fold master assignment.
- Hyperparameters and preprocessing are selected independently inside each
  outer development partition.
- The 98 field-quality-stress spectra and all held-domain outcomes are locked
  evaluations and cannot choose a forest.
- Unsupported test analytes are retained and marked, but excluded from
  supported-class balanced accuracy and macro-F1.

## Candidate family

Each of the three frozen representations is crossed with:

- 300 trees;
- `max_features` in `{sqrt, 0.1}`;
- `max_depth` in `{None, 12}`;
- `min_samples_leaf` in `{1, 2, 4}`;
- ordinary row weighting or inverse-master-frequency sample weighting.

All forests use bootstrap sampling, Gini splits, balanced-subsample class
weights, a fixed run seed, and one estimator thread. This produces 72 declared
candidates and 2,880 grouped inner fits across strict and quality selection.

Inverse-master weighting is a prespecified sensitivity to the archive's
unequal number of spectra per physical sample. It does not alter the held-out
unit or duplicate observations.

## Locked evaluation

The independently selected forest in each outer fold is trained under three
declared seeds. It is evaluated on strict core, quality pass, and field stress.
Probability calibration uses a single temperature fitted to cross-fitted
development predictions. Selective prediction uses calibrated maximum
probability.

The globally inner-selected strict and quality forests are also evaluated on
the frozen held-instrument and held-sensor partitions. Domain outcomes cannot
change the configuration.

## Interpretability

Both native impurity importance and held-out 20 cm⁻¹ band permutation
importance are recorded. Neighboring spectral variables are strongly
correlated, so importance is interpreted as predictive reliance on a region,
not as proof of a molecular assignment or causal Raman band.

## Claim limits

Random forest predicts labels; it neither reconstructs spectra nor removes
noise. Strong performance would support a nonlinear tabular baseline, not
chemical/nuisance disentanglement. Any comparison with published random-split
SERS results must acknowledge that this protocol holds out physical master
samples and is consequently more difficult.
