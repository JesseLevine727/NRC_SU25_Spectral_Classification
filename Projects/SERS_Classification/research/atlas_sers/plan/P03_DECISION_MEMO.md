# P03 pre-fit decision memo

This memo records three outcome-blind protocol decisions that must be frozen
before P03 may fit protected spectra. None of the findings below uses a model
prediction or outer-test outcome.

## 1. Literal compute authorization

The complete manifest, including the controls in section 3, has 260,356
terminal task rows. Mutually exclusive selected-model branches mean that at
most 247,924 estimator fits activate: 210,730 for C00–C12 plus 37,194 for the
predeclared controls. The original registry ceiling was 15,000, so the
implementation correctly refuses to start.

The conservative resource model estimates roughly 14.9–268.0 CPU-hours and
39.6 GB of private artifact storage for the complete design. The workstation
has 32 logical CPUs, but free storage and safe concurrency must be confirmed
with source-only pilot shards because the 1,000-tree ensembles have the
largest memory footprint.

Recommended resolution: authorize an upper ceiling of 250,000 activated fits.
This preserves every registered model, domain, repeat, fold, selection route,
technical seed, and control while leaving 2,076 fits of headroom. This is an
authorization ceiling, not an instruction to fabricate or repeat fits after a
terminal result.

Alternative: replace the literal grid with a staged source-only search. That
would change the estimand and requires a separately versioned design; it may
not be introduced after seeing any outer-test outcome.

## 2. C12 source-only covariance control

Conventional CORAL requires target covariance and is therefore incompatible
with the primary zero-shot rule. Calling an operation “source-only CORAL”
without defining the unseen-row transform would be scientifically ambiguous.

Recommended resolution: define C12 as a CORAL-inspired source-to-source
covariance augmentation control, reported under that explicit name rather
than as target adaptation.

For each authorized source-development fit only:

1. average technical rows within each physical-master/acquisition-unit view;
2. estimate each source unit's mean and regularized low-rank covariance from
   only its master views in that fit role;
3. use rank `min(20, unit_master_views - 1, feature_count, numerical_rank)` and ridge
   `1e-3 * trace(covariance) / feature_count`; mark rank below two unsupported;
4. for each physical master, create one raw view by pooling its available fit
   rows; for every destination source unit, whiten each of that master's
   available origin-unit views in the corresponding origin covariance, color
   them in the destination covariance, and average the transformed origin
   views;
5. retain exactly one raw plus one transformed view per destination unit for
   every physical master, so every master has equal total weight, and fit the
   declared PCA–LDA/RBF-SVM base candidate; and
6. apply the resulting classifier directly to each held-target row without
   computing any target mean, covariance, batch statistic, QC threshold, or
   transform.

The existing source pseudo-domain/master-CV objective selects only among
declared base candidates. C12 is a secondary invariance control; failure to
improve does not invalidate the primary comparator.

Alternative: remove C12 from P03 and reserve conventional CORAL for the later
unlabeled-target adaptation phase, where target statistics are explicitly
allowed. This is methodologically cleaner but removes the planned source-only
covariance control.

## 3. Negative-control scope

The master plan requires master-label permutation, acquisition-metadata-only,
and prior controls but did not freeze their exact fit budget. Defining them
after outcomes would make the validity gate outcome-dependent.

Recommended resolution:

- Run 20 deterministic master-label permutations on the primary C09 T3
  endpoint. Freeze the real-label source-selected candidate and
  hyperparameters first, permute labels among current source-training masters
  within station, refit, and score unchanged real outer-test labels. Do not
  re-run the 126-candidate search. Treat this as a leakage/chance diagnostic,
  not a permutation-test p-value.
- Run one metadata-only elastic-net logistic control on the identical C09
  roles. Select the existing 30-candidate logistic grid using the same source
  pseudo-domain/fallback objective. Allow only source-fitted missingness,
  imputation, scaling, and one-hot encoding.
- Freeze the metadata allowlist to categorical `instrument`,
  `instrument_family`, `sensor_family`, `sensor_variant`, `source_format`, and
  `team`, plus numeric acquisition fields `averages`, `laser_power`,
  `n_points`, `axis_min_cm1`, `axis_max_cm1`, `axis_step_median_cm1`,
  `leading_constant_points`, `trailing_constant_points`, and
  `finite_fraction`. Exclude IDs, row numbers, timestamps, scenario/label
  fields, target-detection logs, and all intensity-derived QC summaries.
- Add empirical and uniform source-station priors to every primary T3 cell.
- Preserve all failed/unsupported control cells, use no control outcome for
  model or preprocessing selection, and report controls separately from the
  primary comparator.

The frozen routing gives an exact worst-case control burden of 37,194 fits:
15,600 permutation refits, 21,074 metadata-selection/refit/calibration fits,
and 520 prior fits. The combined upper bound is 247,924, just below the
recommended 250,000 ceiling.

## Approval record

The study owner explicitly approved all three recommended decisions on
2026-08-10 before any protected P03 estimator was constructed. The binding
contract records decision `P03-AUTH-20260810`, raises only the activated-fit
ceiling to 250,000, resolves C12 as
`source_to_source_covariance_augmentation_v1`, and resolves the controls as
`p03_negative_controls_v1`. The no-fit planner must now be regenerated twice
(`new`, then `verified_skip`), independently validated, and show
`scientific_fitting_authorized=true` before the first estimator is
constructed.
