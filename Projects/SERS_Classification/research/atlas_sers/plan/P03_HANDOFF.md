# P03 immutable handoff

P03 is the first predictive phase. It answers the classical portion of
`RQ-P01` without changing P01 representations or P02 roles. This document is a
consumer contract: if implementation convenience conflicts with it, P03 must
stop and record a versioned deviation rather than rebuild a favorable split.

## Required private inputs

P03 must resolve `p01/LATEST.json` and `p02/LATEST.json`, then require both
latest validators to pass. It must store and verify:

- the P01 run ID, protected-state hash, representation registry hash, and
  `R_MIN_400_1800` array/row-order hashes;
- the P02 run ID, protected-state hash, protected-payload bundle hash, and
  hashes of every P02 registry consumed; and
- the P03 code, configuration, dependency, seed, estimator-grid, and run
  identity hashes.

P03 must load the existing P02 CSV records. Calling `StratifiedGroupKFold`,
reassigning a master, changing a draw, or deriving a new test role in P03 is
prohibited. A P03 validator must compare consumed role UID sets with P02
exactly.

The protected execution is content-addressed and resumable at selection-shard
and outer-cell granularity. Deterministic multi-worker partitioning may reduce
elapsed time, but it may not change any scientific role, seed, candidate,
control, endpoint, or denominator. Corrupt completed shards are quarantined;
they are never overwritten in place.

## Primary data and preprocessing

The population is the 598-observation/69-master P01 primary population. The
primary representation and policy are `R_MIN_400_1800` and `PP-U-MIN`:
row-local interpolation on measured 400–1,800 cm⁻¹ support followed by
per-spectrum min–max scaling to `[0,1]`. Loading this immutable array before a
split is permitted because P01 proved the operation row-local.

`R_SG_400_1800` and `R_ARPLS_400_1800` are not candidates in the primary P03
comparison. They remain identical-role universal sensitivities for P08. P03
may emit the source pseudo-domain RBF-SVM records later needed by the frozen
policy-development panel, but may not select a universal/family/QC action from
held-test performance.

Every population-fitted operation—including PCA, learned standardization,
feature selection, centroids, class weights derived from counts, probability
calibration, CORAL covariance, and any threshold—must be fitted again inside
the currently authorized training role. Pipeline objects must expose the UID
set on which each fitted state was learned.

## T1 consumption

For each station, repeat, and fold:

- test masters are the masters assigned that outer fold in
  `master_split_registry.csv`;
- training masters are the other three folds; and
- all observations inherit the physical-master role.

Hyperparameters are selected only from the P02 inner master folds. Outer-test
labels and predictions remain unavailable until the candidate and calibration
state are frozen. Fold predictions are pooled before station/repeat metrics;
the five seeds are technical repeats, not five independent datasets.

## T3 zero-shot consumption

For every primary domain/repeat/fold, P03 must use the exact
`t3_partition_registry.csv` roles:

- fit only `train_source`;
- predict only `test_target` for the zero-shot endpoint;
- retain but never fit `excluded_train_target`; and
- retain but do not score as the held-target endpoint `excluded_test_source`.

The held instrument is forbidden from preprocessing fitting, estimator
fitting, hyperparameter/model selection, early stopping, calibration, and
thresholding. The family identifier and row-QC features are also unavailable
to the primary universal cell.

Candidate ranking uses supported rows from `inner_selection_registry.csv`.
Where two or more supported pseudo-instruments exist, use the frozen
lexicographic objective: mean pseudo-domain balanced accuracy, worst
pseudo-domain balanced accuracy, macro-F1, lower complexity, and declared
candidate order. In the 132 outer cells marked for `master_cv`, use the exact
three-fold assignments in `inner_master_split_registry.csv`. Never combine a
sparse pseudo-domain with a supported one by weighting spectra.

Individual held-instrument folds may omit a class. Preserve their predictions,
but make the definitive domain/repeat metric only after pooling all four
out-of-fold slices, which restores every station class. Never report a sparse
fold balanced accuracy as though it were the complete three-class endpoint.

## Classical candidate suite

P03 implements the frozen candidates and grids in the master plan and
`hyperparameter_registry.json`: prior dummy, correlation/spectral-angle
matching, nearest centroid, PCA–LDA, PLS-DA, elastic-net multinomial logistic
regression, RBF SVM, 1,000-tree Random Forest, 1,000-tree Extra Trees, and the
declared source-only CORAL sensitivity. Candidate order is stable. Rank- or
sample-limited hyperparameters are marked unsupported, not silently changed.

Stochastic forests use the declared technical seeds. Probability calibration
uses known-development cross-fitted scores only. CORAL in zero-shot may use
source pseudo-domain covariance, never held-target covariance. No deep model,
VAE, adaptation regime, open-set threshold, or preprocessing policy is part of
P03's primary model selection.

## Outputs and failure accounting

Each run must record row predictions, master aggregation inputs, probabilities
where valid, fit/selection/calibration UID hashes, selected hyperparameters,
candidate support/failure reasons, timings, and protected input hashes.
Missing, rank-deficient, nonconverged, or invalid candidates remain in the
declared denominator with terminal reason codes. An outer cell cannot disappear
because its score is poor or a held fold is class-sparse.

The minimum metrics are:

- `M01` balanced accuracy, class recall unweighted and later domains
  unweighted;
- `M03` macro-F1 after pooling fold predictions;
- `M04` per-class recall;
- `M05` worst-domain balanced accuracy;
- `M06` instrument-balanced master balanced accuracy where applicable;
- `M07` negative log likelihood, `M08` Brier score, and `M09` ECE only for
  valid cross-fitted/calibrated probabilities; and
- `M23` training time, `M24` inference latency, and `M25` model size.

P03 is not the final deep-versus-classical primary verdict. It freezes the
classical endpoint and complete out-of-fold evidence needed for the later
paired comparison. Report station, repeat, fold, class, master, instrument
view, and domain reconciliations, but treat physical masters and held domains—not
spectra or technical seeds—as the inferential units.

The final bundle must also contain schema-validated row predictions, complete
and unavailable endpoint coverage, selection frequency/stability/margins,
confusion, reliability, negative-control, spectrum-versus-master, T1-versus-T3,
cost/latency/size tables, and F12/F13/F38–F43 in matching TikZ, PDF, PNG, and
self-contained HTML forms. A machine-readable P04 freeze must map every one of
the 260 primary C09 cells to the exact source-selected classical model
specification and hash the mapping and held-test endpoint identities. P04 may
not reselect the classical comparator using P03 test results.

## Prohibited access

P03 must stop if any of the following occurs:

- a master or UID role differs from P02;
- a held-instrument row enters fitting, selection, calibration, or stopping;
- an outer-test label or outcome chooses preprocessing, a candidate, a score,
  a threshold, an epoch, or a fallback;
- a target-batch statistic is computed in a zero-shot cell;
- `instrument_family` or row-QC selects the primary action;
- a population-fitted transform sees an unauthorized row;
- predictions from different policies use different test UID sets;
- technical repeats are treated as independent biological/chemical samples;
- unsupported candidates or cells are silently removed; or
- P03 overwrites a completed protected run.

P03 may begin only from the validated P02 protected bundle. Its first action
should be a no-fit compute expansion that enumerates every estimator × station
or domain × repeat × fold × inner-selection route, estimates runtime/storage,
and confirms that all fit roles resolve without reading a test outcome.
