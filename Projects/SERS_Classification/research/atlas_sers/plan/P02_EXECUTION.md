# P02 evaluation-design freeze

P02 converts the frozen P01 primary population into deterministic evaluation,
selection, policy-support, adaptation/calibration, and held-chemical roles. It
does not train a classifier, encoder, VAE, calibration model, threshold, or
preprocessing policy. Its only scientific conclusion is that the later
experiments have a reconstructable information boundary.

## Preconditions and authority

P02 is authorized only from a private P01 run whose report and artifact state
are complete, whose validation checks all pass, whose files rehash, and whose
`LATEST.json` hashes agree. P02 reads only the P01 `primary_manifest.csv`; it
does not reopen native spectra or derive a new representation. The immutable
P01 action arrays remain:

- `R_MIN_400_1800`, the primary universal min–max representation;
- `R_SG_400_1800`, the conservative smoothing sensitivity; and
- `R_ARPLS_400_1800`, the baseline-correction sensitivity.

P02 may assign roles to those arrays, but it may not alter their coordinates,
operations, parameters, row order, or hashes. It may use station, physical
master, target, acquisition-unit, and source-only QC metadata only as declared
below. It may not inspect a held-test outcome.

The machine-readable authority is the combination of
`split_contract.json`, `preprocessing_policy_contract.json`, and
`p02_governance_contract.json`. The latter freezes three clarifications that
were left qualitative in the master plan: three fallback inner folds, paired
calibration sizes of 3/5/10 total masters to match UDA, and the exact 124-member
QC gate enumeration. These clarify implementation without changing an
estimand or consulting a predictive outcome.

## Canonical outer split

For each of the five seeds `20260805`, `20260817`, `20260829`, `20260910`, and
`20260922`, P02 constructs four `StratifiedGroupKFold` folds independently
inside each station. The input to the splitter is one row per physical master,
the stratum is `target_analyte`, the group is `master_sample_id`, shuffle is
enabled, and the declared repeat seed is the random state.

This creates 345 master assignments: 69 masters × 5 repeats. Every master has
exactly one outer-test fold per repeat, every observation inherits its
master's fold, and no held-instrument coverage statistic influences the fold.
Training contains every station class in every station/repeat/fold cell.

Some individual held-instrument fold slices omit a class because acquisition
coverage is incomplete. P02 records 41 such slices as nonfatal sparse-fold
cells. It does not move masters to make the held instrument look balanced,
because doing so would use target-domain availability to choose the test split.
The definitive held-domain unit is therefore the complete set of four pooled
out-of-fold slices within one domain and repeat. Every one of the 13 domains ×
5 repeats contains all station classes after that pooling.

## Exact T3 zero-shot roles

For every domain, repeat, and outer fold, all station observations receive
exactly one of four roles:

| Master set | Acquisition view | Role | Use |
|---|---|---|---|
| outer train | nonheld instrument | `train_source` | eligible for source-only fitting |
| outer train | held instrument | `excluded_train_target` | preserved; forbidden in zero-shot development |
| outer test | held instrument | `test_target` | final held-domain evaluation only |
| outer test | nonheld instrument | `excluded_test_source` | preserved; not used in that fold |

The registry contains all 13 primary and four exploratory domains. It never
drops a row to simplify a cell. The validator independently reconstructs the
role from the master fold and instrument identity, then compares the ordered
UID/role/reason records exactly. A held instrument in `train_source`, an outer
test master in a fitting role, a duplicate assignment, or a missing source
training class is fatal.

## Inner selection roles

Within each primary T3 outer cell, each nonheld source instrument is considered
as a pseudo-domain. Its instrument-view observations are validation rows, and
all views of the same physical validation masters are removed from fitting.
A pseudo-domain is supported only if:

1. its validation view contains every station class;
2. the remaining source fitting observations contain every station class; and
3. fitting and pseudo-validation physical masters are disjoint.

When at least two source pseudo-domains pass, later P03 candidates use the
frozen lexicographic pseudo-domain objective: mean balanced accuracy, worst
balanced accuracy, macro-F1, lower complexity, then declared order. When fewer
than two pass, selection uses the already materialized three-fold stratified
master CV fallback. The metadata audit finds 132 of 260 primary outer cells
require this fallback; the other 128 retain pseudo-domain selection. This is a
support fact, not a model result.

## Acquisition-platform family support

`instrument_family` is derived from immutable acquisition-unit metadata by
removing the final hyphen-delimited unit suffix. It is never copied from or
inferred using SERS `sensor_family`. For each outer source partition and each
platform family, P02 evaluates thresholds 2, 3, and 4 masters per class. A
source unit supports a threshold only when it contains every station class at
that count; a family is supported only when at least two distinct source units
support the same threshold. The largest viable threshold is selected using
metadata only.

After excluding each held unit, none of the 260 primary outer cells has two
qualifying same-station source units from the held unit's platform family.
Consequently:

- 159 cells are `known_unsupported_family`;
- 101 cells are `unknown_family` in that source partition; and
- all 260 cells must use the declared `PP-U-MIN` family fallback.

This means `RQ-S02` is not estimable as a supported adaptive family-policy
comparison on ATLAS v1. The fallback-inclusive endpoint remains defined, but
it is identical to the universal primary action and must not be presented as
evidence that family-aware preprocessing works or fails.

## Identity-blind QC gate freeze

P02 freezes a finite 124-candidate gate library:

- one always-minimal candidate;
- 15 single-trigger candidates: five permitted features × three source
  quantiles; and
- 108 dual-trigger candidates: two noise/spike features × three baseline
  features × three noise quantiles × three baseline quantiles × two priority
  orders.

The only permitted inputs are the current row's normalized first-difference
noise, spike fraction, baseline energy fraction, baseline span fraction, and
negative fraction. Instrument identity, platform family, SERS sensor, station,
target, master, batch summaries, predicted class, confidence, and test outcome
are forbidden.

P02 records the exact `train_source` UID-set hash allowed to estimate future
quantiles 0.50, 0.75, and 0.90. It intentionally stores no numerical cutpoint.
P03/P08 must calculate a cutpoint only inside the corresponding future source
training partition. Missing/nonfinite QC, an invalid routed action, or
insufficient pseudo-domains falls back to `R_MIN_400_1800`/`PP-U-MIN` as
declared.

## Target-access regimes

Every primary outer cell has independently frozen alternative scenarios. A
scenario assigns every station master one mutually exclusive role. Evaluation
masters are the unchanged outer-test masters with a held-instrument view.
Adaptation/calibration masters come only from outer training and are removed
from that scenario's source-training-master role. At least one unselected
source master from every class must remain.

| Regime | Requested access | Supported cells |
|---|---:|---:|
| zero-shot | 0 target masters | 260 / 260 |
| UDA | 3, 5, or 10 total unlabeled masters | 780 / 780 |
| paired calibration | 3, 5, or 10 total paired masters | 780 / 780 |
| few-shot | 1 labelled master per class | 260 / 260 |
| few-shot | 2 labelled masters per class | 260 / 260 |
| few-shot | 3 labelled masters per class | 130 / 260 |
| few-shot | 5 labelled masters per class | 10 / 260 |

Unsupported few-shot cells remain in the registry with the exact class-support
reason. P07 may run only the supported draws; it may not reduce `k`, substitute
a spectrum count, or reuse an evaluation master. UDA hides target labels and
pair IDs. Paired calibration exposes pair IDs but not target-view labels.
Few-shot exposes labels only on selected calibration masters. None of these
regimes may reselect preprocessing in protocol v1.

## Held-chemical roles

P02 also materializes all eight station-conditioned open-set tasks. Each row is
`train_known`, `test_known`, `test_unknown`, or
`excluded_train_unknown`. The held nonblank chemical is absent from fitting,
preprocessing fitting, candidate/score selection, calibration, thresholding,
and early stopping. `blank` remains a known pills class. These roles are
frozen now even though P09 remains secondary and cannot begin before the
primary route is frozen.

## Independent protected hashes

Every primary outer cell records four separate hashes:

1. split/role state;
2. estimator-selection state;
3. family-policy support state; and
4. QC-gate source state.

The hashes must all differ. A future estimator run must reference its split
and estimator-selection hashes; a family/QC policy run must additionally
reference the corresponding policy hash. No estimator score may overwrite or
stand in for a policy-selection hash.

The private build serializes all 13 protected CSV payloads twice from fresh
builder invocations and compares bytes and SHA-256 bundles. The artifact store
then commits atomically, rehashes the completed state, and requires an
immediate `verified_skip` for the same run identity.

## Fatal validation gates

P02 fails closed unless all of the following hold:

- the private P01 prerequisite is complete, passing, and fully rehashed;
- 598 observations and 69 physical masters are present;
- every master appears once per repeat and every station/repeat has four folds;
- all 13 primary and four exploratory domains reproduce from metadata;
- every T3 row is assigned once and reconstructs exactly;
- held instruments and outer-test masters are absent from fitting roles;
- source training contains all station classes;
- all repeat-pooled domain tests contain all station classes;
- pseudo-domain fitting/validation masters are disjoint;
- family support uses acquisition metadata only;
- QC roles contain no forbidden identity, label, master, or outcome field;
- the gate library has exactly 124 candidates and no numeric cutpoint;
- adaptation/calibration and evaluation masters are disjoint;
- held chemicals are absent from development;
- test outcomes are unused;
- all four protected selection hashes are distinct;
- both protected materializations are byte-identical;
- F10/F11 share a data hash across TikZ and HTML, compile to PDF, render a
  300-DPI PNG, use native marks, and remain standalone; and
- the final tree passes the restricted identifier, workstation-path, secret,
  symlink, and artifact privacy scan.

Sparse held-instrument fold slices and unsupported target/family cells are
nonfatal only because the design explicitly preserves and labels them. A
failed fatal check can never be converted to an unsupported sensitivity.

## Commands

Install the visualization and development extras and system TikZ/PNG tools,
then run from `research/atlas_sers`:

```bash
python3 -m pip install -e '.[dev,viz]'
python3 scripts/validate_public_scaffold.py
ruff check src scripts tests
pytest -q
python3 scripts/run_p02.py audit
python3 scripts/run_p02.py dry-run

export ATLAS_PRIVATE_ROOT=/private/immutable/atlas-inputs
export ATLAS_ARTIFACT_ROOT=/private/atlas-artifacts
python3 scripts/run_p02.py build
python3 scripts/run_p02.py build
python3 scripts/run_p02.py validate
python3 scripts/publish_p02_figures.py
```

The first build must report `action: new`; the second must report
`action: verified_skip` with the same run ID. `validate` independently rehashes
the run, latest pointer, protected bundle, artifact manifest, and deterministic
rebuild evidence. The publication command copies only F10/F11 aggregate CSV,
native TikZ, PDF, 300-DPI PNG, and standalone HTML after rejecting protected
row/master/QC fields.

## Private output contract

The final directory is `${ATLAS_ARTIFACT_ROOT}/p02/runs/<run_id>/`. Its exact
top-level payload set is frozen in `p02_governance_contract.json` and includes:

- outer master and T3 observation-role registries;
- pseudo-domain and fallback inner-master registries;
- domain and acquisition-family support registries;
- preprocessing-policy and finite QC-gate registries;
- target-access scenario and master-assignment registries;
- held-chemical row roles;
- unsupported-cell and leakage-audit tables;
- P01 prerequisite, provenance, protected state, protected payload hashes,
  deviations, and deterministic rebuild evidence;
- private F10/F11 data/TikZ/PDF/PNG/HTML forms and figure manifest; and
- `P02_VALIDATION_REPORT.json` plus `P02_ARTIFACT_HASHES.json`.

The artifact store adds `_STATE.json` and updates `p02/LATEST.json`. Failed,
conflicting, stale, or corrupt transactions are quarantined; a completed run
is never overwritten.

## Recovery

If P01 evidence fails, repair or deliberately rebuild P01; never point P02 at a
different manifest under the same identifier. If a public contract or split
algorithm changes, record a versioned deviation and expect a new P02 run ID.
If an existing run is corrupt, the artifact store quarantines it before a new
transaction. If TeX or PNG rendering fails, install the declared system tools
and rerun the same protected state; do not remove a required figure assertion.
If support is insufficient, keep the requested cell and reason code; do not
resample until it becomes convenient.

## Interpretation and boundary to P03

P02 contains no accuracy, classifier, deep-learning, VAE, denoising, or
disentanglement result. The all-fallback family finding is a design-support
limitation, not a predictive failure. P02 authorizes P03 only after its private
report and latest validator pass, the identical build returns
`verified_skip`, the public validator/lint/tests pass, and the protected bundle
hash is handed to P03. The complete consumer contract is in
`P03_HANDOFF.md`.
