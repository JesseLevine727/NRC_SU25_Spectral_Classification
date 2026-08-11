# P03 classical benchmark execution

P03 is the first predictive ATLAS phase. It must establish a classical
reference under `PP-U-MIN` without changing the P01 representation or any P02
role. `P03_HANDOFF.md`, the master plan, the P03 registries, and
`p03_governance_contract.json` jointly define the executable contract.

Class order is frozen before fitting in that contract: each station has an
explicit three-class order for T1/T3, and each T2 direction has the declared
two-class order. Sparse inner roles therefore cannot silently redefine score
columns or metric denominators.

## Mandatory no-fit gate

`scripts/run_p03.py plan` validates the immutable P01/P02 latest runs, rehashes
every consumed protected payload, and expands the complete registered design
without importing `atlas_sers.models`, fitting an estimator, or using a test
outcome. It writes a private, atomic, content-addressed plan under
`${ATLAS_ARTIFACT_ROOT}/p03plan/runs/<run_id>/`.

The expansion contains:

- 126 candidate configurations across the nine registered classical
  families;
- 540 T1 family-level outer runs;
- 260 development-selected T3 outer runs;
- 1,040 fixed-family T3 outer runs;
- two directional T2 runs;
- 260 source-only CORAL outer runs; and
- 5,200 master-label-permutation, 260 metadata-only, and 520 prior-control
  outer runs under the exact C09 roles; and
- 80 explicitly visible, non-authorized low-support exploratory cells.

The 260 primary T3 cells reproduce the immutable P02 routing exactly: 128 use
supported source pseudo-domains and 132 use the registered three-fold
source-master fallback. Every fitting, validation, calibration, and test role
is represented by row/master counts and UID-set hashes. The validator checks
fit/test UID disjointness, inner fit/validation master disjointness, complete
experiment coverage, deterministic byte-identical expansion, and zero model
fits.

## Compute-gate finding and resolution

A literal expansion of the proposed complete contract produces 260,356
manifest task rows and at most 247,924 activated estimator fits after declared
cache reuse and mutually exclusive deterministic/forest seed branches. This
exceeded the original classical budget ceiling of 15,000 fits by 232,924. The
largest contributors are the full per-cell T3 candidate search and the
three-seed, 1,000-tree forest grids. This is a design inconsistency discovered
before any predictive outcome was accessed, not a model result.

Storage planning includes the 7,447,084 inner-validation prediction records,
not only final outer-test predictions. These source-only records are needed to
audit instrument-balanced selection and recover the selected model's
cross-fitted calibration evidence; the estimate is deliberately conservative
and must be checked against free private-artifact storage before launch.

Before any protected fit, the study owner approved the recommended versioned
amendment on 2026-08-10. Contract `P03-AUTH-20260810` raises the activated-fit
ceiling to 250,000 while preserving the literal design. The implementation may
not silently use fewer repeats, folds, seeds, families, candidates, or domains.
Authorization becomes operational only after the amended plan regenerates,
content-verifies on repetition, and independently validates.

## C12 source-only covariance resolution

Conventional CORAL transforms an unseen target using target covariance, which
is prohibited in T3 zero-shot evaluation. P03 therefore records the approved
C12 operation explicitly as `source_to_source_covariance_augmentation_v1`, not
conventional target CORAL. It uses only source-development master views, rank
at most 20, the frozen trace-scaled ridge, and direct raw unseen-row inference.
No held-target mean, covariance, batch statistic, QC value, or transform is
permitted.

## Negative-control resolution

The master plan requires master-label permutation, acquisition-metadata-only,
and prior/chance controls, but does not yet freeze the permutation count,
whether model selection is repeated under permutation, the task/domain scope,
or the safe acquisition-feature allowlist and model for the metadata control.
Those choices materially change both the interpretation and compute count.
The outcome-blind proposal is now versioned as 20 frozen-selection C09
master-label permutations, a 30-candidate metadata-only elastic-net control,
and empirical/uniform priors across every primary T3 cell. The study owner
approved that exact design on 2026-08-10 before any protected fit. A
post-outcome control redesign remains prohibited.

## Frozen execution sequence

1. Implement every classical estimator behind one fit/predict/score contract,
   with rank-limited candidates retained as terminal unsupported records.
2. Run T1 nested selection on the immutable master folds and pool four-fold
   predictions within station/repeat.
3. Freeze the development-selected classical procedure without consulting a
   T3 target outcome.
4. Run T3 using only `train_source`, the P02 pseudo-domain/fallback route, and
   the registered lexicographic objective.
5. Run fixed-suite T3, T2, and resolved source-only CORAL as secondary
   analyses.
6. Fit scalar-temperature calibration only from master-grouped cross-fitted
   development scores.
7. Reconstruct row, instrument-view, master, class, station, fold, and domain
   tables; run chance and metadata controls; generate F12/F13; and validate
   all private hashes before freezing the P04 comparator.

## Commands

From `research/atlas_sers`, with private roots configured outside Git:

```bash
python3 scripts/run_p01.py validate
python3 scripts/run_p02.py validate
python3 scripts/run_p03.py plan
python3 scripts/run_p03.py plan
python3 scripts/run_p03.py validate
```

The first plan must be `new`; the identical second invocation must be
`verified_skip`. Validation independently rehashes every artifact. A passing
plan may set `scientific_fitting_authorized=true` only when the approved budget,
C12 method, and controls are simultaneously resolved. That distinction
prevents a structurally correct manifest from bypassing a scientific gate.

After a versioned contract explicitly resolves both gates, selection work is
executed by immutable shard ID with `scripts/run_p03.py execute-selection
--shard-id N`. The command revalidates P01/P02/P03, the current public source
tree, configuration hashes, representation order, fit-ID membership, and the
authorization bit before constructing an estimator. Completed shards are
content-verified skips; corrupt finals and interrupted temporaries are
quarantined rather than overwritten.

The batch commands preserve that exact single-shard behavior while loading
the frozen execution context once per worker. Assignment is deterministic:
an ID belongs to worker `w` of `k` exactly when `ID mod k = w`. `--start-index`
is inclusive, `--stop-index` is exclusive, and `--max-tasks` supports a
resource-only pilot without altering the scientific manifest. Worker
partitions are tested for disjointness and complete coverage. Concurrency may
be chosen from CPU, memory, storage, and elapsed-time observations only; no
prediction, score, or test outcome may change the scope or worker count.
Every selection and outer worker enforces and records one native BLAS/OpenMP
thread, and every forest estimator uses `n_jobs=1`. Parallelism therefore
occurs only across independent immutable task IDs, avoiding nested 32-way
oversubscription and making the resource pilot interpretable.

The complete protected execution is dependency ordered and resumable:

```bash
python3 scripts/run_p03.py execute-selection --shard-id N
python3 scripts/run_p03.py aggregate-selection
python3 scripts/run_p03.py execute-outer --outer-index N
python3 scripts/run_p03.py aggregate-final
python3 scripts/run_p03.py validate-execution
```

For example, after authorization and a successful resource pilot, four
independent shells may execute the complete deterministic partitions with:

```bash
python3 scripts/run_p03.py execute-selection-batch --worker-index 0 --worker-count 4
python3 scripts/run_p03.py execute-selection-batch --worker-index 1 --worker-count 4
python3 scripts/run_p03.py execute-selection-batch --worker-index 2 --worker-count 4
python3 scripts/run_p03.py execute-selection-batch --worker-index 3 --worker-count 4
```

After selection aggregation, the same pattern uses
`execute-outer-batch`. A resource pilot uses `--max-tasks 1` on each worker;
rerunning the full partition then verifies and skips completed work. These
commands do not launch background processes themselves, so logs and process
supervision remain explicit.

All 225 selection shards must validate before `aggregate-selection` freezes
the 2,302 source-development selections. That freeze is required before any
of the 8,082 executable outer indices can run. `aggregate-final` then requires
every selection and outer shard to be present and content-valid, reconciles
all 260,356 planned fit IDs to exactly one terminal record, and reconstructs
the frozen 8,142 fold-level procedure endpoints. It writes the terminal and
failure ledgers, protected row/master predictions, expected-endpoint registry,
pooled metrics, domain summaries, calibration records, shard validation, and
a content-addressed descriptor in one atomic shard. If a selected model or
dependency fails, its pooled endpoint remains visible as `unavailable`; it is
never removed from a denominator.

Final aggregation additionally validates every prediction row against the
public result schema and writes candidate-selection frequency, repeat
stability, entropy, winner margins, endpoint coverage, fixed-family T1-versus-
T3, spectrum-versus-master, confusion, reliability, controls, and cost tables.
It creates F12, F13, and F38–F43 from one hashed table per figure in native
TikZ/PGFPlots, vector PDF, 300-DPI PNG, and standalone self-contained Plotly
HTML. The private report records denominators, failures, controls, limitations,
and artifact hashes. The P04 handoff freezes the exact 260 C09 classical
comparator cells and their selected model specifications; P04 may consume that
freeze but may not select a different classical comparator from P03 test
outcomes.

`validate-execution` independently rehashes the atomic final bundle and checks
all 260,356 fit IDs, all 8,142 expected endpoints, metric reconstruction
coverage, prediction-schema coverage, figure form/parity flags, exact P04
mapping, and report presence. A run is not complete merely because every fit
process exited; this independent validation must pass.

## Completed protected execution

The approved no-fit plan regenerated deterministically (`new`, then
`verified_skip`) and independently validated before fitting. The protected run
then completed all 225 selection shards, froze all 2,302 source-development
selections, and completed all 8,082 executable outer/control shards. Final
aggregation reconciled every one of the 260,356 planned fit IDs to one terminal
status, retained unavailable endpoints in their declared denominators, rendered
F12, F13, and F38–F43 in all required forms, and froze the exact 260-cell P04
classical comparator. The independent final validator passed all required
bundle, endpoint, schema, figure, fit-ledger, report, and handoff checks.

The protected metrics, predictions, selected specifications, figures, gate
outcomes, validation report, and run identities remain outside Git. This
public completion record is evidence that the protocol executed; it is not a
disclosure review or permission to publish predictive values. P04 is now the
next executable phase and may consume only the machine-frozen comparator—not
P03 held-test outcomes for neural development choices.

## Privacy and claim boundary

The public repository contains code, schemas, aggregate plan counts, and
methods only. Spectra, row/master identifiers, role assignments, fit/test
hashes, predictions, QC values, source notes, local paths, and provenance stay
under the private artifact root. Authorization itself is not a result: until
the complete protected execution and independent final validation pass, P03
supports no accuracy, calibration, model-ranking, or classical-versus-deep
conclusion.
