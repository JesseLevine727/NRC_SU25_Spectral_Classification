# P03 completion audit

This matrix prevents implementation readiness from being mistaken for a
completed scientific benchmark. The binding requirements are the master plan,
`P03_HANDOFF.md`, the P03 governance contract, and the registries. A check is
complete only when the named authoritative evidence exists and independently
validates. Source-code presence or a passing unit test alone does not prove a
protected scientific result.

## Current boundary

The outcome-blind plan and execution machinery are validated. The study owner
approved the 250,000-fit ceiling, C12 source-to-source covariance definition,
and frozen negative-control design on 2026-08-10, before any protected P03 fit.
The amended plan must still regenerate and independently validate before the
first estimator is constructed. Until protected execution validates, P03
supports methods and resource-planning claims only—not accuracy, calibration,
model-ranking, classical-versus-deep, or universal-preprocessing claims.

## Requirement-to-evidence matrix

| Requirement | Required authoritative evidence | Pre-fit state | Scientific completion condition |
|---|---|---|---|
| Immutable P01/P02 consumption | passing latest P01/P02 validation; input hashes; exact protected-payload hashes; representation and manifest hashes | Implemented and validated in the no-fit plan | Final execution context revalidates the same latest immutable runs and hashes before every shard and aggregation |
| Exact population and policy | 598 observations, 69 physical masters, `PP-U-MIN`, `R_MIN_400_1800`, frozen class vocabularies | Proven by plan validation | Every protected prediction carries the same policy/representation identity and passes the result schema |
| No resplitting or role inference | exact P02 UID/master role reconstruction; 128 pseudo-domain and 132 master-CV fallback cells; fit/test and fit/validation disjointness | Proven by deterministic no-fit expansion | Every fitted UID hash equals its manifest row and every final endpoint reconciles to the immutable P02 role |
| Complete estimator/grid design | exact 126 standard candidates, 46 covariance candidates, 52 controls, stable candidate order and hashes | Implemented and unit-tested | Every planned candidate fit is complete or has a valid terminal failure/unsupported status; no family is added post-outcome |
| Full task coverage | C00–C12; T1/T2/T3; 13 T3 domains × 5 repeats × 4 folds; 80 labeled exploratory non-authorized cells | 8,162 outer/control records and 260,356 fit-manifest records enumerated | All 8,082 executable outer/control records and all 8,142 expected procedure endpoints are present and terminal |
| Development-only selection | master-grouped nested T1; source pseudo-domain or source-master-CV T3 objective; lexicographic tie-break | Implemented and tested with selection traces | All 225 selection shards validate and the 2,302 selections freeze before any dependent held-target prediction |
| Leakage-safe T3 | source-only fit/selection/calibration; no held-target statistics, QC, preprocessing fitting, stopping, or thresholding | Fail-closed role and hash guards implemented and tested | Final leakage audit shows exact source/test separation for every T3 cell and no target-batch statistic artifact exists |
| C12 covariance control | versioned source-to-source method, rank/ridge rules, source-only fit hashes, direct raw unseen-row inference | Approved and contract-resolved; execution pending | All C12 records are complete or terminal; report labels it as an invariance control, not target adaptation |
| Negative controls | frozen 20 master-label permutations; metadata allowlist and elastic-net grid; empirical/uniform priors; identical C09 roles | Approved and contract-resolved; execution pending | Every planned control is complete or terminal and is reported separately without influencing selection |
| Learned-transform and calibration audit | fit UID/master/domain hashes; selected hyperparameter hashes; master-grouped cross-fitted scalar temperature evidence | Implemented and tested | Every complete probability endpoint has valid development-only calibration evidence; unsupported probability metrics remain unavailable |
| Predictions and aggregation | private row predictions, fixed class order, instrument-view and instrument-balanced master aggregation | Schema normalization and reconstruction tests pass | Every row passes the result schema; master and spectrum metrics reconstruct exactly from protected predictions |
| Metrics and diagnostics | M01/M03–M09/M23–M25/M31, endpoint coverage, confusion, reliability, controls, T1–T3 and spectrum–master comparisons | Implemented and tested, including unavailable endpoints | Every expected endpoint has complete/unavailable metric rows with exact denominators and reconstruction checks |
| Failure and recovery accounting | atomic shards, content verification, retries, warning/exception digests, corrupt-partial quarantine, one terminal record per fit ID | Implemented and tested | All 260,356 fit IDs reconcile exactly once; no corrupt or partial final remains; every failure is reason-coded |
| Runtime resource integrity | content-addressed resource estimate, private storage headroom, deterministic task partitions, one native math thread and `n_jobs=1` per worker | Estimate and execution limits frozen; approved pilot pending | Source-only pilot confirms safe concurrency without outcome-based scope changes; every shard descriptor records the thread limit |
| Figures | F12, F13, F38–F43 from one hashed table each; native TikZ, vector PDF, 300-DPI PNG, self-contained HTML | Renderer determinism, parity, style, and placeholder tests pass | Actual protected aggregates render all four forms; final validator proves hashes, semantic parity, and form requirements |
| P03 report and P04 comparator | private report, limitations, model specs, exact 260-cell C09 mapping, endpoint and metric hashes | Builders and validation tests pass | Final bundle contains validated report/handoff and machine freeze; P04 consumes it without reselecting from P03 test outcomes |
| Privacy and publication boundary | no protected inputs, identifiers, local paths, source codename, predictions, or unreviewed aggregates in Git | Public scaffold/privacy validation passes | Disclosure review approves any aggregate promoted publicly; protected execution bundle remains outside Git |
| Repository delivery | full P00–P03 tests, clean scoped diff, main-only commit, pushed main, green CI | Pre-fit tests pass; delivery intentionally pending | Post-result tests/privacy validation pass, only scoped ATLAS files are committed, main is pushed, and CI is green |

## Frozen execution and verification order

1. Record explicit approval or a versioned alternative for each unresolved
   decision before accessing any protected outcome.
2. Amend the contract, regenerate the no-fit plan twice, require `new` then
   `verified_skip`, and independently validate that fitting is authorized.
3. Run a source-only resource pilot. Concurrency may change only because of
   memory, CPU, storage, or elapsed-time evidence.
4. Execute and validate all selection shards; atomically freeze selections.
5. Execute all outer/control indices. Reruns must verify and skip completed
   content rather than overwrite it.
6. Aggregate once all planned tasks are terminal. Preserve unavailable
   endpoints and every failure in their declared denominators.
7. Run independent execution validation, prediction-to-metric reconstruction,
   privacy review, and the complete P00–P03 test suite.
8. Interpret results only from validated aggregates, freeze P04, commit only
   scoped public files to `main`, push, and require green CI.

## Completion decision

P03 is complete only when every row in the matrix has direct final evidence.
If any fit, endpoint, control, figure, report, hash, privacy gate, repository
delivery, or CI check is missing or merely inferred, the phase remains active.
