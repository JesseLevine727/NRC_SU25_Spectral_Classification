# P06/P11 supervisor review

## Gate 1 — evidence and structural support

**2026-09-29.** The [readiness note](../P06P11_READINESS.md) records the independently authenticated evidence, exact point-estimate reproduction, role checks and support findings. No model was fitted, calibrated or asked for a new prediction. No random resampling or new confidence interval was computed.

DeepSeek V4.1 Flash authored the pure, outcome-blind `p06p11_support` module and its synthetic tests under tool-denied, bounded assignments. It receives an in-memory table for one model's spectrum-level endpoint and selects only identity/class metadata. The supervisor supplies actual private metadata locally; the worker receives no private rows or identifiers. The module returns aggregate support counts and analytical empty-cell risks without file access, scores, optimization or resampling.

The first patch passed 29 independent tests and Ruff. Review found ambiguous prose about originally absent classes, then requested broader validation and cross-domain tests. The second patch's docstring correction and seven-column invalid-value parametrization were accepted. Its added tests were rejected because they assumed a nonexistent output column and used contradictory master/class fixtures. No production behavior was changed to accommodate those tests. The corrected third patch uses the actual aggregate API and consistent fixtures.

The accepted slice passes **74 synthetic tests** and Ruff. Tests cover all required value types, duplicate and conflicting identities, repeated spectra versus masters, shared identities across domains/stations, absent-class handling, deterministic ordering, nonmutation, outcome-column exclusion, and known analytical empty-cell probabilities. Independent direct-loop calculations on full and primary-paired support matched the module's expected empty-cell counts within `1e-12`.

The analytical calculation sums cell-empty probabilities to obtain an expected count; it does not assume independence between cells or estimate the probability of any empty cell. The original point estimator averages only classes present in each original test context. Bootstrap-created empty cells are a distinct problem requiring a declared analysis rule.

The initial clean-archive regression passed 2,545 tests with four CUDA skips, but one historical integration test failed before its scientific logic because the archive had no Git repository metadata. Adding a local Git identity to that private validation snapshot made the isolated test pass without changing production code. The repeated full fixed-snapshot suite passed 2,546 tests with four CUDA skips. The failed invocation remains in the review record. The subsequent documentation-only approval update passed public-boundary validation across 790 files; its numerical additions were reconciled with the approved specification and support audit.

## Gate 2 — approved numerical specification

The [inference protocol](../P06P11_INFERENCE_PROTOCOL.md) fixes the estimator, three positive-weight modes, original hierarchical empty-cell rule, deterministic streams, 34 contrast–endpoint combinations, descriptive sign-flip families, unavailable BCa reasons and finite resources before real resampling. It preserves the original G4 criterion rather than promoting a result through the added interval. Separate tool-denied assignments cover the paired-score engine, frozen-prediction adapter and aggregate diagnostics, with disjoint new-file scopes. None authorizes access to private records or scientific execution.

The first engine assignment reached its bounded timeout without returning code and was not applied. A smaller weighted-engine assignment was issued; hierarchical sampling is deferred to its own slice. The first adapter patch passed 16 tests but failed one test's expected reason-code check. Independent review also identified identity/coverage validation gaps and contradictory synthetic unit identities. It remains unaccepted pending correction. Neither failure involved model fitting or real uncertainty calculations; only the reviewed support auditor belongs to this milestone release.

## Remaining acceptance gates

The owner approved the inference amendment in the readiness note on 2026-09-29. The original P11 specification and G4 criteria are unchanged. The separate inference protocol records exact resampling, shared-identity treatment, sparse-cell rules, interval construction, multiplicity and finite runtime/storage guards before real uncertainty analysis. No real resampling has run at this checkpoint.

This reviewed code is a support-audit milestone, not completion of P06/P11. Publication requires exact-path staging, public-boundary validation, appropriate regression and review; remote push and CI must be verified separately. No later experiment is authorized by this gate.
