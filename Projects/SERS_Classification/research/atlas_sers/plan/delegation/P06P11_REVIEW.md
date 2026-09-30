# P06/P11 supervisor review

**Current checkpoint, 2026-09-30:** the approved frozen-result analysis, deterministic supplementary metrics and four-format figures have passed independent numerical/visual review and full local release validation. The [report](../../reports/NATO_SERS_UNCERTAINTY_REPORT.md) and [browser](../../results/p06p11/index.html) are the reading entry points. The accepted package is ready for exact-path main publication; its remote CI check is the final publication evidence. Historical implementation failures remain part of this record; they do not describe the accepted current code.

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

## Historical remaining gates at the support-audit checkpoint

The owner approved the inference amendment in the readiness note on 2026-09-29. The original P11 specification and G4 criteria are unchanged. The separate inference protocol records exact resampling, shared-identity treatment, sparse-cell rules, interval construction, multiplicity and finite runtime/storage guards before real uncertainty analysis. No real resampling has run at this checkpoint.

This reviewed code is a support-audit milestone, not completion of P06/P11. Publication requires exact-path staging, public-boundary validation, appropriate regression and review; remote push and CI must be verified separately. No later experiment is authorized by this gate.

## Gates 3–4 — implementation, execution and independent numerical review

DeepSeek V4.1 Flash implemented bounded, tool-denied slices for the adapter, positive-weight engine, original hierarchy, diagnostics, guarded runner, supplementary metrics and figures. The supervisor reviewed complete patches, rejected truncated or schema-inconsistent drafts, requested corrections and independently tested the accepted code. Private rows, sample/observation identifiers and credentials were never included in implementation-worker prompts. No substitute model or worker Git write was used.

Review corrected identity/coverage checks, fixed-estimator aggregation, sparse-cell handling, exact 13-domain gate counts, numerical type guards, actual stored vocabulary containers, renderer coordinates and release provenance. Synthetic fixtures that swapped model and endpoint fields or silently mocked missing modules were rejected. A later release-table draft lost the explicitly required skip for partial-support nonprimary selected-classical comparisons; the supervisor restored that rule and added a regression test. Scientific outcomes were not changed to accommodate software tests.

The source engine and adapter passed a read-only real-data integration: all 34 contrasts reproduced saved point scores within 1e-12, and all 4,144 endpoint audits agreed within 2.220446049250313e-16. The fixed implementation snapshot passed 2,808 package tests with four CUDA skips before later reporting additions. The final pre-render focused slice passed 484 tests and package Ruff; the subsequent browser-aspect correction adds one focused regression test.

The separate 100-draw synthetic resource probe passed in 1.405 seconds. The approved full analysis then ran **once**, without retry: 66.590 seconds, peak resident memory 924,200,960 bytes and 84,735,821 output bytes, below the 30-minute/2-GiB/1-GiB limits. It used no CUDA or training. Original outputs and random arrays remain immutable and private.

Independent direct calculations authenticated fourteen output hashes and 140 original code/protocol hashes; checked all 34 point summaries, 102 weighted interval quantiles, five saved draw indices for every weighted mode/contrast, 782 deletion rows and 68 exhaustive sign-flip/Holm results. Maximum arithmetic disagreement was 3.7470027081099033e-16. No new random draws were used for this review. The original hierarchy produced zero defined draws out of 10,000 in each of 34 combinations. Primary mean empty-cell count was 115.7087, consistent with the analytic expectation 115.50587140139439; no interval conditioned on surviving draws was produced.

The added primary M01 interval is 0.0068373247447678955–0.07794773186966983 around a fixed difference of 0.04355730932485324. It is a conditional post-benchmark addition, not the original G4 interval. G4 remains `unassessable`, with four supported, zero failed and two unassessable criteria; `promote` is false.

### Numerical tie-count correction

An exact rational-correctness review identified seven secondary count rows in which floating-point residuals as small as −2.22e-16 had been called losses. Counts now use the existing numerical reproduction tolerance: positive above 1e-12, negative below −1e-12, tied otherwise. This is a computational-zero correction, not a practical-effect margin. The original run is preserved. Derived summary counts and an explicit before/after CSV are released; all noncount summary fields, point scores, intervals, primary counts and G4 are unchanged. Exact DataFrame comparison verified noncount preservation. The original and release-generation code hashes are recorded separately.

### Supplementary metrics

The deterministic metric layer returns six aggregate tables, separately by endpoint and support scope. Its 780 comparable domain-score checks agree with the inference tables to 3.3306690738754696e-16. An independent NumPy/confusion-count calculation checked 2,210 metric values to maximum absolute error 8.881784197001252e-16, all 540 confusion cells and all 600 fixed-width reliability bins. A first reviewer command attempted in-place normalization of a read-only pandas array and stopped before a comparison; the reviewer used an explicit copy. Production metrics and saved predictions were unaffected.

## Gate 5 — figures, reporting and next-phase boundary

The four figures contain 26 domain points, 12 contrast intervals, six weighting intervals and 24 deletion/reference points. Each uses one semantic CSV for native TikZ and offline HTML, with vector PDF and 300-DPI PNG outputs. Visual review corrected overlapping native labels and panel spacing, then constrained browser plot domains so equal-aspect resizing preserves the declared axis limits. These are display corrections only: all inference, supplementary and semantic CSV bytes are identical across rendering attempts.

Three deterministic rendering passes took 45.730 seconds cumulatively, within the separate ten-minute allowance. The highest monitored aggregate process-tree RSS was 345,239,552 bytes; each release bundle was approximately 21 MB. No scientific resampling occurred. All four PDFs were checked with `pdfimages -list` and contain no raster image objects. All four HTML files were rendered with a fresh browser profile and external name resolution disabled. Numerical axes, markers, interval endpoints, captions and visible semantic hashes were reviewed. Browser screenshots remain private review artifacts, not publication sources.

The report uses the Lenarizer evidence-bound style: comparator, operating conditions, metric definitions and limits accompany each result. Preservation checking found only authorized additions: the analysis date, independently checked supplementary metric/support values, numerical tolerance and expanded evidence links. The research-question map's status date advances from 2026-09-29 to 2026-09-30. No original result numbers or citations were removed. Black text and standard LaTeX/Times-compatible typography follow the owner's preference. Scientific-visualization guidance informed accessible redundant markers and interval/scatter layouts; no unvalidated significance marks were added.

The [P08 handoff](../P08_HANDOFF.md) plans universal preprocessing comparisons before family/QC policies and robustness. It requires a separate no-fit model/role/support/budget lock; it is not an execution permit. No G4 promotion, universal substrate independence, causal disentanglement or optimal-preprocessing claim is made. Missing wider T1/T2/exploratory neural evidence remains a programme gap.

## Gate 6 — publication verification

**Local release gate passed, 2026-09-30.** The immutable package-only snapshot passed **2,957 tests**, with **four CUDA-only skips**, in **501.30 seconds**. The 2,026 warnings are retained in the private regression log. Whole-package Ruff passed, and public-boundary validation passed across **862 files**. All 35 manifested release files and the current release code/protocol hashes match their recorded digests. The final documentation-only status update does not alter the tested implementation or generated evidence.

The supervisor accepts the numerical analysis, aggregate metrics, figures and report for publication. Generation manifests deliberately retain `requires_supervisor_review`: a generator does not approve its own work; this dated review supplies the separate acceptance. Only the exact reviewed paths may be staged. Private analysis rows/draws and unrelated legacy work remain excluded.

Remote publication requires a non-forced push to `main`, an exact local/remote commit match and a successful **NATO SERS research validation** check attached to that release commit. A local pass does not substitute for that remote check. This pre-push record does not claim that the remote gate has already completed.
