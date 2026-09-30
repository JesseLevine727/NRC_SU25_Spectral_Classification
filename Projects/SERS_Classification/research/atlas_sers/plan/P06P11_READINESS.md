# NATO SERS: frozen-results inference readiness

**Date:** 2026-09-29 (local). **State:** evidence audit passed; uncertainty specification not yet locked.

## Scope

**Update, 2026-09-30:** the approved analysis has now run once and passed independent numerical review. The [result report](../reports/NATO_SERS_UNCERTAINTY_REPORT.md) and [figure browser](../results/p06p11/index.html) supersede the unexecuted status in this historical readiness audit. The original hierarchy produced no defined draws; the added primary M01 interval is 0.68–7.79 percentage points around a +4.36-point difference. G4 remains unpassed, with two unassessable criteria. The [review record](delegation/P06P11_REVIEW.md) separates numerical acceptance from publication checks. No new training is authorized.

The active goal covers the completed four-recipe benchmark's remaining P06 synthesis and P11 uncertainty analysis. It does not authorize fitting, calibration, new predictions, preprocessing changes, adaptation, or RBF/SOM experiments. The [master plan](MASTER_PLAN.md), [P05 completion audit](P05_COMPLETION_AUDIT.md), [frozen results](../results/p05_comprehensive/P05_RESULTS.md), and [supervised implementation protocol](ORCHESTRATION_PROTOCOL.md) remain authoritative.

DeepSeek V4.1 Flash authors implementation patches from scoped, non-private inputs. The supervisor reviews the code, independently tests it, controls scientific execution, and reviews publication. The first slice is an outcome-blind support audit, not a confidence-interval calculation.

## Evidence checks

The supervisor reauthenticated the reporting receipt and upstream bindings, checked all 159 released file hashes against their private originals, and verified the aggregation, comparison and selection manifests. The P03 and P04 prediction shards matched their saved content hashes. The manifest, context registry and role registry matched the numerical contract's input pins.

The inherited-role validator passed for 598 manifest rows, 320 contexts and 132,392 role rows. Independent checks confirmed physical-master separation in all 320 outer contexts, exclusion of the held instrument from fitting, and exact agreement between predicted observation sets and the registered test roles.

The 51,975 strategy-alias seed-prediction rows produce 17,325 ensemble rows. Every ensemble row has the three registered seeds; direct probability averaging reproduced the saved ensemble with maximum absolute difference 0.0. These row counts include repeated appearances, not independent measurements or new fits.

Recomputing the comparison from saved predictions reproduced all four stored tables exactly: 4,144 endpoint rows, 8,840 paired rows, 2,080 coverage rows and 102 summary rows. The primary common-support contrast remains +4.35573093 percentage points for individual spectra and +3.61373519 points for combined sample predictions, rounded here to eight decimal places. These are reproduced point estimates, not new significance results.

## Coverage and available work

| Evidence | Contexts | Distinct spectra | Physical-master units | Domains / instrument identities |
|---|---:|---:|---:|---:|
| Within-station neural development | 60 | 598 | 69 | Not the held-instrument estimand |
| Held-instrument neural and fixed-reference evaluation | 260 | 557 | 69 | 13 / 10 |
| Selected-classical paired comparison | 252 | 557 | 69 | 13 / 10 |

The eight incomplete selected-classical references remain missing: two CWA and six surfaces contexts. The stored reason is `incomplete_or_missing_reference`; this audit does not invent a more specific cause or refit them. They remove 71 repeated test-row appearances from that comparison, not 71 distinct spectra.

The 41 spectra outside the primary held-instrument evaluation belong to the four metadata-declared exploratory domains: CWA/Agilent-3 (19), CWA/Pendar-3 (13), pills/Pendar-2 (6), and surfaces/Mira-2 (3). They are included in within-station development. They are not newly failed or discarded predictions.

Frozen classical T1, T2 and T3 predictions are present. The neural archive contains within-station development and the 13 primary T3 domains, but no T2 or four-exploratory-domain neural evaluation. The classical T3 shard also contains PCA-LDA and prior-control records, which require their own role/duplicate-control mapping before inclusion in the wider P06 panel. The already-paired P05 comparison covers its registered eight methods, including the selected-classical procedure.

The T1 selected-classical comparator required for a full G4 chemistry-retention assessment is not directly represented by a selected-classical T1 experiment in the inspected final shard. The frozen experiment registry confirms that C00–C08 define individual T1 model families, whereas C09 defines the selected procedure only for T3. No matching predeclared selected-classical T1 mapping was identified in the execution specification. Fixed-family T1 results remain descriptive; they cannot be substituted after outcomes to pass the selected-comparator gate. Unless an authenticated, predeclared mapping is subsequently located, this G4 component remains unassessable. Missing wider-programme evidence is not completed P06/P11 work.

## Structural issue before uncertainty estimation

The published score is an equal-domain mean of equal-context balanced accuracies. Each context averages recall over its originally observed true classes. An originally absent class does not make that context's published score undefined.

The resampling issue is different: drawing masters with replacement can remove every member of a class that was originally represented in a context. Calculating the same fixed-support score then requires an explicit empty-cell rule. Dropping that class, pooling contexts, inserting a zero, or rejecting the draw are different analysis choices; none should occur silently.

| Masters in a context–class cell | All 260 contexts | Primary paired 252 contexts |
|---|---:|---:|
| 1 | 237 | 228 |
| 2 | 430 | 418 |
| 3 | 71 | 68 |
| Total represented cells | 738 | 714 |

For a class pool with N distinct masters and an originally represented cell containing k of them, the probability that N uniform draws with replacement omit the cell is `(1 - k/N)^N`. Summing this probability across cells gives an expected empty-cell count, not the probability of any empty cell. For one such within-domain resample in each of the 13 observed domains, the expected total is 119.57596177 cells on full support and 115.50587140 on primary paired support. These are analytical support calculations; no random draws or uncertainty intervals have been produced.

The grouping is crossed, not strictly nested. Of the 69 masters, 2 appear in one primary domain, 12 in two, 19 in three, 1 in four and 35 in five. Thus 67 masters appear in multiple domains. Agilent-3, Pendar-2 and Pendar-3 each appear in two primary station–instrument domains. A domain-local bootstrap does not, by itself, preserve those cross-domain dependencies.

The [machine-readable support summary](../results/p06p11_readiness/support_summary.json)
contains the aggregate tables for both supports and hashes of the two consumed
prediction/coverage files. It contains no master, observation or context IDs.

The historical P04 bootstrap is not a drop-in solution: it pools repeated correctness before domain scoring, conditions on retained class support, and does not resample domains. Applying it unchanged would not reproduce the newly frozen equal-context point estimator.

## Historical owner-approved analysis amendment — before execution

The owner approved this amendment on 2026-09-29. The primary comparison, published point estimator, model choices, test support and practical margins remain unchanged. The amendment adds an explicitly post-benchmark, support-preserving crossed-weight analysis alongside an audit of the originally planned hierarchical resampling. It must not be presented as the unmodified preregistered hierarchical interval.

The approved additional analysis has 10,000 paired draws:

1. Draw an independent positive, unit-mean exponential weight for each physical master and one for each instrument identity. Reuse each master weight across all of its spectra, contexts, methods and domains; reuse each instrument weight across its station domains.
2. Within each original context and observed class, recompute the spectrum-level recall using master weights. For M06, form the already-defined sample prediction first, then weight its correctness. Average over the same originally observed classes and contexts as the published estimator.
3. Average domain scores using their instrument weights, with normalization over the fixed set of supported domains. Unit weights must reproduce the original point estimate exactly; all methods in a contrast use identical weights and common support.
4. Report approximate percentile uncertainty intervals, grouped leave-out sensitivities and explicit small-sample limitations. Separate fixed-instrument/master-weight-only and instrument-weight-only sensitivities can identify which variation is being represented; their definitions must be frozen before execution. Do not automatically apply a row-wise BCa correction to crossed clusters.

Positive weights retain represented cells, but they cannot create missing chemistry or estimate within-cell variation from a singleton. This analysis conditions on saved fits and the observed measurement design; it does not include all uncertainty from retraining or acquisition of new chemicals. No finite-sample 95% coverage guarantee is claimed.

The method is motivated by factor-wise reweighting for crossed data in [Owen and Eckles (2012)](https://arxiv.org/abs/1106.2125). Their results concern variance under specified crossed-effects assumptions; they do not establish exact coverage for this small, nonlinear SERS score. [SciPy's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) also makes pairing explicit and warns that BCa intervals can be undefined for degenerate distributions. These sources inform the implementation specification, not its empirical validation.

The original 10,000-draw hierarchical procedure is retained as a clearly labelled sensitivity/feasibility analysis with explicit undefined-cell accounting, not repeatedly redrawn until a favourable or computable interval appears. The [numerical analysis contract](P06P11_INFERENCE_PROTOCOL.md) fixes the zero-cell rule and gate interpretation before execution. G4 must not pass merely because an added post-benchmark interval excludes zero; all other criteria and the original uncertainty criterion still apply.

## Historical next boundary at the readiness checkpoint

Finish regression review of the support audit and implement the approved inference specification with synthetic tests and an independent numerical cross-check before real resampling. The audit is not the final research result, a G4 verdict, a preprocessing selection, or a completion claim for the active goal.
