# P08 robustness comparisons and conditional uncertainty

**Specified:** 2026-10-07. **Status:** pre-outcome statistical declaration, not a numerical implementation or execution permit. This document completes the comparison definitions left open by the [stress-test design](P08_PERTURBATION_PROTOCOL.md). It does not change the primary [preprocessing statistical protocol](P08_STATISTICAL_PROTOCOL.md), its hypothesis families or G4.

## 1. Question, support and interpretation

The robustness experiment asks whether a registered preprocessing pipeline reduces identification loss under a specified test-time disturbance, and whether that effect depends on the classifier. The six disturbance families are shift, slope, quadratic background, Gaussian noise, isolated impulses and clipping. Their finite grids, ten stochastic repetitions, matched realization keys, source-only noise reference and shared clean case remain unchanged. Neither disturbance parameters nor model choices are selected from stressed held outcomes.

Owner decision P08-A10 includes RBF-SVM, Random Forest, Extra Trees, D0-M and context-local P05-SELECTED in the MIN/SG/arPLS comparison. P08-A11 adds the labelled fixed-clean-route QC sensitivity with the inherited four-method panel, excluding Extra Trees. QC gate reactions are not tested: each row retains its clean native-QC action, and the already trained mixed-route estimator receives the corresponding disturbed input. Do not replace that estimator with a row-wise mixture of universal predictions.

The [identity-only support audit](../results/p08_readiness/perturbation_inference_support_audit.json) authenticates the existing roles and QC eligibility without reading spectral arrays or predictions.

| Target | Contexts | Domains / instruments | Distinct held spectra / masters | Repeated spectrum appearances | M06 master/context units |
|---|---:|---:|---:|---:|---:|
| Universal and operational QC | 260 | 13 / 10 | 557 / 69 | 2,785 | 1,310 |
| Eligible QC sensitivity | 54 | 3 / 3 | 176 / 24 | 758 | 293 |

The eligible subset has 18 contexts in each of three CWA domains. It is fixed by source-role support, not by the chosen gate or robustness outcome. The 206 unsupported contexts remain complete MIN fallbacks in operational QC. Their disturbed predictions must alias the corresponding **disturbed** MIN endpoint; an old clean prediction is not that fallback. All 260 family-policy contexts are structural MIN aliases, and supported-family transfer remains unavailable. None of these repetitions or aliases creates a new independent sample.

Minimal-input benchmark results have already been examined. These are prospective secondary robustness comparisons, not unseen primary hypotheses or randomized instrument interventions. They neither recover an unmeasured clean chemical spectrum nor validate the imposed disturbance as a particular instrument's physical mechanism.

## 2. Prediction order, curves and area estimands

Retain the primary authentication, class vocabulary, argmax tie rule and calibration order. Forest seed probabilities are averaged before their single source temperature; neural temperatures apply per seed before seed averaging. M01 classifies each spectrum. M06 combines probability vectors within master/instrument and then equally across that master's instruments before one decision; it does not average spectra or hard labels. The held-instrument context has one instrument.

For every stochastic repetition, construct these endpoints and calculate balanced accuracy on the registered context/class support. Average the ten **scores** at each stochastic dose. Do not ensemble probabilities over synthetic repetitions, pool context predictions or add repeats to the physical-master count. All ten repetitions are required; a shorter average is not the registered experiment.

For context c, method m, policy p, endpoint e and family f, define

`L(c,m,p,e,f,x) = BA(c,m,p,e,clean) - mean_r BA(c,m,p,e,f,x,r)`.

For deterministic cases the repetition mean contains one value. At zero dose, use the same authenticated clean endpoint across all families. Numerical preprocessing and prediction parity must pass the design's separate zero-dose gate before accepting either a curve or an area. Keep negative losses rather than clipping accidental improvements.

Integrate L by the trapezoidal rule over the prescribed severity axis normalized to [0,1], giving `A(c,m,p,e,f)`. For signed shift and slope, first average the positive- and negative-direction losses at each absolute dose, then integrate. Retain both directional curves and their separate areas; report the larger directional loss area and every tie descriptively. A worst-dose-direction envelope is a different statistic and is not substituted.

The Gaussian horizontal coordinate is the declared source-quantile rank divided by 0.95, preceded by a distinct no-noise point at zero. It is not a physical noise-amplitude axis or an empirical zeroth quantile. Publish source-proxy amplitude summaries alongside its curves. Do not average areas across different disturbance families or interpret their axes as equivalent physical severity.

The method-specific robustness benefit is

`B(c,m,p,e,f) = A(c,m,MIN,e,f) - A(c,m,p,e,f)`.

Positive B means less loss relative to that policy's own clean accuracy. Calculate this paired context quantity first, average contexts equally within each retained domain, then average domains equally. Report clean accuracy, disturbed absolute accuracy, and clean-minus-stressed loss with B: a weak classifier may have little accuracy left to lose. An area difference of 0.03 is three percentage points averaged over the declared normalized severity axis, not a 3% relative improvement.

For neural strategy n and classical method k, define `I(c,n,k,p,e,f) = B(c,n,p,e,f) - B(c,k,p,e,f)`. Every area in this four-procedure contrast uses the same complete context, dose, repetition and test-row support. Positive I means the preprocessing policy reduces degradation more for the neural strategy; it does not imply higher absolute accuracy or physical nuisance separation.

## 3. Fixed contrast families and structural limits

The [registry](contracts/p08_perturbation_inference.json) enumerates every contrast. Holm adjustment spans **all six disturbance families** within each row below, separately for domain-sign and instrument-sign sensitivities. Do not reset adjustment for each plotted disturbance, endpoint or method.

| Multiplicity family | Fixed slots | Target |
|---|---:|---|
| Universal robustness effects | 6 × 2 policies × 5 methods × 2 endpoints = 120 | All 260 contexts |
| Operational QC robustness effects | 6 × 4 methods × 2 endpoints = 48 | All 260 contexts, including fallback |
| Operational robustness–model interactions | 144 universal + 48 QC = 192 | All 260 contexts |
| Eligible QC robustness effects | 6 × 4 methods × 2 endpoints = 48 | Fixed 54-context CWA subset |
| Eligible QC robustness–model interactions | 6 × 2 neural × 2 classical × 2 endpoints = 48 | Fixed 54-context CWA subset |

There are 456 entries across these five families. The 192-slot operational interaction family retains the primary protocol's grouping of universal and QC interactions. It does not alter that primary family's separate 32 slots. No pointwise dose test, directional-area test, pooled-fold test or Monte Carlo-spread test adds another inferential family. Selected/D0 aliases retain their registered entries; structural duplication does not reduce multiplicity after outcomes.

Use the inherited two-sided absolute equal-domain statistic, exhaustive sign assignments including the observed assignment, and equality tolerance 1e-12. An instrument shares its sign across its station domains. Full operational support has 8,192 domain assignments and 1,024 instrument assignments. The eligible subset has eight assignments for either analysis.

QC effects and QC interactions can be nonzero in only the three CWA domains/instruments. All other operational domains are structural MIN identities. Therefore their two-sided unadjusted sign p-values cannot be below **2/8 = 0.25**, in either the operational or eligible analysis; additional ties can only increase this bound. Counting sign duplicates from fallback-only domains does not improve resolution. This is an algebraic support limit, not a calculated scientific p-value. These analyses cannot establish a conventional 0.05 significance claim for QC; retain the diagnostic and report effect magnitude, direction and support instead. It remains a symmetry sensitivity, not a randomized-treatment test.

An unavailable entry contributes p = 1 to Holm bookkeeping only; report its estimate and raw p-value as unavailable with a reason. Apply the same family sizes even when QC runs later. Family-policy identities receive `structural_identity_no_transfer_test`, not an empirical test. Marginal percentile intervals are not simultaneous intervals and cannot bypass these rules or automatically pass G4.

An [independent declaration audit](../results/p08_readiness/perturbation_inference_declaration_audit.json) verifies the contrast counts and this sign-resolution arithmetic on invented rational effects. It does not calculate a scientific p-value or validate the full numerical inference engine. With complete registered support, equal 18-of-20 eligible-context coverage in each affected domain also makes the operational QC point benefit exactly 27/130 of the eligible-subset point benefit. These are differently weighted views of the same supported effects, not independent evidence. This fixed rescaling need not hold for pooled-fold sensitivities, reduced support or every instrument-weighted draw.

## 4. Conditional uncertainty and pooled-fold sensitivity

Use the P08-A02 global positive-weight arrays: 10,000 Exponential(mean=1) draws, PCG64 master seed 2026093001 and instrument seed 2026093002. Their float64 shapes are (10,000,69) and (10,000,10), with lexicographically sorted identities. Generate each complete array in the inherited allocation order once, then index it for every subset, family, method, policy, dose and repetition. Do not redraw weights for the QC subset or a stochastic repetition.

For each draw, reweight correct prediction units within each original context/class cell as in the primary protocol. Recompute the full loss curves, stochastic score averages, areas, benefits and interactions using shared weights. Apply instrument weights only in the inherited domain aggregation. Synthetic noise vectors, source-noise quantiles, routing choices, temperatures and trained models remain fixed. Average all ten repetition scores within each draw; do not draw an extra repetition bootstrap or recalibrate under a draw.

Report crossed master/instrument, master-only and instrument-only versions. Unit weights must reproduce independent point estimates within absolute tolerance 1e-12. Batch at most 128 draws; reject nonfinite arithmetic or invalid weights without replacement draws or repairs. Use marginal 95% percentile intervals, quantiles 0.025 and 0.975 with linear interpolation. Interactions combine all required areas within each draw, not separately estimated interval endpoints.

These intervals are conditional on observed support, saved fits, source decisions and the particular finite disturbance realizations. They do not include relearning a gate, model selection, retraining, source-noise estimation uncertainty, Monte Carlo integration uncertainty or new instruments. Repetition spread is separately labelled Monte Carlo variability. A degenerate interval is not certainty; BCa remains unavailable because the crossed-cluster acceleration is unspecified.

Retain the pooled-four-fold sensitivity without pooling repeats. At each case/repetition, concatenate four disjoint folds within domain/repeat, form M01/M06 units and calculate BA; then construct the loss curve and area. Average complete repeats equally within domains, then domains equally. Require all four original folds inside the target support: do not fill unsupported QC folds to enlarge the eligible subset.

The operational target has 65 complete domain/repeat groups and 260 contexts. The eligible target has **nine** complete groups and **36** contexts, still covering 176 spectra and 24 masters across the same three domains; it contains 528 repeated spectral appearances. The other 18 eligible contexts remain in the main equal-context analysis but cannot enter this complete-four-fold sensitivity. This metadata-defined pooled subset is fixed before outcomes and is not a post-failure deletion rule. An additional numerical failure leaves the fixed pooled target unavailable, with any smaller complete-paired sensitivity labelled separately.

## 5. Original hierarchy, missingness and stability

Retain 10,000 hierarchical feasibility draws with PCG64 seed 2026093003, reset per registered contrast/endpoint. Use the inherited sorted domain/class/master order and draw the complete domain-index array first. Carry each pair or four-procedure interaction together across all doses and repetitions. A single originally represented context/class cell becoming empty makes that occurrence and draw undefined; do not drop it, retry or change its denominator. Report an unconditional interval only when every planned draw is defined; otherwise retain `hierarchical_fixed_support_undefined`. This hierarchy does not preserve master sharing across domains and is not silently replaced by the positive-weight analysis.

Each family-specific contrast requires the clean cell and every registered dose/repetition for all methods/policies on its complete fixed target. A failed parity check, missing seed, absent calibration, invalid numerical result or incomplete registered test role makes the affected cell unavailable. A missing noise reference can make the Gaussian family unavailable without invalidating a complete shift family. A failure shared by the clean endpoint affects all its families. Do not shorten a curve, interpolate a missing dose, average surviving repetitions or replace a failed universal model with MIN.

An invalid selected QC action retains the explicit row-level MIN-input fallback under the same trained QC estimator. That is distinct from the complete-MIN model fallback in 206 structurally unsupported contexts. Record reason and changed fallback burden; neither is evidence that the frozen gate detected contamination. Invalid MIN input remains fatal. A finite collapsed classifier is a valid outcome, not a reason to rerun it.

If any required cell is missing, the fixed full-target contrast is unavailable. A separately labelled complete-paired-context sensitivity may use the intersection across the **whole curve**, including all repetitions and all four interaction procedures. Publish its exact context/domain/instrument/master/spectrum counts and exclusions; never use a different support at each dose. Its sign diagnostics do not fill missing full-target Holm slots. The eight missing historical C-SELECTED endpoints are a different comparator and are not repaired here.

Publish every domain's paired areas and benefits, mean, median, quartiles, range and positive/negative/tied counts using linear quantiles and tolerance 1e-12. Retain leave-one-domain, leave-one-instrument and leave-one-known-platform-family-out summaries without refits; empty deletions are undefined. For each dose report both pipelines' lowest domain BA and the minimum paired domain effect, distinguishing those quantities. Probability quality and preservation remain descriptive under their inherited definitions. No new preservation threshold or favorable held-test maximum selects a pipeline.

## 6. Figures and remaining gates

P08-F08 uses directional degradation curves with marked doses, paired domain loss-area scatter, clean-versus-stressed accuracy views and the registered benefit/interaction summaries. Show poor domains, structural aliases and unavailable cells. Mark the QC subset as three CWA instruments, state the sign-resolution limit and label its routing as fixed. Keep Monte Carlo spread separate from conditional master/instrument intervals. Native TikZ, offline HTML, vector PDF and PNG share reviewed semantic tables, with black standard-font text and the existing public aggregation boundary.

This declaration and its identity-only audit close the unspecified-comparison gap. Numerical inference and perturbation implementations, exact reconstruction/prediction ledgers, finite resource proposals, parity on scientific inputs and separate execution permission remain incomplete. All model fits, new predictions, resampling draws, preprocessing rebuilds and perturbation runs remain unauthorized. The universal-first, adaptive-second, robustness-later sequence is unchanged.

## 7. Audited comparison bindings and proposed resources

The [contrast-binding audit](../results/p08_readiness/stress_contrast_catalog_audit.json) connects the registered comparisons to the authenticated score-support catalog. It verifies **216 effects and 240 interactions**, retaining all **456** entries and the five original multiplicity families. These are comparison definitions, not measured effects.

| Binding | Main equal-context target | Complete-four-fold sensitivity |
|---|---:|---:|
| Contrast/comparison-unit pairs | 98,784 | 24,264 |
| Signed references to procedure views | 302,592 | 74,352 |

Each effect retains its two signed terms; each interaction retains four. Same-procedure aliases are not cancelled or removed from the registry. The audit checks **9,888 structural QC effect bindings** whose opposite terms refer to the same MIN procedure but retain distinct reporting identities. This does not manufacture a supported QC transfer estimate.

Recorded master IDs keep their integer type in the private catalog. A separate bijection maps their textual identities into the inherited global lexical order for **69 master columns**; **ten instrument columns** are mapped similarly. Subsets retain those global column indices rather than renumbering their members. The authenticated P02 mapping identifies **four** known platform families in the operational target and **three** in the eligible target; platform families are not substrate families. No random weights are loaded or generated by this audit.

The generic adapter binds the externally authenticated registry and platform mapping, but does not establish their provenance itself. The independent actual-record audit verifies their hashes, declared support sizes, signed terms, every context/pooled view target, deterministic reconstruction and unchanged input bytes. Public evidence contains counts and hashes only; identity-bearing catalogs remain private.

The [S1 proposal](P08_RESOURCE_PROPOSAL.md#15-s1-proposal-registered-test-time-robustness) now supplies finite, unapproved limits: **1,820 exact historical MIN reconstruction fits**, no neural optimization or new scalar calibration, **72 hours**, **64 GiB** of artifacts, **24 GiB** of process-tree RAM and **8 GiB** of allocated GPU memory. These limits are not a completion forecast. Full upstream-plus-stress retention exceeds observed capacity; admission requires actual free space and a separate permit.

This closes comparison-to-support binding and the finite-proposal gap, not the full analysis ledger. Exact uncertainty, original-hierarchy, sign/Holm, stability, preservation and rendering operations still require reconciled accounting. Numerical inference, scientific clean-path parity and execution remain unaccepted. No comparison, score, draw, fit or prediction was computed.

## 8. Statistical-operation metadata acceptance

The [operation audit](../results/p08_readiness/stress_inference_plan_audit.json) enumerates the finite inference inventory against the authenticated comparison catalog. It verifies **495,780 planned descriptors**, including **272,340 unconditional slots** and **223,440 conditional slots**. These are future analysis operations, not model fits or executed calculations. Conditional slots remain inactive unless the fixed target is unavailable and the whole-curve paired intersection is nonempty; their existence does not imply that condition has occurred.

| Operation | Allocated descriptors |
|---|---:|
| Shared global-weight authentication | 2 |
| Fixed and candidate support assessment | 912 |
| Point estimates and descriptive curves | 1,824 |
| Unit-weight parity | 5,472 |
| Positive-weight batches | 432,288 |
| Positive-weight summaries | 5,472 |
| Original-hierarchy realization preparation | 456 |
| Original-hierarchy batches | 36,024 |
| Original-hierarchy summaries | 456 |
| Domain/instrument sign sensitivities | 1,824 |
| Fixed-family Holm bookkeeping | 10 |
| Domain stability summaries | 456 |
| Leave-one-domain/instrument/known-platform-family summaries | 10,584 |

Each contrast allocates fixed-context, fixed-pooled, conditional paired-context and conditional paired-pooled views. Each weighted view retains crossed, master-only and instrument-only analyses. Its **10,000** draws occupy **79** batches of at most **128**, with **16** in the final batch. The allocation permits at most **54,720,000 terminal contrast scalars**; no scalar or weight was computed. The full-support sign enumeration contains **3,319,296 assignments** across registered contrasts, with the same upper bound for paired sensitivities. Sign assignments and technical repetitions are not independent samples.

Every emitted operation identifier hashes its complete descriptor, parent-catalog identity and already resolved dependencies. Support assessments retain ordered targets, and every weighted batch depends on both global-weight authentication records. The original hierarchy declares one complete domain-index draw followed by a single multinomial call per sorted domain/class for all selected occurrences, before score batching. It is not replaced with interleaved per-occurrence draws. Numerical parity and these future random-stream operations still require separate implementation acceptance.

The audit traversed every descriptor, checked content identities and topological dependencies, verified ordered support targets and fixed Holm membership, and confirmed unchanged inputs. No scientific values, saved predictions or random weights were loaded. Public evidence contains counts and hashes only; the identity-bearing plan stays private. The [reporting inventory](P08_REPORTING_ACCOUNTING.md) accounts separately for diagnostic reuse and figure delivery. Final cross-ledger reconciliation and requirement-wide readiness review remain open; neither component grants execution authority.
