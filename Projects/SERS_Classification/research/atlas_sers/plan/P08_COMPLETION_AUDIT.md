# P08 no-fit readiness: current requirement-to-evidence audit

**Reconciled:** 2026-10-07. **Status:** scientific specification locked; final release acceptance pending. No scientific execution is authorized.

The input, support, operation-accounting and statistical specifications are documented. P08-A12 resolves the last model-panel choice: Extra Trees is included in both filtered-population sensitivities. Final release validation, remote CI and a separate U0 execution request remain. No further experimental planning layer is required for this no-fit lock. This page is the current status authority; dated checkpoints in the [handoff](P08_HANDOFF.md) and [implementation review](delegation/P08_REVIEW.md) retain their historical scope.

This is a readiness audit, not a preprocessing result. No new P08 model, calibration, prediction, uncertainty draw, QC route, perturbation or outcome figure has been produced. Numerical acceptance of later stages is separate from the no-fit planning lock.

## 1. Requirement coverage

| Original goal requirement | Current evidence and status | Remaining boundary |
|---|---|---|
| G1: authenticate upstream evidence | The [fresh byte audit](../results/p08_readiness/final_evidence_reauthentication.json) passes for 634 unique retained files, including both relevant P00/P01 lineages, P02/P04 planning, P05/P13 releases and the frozen P06/P11 release. | Reauthenticate consumed evidence at launch. The eight missing historical selected-classical endpoints remain missing; no imputation or replacement prediction is permitted. |
| G2: panels and owner decisions | P08-A01–A12 are recorded in the [readiness contract](contracts/p08_readiness_contract.json). Universal preprocessing, robustness and both filtered-population sensitivities use five methods. Adaptive and range analyses use four. N2 retains each context's MIN-selected classical family. | Scope decisions are resolved. They do not authorize resources or execution. |
| G3: frozen inputs and split roles | [Input/support](../results/p08_readiness/input_support_audit.json), [role equivalence](../results/p08_readiness/role_equivalence_audit.json), [nested QC roles](../results/p08_readiness/qc_nested_role_audit.json), and later range/normalization/population audits bind existing arrays and source-only roles. | No array rebuild, new cutpoint estimation or QC routing is authorized. Input integrity does not establish chemical preservation or predictive benefit. |
| G3: support and fallback | All 260 held-family contexts require complete MIN fallback. QC has 54 eligible CWA contexts and 206 complete MIN fallbacks. Metadata-defined unavailable population cells are retained. | Supported-family transfer is unavailable, not a measured zero effect. The QC-supported result cannot establish benefit at pills or surfaces. |
| G4: estimands and statistical rules | The [primary protocol](P08_STATISTICAL_PROTOCOL.md), [range](P08_RANGE_INFERENCE.md), [N2](P08_NORMALIZATION_INFERENCE.md), [population](P08_POPULATION_INFERENCE.md), and [robustness](P08_PERTURBATION_INFERENCE.md) specifications define effects, interactions, support, missingness and multiplicity before new outcomes. The selected population panel has 40 effect and 48 interaction entries across both tiers. | Numerical inference is not accepted or executed. Conditional uncertainty does not pass the original superiority gate or include retraining uncertainty. |
| G4: preservation and weakest domains | The [reporting specification](P08_REPORTING_ACCOUNTING.md) separates authenticated P01 diagnostics, future summaries, probability quality and lowest-domain diagnostics. | No independently measured clean chemical spectrum or validated binary chemistry-preservation threshold is available. |
| G5: exact fit/reuse accounting | The [cross-ledger reconciliation](../results/p08_readiness/readiness_reconciliation.json) verifies stage sums, linked evidence hashes, reporting ownership and additional archived range/N2/population catalogs. The table below separates operation descriptors, fits and reused evidence. | These are finite metadata inventories, not an accepted global numerical executor. Conditional branches remain conditional; aliases and upstream jobs must not be counted as fresh work. |
| G5: finite resource proposals | [Staged ceilings](P08_RESOURCE_PROPOSAL.md) cover U0/U1, QC, range, N2, both population alternatives and robustness. | All ceilings are unapproved. Full upstream-plus-robustness retention requires 574 GiB including reserve, beyond observed capacity. Later capacity gates do not authorize deletion or delay U0 solely because Q1 cannot yet fit. |
| G6: implementation and review | DeepSeek V4.1 Flash implemented the reviewed metadata planners, guards and default-denied U0 launcher. [Invented-data tests](P08_U0_LAUNCH_BOUNDARY.md#7-implemented-boundary-and-tested-limits) exercise source-kernel integration, persistence, interruption and refused relaunch. | CPU fixtures and simulated resource observations do not establish actual GPU enforcement or scientific acceptance. Later numerical runtimes require stage-specific review; they need not all be built to finish this planning goal. |
| G7: release and figures | Contracts, public aggregate audits, review records and the [eleven-bundle figure manifest](P08_FIGURE_PLAN.csv) specify native TikZ, offline HTML, vector PDF and PNG. | Finish exact-state validation, publication, remote-SHA agreement and CI for the revised package. These are prospective figures, not newly generated results. |
| Completion: separate execution request | U0's exact source-only subset and proposed ceilings are documented. | Finish release checks, then request U0 alone. Previous permits do not transfer; successful U0 work cannot automatically launch U1. |

The 634-file check authenticates 17 frozen input pins, 18 inherited source specifications, nine metadata archives and the inference-plan bytes. It verifies 159 P05 public/private file pairs, 11 P13 public release files and 35 P06/P11 release files. A separate reconciliation reauthenticates seven additional range/N2/population archive or binding files. These counts describe distinct audit scopes and are not added as an independent-sample count.

Both retained P01 lineages have matching primary manifest and MIN/SG/arPLS array bytes, despite different upstream provenance. Parent protected-state links use canonical JSON content hashes; retained file inventories use byte hashes. The first private diagnostic incorrectly compared those two digest types, then passed after the interpretation was corrected. No artifact changed.

## 2. Resolved choices and scientific boundaries

The primary dataset remains 598 spectra from 69 physical masters and ten instruments. Held evaluation uses 557 spectra, all 69 masters, 13 station–instrument domains and 260 contexts. The other 41 spectra belong to four exploratory domains; the 60 development contexts remain separate.

MIN, SG and arPLS retain the frozen 400–1,800 cm⁻¹ grid, 1,401 channels, row order and final [0,1] scaling. SG and arPLS both include impulse replacement. Their comparison with MIN therefore tests whole pipelines, not isolated smoothing or baseline-subtraction mechanisms. No despike-only or combined SG/arPLS action is added.

The universal five-method panel is RBF-SVM, Random Forest, Extra Trees, matched ordinary CNN (D0-M) and the context-local P05 source-selected CNN procedure. Its neural recipe identities remain fixed across primary preprocessing policies. The observed held winner does not choose a recipe. Classical hyperparameters are reselected from source data within each policy.

P08-A06 selects N2's four-control, frozen-family sensitivity: 42,368 model fits and 1,040 scalar temperatures. The 525,288-fit N1 full-family alternative is not an additional experiment. The historical reference supports 252 of 260 contexts and 57 complete-four-fold groups; the eight missing endpoints remain missing.

P08-A07 requires fresh registered source-only selection within the notes-clear and Mira-1-excluded populations, then freezes each resulting neural recipe across preprocessing. Notes-clear has 500 spectra and 220 eligible contexts: 215 classical/common and 216 neural contexts are supported. Five classical-calibration and four neural endpoints remain unavailable. Mira-1 exclusion retains 575 spectra and all 240 eligible contexts. Both tiers retain 69 masters. Their acquisition-aware comparison is structurally supported in 93 and 83 contexts; 123 and 157 use the registered ordinary-CNN fallback.

P08-A08–A11 fix disturbance placement before numerical preprocessing, constant extension of unsupported synthetic-shift coordinates, the five-model universal robustness panel and fixed-clean-route QC. The 96 cases feed 456 robustness comparisons. The QC sensitivity tests the pipeline after its original preprocessing choice; it does not test whether the QC gate detects new contamination. Its 54 eligible contexts cover three CWA instruments and 24 masters, so the smallest possible two-sided sign p-value is 0.25. This is an analytical limit, not an observed result.

M01 classifies individual spectra. M06 averages model probabilities within master/instrument and then across instruments before making a sample-level decision; it does not average spectra before classification. Main scores give equal weight to contexts within domains, then equal weight to domains. Pooled-four-fold scores remain a separate sensitivity. The approved 10,000-draw positive-weight analysis retains shared master/instrument identities; the original hierarchy remains a feasibility check. Neither establishes causal nuisance removal.

## 3. Reconciled finite accounting

All values below are prospective. A descriptor is a recorded operation or dependency, not necessarily a model fit, a compute process or an executed job.

| Stage or alternative | Model-fit slots | Scalar calibrations | Existing fit/prediction/selection graph descriptors |
|---|---:|---:|---:|
| U0 source-only smoke, already inside U1 | 78 | 0 | 78 fits + 78 dependent source-validation operations |
| U1: new SG and arPLS | 195,202 | 3,354 | 404,814 |
| Q1: complete nested QC catalog | 1,630,980 | 53,880 | 3,471,416 |
| R1: wider-range sensitivity | 62,981 | 1,417 | 131,199 |
| N2: four frozen-family controls | 42,368 | 1,040 | 89,632 |
| Notes-clear, four methods | 168,150–170,277 | 3,234–4,071 | 349,314–355,521 |
| Notes-clear, five methods | 256,260–258,387 | 3,879–4,716 | 530,763–536,970 |
| Mira-1 excluded, four methods | 183,006–184,749 | 3,600–4,347 | 381,264–386,493 |
| Mira-1 excluded, five methods | 279,135–280,878 | 4,320–5,067 | 579,921–585,150 |

P08-A12 selects the five-method population alternative. The four-method catalog remains historical, not additional work; conditional neural branches remain mutually exclusive. The combined upper fit counts are 355,026 or 539,265; Extra Trees adds 184,239. These ceilings remain unapproved for execution. The U1/R1/N2 descriptor counts include 5,544/1,584/1,776 calibration-prediction alias records, respectively. They are dependencies, not new fits. Historical MIN's 202,407 operation bindings are also reuse evidence, not another retraining arm.

Robustness S1 has 658,876 input, 2,013,605 prediction/reuse, 882,096 score/curve and 495,780 inference descriptors. Their subtotal is 4,050,357; adding its 24,981 reporting descriptors gives 4,075,338. This subtotal excludes upstream U1/Q1 fitting and reporting aliases, and includes 223,440 conditionally activated inference descriptors. It is not a count of new models or evidence of an accepted numerical runtime. S1's only proposed fits are 1,820 exact historical classical MIN reconstructions; clean-probability parity must pass before use. No new neural optimization, hyperparameter search or scalar-calibration fit is included.

The reporting inventory has 25,603 non-alias descriptors allocated once: U1 582, Q1 16, S1 24,981, R1 8, N2 8 and population 8. U1 includes 441 conditional private-example descriptors. The 1,794 historical preservation-row references and 830,208 existing context/pooled case-scoring operations are not added again. Figure rendering consumes frozen scientific outputs rather than recalculating scores.

The reconciliation checks published aggregates and their authenticated parent links. It does not assert global numerical acceptance, resolve conditional activation or replace the per-stage operational checks. Historical component flags that deny full runtime acceptance remain unchanged.

## 4. Next completion path

1. Retain the approved P08-A12 five-method sensitivity panel. The owner selected Extra Trees despite its 184,239 additional fit slots. Do not reopen this decision, add experiments or build later numerical runtimes as a condition of closing this planning goal.
2. Validate the exact revised public package, publish only reviewed paths to main, verify remote agreement and await that revision's CI. Preserve unrelated work and all private evidence.
3. Request a separate U0 permit: at most 78 registered source fits and 78 source-validation operations, 90 active minutes, 8 GiB new artifacts, 16 GiB process-tree RAM and 8 GiB allocated GPU memory. The reviewed schedule is serial. Reserve 30 GiB free beyond the remaining artifact allowance; recheck actual capacity before launch. No automatic retries, hyperparameter winner selection, scalar calibration, final refit, held prediction, QC routing, perturbation or uncertainty calculation.
4. Only after an approved and reviewed U0 result, consider separate U1 approval. Q1, robustness, range, normalization and population stages retain their own numerical-review, resource and execution gates.

The outcome checklist in Master Plan §27 remains future scientific work. This readiness goal defines that work without running it. Minimal min–max is still the reference, not the established best preprocessing pipeline.
