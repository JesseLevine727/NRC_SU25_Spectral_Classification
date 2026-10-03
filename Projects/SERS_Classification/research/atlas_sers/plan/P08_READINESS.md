# P08 preprocessing readiness: evidence and approved direction

**Date:** 2026-09-30. **Status:** no-fit readiness work in progress; not an execution permit.

P08 tests whether defined preprocessing pipelines change chemical identification on unseen instruments and physical samples. Minimal min–max scaling remains the reference, not an established optimum. This task follows [Master Plan §16](MASTER_PLAN.md#16-sub-plan-p08--preprocessing-policy-factorial-and-robustness) and the [handoff](P08_HANDOFF.md). It does not authorize training, calibration, new predictions, resampling, perturbations or reconstruction of preprocessing arrays.

## 1. Authenticated inputs

The read-only audit checked the frozen artifact bytes, representation contents and split metadata. It did not select a model or inspect a new predictive outcome. The [aggregate audit](../results/p08_readiness/input_support_audit.json) contains hashes and counts without sample or observation identifiers.

| Evidence | Audited scope |
|---|---|
| P00, P01, P02 and P04 planning states | 12, 71, 34 and 9 files respectively; complete states and matching file hashes |
| Existing P01/P02 validators | Both passed, including latest-pointer and upstream bindings |
| P03/P04 reference predictions | Match the accepted content pins; no replacement predictions |
| P05 completed release | Accepted reporting receipt and upstream bindings; all 159 public files match private originals |
| P13 completed aggregation | 16 state files and passing validation report; 11 public files match the release manifest |
| P06/P11 release | All 35 manifest-listed files match; accepted base commit `a08c2b06af9049e0181560e216ad733f04ee1327` |

| Population or evaluation scope | Spectra | Physical masters | Additional support |
|---|---:|---:|---|
| Frozen primary dataset | 598 | 69 | 10 instruments; 3 stations |
| Primary held-instrument evaluation | 557 | 69 | 13 station–instrument domains; 260 contexts |
| Within-station development | 598 | 69 | 60 contexts; a different estimand |

Each held context is one domain, outer repeat and outer fold: 13 × 5 × 4 = 260. The 41 spectra outside primary held evaluation belong to four previously declared exploratory domains; they remain in the dataset. Repeats and contexts do not create additional independent samples.

The eight incomplete historical selected-classical contexts remain missing, leaving 252 paired contexts in that historical comparison. The frozen archive records two CWA and six surfaces contexts with `incomplete_or_missing_reference`. This is not evidence that eight samples are missing. The proposed fixed-family P08 panel is not the historical selected-classical pipeline; its support must be audited separately, not inherited as 252 by convenience.

## 2. What the three preprocessing actions actually do

All three frozen arrays have the same ordered 598 rows, 1,401 float32 channels and 400–1,800 cm⁻¹ wavenumber grid at 1 cm⁻¹ spacing. All rows are finite, have final per-spectrum [0,1] scaling, and pass the saved validity records. The new validator independently reproduces these checks and verifies array, axis and row-order hashes.

| Policy | Frozen operation order |
|---|---|
| `PP-U-MIN` | Linear interpolation → per-spectrum min–max scaling |
| `PP-U-SG` | Linear interpolation → isolated impulse replacement → Savitzky–Golay smoothing → min–max scaling |
| `PP-U-ARPLS` | Linear interpolation → isolated impulse replacement → arPLS baseline correction → min–max scaling |

Impulse replacement uses a centered five-point rolling median, residual MAD scaled by 0.67448975, a threshold multiplier of 10, and a three-point neighborhood. A flagged point is replaced only when that neighborhood contains at most two flagged points. Savitzky–Golay uses window 11, polynomial degree 3 and `interp` boundary mode. arPLS uses λ = 100,000, at most 12 iterations, relative-weight tolerance 0.001 and logistic clipping at ±60. The implementation also retains its frozen early exits for insufficient or negligible negative-residual variation. Min–max scaling uses the frozen float64 epsilon before float32 serialization.

These are complete-pipeline comparisons. SG versus MIN does not isolate smoothing from spike replacement; arPLS versus MIN does not isolate baseline correction from spike replacement. There is no authorized despike-only or combined SG-plus-arPLS action. A changed transform would require a separate versioned deviation and P01 rebuild, outside this goal.

## 3. Support audit and an important routing distinction

The P02 and P04 held-context source/test observation sets agree exactly. Fitting excludes both the held instrument and every test physical master. Source pseudo-instrument selection is available in 128 contexts; the other 132 use the registered master-CV fallback for estimator selection. These contain 285 and 396 inner selection units respectively, totaling 681 for held evaluation.

For **the held instrument's platform family**, all 260 contexts require the minimal fallback: 159 have a known but unsupported family and 101 an unknown source family. There are no supported held-family cells. Thus a supported-family transfer effect is unavailable, not zero and not a failed sensor response. Platform family is an acquisition-system grouping, not a SERS substrate family.

However, this does **not** prove that every family-policy fit can reuse the universal-minimal fit. An independent audit of source-family support finds 202 supported family–context combinations in 182 contexts, alongside 637 unsupported source-family combinations. These counts match the frozen family-support registry. A policy that preprocesses source rows by their own families can therefore change training inputs even when every held row falls back to MIN.

The owner has now clarified the fallback: an unsupported held family uses the **complete minimal pipeline**, including the estimator trained on minimal source inputs. All 260 family-policy contexts therefore map to their minimal baseline, subject to full input/specification and artifact-hash verification. This follows the approved fallback definition, not an inference from held-row processing alone. Mixing family-specific source inputs while keeping held inputs minimal is a different, unapproved experiment. Family-specific transfer remains untestable under the current support; the baseline identity cannot demonstrate that such transfer works or fails.

The QC gate library contains 124 candidates: one minimal-only rule, 15 single-trigger rules and 108 dual-trigger rules. QC policy selection requires supported source pseudo-instruments; the 132 master-CV-only contexts therefore require the registered minimal QC fallback. Numeric thresholds must be calculated inside the relevant source fitting role, never from held rows or validation distributions. The six recorded ingredients for the five allowed features are present in the frozen manifest. No numerical cutpoint or gate has been selected during this audit.

## 4. Owner-approved planning decisions

The owner approved the recommended model mapping, the scoring/uncertainty amendment, the explicit addition of Extra Trees and the complete minimal-pipeline family fallback on 2026-09-30. These approvals concern the design, not permission to launch scientific computation. The separately versioned [readiness contract](contracts/p08_readiness_contract.json) records them without altering historical contracts.

| Decision | Approved P08 treatment | Reason and boundary |
|---|---|---|
| Model mapping | RBF-SVM, Random Forest, matched D0-M and the frozen context-local P05-selected recipe | Explicitly amend historical D0-ERM wording; do not select a fixed D3 from held scores |
| Adaptive policy-development panel | Equal-weight RBF-SVM and D0-M under the approved ordinary-CNN mapping | This changes the older named D0-ERM control and is recorded explicitly |
| Point estimator | Equal-context scores within equally weighted domains; pooled-four-fold domain/repeat scores as sensitivity | P02's pooled-fold wording and the completed benchmark's equal-context estimator are not interchangeable |
| Uncertainty | Documented 10,000-draw support-preserving crossed-weight analysis, with original hierarchy retained as feasibility analysis | Sparse class cells and shared masters/instruments invalidate a silent transfer of the original interval wording |
| Extra Trees | Add to the universal-policy panel and budget separately | Explicit owner approval expands that panel to five methods; adaptive-panel expansion is not implied |
| Unsupported held-family fallback | Use the minimal-trained estimator and minimal test inputs | All 260 current family contexts become baseline aliases after verification; no mixed-source training |

The P05 source-selected recipe is D0-M in 221 held contexts, D1 in 14, D2 in 11 and D3 in 14. The corresponding inherited inner-unit totals are 594, 36, 22 and 29. The proposed P08 comparison freezes these recipe identities across policies. Epoch selection and scalar calibration remain source-only and representation-specific. Historical D0-ERM remains separately named and unchanged.

These planning approvals do not authorize fitting. The detailed statistical amendment must state random streams, endpoint aggregation, contrast families, missing-cell handling, paired dependence and reporting limits before new policy outcomes. M01 predicts individual spectra. M06 averages model probabilities from repeated measurements to form a sample prediction; it does not average spectra before classification. Weighted intervals condition on fitted models and observed support and do not pass the original G4 criterion automatically.

## 5. Preliminary accounting, not a launch ledger

The reviewed, synthetic-tested no-fit counter reproduces the following arithmetic from authenticated context/unit metadata; see the [aggregate accounting report](../results/p08_readiness/universal_accounting.json). It cannot authorize execution or authenticate cached scientific results.

The following arithmetic assumes only the 260 primary held contexts, the approved model mapping, the inherited grids and three registered stochastic seeds. It counts logical jobs before any additional hash-proven deduplication. Existing MIN evidence may be reused only after full specification and artifact matching. No MIN retraining is included in these provisional incremental counts.

| Model | Source-selection fits per new universal action | Calibration cross-fit model fits | Final refits | Total model fits per action |
|---|---:|---:|---:|---:|
| RBF-SVM | 681 × 36 = 24,516 | 260 × 3 = 780 | 260 | 25,556 |
| Random Forest | 681 × 16 × 3 = 32,688 | 260 × 3 × 3 = 2,340 | 260 × 3 = 780 | 35,808 |
| D0-M plus frozen selected recipe | (681 + 87) × 3 = 2,304 | Source logits reused, not new neural fits | (260 + 39) × 3 = 897 | 3,201 |
| Core total | 59,508 | 3,120 | 1,937 | 64,565 |
| Approved Extra Trees addition | 681 × 16 × 3 = 32,688 | 260 × 3 × 3 = 2,340 | 260 × 3 = 780 | 35,808 |
| Five-model universal total | 92,196 | 5,460 | 2,717 | 100,373 |

Before adding Extra Trees, the four-model core contributes at most **129,130 model-fit jobs** across the two new universal actions under these assumptions. Neural scalar temperature calibrations add up to 897 per action and must be counted separately. Classical scalar-calibration operations, evaluation predictions and inference jobs also require explicit entries; the table is not the complete execution budget. No P05 guard-unit retraining is included because the recipe decision is frozen rather than reselected.

The approved Extra Trees addition contributes 35,808 model-fit slots per new action, or **71,616** across SG and arPLS. The five-model incremental literal ceiling is therefore **200,746** model-fit slots. Historical T3 selection plus fixed-family calibration/refit timing records sum to 32,246.712 seconds for Extra Trees; doubling that recorded sum gives approximately **17.9 sequential CPU-hours**, excluding orchestration, I/O and new-policy timing changes. Recorded timing sums are an estimating basis, not a measured P08 wall time or an approved ceiling.

The calibration columns above are literal slots, not all necessarily new fits. The inherited master-CV procedure can reuse selected source-validation predictions for its calibration inputs. If exact roles and selected specifications match, that removes 396 SVM and 1,188 slots for each tree model per policy, reducing the two-action model-fit count to **195,202**. This within-policy prospective cache match is separate from reuse of historical MIN evidence. Classical temperature fitting occurs after technical-seed aggregation: 260 scalar optimizations per classical family per action. Together with 897 neural scalar calibrations, that gives **1,677** scalar calibration operations per action, separate from classifier fitting.

The final ledger must enumerate actual context, role, candidate, seed, policy, refit, calibration and prediction identities; demonstrate every reuse; and bind source-only stopping to the same rules across policies. It must also count the second-stage family/QC procedure's nested policy and estimator selection. Multiplying the universal ledger by 124 is not a valid QC budget without specifying its nested roles and reuse. Robustness and exploratory branches require their own explicit ledgers and remain unlaunched.

## 6. Remaining gates and deliverables

**Subsequent no-fit milestone, 2026-09-30:** the [statistical specification](P08_STATISTICAL_PROTOCOL.md) and its [numerical contract](contracts/p08_statistics_contract.json) now define the endpoints, shared random streams, contrast families, missing-cell treatment and preservation boundary. They specify future calculations; none has been executed. The original input/accounting milestone was published as `0ea4202406eb31280c6626f1ea23c4abf8932561`, and its remote CI passed.

The independent [role-equivalence audit](../results/p08_readiness/role_equivalence_audit.json) reconciles 1,721 distinct role definitions: 681 inner-selection units, 780 calibration folds and 260 final fitting roles. All 396 master-CV calibration folds match source-selection fitting and validation observation hashes exactly. Historical labels differ (`calibration_master_cv` versus `master_cv`); role identity, not label spelling, establishes the match. This supports the prospective within-policy calibration-cache savings in Section 5. It is distinct from authentication of old scientific artifacts.

The [minimal classical evidence audit](../results/p08_readiness/minimal_classical_reuse_audit.json) verifies all 260 fixed-family contexts for each of RBF-SVM, Random Forest and Extra Trees. It checks 74 final-bundle files, exact registered test observations, source-fit and final-refit records, selected candidate hashes, seed coverage and original per-context calibration states. Each model has 2,785 prediction appearances across repeated contexts; these are not 2,785 independent spectra. This audit does not substitute the incomplete historical selected-classical comparator. Neural reuse and the complete job-to-artifact bridge remain separate checks.

The universal no-fit planner now records dependencies for fitting, source predictions, selection, scalar calibration, final predictions and neural strategy aliases. Its execution entry always rejects a launch, including a forged authorization flag. Classical predictions retain seed averaging before a single temperature; neural predictions retain per-seed calibration before averaging. A slot plan is not a runtime, a cache receipt or permission to train.

The [expanded universal ledger audit](../results/p08_readiness/universal_slot_ledger_audit.json) records 607,221 operation slots across all three policies, including the historical MIN slots requiring a separate artifact bridge. For SG and arPLS together it records **195,202 model-fit slots**, **3,354 scalar calibrations**, 184,392 source-validation prediction jobs, 5,376 calibration-validation prediction jobs, 5,434 held-prediction jobs and 2,158 ensemble-prediction jobs. The remaining entries record selection and alias dependencies. These are operations on repeated source splits and hyperparameter candidates, not counts of independent samples. The full identity-bearing graph stays private; its aggregate counts, code/specification hashes and archive digest are public. QC, family routing and later branches are not included in this universal graph.

1. Retain the four approved decisions and finish the numerical amendment without editing frozen historical contracts.
2. Lock endpoint, uncertainty, multiplicity, missing-cell and preservation definitions; state that historical benchmark outcomes have already been examined.
3. Resolve source/held routing for adaptive policies and audit nested roles, including source-only cutpoints, stopping and calibration.
4. Have DeepSeek V4.1 Flash implement deterministic no-fit ledgers, routing/support checks and fail-closed launch guards. Independently test each bounded slice.
5. Freeze finite fit, wall-time, RAM, GPU and storage ceilings and a small scientific smoke proposal. Request a separate execution permit.
6. Publish the reviewed readiness package on `main` and verify remote CI. A planning draft or focused test pass is not completion of this goal.

The [figure manifest](P08_FIGURE_PLAN.csv) specifies future paired spectra, domain scatters, policy-effect intervals, model interactions and quality/routing views. Every numerical figure must have native TikZ, offline HTML, vector PDF and PNG derived from one semantic table, with black standard fonts. Individual-spectrum overlays stay private; only separately reviewed aggregate spectral summaries may be published. No new P08 outcome figure exists yet.

The plan does not promise that preprocessing removes instrument nuisance, restores a clean chemical spectrum or proves substrate independence. It tests predictive effects and spectral preservation on the measurement combinations that were actually observed.

## 7. Neural reuse and nested QC support amendment

The [neural MIN audit](../results/p08_readiness/minimal_neural_reuse_audit.json) authenticates 2,304 relevant source fits and 897 unique final fits, calibrations and held predictions. The audit rehashed 16,416 files or frozen source bindings and reproduced all 897 calibration-input semantic hashes from saved source logits. It checked source/test roles, the exact source-selected recipe map, three-seed coverage and the inherited epoch calculation. The two reported neural strategies cover 520 context–strategy cells and share fits where their specifications coincide. No model, temperature or prediction was recomputed.

The owner approved **P08-A05** after the nested metadata audit. Of the 128 source-pseudo-domain contexts, 74 cannot support independent inner estimator selection: at least one registered policy-validation fitting role has only one physical master per class at its limiting class. The remaining 54 contexts retain at least three masters per class in every such role and can support three-fold, master-separated estimator selection. The 285 source pseudo-domain units have minimum class-master counts of one in 177 units, three in 75 units and four in 33 units. Repeated spectra of a master do not increase those counts.

Thus QC adaptation has **54 eligible contexts and 206 complete MIN fallbacks**: 74 for insufficient nested class support plus the original 132 lacking source pseudo-domains. All 260 remain in operational results; the 54-context subset is reported separately. This approved support refinement supersedes the initial 128-context adaptive-subset wording, not the historical source-support counts. The universal SG/arPLS panel and complete held-family fallback are unchanged. Actual nested folds, quantile-fitting roles, per-gate model jobs and resource costs remain to be enumerated before execution.

The [nested support audit](../results/p08_readiness/qc_nested_support_audit.json) locates all 54 eligible contexts at the CWA station. Its other six contexts use the original pseudo-domain fallback. All 100 pills contexts lack source pseudo-domain support. At surfaces, 74 contexts fail nested class support and 26 lack pseudo-domain support. Therefore the adaptive subset estimates transfer only within the supported CWA contexts; it cannot identify adaptive-policy effects at the other two stations. The operational result must retain those stations' fallbacks rather than present the supported subset as dataset-wide evidence.

The [classical source-artifact bridge](../results/p08_readiness/minimal_classical_source_bridge_audit.json) additionally checks 89,892 completed source fits and their exact stored validation observation sets across 131 shards and 524 rehashed files. It complements the earlier fixed-family endpoint audit; a repeated prediction appearance is not a new sample. The final runtime reuse ledger must still bind the calibration, selection and endpoint operations to these records explicitly.

Historical classical prediction files support MIN comparisons and complete-pipeline result aliases, but the P03 run did not persist fitted estimator objects. Its serialized-size field measured in-memory object size. Therefore later test-time perturbation cannot use those predictions to evaluate altered spectra. A separately budgeted, source-only reconstruction of the exact selected estimator would be required, with parity checked against the original MIN predictions and original results preserved. No such reconstruction is authorized here. The future nonminimal pipeline should retain its selected estimators so the later robustness stage can avoid unnecessary refitting.

## 8. Concrete nesting and staged resource proposal

The [QC nesting protocol](P08_QC_NESTING.md) now specifies the three inner master folds, fold-local quality thresholds, gate-ranking panel, final estimator selection, missing-result rules and distinction between supported adaptation and complete MIN fallback. Its [role audit](../results/p08_readiness/qc_nested_role_audit.json) passed for 108 policy-validation units and 324 inner estimator folds. Every fold retains at least two fitting masters and one validation master per class. Original P02/P04 roles are unchanged; these are additional P08 roles inside their source fitting sets. No threshold, gate, fitted model or score was calculated.

The literal full QC ceiling is 1,630,980 model fits and 53,880 scalar calibrations before exact routed-input reuse. This includes 161,316 neural fits. The [resource proposal](P08_RESOURCE_PROPOSAL.md) therefore separates the universal comparison from the adaptive sweep. A full-retention adaptive run is not storage-feasible on the observed free disk: its neural checkpoints alone are estimated at 223.33 GiB. This is a prospective accounting constraint, not a failed experiment or evidence that QC preprocessing is ineffective.

The first proposed scientific smoke contains 78 exact source-fit slots and 78 source-validation prediction slots from the existing universal graph. The [smoke audit](../results/p08_readiness/universal_smoke_proposal_audit.json) fixes the station/recipe coverage from metadata before outcomes. It covers SG and arPLS, the three classical families and D0-M/D1/D2/D3 using the original seeds and training rules. It includes no held predictions, scalar calibration, new policy selection or QC cutpoints. Its proposed 90-minute, 8-GiB artifact and 8-GiB allocated-GPU limits need separate approval after runtime review.

The full readiness goal remains incomplete. Complete runtime reuse bindings, the adaptive operation catalog, later-branch numerical budgets, admission/restart review and final release are still required. The preceding milestone `f17f1f3e96eaa1bb988077dc0978305dd5a60738` passed remote CI run `36750480085`; that result does not validate later uncommitted work.

The [classical outer-bundle audit](../results/p08_readiness/minimal_classical_outer_audit.json) now verifies 780 model–context cells, 1,820 final-fit records, 5,460 calibration-model records and 780 scalar states against 3,979 file hashes. Calibration-model records include inherited cache aliases; they are not all newly executed fits. In 384 cells, fresh calibration predictions retain individual seeds; in 396 cached cells, the outer bundle stores the seed average and the original source shards retain seed-level evidence. Held predictions are stored after seed aggregation. Complete MIN endpoint reuse is supported, but missing per-seed held arrays and fitted estimator objects cannot be fabricated or declared independently reusable.

## 9. Complete MIN operation bindings and fallback endpoints

The [reuse protocol](P08_REUSE_PROTOCOL.md) now distinguishes recorded fit completion, reusable saved predictions and unavailable intermediate objects. Its [read-only audit](../results/p08_readiness/minimal_operation_reuse_audit.json) binds all 202,407 MIN planning slots after 20,934 file checks, including a fresh byte/array hash verification of the immutable MIN input. There are 1,079 distinct complete model–context endpoints. In particular, 1,560 tree seed-level held slots are represented only by the saved ensemble; their individual arrays are not claimed to exist.

The same audit binds all 1,040 family-policy and 824 unsupported-QC endpoint aliases to the complete MIN pipeline. Those aliases require no new fitting or prediction. This completes the historical evidence mapping, not a future executor or blanket cache authorization. The adaptive operation catalog, reviewed admission/routing controls, later-branch choices and budgets, and final execution request remain open.

Three later-branch choices have been submitted to the owner: whether normalization controls repeat the full source-only classical selector, whether regenerated population tiers repeat source-only model selection before freezing their neural recipes, and whether robustness corruption precedes preprocessing or follows it. Those choices affect the scientific question and compute ledger. They do not alter the approved universal SG/arPLS comparison or authorize any scientific operation while pending.

## 10. Audited QC policy-selection subgraph

The [policy-subgraph audit](../results/p08_readiness/qc_policy_subgraph_audit.json) binds 3,456 compact operation blocks for the 54 eligible contexts, 108 policy-validation units and 324 nested estimator folds. The blocks represent 1,620,432 prospective model fits and 53,568 scalar calibrations for gate development. These are planning slots, not executed experiments or an approved compute budget. The full QC ceiling remains 1,630,980 model fits and 53,880 scalar calibrations after final estimator development is included.

The reviewed implementation records inner-fold neural stopping roles, same-gate selection dependencies, per-seed calibration and equal SVM/D0-M gate-ranking weights. Independent checks found no held-test role in this subgraph and no policy-validation role upstream of inner model selection or calibration. It computes no threshold, route, prediction, score or winner. The final-estimator subgraph, complete fallback assembly and runtime admission controls remain unfinished; the complete adaptive catalog is not yet locked.

## 11. Figure delivery boundary

The [P08 figure protocol](P08_FIGURE_PROTOCOL.md) makes the requested black standard-LaTeX/Times-style typography explicit and separates public spectral aggregates from private individual examples. Public spectral curves average within physical masters and then across masters on matched action membership; cells containing only one master are unavailable for that display, not excluded from scientific analyses. The renderer must preserve the shared semantic table across native TikZ, offline HTML, vector PDF and PNG. No figure or new numerical summary has been produced at this readiness stage.

## 12. Audited final-estimator subgraph

The [final-subgraph audit](../results/p08_readiness/qc_final_subgraph_audit.json) now covers the additional 10,548 model fits and 312 scalar calibrations after gate selection in the 54 supported contexts. Together with gate development, the dependency graph represents 1,630,980 prospective model fits and 53,880 scalar calibrations. These counts confirm the existing literal ceiling; they are neither completed experiments nor an execution permit.

The supervisor checked 1,784 exact unit-role bindings and 1,474 protected-stage ancestor chains. Final fitting uses source-only routing; held-test routing depends on every final refit. No held-test role appeared upstream of fitting, selection or calibration. The 176 distinct final endpoints share D0-M where the source-selected strategy has the same recipe. Extra Trees remains outside the adaptive panel.

This internal constructor requires validated inputs. The whole-corpus input validator, complete fallback assembly and runtime admission controls are separate acceptance gates. No QC threshold, routing choice, model, prediction or score has been computed.

## 13. Whole-corpus catalog and fallback evidence

The [complete catalog audit](../results/p08_readiness/qc_complete_catalog_audit.json) now verifies 260 contexts and 1,040 model–context aliases. The 54 supported contexts contribute 216 future aliases; the 206 unsupported contexts contribute 824 complete MIN aliases and no new operation blocks. Every fallback matches the previously authenticated recipe, fitting/test roles, MIN input, model specification and support reason. No new fitting or prediction is needed for these aliases.

The assembler rebuilds the nested role registry from raw membership metadata and checks exact alignment with the compact universal registry, including unsupported contexts. It then seals the combined policy/final graph and fallback aliases. The independent audit checked 5,876 ancestor chains without finding a held-test role upstream of source operations. Calibration-role memberships retain their separately authenticated universal provenance; different hash strings alone do not establish sample independence.

This closes the metadata-construction and evidence-matching portion of the adaptive catalog. Synthetic rejection tests, full regression and publication acceptance are recorded in the [review](delegation/P08_REVIEW.md). Numerical QC routing, admission/restart controls, later-branch decisions and budgets, and a separate execution request remain outside this metadata result. No scientific run is authorized by the catalog.

## 14. Synthetic numerical QC controls

The [numerical implementation specification](P08_QC_NESTING.md#8-numerical-implementation-and-evidence-boundaries) separates threshold estimation from single-row routing. Source-only membership is checked through the audited role registry; a threshold-state hash alone is insufficient. The router uses the unchanged P02 gate library and cannot inspect labels, instrument identity or a target-batch distribution.

The current implementation review exercises hand-calculated quantiles, strict trigger boundaries, fixed priority rules, unavailable-input fallbacks and malformed-state rejection using invented data. Results and acceptance gates are recorded in the [review](delegation/P08_REVIEW.md). No real-data threshold, route, model, temperature, prediction or new classification score has been calculated. Admission/restart controls, later-branch decisions and budgets, the final readiness release and the separate execution request remain open.

## 15. Resource and restart accounting boundary

The [resource proposal](P08_RESOURCE_PROPOSAL.md#7-resource-snapshots-and-cumulative-attempt-accounting) now specifies exact budget boundaries and cumulative attempt semantics. A resource snapshot is a check of supplied measurements, not a reservation. A replayed journal is a check of recorded events, not proof that its storage survived a crash. Both components retain unconditional scientific-execution denial.

Synthetic review covers exhausted budgets, separate GPU memory readings, failed-attempt consumption, clean stop/reopen accounting and unresolved open sessions. The [review log](delegation/P08_REVIEW.md) distinguishes accepted checks from remaining integration work. Exact U0 job admission, durable journal/head handling, exclusive ownership and interruption/restart verification remain open; no full launch-safety claim follows from these pure components.

The [smoke-attempt mapping audit](../results/p08_readiness/smoke_attempt_manifest_audit.json) authenticates all 156 proposed operations and their 78 exact fitting/prediction pairs. The private attempt manifest contains 42 CPU fits and 36 GPU fits, each with its corresponding source-validation prediction. This closes metadata mapping, not live job admission: no execution event or new scientific artifact was generated.

The [exact candidate-check specification](P08_RESOURCE_PROPOSAL.md#8-exact-smoke-candidate-checks) joins that fixed mapping to journal-derived counters and prospective worker capacity. It also requires review after a failed or interrupted U0 attempt. Implementation and test acceptance are recorded separately in the review log. Passing a pure candidate check is not a reservation, proof of exclusive ownership or scientific permission; the live admission/restart gate remains open.

The [local persistence specification](P08_RESOURCE_PROPOSAL.md#9-local-journal-persistence-and-incomplete-sessions) adds exclusive controller ownership, event/head ordering and explicit refusal of incomplete sessions. Functional, filesystem-fault and cross-process tests have distinct review gates. Their acceptance does not establish physical power-loss durability, fresh resource measurements or receipt authenticity, and does not authorize a scientific smoke.

## 16. Later-branch numerical design audit

The [later-branch design note](P08_LATER_BRANCH_DESIGN.md) records two constraints before exploratory experiments are locked. Final min–max scaling would erase the distinction between the frozen SNV, vector and area controls for valid nonconstant rows. Rigid shifts can require values outside the registered input support; no padding or extrapolation rule has been approved. Synthetic tests check the arithmetic without transforming scientific data.

The [full-selector bound](../results/p08_readiness/later_classical_selector_bounds.json) authenticates nine classical families, 126 candidates and 190 candidate–seed pairs across 681 source-selection units. Repeating that selector for one new representation has a conditional ceiling of 131,322 model fits after authenticated within-representation calibration reuse. Applying it to three normalization controls and the derivative control would total 525,288 model fits. These are prospective limits for an unresolved design choice, not executed fits or an approved budget. The universal comparison and approved adaptive fallback are unchanged.

The [conditional N1 resource proposal](P08_RESOURCE_PROPOSAL.md#conditional-n1-proposal-full-selector-normalization-controls) now adds finite wall-time, storage, RAM and worker ceilings for that full-selector interpretation. Its timing basis counts historical failed candidates as well as completed ones. The owner choice and exact later-branch operation ledger remain unresolved; no N1 resource or execution approval is inferred.

Owner decisions on normalization selection, regenerated-population selection and perturbation placement remain pending. The note also identifies the unapproved shift-edge rule and separate range/population accounting requirements. Live runtime integration, finite later-branch resources, final readiness review and a separate scientific execution request remain open.

## 17. Completion-evidence boundary

The [terminal-receipt specification](P08_RESOURCE_PROPOSAL.md#10-terminal-receipt-and-artifact-byte-verification) requires a proposed completion event to match the running attempt and actual saved bytes. A journal entry containing a hash does not prove that its referenced file exists. The read-only check separates byte integrity from checkpoint loadability and source-prediction validity; it cannot append a completion event or grant execution. Synthetic implementation acceptance and the remaining runtime obligations are recorded in the [review log](delegation/P08_REVIEW.md).

## 18. Proposed serial smoke integration

The [serial resource specification](P08_RESOURCE_PROPOSAL.md#11-serial-smoke-measurements) connects the exact candidate guard to process, filesystem, numerical-thread and CUDA measurements. The proposed smoke executes its existing 156 operations sequentially in one Python process. This stays within the existing worker ceilings; it neither changes the scientific comparison nor validates a later parallel benchmark.

Synthetic validation uses invented process and CUDA readings. It does not run the 78 model fits, inspect scientific arrays or initialize a GPU. The adapter's freshness check establishes a bounded observation interval, not a reservation or an execution permit. Durable progress, stage-specific artifact semantics, complete runtime integration, unresolved later-branch choices and the separate scientific approval remain outstanding.

## 19. Audited wider-range input and operation ledger

The [input audit](../results/p08_readiness/range_input_audit.json) authenticates the existing **598 × 1,450** representation over **400–1,849 cm⁻¹**. All rows remain valid and keep their original order. The [operation audit](../results/p08_readiness/range_ledger_audit.json) binds the same **260 contexts**, with **62,981 prospective model fits**, **1,417 scalar calibrations** and **131,199 total operations**. It verifies every retained role, rekeyed dependency and **520 neural strategy aliases** independently. The identity-bearing graph remains private.

This range sensitivity retains four methods without Extra Trees. It changes the complete normalized input; it cannot attribute an effect solely to the additional channels. Synthetic architecture checks and implementation acceptance are recorded in the [review](delegation/P08_REVIEW.md). Existing primary-width guards are unchanged. No range fitting, calibration, prediction, scoring or resampling has occurred.

The [R1 resource proposal](P08_RESOURCE_PROPOSAL.md#12-r1-proposal-frozen-wider-range-sensitivity) is separate and unapproved. Range inference and executor acceptance remain open. At the range-audit storage observation, every proposed scientific stage was infeasible under its initial allowance and reserve. The next permitted work remains implementation review, remaining numerical specifications and a final readiness audit—not training or unapproved cleanup.

## 20. Normalization-selector decision now has both cost bounds

The [fixed-family alternative audit](../results/p08_readiness/fixed_family_alternative_audit.json) verifies **260 complete MIN source-family choices**, but only **252 contexts with historical final predictions**. These are different completion stages. No missing endpoint was reconstructed, scored or imputed. The source map therefore permits a precise fixed-family cost proposal without repairing the old comparison.

For the four exploratory controls, full source-only family reselection has a ceiling of **525,288 model fits**; retaining the MIN-selected families and retuning their hyperparameters has a ceiling of **42,368**. Both allow **1,040 scalar temperatures**. They answer different questions, as defined in the [later-branch note](P08_LATER_BRANCH_DESIGN.md#2-a-full-classical-selector-is-a-substantial-experiment). The [N2 resource proposal](P08_RESOURCE_PROPOSAL.md#13-conditional-n2-proposal-frozen-min-selected-families) is conditional on the narrower choice and cannot be launched alongside N1 by default.

The owner decision remains pending. The new evidence justifies asking for that decision with both costs, not silently adopting the cheaper option. Source records and historical elapsed times were read; no control array, model, calibration, prediction or score was calculated. The other unresolved population and perturbation choices, range inference and runtime gates remain open.

A subsequent [capacity check](P08_RESOURCE_PROPOSAL.md#13-conditional-n2-proposal-frozen-min-selected-families) clears the proposed initial storage thresholds for U0 and conditional N2 only. It supersedes the earlier low-space checkpoint as a point-in-time observation, not as a reservation or authority. The universal full-comparison allowance remains unsupported by that observed capacity.
