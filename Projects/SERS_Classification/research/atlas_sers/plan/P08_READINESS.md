# P08 preprocessing readiness: evidence and approved direction

**Date:** 2026-09-30. **Status:** no-fit readiness work in progress; not an execution permit.

**Current accounting, 2026-10-07:** [statistical-operation metadata](P08_PERTURBATION_INFERENCE.md#8-statistical-operation-metadata-acceptance) now covers uncertainty, the original hierarchy, sign/Holm and stability with 495,780 planned descriptors. [Reporting accounting](P08_REPORTING_ACCOUNTING.md) separates diagnostic reuse from future numerical summaries and native figure delivery. No scientific score, draw or figure has been produced. Final cross-ledger and requirement-wide reconciliation, the separate filtered-population panel clarification and release checks remain before a distinct execution request. Earlier dated checkpoints retain their historical scope.

**Owner update, 2026-10-04:** [P08-A06–A09](P08_LATER_BRANCH_DECISIONS.md) resolve the four later-branch choices and resume the original readiness goal. N2 normalization scope, fresh source-only population-tier selection, pre-pipeline disturbances and constant shift-edge extension are approved as planning decisions. Historical pending-choice checkpoints below are superseded; numerical branch specifications and exact ledgers remain incomplete. No resource ceiling or scientific operation is approved.

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

## 21. Range-specific numerical inference specification

The [range protocol](P08_RANGE_INFERENCE.md) and [registry](contracts/p08_range_inference.json) now specify **eight range effects** and **eight range–model interactions** for the four-method panel and two endpoints. These secondary families remain separate from the universal and QC families. The registry binds the unchanged primary statistical protocol and both public range audits by SHA-256.

The range procedure retains equal-context/equal-domain scoring, shared **10,000-draw** master/instrument weighting, the original hierarchy as a feasibility check and separate Holm adjustment for each range family and sign sensitivity. Missing complete context cells make a full-support contrast unavailable; a separately labelled paired-support sensitivity cannot replace it. No failed range cell receives MIN fallback. Wider-input min–max scaling and pooling prevent a claim that the comparison isolates the added channels.

This closes the range-specific numerical-definition gap identified in the earlier checkpoints, not the inference-implementation or runtime gate. R1 resources remain proposed, and no fitting, prediction or uncertainty analysis has been launched. The four pending normalization/population/perturbation decisions and remaining later-branch job specifications are unchanged.

## 22. Combined component checks, not a scientific runner

The [combined no-fit specification](P08_RESOURCE_PROPOSAL.md#combined-no-fit-validation-boundary) now has synthetic tests across the actual store, journal, serial resource/admission and receipt components. They cover CPU and fake-GPU bookkeeping, dependency ordering, durable progress, corrupt artifact refusal, clean reopen, persistent failure/interruption accounting and refusal of incomplete sessions. All execution entry points still deny scientific work.

The [review](delegation/P08_REVIEW.md) records **11 integration tests** and **13 range-registry tests**, with exact schema corrections reviewed before acceptance. These tests do not establish measured cumulative artifact growth, live permit enforcement, checkpoint loadability or valid source predictions. The next runtime work must enforce the tested ordering and stage-specific semantic checks; no complete launch-safety claim follows from the test helper. Production scientific code, primary inference definitions and historical artifacts are unchanged.

## 23. Neural checkpoint-content checks

The [checkpoint-content specification](P08_RESOURCE_PROPOSAL.md#neural-checkpoint-content-requirement) now has a reviewed implementation using only synthetic CPU checkpoints. It authenticates the supplied bytes before restricted deserialization, then checks the inherited architecture, finite float32 tensors and tensor-state digest. The supervisor verified **107 passing tests**, including both supported class counts, all four recipe interfaces, malformed contents and interruption handling. The complete regression and publication gate is recorded in the [review](delegation/P08_REVIEW.md).

This closes the checkpoint-content component, not scientific source-fit acceptance. A matching architecture does not establish the chemical-label order, source role, seed, stopping history or completed training. Recipes with the same architecture remain indistinguishable by tensor layout. Source-prediction contents and checkpoint agreement, authenticated job binding, live resource/permit enforcement and reviewed recovery remain separate requirements. The byte ceiling is not a hard deserialization-memory bound or a hostile-file sandbox. No scientific checkpoint, spectrum or prediction was read by these tests, and every scientific execution allowance remains zero.

## 24. Source-prediction structure and declared job pairing

The [source-prediction specification](P08_RESOURCE_PROPOSAL.md#source-prediction-content-and-job-binding) now has a shared in-memory checker for classical score tables and neural logits archives. It checks the complete declared source-fit/prediction pair against independently supplied job identifiers, then verifies validation-row order, class-column order and finite float64 scores. It preserves the existing formats and accepts finite collapsed models. The supervisor verified **96 passing synthetic tests** and independent mutation/pair-mismatch probes; the [review](delegation/P08_REVIEW.md) records the correction and regression gates.

The checker reports hashes and dimensions from private snapshots of the validated inputs. It does not authenticate the file loader, prove membership in the audited operation graph, establish physical-sample independence or show that the model generated those scores. Those flags remain false. A separate read-only bounds audit authenticated all **156 proposed job records** and the **598-row manifest** without reading spectra or outcomes. No scientific prediction, metric, threshold or model was computed. Source-fit completion/history, checkpoint agreement, artifact-loader/controller integration, unresolved later branches and the separate execution permit remain open.

## 25. Neural completion-record consistency

The [completion-record requirement](P08_RESOURCE_PROPOSAL.md#neural-completion-record-requirement) now has an in-memory checker for the inherited three-class development kernel. It compares epoch/update counts, checkpoint-selection and stopping decisions, selected metrics, gradient counters and trace/state digests. It retains finite collapsed or zero-gradient outcomes. Its **136 synthetic tests** passed; independent supervisor checks covered **40 recipe/stopping-history cases**, **six negative cases** and **16 gradient-counter boundaries**. The [review](delegation/P08_REVIEW.md#neural-training-record-consistency-review) distinguishes the corrections, focused checks and full-regression gate.

Synthetic composition tests connect the existing result-summary projection, checkpoint-content checks and exact source-fit/prediction pair for all four recipes. They do not run a model forward pass. The record's state digests agree with independently checked invented checkpoints, but invented finite scores can still pass the structural prediction check. Consequently, record consistency is not training authenticity or checkpoint–prediction parity. Those flags remain false, as do live-resource acceptance and execution authority. Artifact loaders, runtime integration, unresolved later-branch choices and the separate scientific permit remain open.

## 26. Combined neural artifact consistency

The [combined artifact requirement](P08_RESOURCE_PROPOSAL.md#combined-neural-source-artifact-requirement) now has a reviewed in-memory implementation. It connects the declared source jobs, projected training record, best/terminal checkpoint bytes and inherited validation-logits archive. **71 synthetic tests** pass, including the actual writer format, all four recipes under both new policies, corrupt archives and the maximum permitted metadata dimensions. Independent checks cover all **eight recipe–policy combinations** and confirm that an oversized array header is rejected before array allocation.

This component checks consistency against supplied hashes; it does not authenticate the controller's source of those hashes or prove that the checkpoint generated the saved scores. External registry membership, physical-role isolation, training completion, prediction parity, live-resource acceptance and execution authority remain false. The [review](delegation/P08_REVIEW.md#combined-neural-source-artifact-review) records implementation corrections and the separate full-regression gate. No scientific fit or prediction was executed.

## 27. Frozen normalization/control inputs verified

The [normalization-input audit](../results/p08_readiness/normalization_input_audit.json) authenticates the stored SNV, vector, area and first-derivative controls. Each contains **598 spectra × 1,401 channels** on the unchanged **400–1,800 cm⁻¹** grid, with the primary row order and zero invalid rows. All **2,392 recorded validity rows** agree with the registry, and independent normalization-invariant checks pass. The files retained their hashes before and after the read-only audit.

This closes an input-authentication gap without computing a new representation, predictive score, QC threshold or route. It does not establish signal preservation or an optimal preprocessing method. The [later-branch note](P08_LATER_BRANCH_DESIGN.md#1-normalization-controls-must-remain-distinct) preserves each control's native scale; the selector choice and other previously submitted decisions remain pending. The approved universal comparison, **54** adaptive contexts and **206** complete MIN fallbacks are unchanged. Final runtime integration and a separate scientific execution request remain outstanding.

## 28. Inherited kernel compatibility and remaining completion gates

The [source-stage mapping](P08_SOURCE_STAGE_MAPPING.md) now distinguishes logical job accounting from internal model calls. Both inherited fitting kernels calculate source-validation outputs before returning; the dependent prediction stage authenticates and verifies those outputs without a second optimization. This preserves the proposed **78 fits** and **78 source-prediction jobs**, original dependencies and zero scientific execution authority. An unavailable in-memory classical estimator cannot be replaced by an unrecorded refit.

DeepSeek's [integration tests](../tests/test_p08_source_kernel_integration.py) passed **19 cases** using four actual neural recipe fits and three classical fits on invented CPU data. They connect existing kernels, writers, artifact checks and numerical verification. Finite tampered logits pass structural checks but fail restored-best equality, establishing why both checks are required. Independent probes confirmed refusal of **four** forbidden fitting/GPU-initialization calls even when their exceptions were swallowed, and verified CPU random-state restoration.

These are synthetic compatibility results, not NATO-data experiments or evidence that SG/arPLS performs better. Live permit/registry binding, complete controller ordering, cumulative accounting and reviewed recovery remain open. The [requirement-to-evidence checkpoint](P08_COMPLETION_AUDIT.md) maps the original goal, pending owner choices and remaining ledgers without replacing the goal with this component milestone. Full-regression and publication evidence is recorded in the [review](delegation/P08_REVIEW.md#inherited-source-kernel-integration-review).

## 29. Fixed U0 jobs bound to recorded source roles

The [source-binding audit](../results/p08_readiness/u0_source_binding_audit.json) links all **156 proposed jobs** to the fixed attempt manifest and recorded physical-sample roles. The **78 fit–prediction pairs** use **five source units in five contexts**, with **42 classical fits** and **36 neural fits** proposed. These remain unexecuted scientific slots. Four units use pseudo-instrument validation; one uses physical-master cross-validation. Every source training and validation role contains the same **three chemicals**.

The reader authenticates all five metadata buffers before parsing, checks exact job pairs and source-set membership, and enforces physical-master and held-instrument exclusions. Returned observations use canonical UID order. An independent reconstruction matched every source row, metadata field and full job record; the original files remained unchanged. Only aggregate counts and hashes appear in the public report.

This closes the fixed job-to-role metadata connection described in the [stage mapping](P08_SOURCE_STAGE_MAPPING.md#5-fixed-graph-and-source-role-binding). It does not load numerical arrays, fit models, verify predictions or authorize execution. Live permit enforcement, array/specification loading, cumulative accounting, stage dispatch and reviewed recovery remain unfinished. The owner-approved **54/206** adaptive/fallback split, outstanding later-branch decisions and original readiness goal are unchanged.

## 30. Source-array preparation and fixed augmentation reference

The [source-array audit](../results/p08_readiness/u0_source_array_audit.json) verifies the exact stored MIN, SG and arPLS inputs and selects only the authenticated source rows needed by U0. Each archive retains **598 × 1,401** values on the **400–1,800 cm⁻¹** grid. The **78 proposed fit–prediction pairs** share **ten prepared policy/source-unit inputs**. Independent read-only checks matched every returned float32 row and fitting-row QC value to the original files, which retained their hashes.

The [stage mapping](P08_SOURCE_STAGE_MAPPING.md#6-fixed-source-arrays-and-inherited-noise-reference) makes the inherited noise reference explicit. Neural augmentation uses the same native-spectrum QC values for a given fitting role across preprocessing policies. Noise is not re-estimated from SG or arPLS residuals. This preserves the existing training recipe; it does not optimize augmentation separately for each pipeline. Its relative strength after preprocessing is a limitation to retain when interpreting future results.

No preprocessing, noise quantile, augmentation, model fit, prediction or score was computed during this input check. The public report contains only counts, hashes and verification flags. It does not establish a complete controller or grant an execution permit. Model-specification loading, live resource/permit enforcement, stage dispatch, reviewed recovery and the four pending later-branch choices remain open. The **54** adaptive contexts and **206** complete MIN fallbacks are unchanged.

## 31. Frozen kernel inputs and substrate-aware pair weighting

The [runtime-input audit](../results/p08_readiness/u0_runtime_input_audit.json) connects all **78 proposed fit–prediction pairs** to frozen model settings and the recorded source substrate metadata. Its **42 classical pairs** use the three predeclared smoke candidates; its **36 neural pairs** retain their registered recipes and **30–200-epoch**, **20-epoch-patience** stopping rule. The inherited inner guards remain **120 seconds** and **4 GiB**. These are configuration checks, not new fits or resource approvals.

The [stage mapping](P08_SOURCE_STAGE_MAPPING.md#7-frozen-model-settings-and-recorded-substrate-metadata) now records the exact `sensor_family`-to-`substrate` mapping. This matters because the inherited contrastive objective uses substrate differences when weighting positive pairs. The adapter preserves the recorded strings rather than replacing them with a generic unknown family. Independent comparisons matched all kernel arguments to the original source arrays, parameters and metadata; every inspected input file remained unchanged.

The supplied bytes of **18 inherited source files** are authenticated, but the adapter does not certify already loaded runtime code. No estimator construction, sampling, augmentation, training or prediction occurred. Live implementation identity, permit/resource enforcement, stage dispatch and reviewed recovery remain required. The **54/206** adaptive/fallback split, four pending later-branch choices and zero scientific execution allowances are unchanged.

## 32. Source-stage numerical bridge

The [backend](../src/atlas_sers/evaluation/p08_u0_stage_backend.py) now connects prepared kernel arguments to the existing fitting routines, artifact formats and numerical output checks. A fit is invoked once. Its complete result must pass structural checks before success; its dependent prediction check authenticates the saved bytes and requires exact agreement with the fitted model. Classical verification retains the fitted estimator rather than refitting if that object is lost.

The [stage mapping](P08_SOURCE_STAGE_MAPPING.md#8-numerical-backend-and-saved-output-verification) distinguishes fit acceptance from prediction acceptance. Serialization preserves the inherited result identity, including failed diagnostics. Writable neural input copies prevent tensor aliasing of the immutable prepared arrays without changing their values. Numerical verification restores the CPU random state and returns only aggregate verification evidence.

An independent check fitted the three classical methods and four neural recipes on invented CPU data. All seven saved outputs matched their models, input arrays remained unchanged, and warnings were treated as errors. No field-trial data were fitted or scored. These checks do not measure preprocessing benefit or validate cross-device numerical equality.

The backend is not a complete run controller. Loaded-code identity, live permit enforcement, durable artifact writes, cumulative resources, stage ordering and interrupted-session handling still require integrated review. All proposed resource limits remain unapproved. The original goal, pending later-branch choices and **54/206** adaptive/fallback rule are unchanged.

## 33. Session accounting and recovery specification

The [session requirements](P08_SOURCE_STAGE_MAPPING.md#9-integrated-session-requirements) define the remaining orchestration boundary. Fit artifacts must be written, read back and verified before fit success; the dependent prediction must authenticate those saved bytes and match the retained model before prediction success. No replacement fit is permitted.

Resource accounting includes journal files and pending writes, not only checkpoints. Conservative charges must remain distinct from physically written bytes and cannot weaken free-space checks. The protocol also distinguishes measured finalization time from the preceding durable timestamp. The initial core must refuse previous-session entry pending review, preserving recorded counters and unfinished evidence.

The bounded controller-authoring task returned no patch before its **600-second** deadline and was terminated. No controller implementation was accepted, no scientific process ran, and the published backend is unchanged. The next implementation assignment requires revised packaging of the same integration requirements, not a replay of the failed request or a reduction in the original goal. Scientific permissions and unresolved owner choices remain unchanged.

## 34. Internal session composition

The [internal session core](P08_SOURCE_STAGE_MAPPING.md#10-internal-session-integration-evidence) now connects the reviewed inputs, backend, durable journal and saved-artifact checks. An invented-data RBF-SVM pair completed one fit and its dependent verification without refitting. Actual temporary files and receipts were checked; resource observations were simulated. This is integration evidence, not a new field-trial result or a preprocessing comparison.

Independent fault diagnostics checked interrupted and unfinished attempts, altered bytes, partial writes, resource-measurement freshness, storage limits and previous-session refusal. A fit cannot be treated as safely closed while its prediction still depends on the retained estimator. Durable counters remain distinct from measured file sizes and the post-close finalization interval. Review and fixed-package validation are recorded in the [implementation review](delegation/P08_REVIEW.md#internal-source-session-integration-review).

Loaded-code authentication, an independently approved permit, later-branch decisions and final goal-wide acceptance remain open. The **54** adaptive contexts, **206** complete MIN fallbacks and proposed budgets are unchanged. No real-data fitting, new prediction, resampling, QC routing or preprocessing rebuild is authorized.

## 35. Fresh-process import evidence

The [import audit](P08_SOURCE_STAGE_MAPPING.md#11-fresh-process-project-source-import-check) authenticates **174** project source files before importing the fixed session/kernel entry modules. The actual-source check loaded **41** project modules from those exact captured bytes, with CUDA uninitialized. It rejected preloaded project modules and bypassed project bytecode caches; external dependencies and the interpreter remain trusted rather than authenticated by this check.

The observed run did not read private scientific inputs, write files, fit models or produce predictions. This is code-identity evidence for one isolated audit process, not permission to execute a later scientific process. The future entry point must combine the same-process import guard with independent permit binding and existing input/session checks. Later-branch decisions and final goal-wide review remain open. The **54/206** adaptive/fallback rule, scientific zero-execution limits and proposed budgets are unchanged.

## 36. Retained imports and the remaining launch boundary

The [owned import scope](P08_SOURCE_STAGE_MAPPING.md#12-retained-import-scope-and-outer-composition) retains authenticated source loading for an entire controller block, rather than only startup inspection. An actual-source diagnostic verified captured-byte access without rereading files and refused use after closure. This extends code-identity evidence, not scientific authority.

The [outer launch specification](P08_U0_LAUNCH_BOUNDARY.md) now records the full remaining composition. A separate permit must precede private-input reads and mutations; failed setup must retain its launch attempt; pre-session time and outer control-file bytes must enter cumulative accounting. The existing session cannot infer those external costs. No permit-bound launcher or automatic recovery is accepted by this specification. The original scientific scope, pending decisions and zero-execution limits remain unchanged.

## 37. Outer-entry integration evidence

The [default-denied launcher](P08_SOURCE_STAGE_MAPPING.md#13-default-denied-outer-entry-point) now composes the owned importer, fixed input adapters, existing session and outer accounting. Its permit pin remains unset. Tests on invented data completed one Random Forest fit and dependent saved-output verification, preserved an intentional interruption, and refused reuse of the same destination in a new process.

The eight entry-point tests passed in **7.92 seconds**, with warnings treated as errors in the parent test process. Resource and CUDA observations were fixtures; no GPU kernel or real SERS training occurred. Separate accounting and fault tests cover control-file charges, setup time, altered files, bounded inventories, resource limits and exception preservation. The [review](delegation/P08_REVIEW.md#outer-launcher-and-accounting-review) records corrections and release checks.

This evidence does not complete the original readiness goal. The **54** adaptive contexts and **206** complete MIN fallbacks remain fixed, all proposed budgets remain unapproved, and later-branch decisions remain open. No new chemical-identification result, preprocessing winner, held prediction, QC route or resampling result is reported here.

## 38. Approved later branches and exact normalization accounting

The owner approved [P08-A06–A09](P08_LATER_BRANCH_DECISIONS.md) on 2026-10-04 and resumed this no-fit goal. These decisions supersede the pending-choice checkpoints above. N2 retains the saved MIN-selected classical family in each context and retunes its registered hyperparameters separately for the four normalization controls. Regenerated population tiers repeat source-only model selection. Synthetic disturbances precede numerical preprocessing; unsupported shift coordinates use labelled endpoint extension.

DeepSeek implemented the metadata-only N2 planner; **45 synthetic tests passed** after supervisor review. A separate [actual-record audit](../results/p08_readiness/normalization_slot_ledger_audit.json) enumerated **89,632 operation slots**, including **42,368 model fits** and **1,040 scalar calibrations**, across **260 contexts**. All **681** source units matched saved source roles. The eight missing historical endpoints remain missing, limiting paired historical-reference coverage to at most **252 contexts**. No new model was fitted or selected.

This closes N2's exact metadata-accounting gap, not its inference specification, numerical runtime acceptance or resource approval. Population/perturbation specifications and ledgers remain unfinished. The primary recipe map, **54/206** adaptive/fallback rule, universal-first order and zero scientific execution authority are unchanged.

## 39. Normalization inference and filtered-population support

The [N2 inference specification](P08_NORMALIZATION_INFERENCE.md) now fixes eight exploratory contrasts on the **252** historical-reference-supported contexts. Its separate pooled-fold sensitivity uses **57** complete domain/repeat groups, comprising **228 contexts**. The [identity-only audit](../results/p08_readiness/normalization_reference_support_audit.json) verified complete recorded row membership without loading prediction values. Both sets retain **69 masters**, **557 distinct evaluation spectra** and all **13 domains**, but contain different repeated appearances. The full **260-context** paired effect remains unavailable, and new-control failures cannot silently shrink either registered set.

The [filtered-population prescreen](P08_POPULATION_SUPPORT.md) authenticates the stored **500-row** notes-clear and **575-row** Mira-1-excluded manifests. Both retain all **69 masters**, but only **11** and **12** original primary domains satisfy the pooled support rule. These are metadata findings, not regenerated source-role support or predictive results. The next population slice must reconstruct roles and audit source-only model selection without changing the original split definitions.

N2 numerical inference implementation and runtime acceptance remain separate from declaration tests. Population and perturbation specifications, exact accounting and resource gates are still incomplete. No real-data model, QC threshold, new prediction, uncertainty draw or preprocessing array has been calculated.

## 40. Filtered-population roles and support

The [regenerated-role audit](P08_POPULATION_SUPPORT.md#4-regenerated-role-support) verifies unchanged outer master assignments and population-specific spectral roles. Each tier reproduces all **345** historical master/repeat assignments. The **220** eligible notes-clear contexts include **216** with model-selection support and **215** with registered classical calibration support. Five CWA/Mira-2 contexts lack classical calibration support; four also lack supported model selection. All **240** eligible Mira-1-excluded contexts support both audited stages. Neural guard-role and source-logit calibration readiness remain separate checks.

The planner retains all original domain contexts, including pooled-ineligible cases and sparse held folds. It checks physical-master isolation, held-instrument exclusion, exact role identities and separate selection/calibration support. The supervisor's independent audit authenticated **14** input files and verified all **24** private generated tables after readback. Public outputs contain aggregate counts and hashes, not sample or observation identifiers.

These checks close the outer/selection/classical-calibration role gap, not the additional neural-role audit, population model-operation ledger, inference specification or resource proposal. No model recipe has been selected, no new model fitted and no prediction generated. The primary **260-context** benchmark, **54/206** adaptive/fallback rule, approved N2 scope and universal-first sequence remain unchanged. Scientific execution is still unauthorized.

## 41. Filtered-population neural support

The [neural-role audit](P08_POPULATION_SUPPORT.md#5-neural-guard-and-calibration-role-support) now closes the metadata gap identified in Section 40. Notes-clear supports ordinary-CNN and source-logit calibration roles in **216/220** eligible contexts, compared with **215/220** for classical calibration. Its acquisition-aware selection has **93** structurally comparable contexts and **123** required ordinary-CNN fallbacks; **four** contexts remain unavailable. Mira-1 exclusion supports all **240** eligible ordinary-CNN contexts, with **83** structurally comparable and **157** required fallback contexts.

The independent audit authenticated **33** inputs, reconstructed the population-bound guard assignment and retained the single unsupported notes-clear guard. Maximum batch capacities are **38**, **37** and **38** for the primary reference and two filtered tiers, below the existing **48** limit. These are source-role feasibility checks, not fits, logits, calibration values or model-selection outcomes. Classical/neural comparisons require their common supported context set; each method's broader coverage remains separately reported.

DeepSeek authored the adapter and its correction; supervisor review and invented-data tests remain separate from this metadata evidence. Exact population operation ledgers, inference rules, budgets, perturbation specifications and final readiness review remain required. The numerical launch permit is still unset.

## 42. Conditional filtered-population operation catalogs

The [population operation audit](P08_POPULATION_SUPPORT.md#6-conditional-population-operation-accounting) verifies fresh source-only selection and all three universal preprocessing actions within each filtered tier. Four-method model-fit ceilings are **170,277** for notes-clear and **184,749** without Mira-1; five-method ceilings are **258,387** and **280,878**. The **184,239-fit** Extra Trees increment is explicitly costed as a panel alternative, not silently authorized.

The graph retains all registered MIN neural-development recipes, ordinary-CNN fallbacks, unsupported guards and distinct classical/neural coverage. Conditional D1/D2/D3 branches share one future context-local selection decision; only one candidate branch can activate. Classical master-CV calibration uses exact-role aliases, and neural temperatures use only inherited source-selection predictions. An unknown or failed selection is not a valid fallback result.

DeepSeek authored the planner and tests; independent review checked arithmetic, dependency ordering and saved-graph readback. The full graphs remain private. No scientific fit, new prediction or preprocessing result was produced. The population-panel clarification, inference specification, finite resource proposals, perturbation specifications and final goal-wide review remain open. The launch permit is still unset.

## 43. Conditional population inference and matched support

The [population inference specification](P08_POPULATION_INFERENCE.md) now defines how the two filtered tiers will be compared. Its [identity-only audit](../results/p08_readiness/population_inference_support_audit.json) authenticates **32** inputs and verifies all **260** primary-to-population context mappings for each tier. Every retained test row has the same recorded master, label and instrument. No intensity array, prediction value or score was loaded.

Notes-clear policy effects retain **215** classical or **216** neural contexts; model comparisons and interactions use the common **215**. Complete-four-fold sensitivities retain **52** classical/common groups (**208 contexts**) or **53** neural groups (**212 contexts**), covering all **11** eligible domains. Mira-1 exclusion retains **240 contexts** in **60** complete groups across **12 domains**. These denominators are fixed before new outcomes and cannot shrink silently after a numerical failure.

The conditional registry keeps both panel alternatives: **32** effects and **32** interactions for four methods, or **40** effects and **48** interactions for five, across both tiers. The separate panel clarification remains open. Shared uncertainty settings, missingness rules and a future native-TikZ/offline-HTML scatter figure are specified; no inference or figure result has been generated. Numerical implementation, finite population resources, perturbation accounting and final readiness closure remain required. This milestone does not authorize training.

## 44. Conditional population resource proposals

The [finite proposals](P08_RESOURCE_PROPOSAL.md#14-filtered-population-resource-proposals) now cost both panel alternatives using historical durations and the audited operation catalogs. The neural cost audit authenticates **14,428 files** and counts **12,780 source fits** plus **1,635 refits**, excluding development contexts and interrupted-run accounting charges. Checkpoint sizes use authenticated manifests and current file sizes; tensors and prediction archives were not loaded.

Notes-clear proposes **120 hours / 64 GiB** for four methods or **168 hours / 96 GiB** for five. Mira-1 exclusion proposes **144 hours / 80 GiB** or **192 hours / 96 GiB**, respectively. These are unapproved stopping ceilings derived from historical cost estimates and explicit planning margins, not measured P08 requirements or completion promises. Both alternatives retain the same conditional operation counts and source-only selection rules.

The **2026-10-07** capacity check found approximately **179.3 GiB** free. Each population stage clears its initial threshold individually, but retaining the full U1 allowance plus both sensitivities and reserve requires **254 GiB** or **302 GiB**. Preserve evidence and recheck capacity between stages; neither cleanup nor additional resources are authorized. The separate panel choice, perturbation specifications, numerical runtime review and final goal-wide audit remain open. Universal-first sequencing and denied scientific execution are unchanged.

## 45. Conditional synthetic stress-test design

The [perturbation protocol](P08_PERTURBATION_PROTOCOL.md) now defines six disturbance families, the precise operation order, paired stochastic repetitions and normalized loss areas. Its **96** unique cases comprise **95** nonzero-labelled realizations and one clean reference shared across families. The design preserves the original shift, baseline, noise, impulse-count and clipping ranges. It does not create scientific inputs, predictions or scores.

Zero-dose checks must reproduce the original cropped numerical preprocessing and authenticated model predictions before a disturbed comparison is accepted. Classical MIN estimators require separately budgeted reconstruction because the old run retained predictions, not fitted estimators. Synthetic repetitions are averaged as scores after each repetition's normal M06 combination; they are not additional masters or an extra probability ensemble.

The two scope choices are now approved as [P08-A10–A11](P08_LATER_BRANCH_DECISIONS.md#6-robustness-scope-decisions-approved-2026-10-07): all five universal models and a clearly labelled fixed-clean-route adaptive sensitivity. The adaptive model panel remains unchanged. Frozen native QC cannot be silently redefined on the interpolated grid, and gate reactions to contamination are not tested. These choices do not reopen the unperturbed universal/adaptive procedures or answer the separate filtered-population panel question. The exact operation ledger, finite resource proposal, complete contrast registry and numerical implementation still require review; no execution permit follows from the [case inventory](../results/p08_readiness/perturbation_case_inventory.json).

## 46. Robustness inference and audited comparison support

The [inference specification](P08_PERTURBATION_INFERENCE.md) now fixes 456 secondary comparisons across five multiplicity families: 120 universal effects, 48 operational QC effects, 192 operational interactions, 48 eligible-QC effects and 48 eligible-QC interactions. Adjustment spans all six disturbances within each family, without changing any primary family. The source-noise reference, chosen models, clean routing and finite disturbance realizations remain fixed during the inherited conditional uncertainty analysis.

The independent identity-only audit verifies all 260 operational contexts and the fixed 54-context QC subset. The subset contains 176 spectra from 24 masters across three CWA instruments, with 18 contexts per domain. Its complete-four-fold sensitivity retains nine domain/repeat groups and 36 contexts; the other 18 contexts remain in the main equal-context analysis. Operational support retains 65 complete groups, 557 held spectra and 69 masters.

QC can change predictions in only those three domains/instruments; elsewhere it is a complete MIN alias. Its two-sided sign sensitivity therefore has a minimum unadjusted p-value of 0.25, even in the operational analysis. This is a mathematical resolution bound, not a measured p-value. Effect sizes and curves remain informative within the observed support, but cannot establish conventional significance or independent substrate chemistry. Exact job accounting, finite resources, numerical implementation and separate execution permission remain open.

## 47. Universal robustness model references

The [procedure-reference audit](P08_PERTURBATION_PROTOCOL.md#9-universal-model-reference-accounting) verifies **3,237 universal procedures** and **8,151 seed-specific estimator slots**. These comprise **1,820** historical classical reconstruction slots, **897** historical neural checkpoint slots and **5,434** future SG/arPLS estimator slots. Future estimators belong to the original universal fitting stage; robustness must retain and reuse them rather than silently adding replacement fits.

DeepSeek authored the metadata adapter and synthetic tests. Supervisor review corrected seed accounting, evidence scope, reference shape and upstream hash serialization. The independent actual-record audit reconciled original operation references and rehashed **6,826 existing files** without parsing predictions or checkpoint tensors. It did not fit a model, test numerical parity or evaluate robustness.

The catalog preserves context-local source selection, original fitting/test hashes, calibration order and neural reporting aliases. It is not the full disturbance-job ledger: fixed-route QC extraction, case-level dependencies, finite resources and numerical acceptance remain open. The original readiness goal and separate execution-permit boundary are unchanged.

## 48. Fixed-route QC references and complete MIN fallbacks

The [QC procedure-reference audit](P08_PERTURBATION_PROTOCOL.md#10-fixed-route-qc-model-reference-accounting) closes the extraction gap in §47. It binds **176 mixed-route procedures** in **54 eligible contexts** to **420 seed-estimator references** and **312 calibration references**. Their clean routing decisions remain unresolved upstream dependencies; no gate or route was calculated.

The remaining **206 contexts** contribute **824 reporting aliases** to **643 MIN procedures**. Each fallback must use the same disturbed case as its adaptive comparison, not the saved clean prediction. The **1,517 referenced MIN seed slots** are reused, not added to the fit count. Independent verification reconciled **920 blocks** and **1,544 operation slots**, preserving original roles, calibration order and complete-fallback semantics.

DeepSeek authored the adapter and regression tests; supervisor review corrected action-array binding and calibration-parent validation. The fixed-route interpretation implements P08-A11 and does not test native-grid gate reactions. The complete disturbance ledger, finite resources, numerical acceptance and requirement-wide readiness closure remain open. No new scientific operation or execution permission follows.

## 49. Exact stress-input graph and source-ID binding

The [input-operation audit](P08_PERTURBATION_PROTOCOL.md#11-exact-input-operation-dependencies-and-reuse) verifies **658,876 descriptors**, including **142,592 raw-row cases**, **427,776 action transforms** and **74,880 context/action assemblies**. These counts include explicit clean-path checks and separate context-local noise references. They are planned operations, not executed transforms or model fits.

The independent audit verified every content ID and dependency, source/test master separation, held-instrument exclusion and the raw-to-logical observation-ID mapping. It authenticated the raw intensity header without loading its values. Non-Gaussian row transforms are shared across repeated appearances; Gaussian amplitudes retain their original source-context binding. Input files and frozen preprocessing remain unchanged.

The next ledger layer must join these input dependencies to retained models, reconstruction/parity, calibrated predictions, fixed-route QC composition and same-case aliases. Scoring/inference accounting, finite resources and the final readiness review remain open. No scientific operation or execution permit is implied by this metadata checkpoint.

## 50. Joined prediction, reuse and same-case reporting graph

The [prediction/reuse audit](P08_PERTURBATION_PROTOCOL.md#12-exact-prediction-reuse-and-reporting-dependencies) closes the model-and-prediction dependency gap in §49. It joins **3,413 procedures**, **8,571 seed-estimator slots** and **96 cases** into **2,013,605 planned operation descriptors**. Its **574,080 reporting aliases** preserve universal strategies, eligible QC, complete QC fallback and family fallback without inventing extra fits. The separate input graph is not counted again.

Only **1,820 historical classical MIN slots** require estimator reconstruction; all other final estimators must be retained from their original upstream stage. The stress graph adds no temperature fitting. Every nonclean prediction depends on a clean replay check against its recorded reference, preserving classical and neural calibration order. No such numerical check was performed during this metadata audit.

QC composition retains each row's native clean action and distinguishes an invalid-action MIN-input fallback under the QC model from a complete MIN-model fallback. The latter always targets the same disturbance case. Independent enumeration verified every dependency and alias target. DeepSeek implemented the join, graph and synthetic tests; supervisor review corrected nested-reference sharing and a mistaken test field.

Scoring, conditional uncertainty, stability and rendering operations still require exact accounting before the finite robustness resource proposal and final readiness lock. The distinct filtered-population panel choice also remains open. No model, spectrum, prediction or routing outcome was computed, and the scientific permit remains unset.

## 51. Sample-bound scoring and complete-fold sensitivity

The [score-support audit](P08_PERTURBATION_PROTOCOL.md#13-sample-membership-pooled-scoring-and-curve-dependencies) closes context scoring, complete-fold pooling and curve dependency accounting. It binds 557 held spectra and 69 masters to their original 260 contexts. Recorded numeric master IDs remain unchanged; the private metadata archive, not the public aggregate, retains them.

The graph contains 882,096 score/curve descriptors and 1,539,588 reporting aliases. Its 911 distinct pooled procedures support 1,567 reporting views without duplicating model predictions. All nine eligible-QC complete groups reuse operational procedures. Their 36 contexts form the pooled sensitivity; the other 18 remain in the main 54-context analysis and are not filled in artificially.

Pooled scoring reconstructs M01/M06 from four disjoint folds within a domain/repeat. It does not average fold balanced accuracies, pool repeats or treat technical repetitions as new masters. Every curve depends on its full registered case list and shared clean reference. The audit verifies metadata dependencies, not numerical scores, clean parity or uncertainty.

DeepSeek authored the adapters and synthetic tests. Supervisor review corrected shared-clean validation, strict input checking, recorded ID types and test assumptions about support aliases. Conditional inference still needs explicit mapping from recorded numeric IDs to the inherited lexical bootstrap identity order. Contrast/uncertainty, stability, preservation/rendering accounting, finite resources, the distinct population-panel choice and final requirement-wide acceptance remain open. No scientific execution is authorized.

## 52. Bound contrasts and finite robustness proposal

The [comparison audit](P08_PERTURBATION_INFERENCE.md#7-audited-comparison-bindings-and-proposed-resources) verifies all 456 registered effects/interactions, each signed model/policy term, complete-fold support and the explicit mapping from recorded master IDs to global lexical uncertainty columns. The operational and eligible subsets use the same global identities; neither aliases nor repeated measurements create new independent samples. This is metadata acceptance, not a calculated contrast or validated inference runtime.

The [S1 proposal](P08_RESOURCE_PROPOSAL.md#15-s1-proposal-registered-test-time-robustness) now bounds historical classical reconstruction and later retained-model inference. Its limits remain unapproved. The full U1/Q1/S1 artifact allowances plus reserve exceed current capacity; no cleanup or additional storage is assumed.

Next, reconcile exact uncertainty, hierarchy, sign/Holm, stability, preservation and rendering operations, then the original requirement-wide readiness audit and separate U0 request. The distinct filtered-population Extra Trees choice remains unresolved. The current goal stays active and incomplete; no scientific operation has been authorized.
