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
