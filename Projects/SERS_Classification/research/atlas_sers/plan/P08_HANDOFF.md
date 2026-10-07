# P08 handoff: does preprocessing change acquisition-shift performance?

**Prepared:** 2026-09-30. **Status:** planning handoff, not an execution permit.

**Score-support accounting, 2026-10-07:** the [membership and scoring audit](P08_PERTURBATION_PROTOCOL.md#13-sample-membership-pooled-scoring-and-curve-dependencies) verifies 882,096 score/curve descriptors and 1,539,588 reporting aliases. Pooled scoring uses four disjoint folds, not an average of fold scores. The nine eligible-QC pooled groups reuse existing operational procedures without adding independent evidence. Recorded master IDs retain their source type; later weighted inference must use an explicit identity-to-column mapping. Next, complete contrast/uncertainty, stability, preservation/rendering accounting and finite resource reconciliation. The distinct filtered-population panel clarification and final requirement-wide audit remain open. No scientific prediction, score or draw was calculated.

**Prediction/reuse accounting, 2026-10-07:** the [joined graph audit](P08_PERTURBATION_PROTOCOL.md#12-exact-prediction-reuse-and-reporting-dependencies) verifies 2,013,605 planned operation descriptors and 574,080 reporting aliases across 96 cases. It binds model reuse, calibration order, clean-replay prerequisites, fixed-route QC composition and same-case MIN fallbacks. These are metadata counts, not executed predictions or fits. Next, complete scoring, uncertainty, stability and rendering accounting, then reconcile finite resources and the remaining filtered-population panel choice. The original readiness goal remains incomplete; no scientific execution is authorized. Earlier dated next-step statements below are historical checkpoints.

**Stress-input accounting, 2026-10-07:** the [input-catalog audit](P08_PERTURBATION_PROTOCOL.md#11-exact-input-operation-dependencies-and-reuse) verifies all 658,876 planned input descriptors and their dependencies. Raw-source identifiers are explicitly mapped to frozen dataset identifiers; Gaussian amplitude references remain source-context-specific. No intensity values or perturbations were computed. Next, join this graph to model reconstruction/reuse, per-seed prediction, calibration, fixed-route QC composition and same-case aliases, then finish scoring/inference accounting and finite resources. Earlier dated checkpoints describe completed components, not new execution authority.

**Fixed-route QC accounting, 2026-10-07:** the [QC reference audit](P08_PERTURBATION_PROTOCOL.md#10-fixed-route-qc-model-reference-accounting) binds 176 procedures in 54 supported contexts and all 824 reporting aliases in the 206 fallback contexts. Clean routes are future dependencies, not calculated results. Each fallback must reuse MIN for the same disturbance case. Next, expand the shared case-level dependency ledger, account for transforms, prediction/calibration applications and inference, and propose finite resources. The full no-fit readiness goal remains incomplete; no scientific execution is authorized.

**Universal robustness accounting, 2026-10-07:** the [procedure-reference audit](P08_PERTURBATION_PROTOCOL.md#9-universal-model-reference-accounting) now separates classical MIN reconstruction, saved neural MIN checkpoints and future retained SG/arPLS estimators. Its 3,237 procedures contain 8,151 seed-specific estimator slots; no new fit or prediction was performed. Next, bind the fixed-route QC procedures and disturbed-MIN fallbacks, expand the case-level dependencies and complete the finite resource proposal. This is progress within the existing readiness goal, not completion of the full robustness ledger.

**Robustness checkpoint, 2026-10-07:** P08-A10–A11 approve all five universal robustness methods and the fixed-clean-route QC sensitivity. The [disturbance design](P08_PERTURBATION_PROTOCOL.md) defines 96 cases, and the [inference specification](P08_PERTURBATION_INFERENCE.md) fixes 456 secondary comparisons and their uncertainty, missingness and multiplicity rules. QC support spans three CWA instruments and 24 masters; its two-sided sign p-value cannot be below 0.25. This analytical limit is not a measured result. The original no-fit goal remains incomplete: exact robustness operation accounting, finite budgets, the distinct filtered-population panel choice and final requirement-wide reconciliation still precede a separate execution request. Earlier checkpoints below retain their dates and are not current pending-choice instructions.

**Resumed, 2026-10-04:** the owner approved [P08-A06–A09](P08_LATER_BRANCH_DECISIONS.md): N2 fixed-family normalization controls; fresh source-only selection within regenerated populations; perturbations before numerical preprocessing; constant endpoint extension outside available shift support. These choices supersede the later-branch pending-choice wording below. The original no-fit goal is active; exact branch ledgers, remaining numerical definitions and proposed resources still require completion. Training remains unauthorized.

**Readiness has begun:** the [input/support audit and approved direction](P08_READINESS.md) now record the owner's model-mapping, Extra Trees, scoring/uncertainty and complete minimal-pipeline fallback decisions. Those dated decisions supersede pending recommendations below. The original specifications remain preserved; exact execution ledgers, numerical locks and a separate scientific permit are still required.

**Subsequent checkpoints, 2026-10-02:** universal and nested-QC metadata catalogs and the wider-range ledger have been audited; none authorizes execution. QC retains 54 eligible contexts and 206 complete MIN fallbacks. The [normalization decision audit](P08_READINESS.md#20-normalization-selector-decision-now-has-both-cost-bounds) now distinguishes 260 complete source-family choices from 252 complete historical final endpoints and costs both selector interpretations. The [later-branch note](P08_LATER_BRANCH_DESIGN.md) and [resource proposal](P08_RESOURCE_PROPOSAL.md) identify the remaining choices, numerical specifications and resource gates. The recommendations and first-gate wording below are retained as the original handoff, not evidence that the completed metadata work remains unstarted.

The next experimental question is whether smoothing or baseline correction improves chemical identification on an unseen instrument, and whether its effect depends on the classifier. Minimal min–max scaling is the completed benchmark's controlled reference; it has not been established as the best preprocessing policy.

The [requirement-to-evidence checkpoint](P08_COMPLETION_AUDIT.md) separates completed readiness components from the remaining decision-dependent specifications and runtime acceptance gates. The [N2 metadata ledger](../results/p08_readiness/normalization_slot_ledger_audit.json) now verifies the approved normalization branch's exact operations; it is not a numerical executor. The [source-stage mapping](P08_SOURCE_STAGE_MAPPING.md) records how inherited within-fit validation relates to the distinct fit and prediction jobs. None of these documents authorizes a scientific run.

The [outer launch specification](P08_U0_LAUNCH_BOUNDARY.md) connects those components to the independent-permit, retained-import, failed-setup and cumulative-accounting requirements. The [default-denied implementation](P08_SOURCE_STAGE_MAPPING.md#13-default-denied-outer-entry-point) has now passed an invented-data CPU-pair test with actual persistence, deliberate interruption and fresh-process relaunch refusal. An import-audit report cannot substitute for scientific authority. Actual capacity checks, a separately approved permit, remaining branch specifications and final readiness review remain required before execution.

This handoff implements the ordering in [Master Plan §16](MASTER_PLAN.md#16-sub-plan-p08--preprocessing-policy-factorial-and-robustness). The current frozen-prediction uncertainty analysis does not authorize new fitting, policy selection, calibration, perturbations, or predictions. Its release must be reviewed before P08 begins. The original plans and immutable registries remain unchanged.

## 1. Scientific questions and reading order

| Question | Controlled comparison | Main figure and interpretation |
|---|---|---|
| RQ-S01: universal preprocessing | Minimal, conservative Savitzky–Golay smoothing, and arPLS baseline correction, each applied universally | Paired domain scatter and policy-minus-minimal intervals. Show all domains, including deterioration. |
| RQ-S02: platform-family policy | A rule chosen on source instruments, using only the deployed instrument's family metadata | Domain effects plus selected-action/support views. Separate all-domain fallback-inclusive performance from supported-family results. |
| RQ-S03: row-quality policy | A source-selected rule driven only by permitted quality measurements from the current spectrum | Quality/action scatter, representative before/after spectra, domain effects and fallback burden. No instrument identity or target-label selection. |
| RQ-S05: robustness and preservation | Fixed, prespecified spectral perturbations and preservation diagnostics | Degradation curves and paired spectral overlays. Improved average accuracy must not hide damaged peaks or a poorer weakest domain. |

Every figure requires native TikZ, offline HTML, vector PDF and PNG from the same semantic table. Use black text and standard LaTeX or Times-compatible fonts. Scatter, paired-dot and interval views take precedence over bars when they convey the comparison directly.

## 2. First gate: a no-fit implementation and support audit

Before launching any numerical experiment:

1. Authenticate the accepted P00–P05/P13 evidence and the completed frozen-result release. Record their hashes and preserve their scores.
2. Resolve the model identities below. Compile an explicit policy × model × context × seed ledger, including legitimate reuse and unsupported cells.
3. Verify the existing 400–1,800 cm⁻¹ representations and final per-row [0,1] scaling. Do not change SG/arPLS settings or choose new baseline parameters from held results.
4. Reconcile all inherited split roles, physical-master exclusions, held-instrument exclusions, inner selection roles and test-unit sets. Account for the eight incomplete historical selected-classical references without filling them retrospectively.
5. Audit platform-family and QC selection support from source roles only. Record where the frozen minimal fallback is mandatory.
6. Define the paired policy effects, model interactions, missing/failed-cell treatment, uncertainty method, multiplicity families and preservation rules before P08 outcomes. The current weighted-bootstrap addition is evidence for that discussion, not automatic authority to reuse it for a new hypothesis family.
7. Count actual fits, refits, calibration operations and prediction jobs. Set a finite wall-time, memory, storage and execution ceiling; obtain a separate owner-approved permit before a scientific smoke test or expansion.

DeepSeek V4.1 Flash implements bounded slices; the supervisor reviews code, synthetic tests and role/support evidence before controlling execution. A worker's proposed command or self-assessment does not pass a gate.

## 3. Model identities that must be resolved before the ledger is locked

The master plan names RBF SVM, Random Forest, D0 and the frozen acquisition-aware candidate. Two names now require an explicit mapping, because the completed benchmark distinguishes historical D0-ERM from matched D0-M and a fixed D3 recipe from a source-selected procedure.

**Recommended mapping, pending the next lock:** use matched D0-M as the ordinary CNN control and retain the P05 source-selected context-local recipe ledger as the acquisition-aware procedure. Keep that recipe identity fixed across preprocessing policies. Architecture, losses, sampling rules, seeds, source-only calibration rules and epoch-selection procedure must be stated explicitly. Retraining on a different representation is required; reusing old weights without an explicitly authorized transfer experiment is not the same comparison.

This recommendation is a post-benchmark planning decision, not a claim that D0-M was the historical meaning of every occurrence of “D0.” It requires an explicit contract amendment before execution. Fixed D3 cannot replace the selected procedure because its held mean was larger. Historical D0-ERM, Extra Trees and a fixed D3 sensitivity may be retained only if separately enumerated and budgeted; the core panel must not expand silently.

Classical hyperparameters are selected afresh using the declared source-only procedure for each policy. Deep architecture/loss identity stays fixed; any source-only stopping/refit-duration rules must be identical across policies. Where two locked specifications are genuinely identical, reuse must be proven by specification and input hashes, not by similar scores.

## 4. Staged experimental sequence after approval

**Universal policies first.** Complete and review minimal, SG and arPLS on the locked model panel and matched splits. Report spectrum-level and combined-prediction endpoints separately. For each model, estimate policy-minus-minimal effects; then compare those effects across classical and deep models. A positive interaction means preprocessing changed the relative model comparison, not that a physical nuisance was causally removed.

**Family-aware and row-QC policies second.** Use the fixed action library and source pseudo-instrument support rules. Family selection uses the frozen equal-weight RBF-SVM/D0 panel and lexicographic objective. Row-QC uses only noise-to-range, spike fraction, baseline energy fraction, baseline span fraction and negative fraction; its source quantiles are 0.50, 0.75 and 0.90. Preserve minimal fallbacks for unsupported families, sparse source support, missing QC and invalid actions. Report operational all-domain performance as well as supported subsets.

**Robustness and exploratory branches last.** Retain the master plan's test-time shifts, baseline slopes, broad backgrounds, noise, impulses and clipping. The 400–1,849 cm⁻¹ range, SNV/vector/area controls, derivative destructive control, notes-clear tier and Mira-1-excluded tier remain separately labelled branches. Regenerated population-tier splits need new support audits; they are not interchangeable with the primary partitions. Enumerate and budget these branches before execution rather than adding them in response to a disappointing result.

## 5. Required records and claim boundary

The final P08 package must retain input/representation hashes; roles and support; selected actions and reasons; all fallbacks; classical selections; fixed neural recipe identities; epochs and actual computational cost; per-domain paired effects; weakest-domain changes; preservation outcomes; calibrated probability metrics; and figure provenance. Missing measurements remain missing, not zeros or failed sensor responses.

The completed minimal-input benchmark supports a classical/deep acquisition-shift comparison, not a claim of chemical–nuisance disentanglement. P08 can test whether defined preprocessing policies change predictive performance and spectral preservation on observed support. It cannot reconstruct an unmeasured clean chemical spectrum or establish instrument-independent substrate chemistry from unsupported factor combinations.

A defensible publication can report competitive tree ensembles, small or inconsistent neural-loss gains, and policy-dependent transfer. It need not claim a new deep model wins. Final positioning should follow the reviewed P08 evidence and a separate literature refresh; publication novelty or venue acceptance is not established by this handoff.
