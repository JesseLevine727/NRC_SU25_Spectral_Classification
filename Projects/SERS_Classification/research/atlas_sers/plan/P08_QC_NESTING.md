# P08 QC policy: nested selection and finite accounting

**Specified:** 2026-09-30. **State:** concrete nested roles and complete metadata catalog audited; threshold/routing controls passed synthetic review, while resource authorization remains pending. **Authority:** P08-A01–P08-A05, not a scientific execution permit.

This procedure asks whether row-local quality indicators can select preprocessing for an unseen instrument. It compares a learned routing rule with the complete minimal pipeline. A routing rule chooses an immutable MIN, SG or arPLS row; it does not combine predictions from separately trained universal models. Models must be trained on the routed inputs used by that procedure.

## 1. Population and nesting

The [support audit](../results/p08_readiness/qc_nested_support_audit.json) identifies 54 eligible contexts, all at CWA, with two source pseudo-instrument units each. The other 206 contexts use complete MIN fallback: 132 lack source pseudo-instrument support and 74 lack nested class support. All 260 remain in operational results. The eligible-subset estimate applies only to the 54 supported CWA contexts.

For an eligible context, let S be its registered outer-source fitting observations and T its held test observations. Each registered source pseudo-instrument unit separates S into a policy-fitting role F and policy-validation role V, with every view of V's masters excluded from F. Additional unused source views remain excluded where required by the original registry. No role is filled using T or another context's held results.

| Level | Fitting information | Validation or application | Permitted choice |
|---|---|---|---|
| Inner estimator | Two master-separated folds inside F | Remaining fold inside F | SVM hyperparameters; D0-M checkpoint and refit duration; source calibration |
| Policy validation | All F, after inner choices freeze | V, a different source instrument and disjoint masters | Rank the 124 gate candidates with equal SVM/D0-M weight |
| Final estimator development | Registered source-only fitting roles inside S, after the gate identity freezes | Registered source validation/calibration roles inside S | Final classifier hyperparameters, stopping and temperature |
| Held evaluation | Final model fitted on S | Original T only | No choice; record predictions and outcomes |

Policy validation must evaluate the estimator selected inside F. Selecting its hyperparameters or best epoch using V would reuse the same observations for model tuning and for judging the gate. The deeper estimator folds prevent that error. The final estimator is then developed on the full available source population using the frozen gate identity. That final development may reuse source observations previously involved in gate selection; it is not an independent replication of gate validation. Only T is the untouched outer evaluation. No score from T selects a gate, cutpoint, model or training duration.

## 2. Additional estimator folds

Construct three folds separately inside each of the 108 registered policy-fitting roles. These are new P08 inner roles, not replacements for any P02/P04 split. The original registries retain their original `StratifiedGroupKFold` construction.

Within each class, list unique physical master IDs. Rank them by the canonical SHA-256 of the namespace `p08-qc-inner-master-v1`, salt 2026093004, context ID, parent unit ID, class label and master ID; break a hash tie by master ID. Assign rank i to validation fold i modulo 3. Every stored view of a master follows that assignment. Sort serialized identities lexicographically. The fitting set is the parent fitting role minus the validation masters, not a subsample of convenient observations.

This fixed, class-wise assignment gives every fold at least one validation master and at least two fitting masters per class when the parent has three. It uses no spectral value, instrument availability, classifier outcome or random search to improve the split. Repeated spectra do not change a master's fold. The inner folds assess master transfer within the source data; they are not independent instrument-transfer folds. The original surrounding pseudo-instrument role provides that transfer validation.

The registry must prove exact parent coverage, disjoint fit/validation masters and observations, class completeness, exclusion of policy-validation masters, and absence of outer-test masters and the held instrument. It contains 324 nested estimator folds before model, gate or seed expansion. Unsupported contexts have no nested folds and retain their reason-coded complete-pipeline aliases.

The [concrete role audit](../results/p08_readiness/qc_nested_role_audit.json) passed these checks on all 324 folds. The supervisor independently reproduced the assignment and canonical hashes, reversed input ordering, checked nonmutation, and verified rejection of a forged execution flag. The private registry digest binds the exact assignments; the public report contains only aggregates and hashes. Fold construction uses metadata only and does not fit a QC threshold or estimator.

## 3. Fold-local quality thresholds and routing

Use only the frozen permitted QC ingredients: noise MAD divided by positive intensity range, spike fraction, baseline energy fraction, baseline span fraction and negative fraction. No instrument, platform, substrate, station, master, label or prediction-confidence field enters a gate. Identity and class metadata may define splits; they may not define the routing rule.

A row's QC vector is available only when all six recorded ingredients are finite, its intensity range is positive and the resulting five features are finite. Otherwise route that row to MIN with a missing/nonfinite-QC reason. Do not impute QC from its instrument, neighboring rows or the target batch. The existing MIN row must itself be valid; an invalid minimal input is a fatal data defect, not an additional fallback.

For a fitting role, calculate the three registered quantiles (0.50, 0.75 and 0.90) over its complete finite QC rows, giving each stored row one contribution. Use float64 linear quantile interpolation. This is a row-based threshold estimator, not a master-balanced estimator. If no complete fitting QC vector exists, all rows governed by that threshold object use MIN with `no_finite_source_qc`; target rows never supply the missing threshold.

During an inner estimator fold, only its fitting observations estimate cutpoints. Apply those unchanged cutpoints to its fitting and validation rows. For a policy-validation refit, estimate cutpoints from F and apply them to F and V. During final estimator selection or calibration, estimate cutpoints inside each actual fitting role. For the final model, estimate cutpoints from S and apply them to S and each T row independently. No validation/test distribution estimates a cutpoint, even without labels.

An active trigger means the current feature is **strictly greater** than its source quantile. Equality does not trigger. A single noise trigger chooses SG; a single baseline trigger chooses arPLS. Dual-trigger gates follow their frozen priority order when both triggers fire. If no trigger fires, choose MIN. An invalid selected action uses the existing row-level MIN fallback and records its reason. All current frozen action rows are valid; this rule is not permission to regenerate an action.

The unchanged P02 library has one minimal-only, 15 single-trigger and 108 dual-trigger candidates. Threshold calculations shared by gates may reuse an identical source-role/QC-input hash. Model reuse additionally requires exact routed fitting/validation inputs, labels, roles, model specification, seed, stopping and calibration bindings. Similar gate names, action counts or scores are insufficient. No numerical cutpoint or routing decision has been calculated in this readiness work.

## 4. Gate selection

For each candidate gate and each F/V unit:

1. Run the inherited 36-candidate RBF-SVM grid across the three inner master folds. Select by the inherited lexicographic classifier objective: mean balanced accuracy, worst balanced accuracy, macro-F1, complexity, then declared candidate order. Require complete fold coverage for an eligible model candidate.
2. Train D0-M with the three frozen technical seeds, using the same 30–200 epoch limits, patience 20 and checkpoint rule as the universal comparison. Its early stopping uses the inner validation fold, never V. The final F-refit duration for each seed is the clipped, Python-rounded median of that seed's three inner best epochs.
3. Fit source temperatures from the chosen SVM's inner validation scores and D0-M's per-seed inner validation logits. Apply the inherited master-equal calibration convention. Calibration does not inspect V. Record its complete input hash.
4. Refit the selected SVM and the three fixed-duration D0-M models on F, with cutpoints estimated from F. Predict V. Calibrate neural probabilities per seed before averaging; SVM retains its single source-fitted temperature.

The policy uses individual-spectrum balanced accuracy (M01). For each pseudo-instrument unit, first average the SVM and D0-M scores equally. Rank each gate by: mean of these unit scores descending; lowest unit score descending; fraction of units strictly improved over the nested minimal-only gate descending; mean nonminimal routing fraction on V ascending; and original declared gate order ascending. Average units equally, including the nonminimal routing fraction. Do not pool rows across differently sized pseudo-units to choose a gate. Use the unrounded values with the inherited deterministic sorting convention.

The minimal-only gate is evaluated with this same nested procedure. Existing universal MIN scores do not replace that policy-validation control: its estimator selection occurred at a different nesting level. No model is selected because it was the held-test winner. Gates share the same development panel, folds, seed set and class vocabulary.

A model candidate missing any required inner fit is ineligible for its source selection and remains reason-coded in the ledger. A gate missing a complete SVM or D0-M policy-validation result for any registered unit is ineligible; do not average remaining seeds or units. If the minimal-only gate cannot be evaluated completely, stop that context's policy selection as unavailable rather than promote another gate against an incomplete reference. A collapsed but finite, valid classifier remains a scored result. Missing execution evidence does not become a sparse-support MIN fallback or authorize a retry.

## 5. Final classifier development and calibration

Freeze one gate identity per eligible outer context, then use it for all four adaptive evaluation methods: RBF-SVM, Random Forest, D0-M and the context's P05-selected recipe. Extra Trees remains outside this adaptive panel. Classical grids are reselected using the original registered source selection units; forest calibration retains seed averaging before its single source temperature. Neural architecture/loss identities are not reselected. Their stopping, final duration and per-seed calibration use source evidence for the frozen gate and retain the universal procedure's rules.

In the 54 eligible contexts, the frozen selected recipe is D0-M in 40, D1 in one, D2 in seven and D3 in six. Thus D0-M and the selected strategy require 68 distinct context–recipe procedures, not 108 independent neural architectures. The 108 inherited source units plus 28 additional non-D0 units give 408 neural source fits and 204 final refits under three seeds before exact reuse.

Each final model must receive a training matrix assembled by the frozen gate, not by applying the gate only to test inputs. Thresholds may differ across fitting roles because their source rows differ; the rule identity stays fixed. The resulting policy is a learned preprocessing-and-estimation procedure. Its effect cannot be attributed to a single transform, nor interpreted as chemical/nuisance disentanglement.

The operation graph must separate final source routing from held-test routing. The source-routing block contains S only and precedes final fitting. After every required final model is frozen, a separate test-routing block applies the source-fitted thresholds to T. Neither final fitting nor any upstream selection or calibration operation may depend on that test-routing block, directly or transitively. This makes the access boundary explicit in the graph, not merely an instruction to ignore part of a combined source/test object.

If the winning gate is minimal-only, the complete final procedure can reference the existing MIN evidence after exact specification and input matching. Other gate procedures may reuse artifacts only when the full relevant input/selection chain is identical. A row-wise hybrid of universal model predictions is not this experiment. All 206 unsupported contexts use the approved complete MIN pipeline independently of gate outcomes.

## 6. Literal compute ceiling before exact reuse

The table enumerates structural model-fit slots, not independent samples, executed fits or an authorized budget. Full job identities, runtime limits and an execution permit are separate release requirements.

| Block | Model-fit slots |
|---|---:|
| Policy-validation SVM inner grid: 124 gates × 108 units × 3 folds × 36 candidates | 1,446,336 |
| Policy-validation D0-M inner fits: 124 × 108 × 3 folds × 3 seeds | 120,528 |
| Policy-validation refits: 124 × 108 × (1 SVM + 3 D0-M seeds) | 53,568 |
| Frozen-gate final development: SVM source/calibration/refit | 4,104 |
| Frozen-gate final development: RF source/calibration/refit | 5,832 |
| Frozen-gate final development: distinct neural source/refit procedures | 612 |
| Literal complete QC model-fit ceiling | **1,630,980** |

The gate-ranking block also contains 53,568 scalar temperature fits; final development adds 312, totaling 53,880 scalar calibrations. These are separate from model fitting. Predictions, score aggregation, gate selection, threshold fitting, routing and aliases require their own operation records. No full 124-gate multiplication applies to final four-model development: only the source-selected gate reaches that stage.

Identical routing can substantially reduce execution, but no reduction is assumed before its complete hash proof. The literal ceiling contains 161,316 neural fits. Keeping the previous full checkpoint-retention pattern may require more space than the currently available disk. A finite staged resource proposal must state retention and storage feasibility; neither successful universal experiments nor passing this role audit automatically authorizes the adaptive sweep.

The [staged resource proposal](P08_RESOURCE_PROPOSAL.md) specifies universal-first smoke/full stages and a separate adaptive ceiling. Its checkpoint estimate exceeds current free storage. Those are proposed limits, not inherited authority, and the adaptive sweep cannot start until its resource gate is resolved.

## 7. Release and interpretation

The implementation worker supplies pure role/ledger/routing guards against synthetic fixtures. The supervisor authenticates private inputs, checks exact roles and arithmetic, validates fail-closed behavior and controls publication. No scientific fitting, calibration, thresholds, gate selection, predictions or resampling occurs under this document's current authority.

Report operational and eligible-subset outcomes, source-frozen gates, cutpoints, action fractions, reasons, selection stability and paired policy effects under the [statistical protocol](P08_STATISTICAL_PROTOCOL.md). Unsupported stations remain in operational denominators. A source-selected QC rule can be evaluated for predictive transfer; it does not identify physical background removal or restore a measured clean chemical spectrum.

## 8. Numerical implementation and evidence boundaries

The threshold kernel consumes the six frozen QC ingredients; it does not recompute them from processed spectra. It copies inputs to little-endian float64, sorts source rows with their identifiers and binds all recorded ingredients, including unavailable rows, to a content digest. Only complete finite vectors contribute to the specified linear quantiles. A nonfinite computed quantile is an error, not an empty-source fallback. Local numerical-error handling leaves the caller's NumPy settings unchanged; explicit validity checks still reject unavailable vectors.

The threshold state records source membership and QC-input hashes, row counts, feature order, quantiles and cutpoints. State validation checks the original digest before interpretation and returns an independent copy. Neither a digest nor a self-consistent state proves source-only membership: the separately authenticated role registry and source-file bindings remain mandatory at the future runtime boundary.

The row router uses one authenticated threshold state, one fixed P02 gate and the validity of the three immutable actions. It accepts no label, acquisition identity or target-batch summary. For deterministic reason logging, an empty source threshold takes precedence over unavailable row QC; either condition selects MIN. Otherwise strict threshold comparisons and the registered trigger priority choose the requested action. An invalid requested action falls back to MIN without trying another nonminimal action. An invalid MIN row always stops processing, including when another action would otherwise be selected.

Each private route record binds the threshold state, exact gate definition, original normalized QC ingredients and action-validity flags. It records available features, trigger states, requested and selected actions, and the fallback reason. Actual row-level values and records remain private under the [figure protocol](P08_FIGURE_PROTOCOL.md). Hashes bind these inputs; they do not authenticate source files or grant execution authority.

Current numerical tests use invented arrays only. The modules have no dataset loader, model executor or scientific permit, and their execution entry points always reject a launch. Passing these tests does not authorize applying thresholds or gates to the dataset, choosing a winning rule or reducing the literal execution budget through numerical deduplication. Independent review and validation results are recorded in the [review log](delegation/P08_REVIEW.md).
