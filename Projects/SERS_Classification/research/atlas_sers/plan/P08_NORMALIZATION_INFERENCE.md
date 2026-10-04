# P08 conditional normalization comparison

**Date:** 2026-10-04. **Status:** pre-outcome numerical specification under P08-A02 and P08-A06; not an execution permit. No normalization-control model, score or uncertainty draw has been calculated.

## 1. Question and conditioning

N2 asks whether a different input normalization changes identification for the classical family already selected from each context's MIN source data. It does not search for the best family under each normalization. The MIN-selected family may be poorly suited to another representation; that limitation is part of the experiment, not permission to change families after seeing results.

The four controls are SNV, vector normalization, area normalization and the first-derivative destructive control. Their frozen arrays retain their native scaling. Do not append min–max scaling: that would erase the intended normalization contrast. The derivative control is not a candidate for promotion to the primary preprocessing policy. No neural model, additional classical family search or normalization-specific candidate grid is added.

The [N2 operation graph](../results/p08_readiness/normalization_slot_ledger_audit.json) binds all 260 primary contexts and their saved source-family decisions. For each new control, retune every registered hyperparameter candidate within that frozen family using source data only; train fresh estimators and fit the inherited scalar calibration. Do not reuse the MIN winning hyperparameters or its fitted model on changed inputs. If the entire family grid is unavailable, report the context unavailable rather than switching families.

The minimal benchmark and its missing outcomes have already been examined. This specification precedes new normalization outcomes; it is not retrospective preregistration of the benchmark. All effects are exploratory and conditional on the MIN selection procedure, observed support and retained reference outcomes. They cannot establish chemical–nuisance disentanglement, a universal normalization optimum or passage of the original G4 superiority gate.

## 2. Fixed comparison support

The [identity-only support audit](../results/p08_readiness/normalization_reference_support_audit.json) loaded recorded prediction keys, not scores or probabilities. It authenticated complete reference row membership against the fixed held-test roles. The eight missing historical C-SELECTED endpoints remain missing; no new MIN fit is scheduled to repair them.

| Support definition | Contexts | Domains | Distinct masters | Distinct spectra | Spectrum/context appearances |
|---|---:|---:|---:|---:|---:|
| Registered primary held evaluation | 260 | 13 | 69 | 557 | 2,785 |
| Fixed reference-available N2 comparison | 252 | 13 | 69 | 557 | 2,714 |
| Fixed complete-four-fold sensitivity | 228 | 13 | 69 | 557 | 2,499 |

The last row consists of 57 complete domain/repeat groups out of 65; each group contains four disjoint test folds. The reference-available set retains 18–20 contexts per domain. The complete-four-fold set retains 12–20 contexts, or three to five complete repeats, per domain. Both sets retain all ten instruments and three stations. A spectrum can appear in several repeats; appearance counts do not increase the number of independent physical samples.

Freeze both context sets and the complete domain/repeat group set using the audit's hashes before any new-control outcome. The principal reported N2 sensitivity uses the fixed 252-context set, clearly labelled **reference-support conditional**, not a complete 260-context estimate. The intended all-context paired effect is unavailable under the current historical lock. Reference availability may reflect prior fitting/calibration failures; missingness is not assumed random.

All 260 new-control contexts remain in the fitting ledger and availability report. The eight without references can contribute descriptive new-control scores, but not paired effects. Do not let performance on those eight influence the frozen family, normalization choice or paired-support definition. The comparison is not a comparison restricted to cases helped by a control.

## 3. Predictions and effects

Use the inherited prediction authentication, class vocabulary and tie rules in [P08 statistics §§1–2](P08_STATISTICAL_PROTOCOL.md#2-predictions-and-point-estimates). M01 scores individual spectra. M06 averages predicted probabilities for repeated measurements of a physical sample within a context before one class decision; it does not average spectra. Each held context contains one instrument. Across the fixed reference-support contexts, these definitions yield 2,714 spectrum/context appearances and 1,268 master/context prediction units.

Random Forest and Extra Trees average uncalibrated technical-seed probabilities, convert the result to the inherited score representation, and apply one source-fitted temperature. Deterministic families use their inherited single-score procedure. Missing seeds are not averaged away. A finite model that predicts only one class remains a valid, possibly poor, result.

For context c, control p and endpoint e, use

`delta(c,p,e) = BA(c,MIN-selected-family retuned on p,e) - BA(c,historical C-SELECTED MIN,e)`.

Within a context, balanced accuracy averages recalls of the true classes originally present in its test role. Average context effects equally within each domain over the fixed reference-support set, then average the 13 domain effects equally. Positive values favor the control. Report proportions and percentage-point changes, not relative-percent improvements. Model families may differ across contexts by design; this is an effect on the frozen selection procedure, not a pure within-one-family global comparison.

The pooled-fold sensitivity uses the fixed 57 complete groups. Concatenate the four held folds within each domain/repeat, score that repeat once, average the available registered complete repeats equally within its domain, then average domains equally. Never pool across repeats. This estimator changes class weighting and uses 228 contexts rather than 252; label both differences. For M06, combine probabilities within master/instrument within each repeat before scoring. Different folds may legitimately use different frozen families.

## 4. Contrasts, uncertainty and multiplicity

The [inference registry](contracts/p08_normalization_inference.json) fixes eight contrasts: four controls × two endpoints. All eight belong to one exploratory normalization-effect family, including the destructive derivative control. These contrasts refer to the fixed 252-context set. There is no neural interaction family, new primary hypothesis or extra inferential family for the pooled-fold sensitivity. The existing universal, QC, interaction and range families are unchanged.

Apply the P08-A02 positive-weight procedure with exactly 10,000 draws. Reuse the P08 global float64 Exponential(mean=1) arrays with shapes (10,000, 69) and (10,000, 10), lexicographically ordered master and instrument columns, and PCG64 seeds 2026093001 and 2026093002. Share those arrays across all controls, MIN, both endpoints and the pooled-fold sensitivity; subsets index the same columns without redraws. No array is generated during readiness.

Report crossed master/instrument, master-only and instrument-only weighting. Follow [P08 statistics §§4–5](P08_STATISTICAL_PROTOCOL.md#4-paired-support-preserving-uncertainty): weighted recall within each original context/class cell, equal class and context weights, and instrument-weighted domain means. Unit weights must reproduce independently calculated point estimates to absolute tolerance 1e-12. Use at most 128 draws per scoring batch and linear 0.025/0.975 quantiles for marginal 95% intervals. Reject nonpositive/nonfinite weights and nonfinite arithmetic without redraw or repair.

The intervals are conditional on saved fits, MIN family choices, new source-only choices and reference availability. They do not propagate family selection, retraining, historical missingness or acquisition of new instruments. A degenerate interval is labelled as such, not interpreted as certainty. BCa remains unavailable because the required crossed-cluster acceleration is unspecified.

Retain the original 10,000-draw hierarchical feasibility check, using PCG64 seed 2026093003 reset per contrast/endpoint and the inherited P06/P11 draw order on the declared support. Carry both methods together. If an originally represented context/class cell becomes empty, the draw is undefined; do not drop, refill, retry or change its denominator. An unconditional interval requires every planned draw to be defined. Otherwise report `hierarchical_fixed_support_undefined`; conditional intervals over surviving draws do not replace that result.

Enumerate all 8,192 domain-sign and 1,024 instrument-sign assignments on the fixed 13-domain/ten-instrument support, including the observed assignment. Use the inherited absolute equal-domain effect statistic, two-sided comparison and equality tolerance 1e-12. An instrument shares its sign across stations. Apply Holm adjustment across all eight registered contrasts, separately for each sign scheme. A planned unavailable contrast contributes p = 1 only to adjustment bookkeeping; its estimate and raw p-value remain missing. These symmetry-based sensitivities are descriptive, not randomized-treatment tests. Bootstrap tail fractions are not p-values, and marginal intervals do not bypass multiplicity.

## 5. Additional missingness and reporting

Every registered reference-support context requires the full test role, complete technical-seed ensemble, valid source selection and calibration. A new failed fit, absent calibration, nonfinite prediction or missing row makes the corresponding fixed-support contrast unavailable. Do not silently redefine the 252-context analysis using the remaining successes. The pooled-fold sensitivity independently requires every registered cell in its 228-context set; incomplete groups are not relabelled as complete.

A further complete-paired-context result may be shown only as a separately labelled descriptive sensitivity, with exact denominators, reasons and changes in model-family composition. It cannot replace the fixed-support effect or its Holm slot. A cross-control comparison on that reduced set requires a common intersection across MIN and all four controls, not four different pairwise subsets. Do not report reduced-support p-values as the registered family tests. No MIN fallback, family substitution, partial-seed average or automatic retry is authorized.

Publish paired domain scores, mean/median/quartile/range effects, positive/negative/tied domain counts, lowest-domain BA for each procedure and the minimum paired effect. Include leave-one-domain, instrument-identity and known-platform-family-out diagnostics without refitting. Report NLL, Brier score, macro-F1, confusion and class recall descriptively using inherited definitions; held labels cannot tune a temperature or a threshold.

Input validity follows the frozen normalization audit: SNV and derivative inputs are standardized, vector inputs have unit L2 norm, and area inputs have unit nonnegative area. They are not all [0,1] inputs. Do not enforce universal min–max invariants on these controls or interpret differences in signed/scaled arrays as independently verified chemical preservation. Spectral diagnostics remain relative to recorded spectra, not clean chemical ground truth.

Figure P08-F10 will show paired MIN-versus-control domain scatter with an identity line, faceted by control and endpoint, and conditional effect intervals where defined. Its caption will name the reference-supported population, missing historical endpoints and MIN-selected-family conditioning. Use native TikZ, offline HTML, vector PDF and PNG from one semantic table, black standard fonts and no private row identifiers or individual spectra. This specification creates no result figure.

## 6. Remaining execution gates

This document locks N2's numerical comparison rules without changing P08-A02 or the primary statistical families. Declaration tests are not numerical inference implementation, validation of an N2 training runtime or permission to execute. The operation graph still contains 42,368 prospective model fits and 1,040 scalar calibrations. The separate 24-hour/20-GiB N2 resource proposal remains unapproved. Complete runtime review, input authentication and fresh capacity checks precede a separate scientific request; universal preprocessing still runs first.
