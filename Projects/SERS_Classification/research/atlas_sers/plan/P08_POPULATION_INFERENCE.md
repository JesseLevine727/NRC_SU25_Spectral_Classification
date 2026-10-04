# P08 filtered-population comparisons

**Date:** 2026-10-04. **Status:** pre-outcome specification, conditional on the outstanding four- versus five-method panel clarification. This document authorizes no fitting, prediction, calibration or resampling. It specifies how the approved regenerated populations will be compared; numerical implementation and finite resources remain separate gates.

## 1. Question and model selection

The notes-clear and Mira-1-excluded sensitivities ask whether the preprocessing conclusions depend on the recorded dataset filters. They retain 500 and 575 spectra, respectively, from the same 69 physical masters. Neither tier replaces the 598-spectrum primary population. Removing repeated measurements changes the available source information and held-domain coverage even when every physical sample remains.

Under P08-A07, each filtered population receives fresh, registered source-only classical tuning and MIN neural-recipe selection. Freeze that population's context-local neural recipe across MIN, SG and arPLS. Do not copy the primary recipe choices, winning classical hyperparameters or fitted estimators. Each policy retains its own source-only stopping and calibration. Source neural calibration uses inherited selection units, not the additional guard units. A structural ordinary-CNN fallback still requires a valid completed selection/readiness record; failed or unknown selection does not trigger that fallback.

The four-method alternative contains RBF-SVM, Random Forest, D0-M and P05-SELECTED. The five-method alternative adds Extra Trees. The [registry](contracts/p08_population_inference.json) and [operation catalog](../results/p08_readiness/population_slot_ledger_audit.json) retain both alternatives until the owner resolves the panel. Neither is silently activated. This clarification does not change the approved primary universal five-method panel or reopen P08-A07.

The minimal benchmark's outcomes have already been examined. These rules precede new filtered-population outcomes; they are not retrospective preregistration of the earlier benchmark. The sensitivities cannot establish a causal chemical-cleaning effect, instrument-independent substrate chemistry or passage of the original G4 superiority gate.

## 2. Fixed, metadata-defined comparison sets

The [support audit](../results/p08_readiness/population_inference_support_audit.json) authenticated 32 input files and checked all 260 domain/repeat/fold mappings for each population against the original held roles. Every filtered held role equals the corresponding primary role intersected with that population's recorded observations. Recorded labels, instruments and physical masters are unchanged. No spectral arrays, scores or probabilities were loaded.

| Population and comparison | Fixed contexts | Complete four-fold groups | Pooled contexts | Domains | Instruments |
|---|---:|---:|---:|---:|---:|
| Primary metadata reference | 260 | 65 | 260 | 13 | 10 |
| Notes-clear: classical effects and common model comparisons | 215 | 52 | 208 | 11 | 10 |
| Notes-clear: neural effects | 216 | 53 | 212 | 11 | 10 |
| Mira-1 excluded: all methods and common comparisons | 240 | 60 | 240 | 12 | 9 |

All listed sets retain 69 distinct masters. Notes-clear retains 449 distinct held spectra in each set; Mira-1 exclusion retains 534. A context is one held domain, repeat and fold, not an independent sample. The primary row describes role support, not a claim that all historical C-SELECTED predictions exist.

Notes-clear has 220 pooled-eligible contexts. Five lack registered classical calibration support, while four lack neural selection support. The common classical/neural set therefore contains 215 contexts. Its unavailable contexts occupy three domain/repeat groups, not five; requiring all four folds leaves 52 of 55 groups. Neural support leaves 53 complete groups. Both pooled sets retain every one of the 11 eligible domains, although not equally many complete repeats per domain. Mira-1 exclusion has 60 complete groups with no metadata exclusions among its 240 eligible contexts.

Freeze the context-set and complete-group hashes in the audit before new outcomes. Report all 260 context audit records per tier, including pooled-ineligible domains and method-specific unavailable roles. Distinguish these structural exclusions from later numerical failures. The eight missing historical primary C-SELECTED endpoints do not define support for fresh population-bound fits and are not repaired by this work.

## 3. Endpoints, paired effects and interactions

Use the prediction authentication, vocabulary, tie rules and calibration order in [P08 statistics §2](P08_STATISTICAL_PROTOCOL.md#2-predictions-and-point-estimates). M01 gives one class decision per stored spectrum. M06 averages the calibrated probability vectors for repeated measurements of a physical master before one class decision; it does not average the spectra. Each held context contains one instrument. Do not combine predictions across outer repeats.

For notes-clear, the classical/common fixed set contains 2,135 spectrum/context appearances and 1,033 master/context prediction units. The neural fixed set contains 2,152 appearances and 1,040 master/context units. Mira-1 exclusion contains 2,670 appearances and 1,225 master/context units. These repeated appearances do not increase the 69 independent physical-master units.

For each population, method, nonminimal policy and endpoint, calculate

`effect = BA(policy) - BA(MIN)`.

Score originally represented true classes equally within each context, average the fixed supported contexts equally within a domain, then average the domains equally. Report positive effects as benefits of the complete pipeline, in percentage points. Both nonminimal pipelines include impulse replacement; do not attribute the difference solely to smoothing or baseline subtraction.

Classical policy effects use that tier's classical set; neural policy effects use its neural set. A direct model comparison or neural-minus-classical preprocessing interaction uses the common set. For an interaction, recalculate all four cells on that same set:

`interaction = [BA(neural,policy) - BA(neural,MIN)] - [BA(classical,policy) - BA(classical,MIN)]`.

In particular, do not subtract a 215-context classical mean from a 216-context neural mean. A positive interaction means preprocessing helped the neural procedure more, or harmed it less. It does not establish higher absolute neural accuracy. P05-SELECTED and D0-M remain separate procedure labels even where they alias the same fitted model.

The pooled-fold sensitivity uses the method-specific complete groups in Section 2; interactions use the common complete groups. Pool exactly four disjoint test folds within a domain/repeat, score once, average the retained complete repeats equally within that domain, then average domains equally. Recombine M06 probabilities within the repeat before scoring. Never pool across repeats. This changes class weighting and, for notes-clear, context support; report both changes rather than calling it a correction of the main estimator.

## 4. Inference and multiplicity

| Conditional panel | Population policy effects | Population model–policy interactions |
|---|---:|---:|
| Four methods | 2 tiers × 4 methods × 2 policies × 2 endpoints = 32 | 2 tiers × 2 neural × 2 classical × 2 policies × 2 endpoints = 32 |
| Five methods | 2 tiers × 5 methods × 2 policies × 2 endpoints = 40 | 2 tiers × 2 neural × 3 classical × 2 policies × 2 endpoints = 48 |

Use one exploratory effect family across both tiers and one exploratory interaction family across both tiers, for the panel selected before new outcomes. Apply Holm adjustment within each family separately for the domain-sign and instrument-sign sensitivities. Retain structurally duplicated entries. An unavailable planned contrast contributes p = 1 only to adjustment bookkeeping; its estimate and raw p-value remain missing. Pooled sensitivities add no hypothesis family. The primary universal, QC, range and normalization families are unchanged.

Inherit P08-A02's 10,000 positive-weight draws. Reuse the global float64 Exponential(mean=1) arrays for 69 lexicographically ordered masters and ten instrument identities, generated with PCG64 seeds 2026093001 and 2026093002. Subsets index those same arrays without redraw; the Mira-1-excluded tier uses nine instrument columns. Share weights across tiers, methods, policies, endpoints and contrasts. Generate no weights during readiness.

Report crossed master/instrument, master-only and instrument-only intervals. Preserve original context/class cells, equal class and context weighting, and instrument-weighted domain means. Unit weights must reproduce independent point estimates to absolute tolerance 1e-12. Use batches of at most 128 draws and linear 0.025/0.975 quantiles. Reject nonpositive/nonfinite weights or nonfinite arithmetic without retry or repair. For interactions, combine all four cells within the same draw.

These marginal intervals condition on the saved fits, source selection and fixed observed support. They do not propagate retraining, source-selection or missingness uncertainty. Label degenerate distributions; they do not imply certainty. BCa remains unavailable because crossed-cluster acceleration is unspecified. Positive weights preserve sparse support but do not create chemical information.

Retain the original 10,000-draw hierarchical feasibility check with PCG64 seed 2026093003 reset per contrast/endpoint and the inherited P06/P11 draw order. Carry paired cells together. An empty originally represented context/class cell makes that draw undefined; do not refill, drop, retry or change the denominator. Every planned draw must be defined for an unconditional interval; otherwise report `hierarchical_fixed_support_undefined`.

Enumerate domain and instrument signs exhaustively, including the observed assignment, using the absolute equal-domain effect and tolerance 1e-12. Share each instrument's sign across stations. Notes-clear has 2,048 domain assignments and 1,024 instrument assignments; Mira-1 exclusion has 4,096 and 512. These counts also apply to their registered pooled supports. The signs give descriptive symmetry sensitivities, not randomized-treatment tests. Bootstrap tail fractions are not p-values.

## 5. Missingness, cross-population views and figures

A fixed-support contrast requires every registered test row, seed, selection and calibration record in its declared set. A new failure makes that contrast unavailable; do not silently reduce its denominator, average surviving seeds, substitute a different family or invoke an unregistered MIN fallback. The pooled check independently requires every cell in its fixed complete groups.

A further complete-case sensitivity may be shown descriptively with exact exclusions and denominators, but cannot replace a registered effect or its Holm entry. Cross-policy displays on reduced support require a common MIN/SG/arPLS intersection, not different successful subsets for each pair. Common model comparisons must also intersect all required model cells. Missingness is not assumed random.

Report domain effects, mean/median/quartiles/range, positive/negative/tied counts, lowest-domain BA for both procedures and the minimum paired effect. Include the inherited leave-one-domain, instrument-identity and known-platform-family-out diagnostics without refitting. Calibration, class recall and confusion summaries remain descriptive; held labels cannot select a temperature or threshold.

An optional descriptive primary-versus-filtered view must use the audited domain/repeat/fold bridge and exactly the filtered test observations on both sides. Restrict primary saved probabilities to those observations, then recompute M06 from the same retained views. Require complete, content-validated predictions for each displayed pair. No additional fitting or prediction jobs are added, and this view introduces no p-values or intervals. Whole-population means with different domain or row coverage are not paired filtering effects. Even a matched view changes source information and potentially source-selected recipes; it does not isolate physical cleaning.

Figure P08-F11 will show MIN-versus-policy domain scatter, faceted by population, method and endpoint, with an identity line and declared support. Use the common set for panels intended to compare methods directly; do not overlay unmatched neural and classical means. A companion matched-population scatter may show the descriptive comparison above. Native TikZ, offline HTML, vector PDF and PNG must use one reviewed semantic table, black standard fonts and no private row identifiers or individual spectra. This document generates no outcome figure.

## 6. Remaining gates

The panel clarification, finite resource proposals, numerical implementation review and execution authority remain outstanding. Declaration tests check agreement between documents and metadata; they do not validate a training or inference engine. Conditional operation ceilings remain 355,026 fits for the four-method alternative and 539,265 for the five-method alternative across both tiers. Neither ceiling is approved for execution. Universal preprocessing remains first, adaptive work second, and these later sensitivities require separate stage-specific permission.
