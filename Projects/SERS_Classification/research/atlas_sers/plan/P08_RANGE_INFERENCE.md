# P08 wider-range inference specification

**Date:** 2026-10-02. **Status:** pre-outcome numerical specification for the existing range sensitivity, not an execution permit. No range model, prediction, score or uncertainty draw has been calculated.

## 1. Question and evidence boundary

This sensitivity asks whether the complete 400–1,849 cm⁻¹ input changes identification relative to the complete 400–1,800 cm⁻¹ minimal input. It does not isolate the contribution of the additional 49 channels. Wider row-wise min–max scaling can change values inside the shared interval; neural adaptive-pooling boundaries also depend on input width.

The [input audit](../results/p08_readiness/range_input_audit.json) verifies the same 598 spectra, 69 physical masters, ten instruments and row order. The [operation audit](../results/p08_readiness/range_ledger_audit.json) binds the same 260 primary contexts and their source/test roles. The panel remains RBF-SVM, Random Forest, D0-M and the frozen context-local P05-selected recipe. Extra Trees, adaptive routing, normalization controls and regenerated populations are not added to this branch.

As in the primary protocol, held evaluation covers 557 spectra in 13 domains. The other 41 spectra belong to four declared exploratory domains: they remain in the dataset but are not new primary test observations. Repeated contexts do not increase the independent sample count.

Classical hyperparameters are selected within the existing source-only grid for the wider representation. Neural architecture/loss identities remain fixed; source stopping, refit duration and calibration follow the inherited procedure. No primary-input fitted estimator is applied to wider inputs, and no wider outcome changes the recipe map. Exact strategy aliases remain aliases, not independent fits.

The minimal benchmark and the P06/P11 analysis have already been inspected. This is a prespecified secondary sensitivity before new range outcomes, not a previously unseen primary hypothesis. Its separate families cannot establish overall superiority or pass the original G4 gate.

## 2. Endpoints and paired effects

Use the prediction authentication, calibration order, class vocabulary, tie behavior and scoring definitions in [P08 statistical protocol §§1–2](P08_STATISTICAL_PROTOCOL.md#2-predictions-and-point-estimates). M01 scores individual spectra. M06 averages prediction probabilities within physical master/instrument, then equally across that master's instruments before one class decision; it does not average spectra. A held-instrument context contains one instrument.

For domain d, method m and endpoint e, define

`range_delta(d,m,e) = BA(d,m,R_MIN_400_1849,e) - BA(d,m,R_MIN_400_1800,e)`.

First average the paired context balanced accuracies equally within each domain, then average the 13 domain effects equally. Positive values favor the complete wider-range pipeline. Report differences both as proportions and percentage points without calling them relative-percent improvements.

For neural strategy n and classical model c, define

`range_interaction(d,n,c,e) = range_delta(d,n,e) - range_delta(d,c,e)`.

Each interaction requires both methods at both ranges on the same complete registered context/test support. A positive value means the wider range benefits the neural strategy more, or harms it less. It does not necessarily mean that strategy is the more accurate classifier.

Retain the pooled-four-fold sensitivity from the primary statistical specification, without pooling repeats. Require all four folds within each domain/repeat. This sensitivity changes class weighting; it does not replace the equal-context estimate or create an additional inferential family.

## 3. Fixed contrast families

The [range inference registry](contracts/p08_range_inference.json) enumerates every contrast before outcomes.

| Family | Entries | Comparison |
|---|---:|---|
| Range effects | 4 methods × 2 endpoints = 8 | Wider minus primary minimal input |
| Range–model interactions | 2 neural strategies × 2 classical methods × 2 endpoints = 8 | Difference of the paired range effects |

Keep these families separate from the existing 20 universal effects, eight QC effects and 32 policy–model interactions. No primary family changes size. Apply Holm adjustment over all eight slots of each range family, separately for domain-sign and instrument-sign sensitivities. Do not reduce a family when a selected/D0 alias is structurally identical or a contrast is unavailable. An unavailable contrast contributes p = 1 to adjustment bookkeeping only; its estimate and raw p-value remain missing with a reason.

The sign sensitivities use the inherited two-sided absolute equal-domain statistic, equality tolerance 1e-12, exhaustive assignments and inclusion of the observed assignment. Complete support has 2^13 domain-sign assignments or 2^10 instrument-sign assignments; an instrument shares its sign across stations. On an explicitly labelled paired-support sensitivity, enumerate only its retained domains/instrument identities. Shared physical masters and observational acquisition conditions preclude a randomized-treatment interpretation.

## 4. Conditional uncertainty and stability

Apply the approved P08-A02 procedure using exactly the numerical settings in [P08 statistical protocol §§4–5](P08_STATISTICAL_PROTOCOL.md#4-paired-support-preserving-uncertainty). Reuse the same global arrays of 10,000 positive Exponential(mean=1) draws: PCG64 seed 2026093001 for the sorted 69 masters and seed 2026093002 for the sorted ten instruments. Share weights across both ranges, all four methods, both endpoints and all interactions. Subsets index these arrays; they do not draw new weights or reset the random stream per model.

The arrays have shapes (10,000, 69) and (10,000, 10): rows are draws, and columns follow the respective sorted identities. Generate each complete float64 array with the inherited `positive_weights` allocation order before batching scores. This clarifies the existing procedure; it does not reuse the earlier analysis's different random seeds or generate weights during readiness.

Report crossed master/instrument weighting, master-only weighting and instrument-only weighting. Average classes, contexts and domains in the inherited order. Recompute each four-cell interaction within its shared draw. Unit weights must reproduce independent point estimates within absolute tolerance 1e-12. Use batches of at most 128 draws, marginal 95% percentile intervals at 0.025 and 0.975 with NumPy's linear quantiles, and no redraw or nonfinite-value repair. These are not simultaneous intervals or an alternative significance test.

Retain the original 10,000-draw hierarchical feasibility check with PCG64 seed 2026093003, reset per contrast/endpoint, in the inherited draw order. Carry the pair or all four interaction cells together. An empty originally represented context/class cell makes its occurrence and draw undefined; do not omit, retry or change its denominator. Report an unconditional interval only if every planned draw is defined. Otherwise retain `hierarchical_fixed_support_undefined`. BCa remains unavailable because a crossed-cluster acceleration is unspecified.

Report point, mean, median, quartiles, range, positive/negative/tied domain counts and leave-one-domain/instrument/platform-family-out stability as specified in the primary protocol. Include lowest-domain BA for each pipeline and the minimum paired domain effect; these are different quantities. No deletion refits a model. The intervals remain conditional on saved fits and observed support: they do not include retraining, model-selection or new-instrument uncertainty. A degenerate interval does not imply certainty.

## 5. Missingness and secondary diagnostics

Each full-support range effect targets all 260 contexts; an interaction targets all four required cells in every context. A missing seed, failed fit, absent calibration, nonfinite output or incomplete registered test role makes the affected cell unavailable. A finite model predicting only one class remains valid. Neither complete-MIN fallback nor replacement by a different model is authorized for a failed range cell.

If any required cell is missing, the corresponding full-support headline contrast is unavailable. A separately labelled complete-paired-context sensitivity may report its exact retained domains, contexts, masters and spectra, with every exclusion reason. Interactions use one common four-cell support, not two different pairwise intersections. The eight missing historical C-SELECTED endpoints do not define this fixed-family range panel and are not repaired by it.

Paired-support sign sensitivities remain separately labelled diagnostics; they do not fill missing full-support Holm adjustment slots.

Probability-quality, confusion and class-recall summaries follow [P08 statistical protocol §7](P08_STATISTICAL_PROTOCOL.md#7-weakest-domains-probability-quality-and-preservation), using each contrast's stated support. They remain descriptive and do not add unregistered hypothesis families. Do not tune calibration or thresholds on held predictions. Representation validity follows the authenticated range input; unequal-width shape diagnostics must not silently truncate, align or renormalize the arrays to manufacture a preservation score.

Figure P08-F09 will show primary versus wider-range domain balanced accuracy with the identity line, faceted by method and endpoint. Add paired domain differences and the specified conditional intervals when available. Use native TikZ, offline HTML, vector PDF and PNG from the same semantic tables, with black standard fonts. Keep individual spectra, row identities, checkpoints and source paths private. No result figure is generated by this specification.

## 6. Release and execution gates

This document supplies the previously missing range-specific contrasts, multiplicity, missingness and uncertainty definitions without changing the primary statistical families or the owner's approved weighted amendment. The machine-readable registry must agree with this text and the authenticated four-method operation graph.

Numerical inference implementation and synthetic verification still require review. The separate R1 resource proposal remains unapproved: 62,981 prospective model fits and 1,417 scalar calibrations are accounting limits, not completed work or permission. Input/specification authentication, runtime acceptance, a fresh resource check and a separate scientific permit precede any range execution. No scope decision for normalization, population or perturbation is inferred from this range specification.
