# P08 preprocessing comparisons and uncertainty

**Version:** 1, with the dated P08-A05 support amendment below. **Specified:** 2026-09-30. **Authority:** planning only, under owner decisions P08-A01–P08-A05. **Scientific execution authorized:** none.

This specification tests whole preprocessing pipelines on the frozen held-instrument splits. It does not assume that minimal scaling is optimal. The previous minimal-input benchmark and its uncertainty results have already been examined. These rules are prospective for the new preprocessing outcomes, not a preregistration of the earlier benchmark. The [readiness contract](contracts/p08_readiness_contract.json) records the approved amendments; historical contracts remain unchanged.

## 1. Population, pipelines and model identities

The primary population contains 598 spectra from 69 physical masters. Held evaluation uses 557 distinct spectra, all 69 masters and ten instruments, arranged in 13 station–instrument domains and 260 outer contexts. A context is one domain, repeat and fold. The 60 development contexts and four exploratory domains are different analyses and do not enter these primary policy effects.

Compare `PP-U-SG` and `PP-U-ARPLS` separately with `PP-U-MIN`. Use the immutable P01 arrays and exact operation order in [P08 readiness](P08_READINESS.md#2-what-the-three-preprocessing-actions-actually-do). Both nonminimal pipelines include impulse replacement. Their effects cannot be attributed solely to smoothing or baseline subtraction.

The universal panel comprises RBF-SVM, Random Forest, Extra Trees, D0-M and P05-SELECTED. P05-SELECTED is the frozen context-local source-selected recipe, not a separately specified architecture. Keep each context's recipe identity fixed across policies. Where it equals D0-M, both strategy names reference the same predictions; they are not independent fitted models. Architecture, loss, sampling and seed specifications are inherited, not optimized using new held results.

Classical hyperparameters are selected anew within each policy, using the same source-only grid and roles. Neural stopping and scalar calibration use only that policy's source-validation predictions. Training permits 30–200 epochs and patience 20. An inner best checkpoint can precede epoch 30: the minimum constrains stopping, not checkpoint eligibility. Final duration uses the inherited per-seed median of inner best epochs, Python rounding and clipping to [30,200]. No held outcome chooses a duration.

The adaptive evaluation panel excludes Extra Trees unless separately approved. Its four methods are RBF-SVM, Random Forest, D0-M and P05-SELECTED. The policy-development panel has equal RBF-SVM and D0-M weight. Neither an adaptive policy nor its source thresholds may be chosen from target spectra, target-population statistics or held outcomes. The numerical selection and routing ledger must pass its separate review before adaptive execution.

## 2. Predictions and point estimates

Authenticate every prediction against the complete registered test role, class vocabulary, model specification, input action, seed and source-selection/calibration evidence. Equal row counts or matching scores do not establish equivalence. No convenient intersection of available rows replaces the registered test set.

Preserve the existing calibration order. For Random Forest and Extra Trees, average uncalibrated seed probabilities first, convert that average to the inherited score representation, and apply the single source-fitted temperature. RBF-SVM uses its deterministic score and single temperature. For neural models, apply the source-fitted temperature for each seed and then average the three calibrated probability vectors. Averaging separately calibrated forest predictions would be a different procedure and is prohibited.

- **M01, individual spectra:** each registered spectrum contributes one final probability vector and one class decision.
- **M06, combined predictions:** within a context, average probabilities within each physical master/instrument, then equally across that master's instruments. Make one class decision per master. This combines model predictions; it does not average spectra before classification. In a held-instrument context only one instrument contributes.

Argmax uses the frozen sorted class vocabulary and inherited tie behavior. Neither endpoint pools predictions across outer contexts or averages hard class labels.

For each context, calculate recall for every true class originally represented in its test role and average those recalls equally. This is context balanced accuracy (BA). Average context BAs equally within each domain, then average domain BAs equally. Originally absent classes remain absent; do not insert a nominal three-class denominator. Repeated folds, seeds and measurements do not create new independent masters.

For model m, policy p and endpoint e, the domain effect is

`delta(d,m,p,e) = BA(d,m,p,e) - BA(d,m,MIN,e)`.

The reported mean effect is the equal-domain mean of these paired differences. Positive values favor the nonminimal policy. Report proportions and percentage points consistently: a difference of 0.03 is three percentage points, not a 3% relative increase.

For deep strategy n and classical model c, the interaction is

`interaction(d,n,c,p,e) = delta(d,n,p,e) - delta(d,c,p,e)`.

All four cells must use the same complete context/test support and draw weights. A positive interaction says preprocessing helped the deep strategy more, or harmed it less. It does not establish that the deep strategy has higher absolute accuracy or removed a physical nuisance.

**Pooled-fold sensitivity:** within each domain/repeat, concatenate its four disjoint outer-test folds and calculate BA once from the pooled prediction units. Average repeats equally, then domains equally. For M06, combine probabilities within master/instrument within that repeat before scoring. This changes class weighting relative to the main equal-context estimator; it is not a correction of that estimator. Require all four folds. Do not pool across repeats or treat repeated appearances as new samples.

## 3. Contrast registry and multiplicity

No new primary superiority hypothesis is created. The completed benchmark's primary comparison and G4 rules remain unchanged. P08 comparisons are prespecified secondary analyses.

| Family | Fixed entries | Interpretation |
|---|---:|---|
| Universal policy effects | 5 methods × 2 nonminimal policies × 2 endpoints = 20 | Policy minus MIN for each method |
| QC policy effects | 4 methods × 1 QC policy × 2 endpoints = 8 | Operational, fallback-inclusive QC minus MIN |
| Policy–model interactions | 24 universal + 8 QC = 32 | Two deep strategies versus each applicable classical method |
| Platform-family policy endpoints | 4 methods × 2 endpoints = 8 structural aliases | Complete MIN fallback throughout; no supported transfer test |

The 24 universal interactions use two deep strategies, three classical methods, two nonminimal policies and two endpoints. The eight QC interactions use two deep strategies, two classical methods and two endpoints. The interaction family retains all 32 slots even if QC execution occurs later. Family-policy interactions are structural zero aliases, not an additional empirical test family.

Apply Holm adjustment within each empirical family, separately for domain-sign and instrument-sign sensitivities below. A planned but unavailable entry contributes p = 1 to adjustment bookkeeping only; its reported estimate and raw p-value remain missing with a reason. Do not present that bookkeeping value as a measured test. Structurally identical selected/D0 cells remain visible; do not reduce multiplicity after seeing results. Family aliases receive `structural_identity_no_transfer_test`, not evidence-based significance.

The 95% percentile intervals below are marginal, not simultaneous intervals. Do not use whether one excludes zero to bypass the multiplicity definition. Robustness, adaptation and exploratory branches do not enter these families. They require their own numerical specifications and finite execution permits before outcomes.

## 4. Paired, support-preserving uncertainty

Use exactly 10,000 draws with the positive-weight procedure approved in P08-A02. Generate independent Exponential(mean=1) arrays for the fixed, lexicographically sorted global lists of 69 master IDs and ten instrument identities. Use NumPy `Generator(PCG64(2026093001))` for masters and `Generator(PCG64(2026093002))` for instruments. Generate each complete array once and share it across policies, models, endpoints, contrasts and supported-subset sensitivities. Subsets index the global arrays; they do not regenerate weights. Seeds do not depend on outcomes, row order or Python hash.

Within an original context/class cell, weighted recall is the sum of master-weighted correct prediction units divided by the sum of master weights over those units. A master's weight is shared across its spectra, repeated contexts, domains and all methods. M01 retains each stored spectrum; M06 contributes one master/context prediction. Average classes and contexts as in Section 2. Average domain scores using their shared instrument weights, normalized over the fixed supported domain set. Instruments appearing in several stations retain the same weight.

Compute crossed master-and-instrument weighting, master-only weighting and instrument-only weighting from the same arrays. Set the unused factor to one for the one-factor sensitivities. The unit-weight calculation must reproduce the independently computed point estimate to absolute tolerance 1e-12. The same rule applies to interactions and pooled-fold sensitivities. Calculate in batches of at most 128 draws; batching must not alter results.

Reject nonfinite or nonpositive weights, inconsistent identities and nonfinite score arithmetic as execution failures. Do not redraw or silently repair. Report empirical 0.025 and 0.975 quantiles with NumPy `method="linear"`. For an interaction, combine the four cell scores within each shared draw, not their separately calculated interval endpoints.

These are approximate uncertainty intervals conditional on the saved fits, selected recipes, source decisions and observed support. They do not include uncertainty from retraining, resampling the source-selection process or observing new instruments. Positive weights retain sparse class support but cannot create information in singleton cells. A point interval is labelled `degenerate_distribution`, not certainty. BCa is unavailable (`crossed_cluster_acceleration_not_specified`).

## 5. Original hierarchy and descriptive stability

Retain the original hierarchical method as a feasibility analysis, with 10,000 planned draws and `Generator(PCG64(2026093003))`. Reset this stream per contrast/endpoint, sort domains, classes and master IDs, and use the draw order in [P06/P11 protocol §4](P06P11_INFERENCE_PROTOCOL.md#4-original-hierarchical-sensitivity-and-feasibility): draw the full domain-index array first, then domain occurrences and within-class multinomial master counts. Carry paired methods together. For an interaction carry all four cells together. This hierarchy does not preserve the sharing of masters across domains; report that limitation.

If any originally represented context/class cell becomes empty, mark the occurrence and draw undefined. Do not drop the cell, change the denominator, replace it with zero, omit its context or retry. Report planned, defined and undefined counts and affected domains. Issue an unconditional percentile interval only when every planned draw is defined. Otherwise report `hierarchical_fixed_support_undefined`; conditional quantiles of surviving draws are not the original interval. No BCa interval is reported without an appropriate multilevel acceleration specification.

Publish all paired domain scores and effects, positive/negative/tied domain counts, mean, median, quartiles, minimum and maximum. Use linear quantiles and equality tolerance 1e-12. Include leave-one-domain-out, leave-one-instrument-identity-out and leave-one-known-platform-family-out effects. Deletions recalculate the mean over retained domains without refitting. Unknown families are not treated as one biological or technical family; their exclusion is covered by instrument-identity deletion. A deletion leaving no supported domain is undefined.

Enumerate domain sign assignments and instrument-identity sign assignments, sharing an instrument sign across its station domains. Use the absolute equal-domain effect as the two-sided statistic and tolerance 1e-12 for ties. Include the observed sign assignment; no Monte Carlo correction is needed. For complete support this is 2^13 and 2^10 assignments respectively. Shared masters and observational acquisition conditions prevent a claim of independent, randomized treatment assignment. These are symmetry-based descriptive sensitivities, even after Holm adjustment. A bootstrap tail fraction is not a null p-value.

## 6. Completeness, support and fallback

The fixed-family universal panel targets all 260 contexts. The historical C-SELECTED comparator with only 252 complete contexts is not silently substituted for any of these fixed families. Its eight missing contexts remain recorded in historical summaries.

A failed fit, missing seed, nonfinite prediction, missing calibration or incomplete registered row set makes the affected method/policy/context unavailable. Do not replace it with MIN unless a specific policy fallback authorizes that action. Do not average the remaining seeds. A valid model predicting only one class remains a valid, possibly poor, result rather than a missing run.

Report all planned context/row counts, failures and reason codes. The full-support headline effect is unavailable if a required cell is missing. A complete-paired-context sensitivity may be reported separately with its exact domain/context/master/spectrum denominator and deletion pattern; it is not a successful complete benchmark. For interactions use the common support of all four cells, not two different pairwise intersections. No automatic scientific retry is implied.

Approved complete-pipeline family fallback applies to all 260 current held contexts. Once the complete MIN evidence is authenticated, the operational family prediction is an alias of that baseline. Its paired difference is zero by construction. Supported-family-only performance is unavailable (`no_supported_held_family`), not zero, and there is no identifiable family-transfer effect. Source-family support alone cannot change this conclusion or authorize mixed-source-family training.

**QC support amendment P08-A05, approved 2026-09-30:** the 128 contexts with source pseudo-domain support are an initial eligibility set, not the final adaptive subset. Independent estimator tuning inside each policy-validation fitting role requires three-fold, master-separated selection. Every registered pseudo-domain fitting role must retain at least three distinct masters per class. The metadata audit finds 54 eligible contexts; the other 74 fail nested class support. Together with the original 132 insufficient-pseudo-domain contexts, 206 therefore use the complete MIN pipeline, including its minimal-trained estimator. No failed pseudo-domain unit is silently dropped to qualify a context.

Retain all 260 contexts in operational QC results, and report the metadata-defined 54-context adaptive subset separately. The previously specified 128-context supported-subset wording is superseded by P08-A05 before any new QC outcomes. Eligibility does not depend on which gate wins; choosing the minimal-only gate in an eligible context does not make it unsupported. Missing/nonfinite permitted QC or an invalid selected action uses the specified row-level MIN fallback. Publish fallback reason, routing coverage and every source-frozen selection hash. Do not restrict the main result to rows helped by a nonminimal action. The three-fold support rule does not make the inner estimator folds instrument-independent: those inner folds group physical masters, while the outer source pseudo-domain evaluates transfer.

All 54 eligible contexts are CWA contexts. The pills and surfaces stations contribute only complete MIN fallbacks under this approved rule. Report that station restriction beside every supported-subset result; do not infer adaptive-policy effectiveness at unsupported stations. The operational equal-domain estimand remains unchanged and includes these fallback-only domains.

Report action fractions and fallback fractions per context, then equally per domain. Report between-repeat gate/action variability as technical selection stability, not new independent evidence. Conditional master/instrument-weight sensitivity for routing fractions can share the Section 4 weights, but it does not measure uncertainty from relearning a gate. A mixed-route QC input is a model-specific training/evaluation procedure; it cannot be replaced by selecting among already trained universal predictions without separate approval.

## 7. Weakest domains, probability quality and preservation

For every paired comparison report both models' lowest domain BA on the same support and their difference. Also report the minimum paired domain effect; it need not occur in either model's lowest-scoring domain. Identify ties and report all affected domains. A favorable mean does not conceal deterioration on one instrument. Weighted versions recompute each minimum within the shared draw and remain conditional diagnostics, not proof of a worst-case guarantee.

Report supported macro-F1, NLL, Brier score, confusion and class recall using the inherited definitions and complete support. Reliability views use ten fixed equal-width confidence bins, with an inclusive final bin. Distinguish pooled repeated appearances from equal-context calibration summaries. Do not fit a new calibration from held predictions or change thresholds to improve a plot.

Preservation is not clean-spectrum recovery. First audit exact row order, grid, finite values, [0,1] scaling, validity and absence of unauthorized extrapolation. A violated representation invariant is a fatal input defect. Then report the existing P01 diagnostics: shape correlation, spectral angle, rank correlation, first-difference variation, baseline-span proxy, changed-point fraction, boundary-value fraction, reference-peak recall and peak displacement. Use saved P01 definitions and evidence; no new threshold is chosen from classifier performance.

The inherited peak diagnostic smooths reference and candidate with SG 11/3, uses reference prominence 0.04 and distance 4, candidate prominence 0.025 and distance 3, up to ten reference peaks, and a match tolerance of 5 cm⁻¹. An absent reference peak set gives an undefined peak metric, not perfect preservation. The diagnostic is relative to interpolated observations, not independently measured chemical peaks. Report distributions, lower tails and representative overlays by instrument; record support rather than interpreting missing instrument/substrate combinations as losses.

There is no validated numerical chemistry-preservation cutoff in the inherited record. Therefore separate hard representation-invariant violations from continuous preservation diagnostics. Do not invent a binary chemical-preservation pass, a destructive-transform threshold or a claim that fluorescence and analyte signal were causally separated. A new chemistry-based acceptance threshold requires external evidence and a versioned decision before use. The derivative control remains a declared destructive control, not a policy candidate.

## 8. Reporting and execution boundary

Publish every prespecified effect, including negative effects and unavailable results. The figures in [P08_FIGURE_PLAN.csv](P08_FIGURE_PLAN.csv) will pair spectra with domain scatter, paired effects, model interactions and QC-routing views. Native TikZ and offline HTML must use the same reviewed semantic tables as PDF/PNG. Private individual spectra and identifiers do not enter public figures; aggregate publication needs its own boundary review.

No P08 policy is promoted by a held-test maximum. No interval here replaces G4's original hierarchy requirement or supplies its missing evidence. Predictive gains do not establish chemical/nuisance disentanglement or instrument-independent substrate chemistry. The inference code, launch ledger, hash-proven reuse, resource ceilings and synthetic checks must be reviewed before a separate execution permit. This document authorizes zero fitting, calibration, prediction generation, resampling, perturbation or preprocessing rebuilds.
