# NATO field-trial SERS: what the uncertainty analysis changes

**Analysis date:** 2026-09-30. Frozen-prediction analysis; no new model training. Release status and provenance are recorded in the [supervisor review](../plan/delegation/P06P11_REVIEW.md).

## Main result

The compact CNN remains competitive, but the additional training losses have not shown a reliable advantage over its matched ordinary classifier. Random Forest and Extra Trees remain strong alternatives. The new analysis checks the uncertainty of existing predictions; it does not train another model or change preprocessing.

The source-selected CNN exceeded the source-selected classical procedure by **4.36 percentage points** for individual spectra. The added sample-and-instrument weighting analysis gave an approximate 95% interval of **0.68 to 7.79 points**. This comparison is not against the strongest fixed classical model: the selected CNN's differences from Random Forest and Extra Trees were **+0.29** and **−1.19 points**, with intervals spanning zero.

These intervals condition on saved fitted models and the observed sample/instrument design. They do not include retraining uncertainty or establish performance on every new instrument. The analysis was explicitly added after the benchmark outcomes; it is not presented as the original registered bootstrap.

## What was compared

The dataset contains 598 spectra from 69 recorded physical-master grouping units. The held-instrument evaluation covers 557 distinct spectra and all 69 masters, across 13 station–instrument domains representing ten instruments. The other 41 spectra belong to four exploratory domains and are included in development, not this primary held evaluation.

Each domain has 20 repeated train/test contexts. Fitting excludes the test physical samples and test instrument. These repeated contexts are not new independent samples. Eight historical selected-classical references are incomplete or missing, so that paired comparison uses 252 of 260 contexts; no missing reference was filled in.

All models here use minimal per-spectrum min–max scaling over 400–1,800 cm⁻¹. No baseline-subtraction or smoothing comparison has yet been performed in this benchmark.

- **Individual spectra (M01):** predict each stored spectrum separately, using the previously saved three-seed mean neural probabilities.
- **Combined predictions per sample (M06):** average the class probabilities from a sample's repeat measurements, then choose the largest probability. The raw spectra are not averaged. In these held contexts, only one instrument is present; the established multi-instrument balancing rule applies when several instruments are present in development.

Balanced accuracy is the mean chemical-class recall within a test context, using its originally represented classes. Contexts are averaged equally within a domain; domains are then averaged equally. This prevents a densely measured domain from dominating the final score. M01 and M06 are different endpoints, not additional independent datasets.

## How large are the differences?

Entries are percentage-point changes, followed by the added crossed-weight interval. Positive values favour the first method. All intervals are marginal, conditional uncertainty summaries, not simultaneous confidence bounds.

| Comparison | Individual spectra | Combined sample predictions |
|---|---:|---:|
| Selected CNN − selected classical | +4.36 [0.68, 7.79] | +3.61 [−1.88, 8.33] |
| Combined-loss CNN (D3) − matched CNN (D0-M) | +1.68 [−0.09, 4.13] | +1.32 [−0.26, 3.50] |
| Selected CNN − matched CNN | +0.17 [−0.40, 0.69] | +0.04 [−0.55, 0.57] |
| Selected CNN − Random Forest | +0.29 [−3.48, 3.47] | −1.82 [−7.88, 1.99] |
| Selected CNN − Extra Trees | −1.19 [−5.19, 2.35] | −1.72 [−8.36, 3.90] |
| Combined-loss CNN − Random Forest | +1.80 [−2.32, 5.17] | −0.53 [−6.52, 3.84] |

The first row uses the 252 common complete contexts. The other rows use 260. The sole primary uncertainty contrast is the first row's individual-spectrum endpoint; the remaining comparisons are secondary or exploratory. All 17 registered comparisons at both endpoints must remain available in the released tables, not only this reading-order subset.

The combined-loss CNN adds supervised contrastive learning and same-sample cross-instrument consistency to ordinary classification. Its small mean benefit is not uniform: at M01, six domains improved, six worsened and one tied. At M06, seven improved, four worsened and two tied. Numerical ties are distinguished from floating-point residuals; no scientific score is rounded or changed to obtain these counts.

## What the new checks reveal

The primary selected-CNN comparison improves eight of 13 domain means at M01 and worsens five. Removing any one domain leaves a positive mean difference of 3.31–5.19 points; removing any one instrument identity leaves 3.36–5.19 points. The mean advantage therefore does not depend on a single domain or instrument deletion. These are sensitivity checks without refitting, not independent experiments.

Reweighting physical samples alone produces a much narrower primary M01 interval, 3.51–5.14 points, than reweighting instrument identities alone, 1.00–7.61 points. This indicates that the represented acquisition variation matters substantially in this calculation. It is not a variance-component decomposition: many tiny class cells contain only one or two samples, which limits what sample reweighting can reveal.

The original hierarchical procedure could not provide its planned interval. Every one of its 10,000 draws, for each of the 34 comparison–endpoint combinations, emptied at least one originally represented context–class group. The fixed score was therefore undefined. Omitting that group would change the score definition; repeatedly drawing until a score exists would condition on a selected subset. Neither was done. The new positive weights retain every observed group and share sample/instrument weights across repeated appearances, but cannot invent unmeasured chemistry or missing combinations.

Descriptive sign-flip checks give primary M01 values of 0.067 for domain signs and 0.072 for shared instrument signs. These are symmetry-based sensitivities, not confirmatory randomized tests: physical samples are shared across domains. They answer a different question from the weighted intervals. No secondary claim is promoted from a small unadjusted value.

## Decision and next steps

Four of the six original advancement criteria are supported: a mean gain of at least three points, improvement in at least eight domains, the specified weakest-domain tolerance, and unchanged inputs. Two remain unassessable: the original hierarchical interval and the chemistry-retention comparison against a selected-classical T1 procedure that was not defined in the inspected frozen registry. The added interval does not substitute for either missing criterion. **The advancement gate is therefore not passed.** This is not evidence that the methods are equivalent, and it does not show causal chemical–nuisance disentanglement or universal substrate independence.

The next experiment should test the preprocessing question directly. The planned sequence is universal minimal/SG/arPLS policies on matched classical and compact-neural models, followed by source-selected family-aware and row-quality rules. Policy-specific effects, changes in the relative model comparison, spectral preservation and weakest-domain behaviour must all be recorded. Model identities, support, finite execution budget and the new uncertainty family require a separate lock before training.

A useful research outcome does not require a new neural classifier to win. This dataset can support a carefully controlled account of acquisition shift, competitive classical methods, limited gains from extra neural losses, and preprocessing-dependent transfer. That remains the publication direction; novelty and venue positioning require a separate literature refresh.

## Confidence as well as classification

On the 252 common contexts, the selected CNN's individual-spectrum macro-F1 was **0.630**, versus **0.574** for the selected classical procedure. Macro-F1 balances precision and recall across each station's three declared chemicals; it differs from balanced accuracy, which averages recalls for chemicals represented in each context.

Probability quality gives a mixed picture. The selected CNN had a lower Brier score (**0.440 versus 0.507**; lower is better), but a higher negative log loss (**1.464 versus 0.964**; lower is better). Log loss penalizes assigning very low probability to the correct chemical strongly. The comparison therefore does not establish that the CNN's confidence estimates are uniformly better.

The [supplementary tables](../results/p06p11/release/metrics/model_summary.csv) retain both endpoints, all eight methods, confusion counts, per-chemical recall and reliability summaries. Confusion and reliability counts include repeated prediction appearances, not independent physical samples. A nominal sample unit can recur under several held instruments and split contexts: the M06 full-support total is 1,310 appearances of 262 master/domain units, not 1,310 independent samples. The independent grouping remains 69 physical masters.

Reliability tables use ten fixed-width confidence bins; the final bin includes confidence 1. Their pooled calibration error is not the same as the separately reported mean-context equal-mass ECE. Neither is a new calibration fit, and small test contexts make calibration summaries sensitive to binning.

## Figures to read first

1. [Domain scatter (HTML)](../results/p06p11/release/figures/F_P06_primary_scatter.html) · [PDF](../results/p06p11/release/figures/F_P06_primary_scatter.pdf). Each dot is one station–instrument domain. Above the diagonal favours the selected CNN; below favours selected classical. Both methods are evaluated on the same contexts. Colour and shape identify the three stations.
2. [Paired differences (HTML)](../results/p06p11/release/figures/F_P06_effect_intervals.html) · [PDF](../results/p06p11/release/figures/F_P06_effect_intervals.pdf). A dot is the mean difference; a horizontal line is the conditional weighted interval. A line crossing zero does not establish a directional advantage or equivalence. Blue marks the selected pair; grey marks secondary comparisons.
3. [Sample/instrument sensitivity (HTML)](../results/p06p11/release/figures/F_P06_weight_sensitivity.html) · [PDF](../results/p06p11/release/figures/F_P06_weight_sensitivity.pdf). The centre stays fixed while the uncertainty calculation changes. Instrument weighting widens the interval appreciably.
4. [Deletion scatter (HTML)](../results/p06p11/release/figures/F_P06_deletion_stability.html) · [PDF](../results/p06p11/release/figures/F_P06_deletion_stability.pdf). Each coloured point removes one domain or instrument identity; the black cross retains all paired support. Upper-right points mean that both endpoint differences remain positive. Some domain/instrument points coincide because those instruments occur in only one domain.

All four figures have editable native TikZ, semantic CSV and PNG counterparts beside the HTML/PDF files. The [release browser](../results/p06p11/index.html) links every format. Existing [held-spectrum reliability plots](../results/p05_comprehensive/release/figures/diagnostic_bundle/figures/P05D_reliability_held_evaluation_M01.html) remain descriptive companions; their documented pooling is not substituted for this release's common-support estimator.

## Evidence and reproducibility

The [frozen analysis specification](../plan/P06P11_INFERENCE_PROTOCOL.md) documents the post-benchmark amendment, fixed estimator, random streams, missing-cell rules and limits. The [release tables](../results/p06p11/release/tables/summary.csv) preserve all 34 comparisons. Seven secondary count rows distinguish numerical ties from floating-point residuals using the existing 1e-12 numerical tolerance; the original run is preserved, all score/interval values are unchanged, and [before/after counts](../results/p06p11/release/tables/tie_corrections.csv) are explicit.

The [P08 handoff](../plan/P08_HANDOFF.md) defines the next no-fit planning gate. It does not authorize training or designate minimal preprocessing as optimal. Broader chemistry-retention and exploratory neural evaluations remain incomplete; this release completes the scoped saved-prediction analysis, not the whole research programme.
