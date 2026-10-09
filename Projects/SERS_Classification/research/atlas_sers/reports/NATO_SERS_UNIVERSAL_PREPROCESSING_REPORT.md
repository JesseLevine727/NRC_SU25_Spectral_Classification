# NATO field-trial SERS: preprocessing and unseen-instrument classification

The baseline-correction pipeline improved average classification accuracy across all five methods. Smoothing gave small, mixed changes. These results support further testing of baseline correction, not a claim that one pipeline is best for every instrument.

## Scope and data

The frozen benchmark contains 598 spectra from 69 physical samples, 10 instruments, and 3 stations. The primary held evaluation covers 557 spectra, all 69 physical samples, all 10 instruments, and 13 station-instrument domains. The remaining 41 spectra form four exploratory domains held outside the primary analysis. Five repeats of four folds give 260 contexts; these are not 260 independent samples. Training excludes both the held instrument and the held physical sample identifiers, so no spectrum is scored by a model that saw its own instrument or sample. Hyperparameter selection, probability calibration, and stopping decisions used source data only. A total of 195,202 unique fit slots completed, including 41,568 verified reused slots and 153,634 new final-phase slots, with 3,354 scalar calibrations. Many fits are cross-validation, seed, or grid jobs and are not independent chemical examples. No retraining occurred after evaluation.

## Preprocessing arms

All inputs were linear-interpolated to 400–1800 cm⁻¹ on 1401 channels, then per-spectrum min–max scaled to [0,1] as the final step. Three primary arms were compared:

- **MIN** — interpolation plus scaling only.
- **SG** — impulse replacement, Savitzky–Golay smoothing (11-point window, degree 3), plus scaling.
- **arPLS** — the same impulse replacement, arPLS baseline correction (λ = 100,000, at most 12 iterations, tolerance 0.001), plus scaling.

There was no combined SG-plus-arPLS arm. Both nonminimal pipelines include impulse replacement, so differences from MIN cannot be attributed solely to smoothing or baseline subtraction.

## Classifiers

Five strategies were evaluated: a radial-basis-function support vector machine (RBF-SVM), Random Forest, Extra Trees, an ordinary compact convolutional neural network (CNN), and a source-selected CNN recipe fixed separately for each split. Classical settings were reselected within each preprocessing pipeline using source data only. CNN recipe identities stayed fixed; training duration and probability calibration used source data. The selected recipe can equal the ordinary CNN, so these strategy labels do not always represent independent fits. Neural training allowed 30–200 epochs with patience 20. A best checkpoint could precede epoch 30; final training duration was source-selected and clipped to 30–200. Held-test results selected none of these settings.

## Metric

Balanced accuracy is the average fraction correct within present chemical classes, with contexts weighted equally within a domain and domains weighted equally. M01 uses one prediction per spectrum. M06 averages model class probabilities for repeat measurements of a physical sample, then chooses a class; it does not average spectra. Within a held context, one instrument is present. Results are shown as percentages to one decimal; differences may be quoted in percentage points to two decimals.

## Primary results

Table 1. Individual spectra: balanced accuracy (%).

| Model | MIN | SG | arPLS |
| --- | --- | --- | --- |
| RBF-SVM | 67.0 | 67.9 | 72.8 |
| Random Forest | 70.1 | 69.4 | 76.8 |
| Extra Trees | 71.6 | 70.6 | 77.9 |
| Ordinary CNN | 70.2 | 70.1 | 75.9 |
| Source-selected CNN | 70.4 | 70.4 | 76.1 |

Table 2. Combined model predictions per sample: balanced accuracy (%). The spectra themselves are not averaged.

| Model | MIN | SG | arPLS |
| --- | --- | --- | --- |
| RBF-SVM | 70.2 | 72.0 | 75.9 |
| Random Forest | 76.7 | 75.1 | 83.5 |
| Extra Trees | 76.6 | 75.4 | 84.3 |
| Ordinary CNN | 74.9 | 75.1 | 84.0 |
| Source-selected CNN | 74.9 | 75.5 | 84.2 |

## Interpretation

The arPLS pipeline improves mean unseen-instrument balanced accuracy for every method and both endpoints. M01 gains range from +5.70 to +6.70 percentage points; M06 gains range from +5.72 to +9.31 points. Extra Trees shows the highest observed primary means (77.9% M01, 84.3% M06), but this is not a selected winner for future deployment. After baseline correction the CNNs are competitive; the evidence establishes neither classical superiority nor deep-learning superiority. Every arPLS-versus-MIN deep/classical interaction interval includes zero: these comparisons do not establish that baseline correction benefits CNNs more than classical methods. Smoothing effects are small or mixed. The study is not proof of a universally best pipeline.

## Sensitivity and individual effects

A sensitivity analysis scores each repeat after pooling its four test folds. All arPLS gains remain positive, although absolute accuracy and model rankings change: combined source-selected CNN predictions reach 83.70% versus 74.07% with MIN, and Extra Trees reaches 83.03% versus 75.64%.

For Random Forest, the individual-spectrum gain is +6.70 percentage points, with a 95% conditional interval of [+2.51, +13.09]. Eleven of 13 domains improve; two worsen. The cwa station measured with Pendar-2 loses 11.20 points. The lowest domain accuracy rises from 37.39% to 49.58%, but that lowest-scoring domain changes from pills/Mira-3 to surfaces/Agilent-3. An improved average therefore does not mean every instrument benefited.

Random Forest also improves its individual-spectrum negative log-likelihood from 0.761 to 0.641; the ordinary CNN improves from 1.453 to 1.216. This metric penalizes incorrect, overconfident predictions; lower is better. Similar classification accuracy does not imply equally reliable probabilities.

## Uncertainty

The 10,000 positive-weight resamples are shared across physical samples and instruments, conditional on saved fits and support. They are not new-instrument or retraining uncertainty. All primary arPLS policy crossed intervals have lower bounds above zero, but most descriptive sign tests are not low after Holm correction. All 20 primary domain-sign Holm p-values exceed 0.05; only the Random Forest individual arPLS instrument-sign test reaches p = 0.0390625, and it does not hold in the pooled sensitivity (0.45703125). These sign tests are observational and descriptive, not randomized proof. The original hierarchy produced 10,000 of 10,000 draws undefined because of empty cells; the pooled sensitivity produced 10,000 defined draws under a different estimand, which cannot rescue the primary gate.

## Platform-family deletions

Preplanned leave-one-known-platform-family deletions were completed from saved paired domain effects across four frozen families (Agilent, Mira, Pendar, RMX): 352 supplement rows (2 estimands, 44 contrasts, 4 families). All primary arPLS policy means remain positive after deleting any single family, with the smallest at +1.9722 points (SVM combined without Mira). This does not show instrument-family-adaptive preprocessing. A technical audit found that a manifest lacked a derived family column and that original diagnostic maps were empty; the supplement completes omitted descriptives using a frozen mapping, with no change to fits, resampling, scores, or intervals.

## Preservation diagnostics

Preservation diagnostics are relative to observed spectra, not clean chemical truth. For example, median arPLS shape correlation is 0.252630 for pills: Mira-3 and 0.995142 for surfaces: Agilent-3, showing different degrees of change; neither measures chemical truth or proves damage. Spectral plots cover 49 cells, of which 46 are eligible (each with at least 2 masters), totalling 138 curves; within-master, then master-equal means are used for display only, without renormalization. Four exploratory domains are labelled; missing cells are not failed measurements. Training plots cover 6,402 expected neural fits, of which 6,366 are monitored; 36 reused pilot fit epoch histories are unavailable and shown as explicit missing values, not interpolated. The 24 groups include 8 structurally empty neural calibration groups. Training-curve bands (10th–90th percentile) describe variation across fits and epoch attrition, not chemical uncertainty.

## Evidence

Start with the [figure browser](../results/p08_universal/release/figures/index.html). Download the repository and open HTML locally; GitHub's file viewer does not execute interactive figures. Each panel also has native TikZ, vector PDF, PNG, and aggregate data.

- [Spectra before and after preprocessing](../results/p08_universal/release/figures/P08-F01/html/P08-F01-D08.html): pills measured with Mira-3; these are display averages, not model inputs.
- [Random Forest paired scatter](../results/p08_universal/release/figures/P08-F02/html/f02-equal-context-m01-c-random-forest.html): each point is a station–instrument domain. Above the diagonal means preprocessing helped.
- [Effects across all 5 methods](../results/p08_universal/release/figures/P08-F03/html/p08-f03-equal_context-M01-crossed.html): gray dots are the 13 domains; colored markers show the mean and conditional interval.
- [CNN-versus-classical preprocessing effects](../results/p08_universal/release/figures/P08-F04/html/p08-f04-equal_context-M01-D0-M-crossed.html): differences in benefit, not absolute accuracy.
- [Peak diagnostics versus classification](../results/p08_universal/release/figures/P08-F07/html/f07-equal-context-m01-c-random-forest.html): observed-spectrum preservation, not clean-chemistry recovery.
- [Training curves](../results/p08_universal/release/figures/P08-U1-training-diagnostics/built_index.html): losses, source-validation performance, and the number of contributing fits at each epoch.

Exact values are in the [model summary](../results/p08_universal/release/tables/model_summary.csv), [paired-effect summary](../results/p08_universal/release/tables/contrast_summary.csv), and [family-deletion supplement](../results/p08_universal/release/family_sensitivity/family_deletions.csv).

## Current state

Next planned, separate stages are source-selected QC and instrument-family rules, plus controlled-perturbation robustness, before any deployment recommendation. The family rule falls back to the complete minimal pipeline in all current held contexts; it cannot test family-specific transfer here. No new architectures or retraining are authorized by this report, and the universal benchmark does not establish chemical/nuisance separation or independent substrate efficacy. This report covers the completed universal numerical benchmark; release verification is recorded in the [completion audit](../plan/P08_U1_COMPLETION_AUDIT.md).
