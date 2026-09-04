# P04 compact deep baseline results

P04 passed its locked training-validity gate and produced a complete compact ordinary deep-learning baseline. The one-dimensional residual model has **208,691 trainable parameters**, consumes the frozen 1,401-channel `R_MIN_400_1800` representation, and uses no BatchNorm. All endpoint selection and stopping evidence came from authorized source-only inner roles. P03/P13 results were already known to investigators, but the P04 fitting and selection path could not load them; they entered only the separate post-freeze comparison. This is procedural separation, not analyst blinding.

## Main result

- Within-station development (`EXP-N00-DEV`, M01): mean balanced accuracy **0.781** across 60/60 complete outer cells.
- Unseen-instrument evaluation (`EXP-N00-T3`, M01): mean domain-balanced accuracy **0.711**, worst-domain balanced accuracy **0.379**, and endpoint coverage **100.0%**.
- After instrument-balanced physical-master aggregation (M06), unseen-instrument mean domain-balanced accuracy was **0.765**, with worst-domain balanced accuracy **0.367**.
- Unseen-instrument spectrum probability quality: mean negative log likelihood **1.635**, Brier score **0.446**, and expected calibration error **0.266**; lower is better for all three.
- Against frozen C-SELECTED on the common M01 denominator: D0 **0.710** versus classical **0.659**, paired difference **+0.051**.
- Pooled out-of-fold, physical-master-clustered D0-minus-C-SELECTED M01 difference: **+0.050**, with 95% interval **[+0.022, +0.078]**.
- Frozen conclusion: **`D0_adds_value_over_C_SELECTED`**.
- This advantage is comparator-specific: the paired intervals do **not** establish an advantage over fixed Random Forest or Extra Trees. Extra Trees has a slightly higher fold-mean BA, while the pooled D0-minus-Extra-Trees estimate is slightly positive; both are consistent with a small, uncertain difference rather than a general deep-learning win.
- On the exact P13 substrate-stratified PP-U-MIN test views, frozen D0 was scored in **15/16** domains, including **13/13** confirmatory domains; **7/13** confirmatory domains had a 95% lower bound at or above the P13 held-recovery threshold of 0.60. P13-DOM-005 (CWA/Agilent-3, exploratory low support) is outside the 13-domain P04 eligibility set. This is a post-freeze stratification of the same D0 predictions, not a second model-selection exercise. It cannot issue the full P13 portability verdict because P04 did not generate the matched-source loss endpoint required by that dual-margin rule.

The P13 reuse also has a training-support difference: D0 learned from all source substrates within the station, whereas P13 classical models fitted only the substrate family being evaluated. Their common held-test views permit descriptive recovery comparisons, but any D0-minus-P13-classical difference combines learning strategy with training-data scope. A controlled P13 learner comparison requires new deep refits on the exact substrate-restricted P13 source roles. The primary P04-versus-P03 comparison above uses the shared P02 design and is unaffected by this limitation.

## Training behavior

The source-selected median best epoch was **32** in development and **23** in T3 source-only selection. These checkpoint epochs differ from the number of epochs executed while waiting for early stopping. Final refits used a per-seed median of the selected candidate's inner checkpoint epochs, clipped to 30–200; their median durations were **30** epochs in development and **30** epochs in T3. No outer-test result chose these durations.

Diagnostic labels describe optimization traces rather than discarded runs: 4343 fits met the locked overfit diagnostic and 176 met the collapse diagnostic. A finite fit can still overfit or collapse. Every failed or collapsed fit remains in coverage and failure-sensitive accounting. The full fit counts, diagnostic categories, optimizer selections, and execution costs are in `tables/fit_summary.csv`, `tables/training_diagnostics.csv`, and `tables/selected_candidate_frequency.csv`.

| Inner-fit diagnostic | Development fits | T3 source-selection fits |
| --- | ---: | ---: |
| none | 2824 | 8092 |
| overfit | 416 | 3927 |
| collapse | 0 | 176 |
| underfit | 0 | 30 |
| optimization_instability | 0 | 33 |

The locked diagnostics mean: **overfit**, checkpoint training BA exceeds validation BA by more than 0.20; **collapse**, the best validation prediction contains fewer than two predicted classes; **underfit**, best training BA is at most chance plus 0.05; **optimization instability**, last-ten validation BA standard deviation exceeds 0.15 or more than half the optimizer steps are gradient-clipped. These are assigned diagnostic categories, not independent hypothesis tests, and concern source-validation runs. Final refits have no test-based diagnostic or early stopping. A `none` label means none of these rules fired, not that generalization is guaranteed.

The registered execution contains **16,458** fits, of which **16,458** completed. The three retained pre-outcome implementation smoke failures are recorded separately in the deviations registry. Timing columns sum elapsed time within individual fits; because fits run concurrently, that sum is neither end-to-end elapsed time nor a measurement of exclusive GPU hours.

This is repeated evaluation of one architecture. The 15,498 inner fits compare six optimizer settings across source-validation units and three seeds. The 960 final refits are 320 evaluation contexts times three seeds; their calibrated probabilities are averaged within context. The 208,691 parameters belong to each individual model. More fits or repeated spectra do not create additional independent physical samples.

## Fair classical comparison and metric definitions

| Classical comparator | Complete paired cells | D0 BA | Classical BA | Difference |
| --- | ---: | ---: | ---: | ---: |
| C-EXTRA-TREES | 260/260 | 0.711 | 0.716 | -0.005 |
| C-RANDOM-FOREST | 260/260 | 0.711 | 0.701 | +0.010 |
| C-RBF-SVM | 260/260 | 0.711 | 0.670 | +0.041 |
| C-SELECTED | 252/260 | 0.710 | 0.659 | +0.051 |

The table averages the paired outer-fold balanced accuracies on each comparator's exact common-success cells. Classical scores are recomputed from frozen P03 row predictions using the same P04 aggregation as D0; they can differ from P03's earlier pooled-repeat summaries. Missing classical endpoints are retained in the separate coverage and failure-sensitive columns of `tables/comparison_summary.csv`; common-success means alone cannot establish operational reliability.

| Classical comparator | Pooled M01 difference | Conditional 95% interval |
| --- | ---: | ---: |
| C-SELECTED | +0.050 | [+0.022, +0.078] |
| C-RBF-SVM | +0.039 | [+0.018, +0.062] |
| C-RANDOM-FOREST | +0.022 | [-0.017, +0.057] |
| C-EXTRA-TREES | +0.005 | [-0.023, +0.032] |

Balanced accuracy (BA) averages class recall. Each station has three classes, for which uniform random prediction has expected BA 1/3 and perfect classification scores 1. The inherited endpoint metric averages recall only over classes actually present in that outer-test cell; some small instrument-specific folds lack one or more classes. The pooled out-of-fold domain comparison below instead requires all three classes. Consequently, the fold-mean table is not a substitute for that pooled three-class comparison. M01 scores spectra. M06 first averages probabilities within each instrument view of a physical master, then weights the available instrument views equally. The same 69 physical masters underlie both summaries. Macro-F1 uses the full station class vocabulary, assigning zero to an undefined class F1. Negative log likelihood, Brier score, and expected calibration error assess predicted probabilities; lower values are better. Calibration on small test cells is uncertain and is not a substitute for discrimination.

The P04 interval and F48 use 5,000 paired bootstrap draws of physical masters, stratified by station and class, carrying each sampled master's repeated predictions and instrument views together. They condition on the observed domain set; draws missing a required class in any included domain are rejected, so the intervals additionally condition on retaining three-class support. Repeats are averaged before inference, and each domain receives equal weight. Their point estimate pools out-of-fold class recalls within each domain, so it can differ from the table's mean of fold-level BAs. This baseline diagnostic does not implement the P11 final primary interval, which additionally resamples domains with 10,000 draws. It cannot by itself pass G4, demonstrate equivalence, or justify a general claim about arbitrary unseen instruments.

## Interpretation

D0 tests whether a compact location-preserving convolutional model adds predictive value over classical methods under the same minimal preprocessing and master-grouped held-instrument design. It does **not** test whether a network removed noise, recovered clean Raman spectra, or disentangled chemistry from acquisition nuisance. Passing G2 means the architecture trained reproducibly enough to serve as the D0 control for P05; it does not by itself establish superiority.

Probability quality is a separate limitation. Uniform three-class probabilities have negative log likelihood 1.099 and Brier score 0.667. D0's held-instrument spectrum NLL of 1.635 is worse than that uniform reference, although its Brier score of 0.446 is better. Log loss penalizes highly confident mistakes particularly strongly. Source-fitted temperature scaling therefore did not deliver uniformly reliable probabilities under acquisition shift, and classification improvement must not be presented as solved calibration. D0's mean NLL was also worse than each of the four classical comparators on their common endpoint sets.

P05 is the next planned phase: test the predeclared supervised-contrastive and paired-consistency successors against this frozen D0. Its exact no-fit expansion and source-only advancement checks must precede fitting. Those models must improve source pseudo-instrument performance without sacrificing worst-domain or within-source performance before any definitive held comparison.

## Figures

| Figure | View | Editable source | Vector export |
| --- | --- | --- | --- |
| F19: architecture and tensor flow | [HTML](../../plan/figures/html/F19_deep_architecture.html) | [TikZ](../../plan/figures/tikz/F19_deep_architecture.tex) | [PDF](../../plan/figures/pdf/F19_deep_architecture.pdf) |
| F20: source-only learning curves | [HTML](../../plan/figures/html/F20_learning_curves.html) | [TikZ](../../plan/figures/tikz/F20_learning_curves.tex) | [PDF](../../plan/figures/pdf/F20_learning_curves.pdf) |
| F48: D0-minus-classical held-domain effects | [HTML](../../plan/figures/html/F48_deep_classical_comparison.html) | [TikZ](../../plan/figures/tikz/F48_deep_classical_comparison.tex) | [PDF](../../plan/figures/pdf/F48_deep_classical_comparison.pdf) |

In F20, only fits still running contribute at later epochs, so changes in the curve can reflect which fits remain. The interquartile band describes variation across fits, not a confidence interval. The vertical line marks the median selected inner checkpoint epoch. In F48, points to the right of zero favour D0, points to the left favour the named classical comparator, and each horizontal interval resamples physical masters.

## Boundaries

The independent chemical evidence remains 69 physical masters, not 598 independent spectra. Results apply to the observed stations, analytes, substrates, and instruments under `PP-U-MIN`; they do not establish broad instrument independence or substrate superiority.
