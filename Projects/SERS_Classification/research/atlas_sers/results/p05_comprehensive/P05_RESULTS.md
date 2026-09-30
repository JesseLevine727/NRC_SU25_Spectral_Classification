# NATO field-trial SERS: acquisition-aware CNN benchmark

## Headline findings

The completed P05 core benchmark compared four neural training recipes using fixed, repeated evaluation splits. Sample balancing and random streams were matched across recipes. Source selection, final fitting, held evaluation, comparison and reporting are complete; formal publication-level inference remains pending.

First, the fixed combined-loss recipe D3 produced a small mean benefit over the matched cross-entropy control D0-M: +1.68 percentage points (pp) at the per-spectrum endpoint (M01) and +1.32 pp at the per-master endpoint (M06), averaged over the thirteen held station–instrument domains. The benefit is not uniform: D3−D0-M was positive in 6 domains, negative in 6 and tied in 1 at M01, and 7 positive, 4 negative and 2 tied at M06.

Second, the predeclared source-only selector added little: source-selected − D0-M was +0.17 pp (M01) and +0.04 pp (M06).

Third, classical comparators remained competitive. Random Forest reached 70.10/76.74% and Extra Trees 71.57/76.65%, against D3 at 71.90/76.21% (M01/M06). D3 exceeded Random Forest at M01 (+1.80 pp) but trailed at M06 (−0.53 pp). It also exceeded Extra Trees at M01 (+0.32 pp) but trailed at M06 (−0.44 pp). Historical CNN D0-ERM reached 71.12/76.55%. These held results do not establish a universal deep-learning advantage or authorize replacing the source-selected procedure with D3.

Fourth, domains were nonuniform. Station and domain results differ substantially, and a positive mean benefit coexists with a worse weakest domain.

## Question and recipe ladder

The question was whether a compact one-dimensional convolutional neural network (CNN), which extracts local spectral patterns, could use repeated measurements to improve classification across acquisition conditions. Four recipes were fixed before execution. Each included auxiliary term has coefficient 0.3; absent terms have coefficient zero:

- D0-M: matched ordinary cross-entropy (classification-error) control.
- D1: cross-entropy plus supervised contrastive learning (same chemical brought closer in representation).
- D2: cross-entropy plus same-physical-sample cross-instrument consistency.
- D3: both auxiliary terms.

D0-M is a new matched control, not the historical D0-ERM. All four share the same class → physical-master → sampled-instrument-view balanced objective and common random streams. Missing sparse branches are explicitly zero and reason-coded; no held data is borrowed.

## Data, inputs and evaluation context

The dataset contains 598 spectra from 69 recorded physical-master grouping units. Repeats are not independent chemistry examples. Inputs were fixed: minimal per-spectrum min–max PP-U-MIN, 400–1800 cm^-1, 1401 channels. No baseline subtraction or smoothing comparison occurs here.

There are 320 evaluation contexts: 60 within-station development contexts plus 260 held unseen-instrument contexts. Held fitting excludes the test physical masters and instrument. The thirteen station–instrument domains (3 CWA, 5 pills, 5 surfaces) each contribute 20 repeated contexts. These are repeated evaluation splits, not 13 independent instruments and not 260 independent chemical samples. Held true-class support was 220 three-class, 39 two-class and 1 one-class context. Balanced accuracy (BA) is the mean recall over observed true classes only, not ordinary accuracy; support therefore matters and no single chance baseline applies.

Eight historical C-SELECTED references are incomplete or missing (2 CWA, 6 surfaces). Every new strategy and every other reference is complete at 260/260, with no imputation or refitting.

## Model, objective and fitting

The classifier is a compact one-dimensional CNN with 208,691 base trainable parameters. D1/D3 add a bias-enabled 64→64 projection head of 4,160 parameters, for 212,851 total. Optimization used AdamW, learning rate 0.0003, weight decay 0.0001, gradient clipping 5.0 and four draws per epoch. Seeds were 20260805, 20260817 and 20260829. Source fits ran 30–200 epochs with patience 20. An epoch here is four sampled batches, not necessarily an exhaustive pass, and the best checkpoint may precede epoch 30 despite the minimum training duration. Refits inherit context/recipe/seed-specific clipped median best epochs, actually 30–118, with no test early stopping. These facts do not license inference about undertraining or an ideal epoch count.

## Held performance

Balanced accuracy percentages below are arithmetic means of the thirteen held domain means on all 260 complete contexts — not the mean of the three station means and not pooled predictions.

| Model | M01 BA (%) | M06 BA (%) |
|---|---|---|
| RBF-SVM | 67.01 | 70.19 |
| Random Forest | 70.10 | 76.74 |
| Extra Trees | 71.57 | 76.65 |
| Historical CNN D0-ERM | 71.12 | 76.55 |
| Matched CNN D0-M | 70.21 | 74.88 |
| Source-selected CNN | 70.39 | 74.93 |
| Fixed combined CNN D3 | 71.90 | 76.21 |

For M01, each of the three seed models produces a class-probability vector for a spectrum. A temperature fitted only on source validation calibrates each model's probabilities. The three vectors are averaged, and the class with the largest mean probability is predicted. M06 then averages these spectrum-level vectors within each instrument for a physical sample, averages equally across instruments, and chooses a class. Raw spectra are never averaged. Held-instrument contexts contain one instrument, so M06 combines its repeat measurements; multi-instrument averaging applies in development. The endpoints differ in prediction and weighting, and averaging does not guarantee improvement.

Evidence: [new-strategy summaries](release/tables/strategy_summary.csv), [domain scores](release/tables/strategy_domains.csv) and [aligned comparator scores](release/tables/paired_domains.csv). View the [matched-control scatter](release/figures/paired_bundle/figures/P05B_D0_M_M01.html) and [Random Forest comparison](release/figures/paired_bundle/figures/P05B_C_RANDOM_FOREST_M01.html).

## Paired effects

Paired changes are computed before rounding; subtracting the displayed percentages can differ by 0.01 pp. On the separate common-complete support where historical C-SELECTED records exist (252 contexts across 13 domains), the source-selected CNN scored 70.20% versus 65.84% at M01 and 74.79% versus 71.18% at M06: +4.36 and +3.61 pp. That is not the same support as the full 260-context table, nor evidence of superiority over all classical methods. The [coverage table](release/tables/coverage.csv) retains the missing-reference denominators.

The [domain-change scatter](release/figures/diagnostic_bundle/figures/P05D_paired_D0_M_M01.html) shows change relative to D0-M on the horizontal axis; points to the right of zero favour the added losses. In the paired performance scatters above, each dot is a station–instrument domain mean over common contexts; the diagonal marks equality and points above it favour the new CNN. Dot overlap can hide identical results.

## Station results and limits

All entries below are balanced accuracy percentages. Station means assign equal weight to their contributing domains.

| Station | Domains / contexts | D0-M M01 | Selected M01 | D3 M01 | D0-M M06 | Selected M06 | D3 M06 |
|---|---|---|---|---|---|---|---|
| CWA | 3 / 60 | 53.65 | 54.16 | 54.32 | 58.06 | 58.33 | 58.24 |
| Pills | 5 / 100 | 80.72 | 80.72 | 82.72 | 84.08 | 84.08 | 85.92 |
| Surfaces | 5 / 100 | 69.64 | 69.79 | 71.62 | 75.78 | 75.72 | 77.28 |

The worst held domain at M01 moved from D0-M to D3 as follows: CWA 48.29% → 51.22%, pills 40.64% → 49.50%, surfaces 50.00% → 48.97%. This is the lowest domain mean, not the lowest split. Mean benefit therefore coexists with a worse weakest domain in one station.

Development results are separate. D0-M/D3 at M01: CWA 62.73/62.81%, pills 82.24/83.96%, surfaces 78.55/81.33%. At M06: CWA 80/83.33%, pills 100/100%, surfaces 93.33/91.39%. Development selection returned D0-M by rule. Source best-validation is another estimator used in selection, so source-versus-outer differences are descriptive, not an unbiased single transfer gap.

## Selection and source diagnostics

The source-only selector assigned 281 contexts to D0-M, 14 to D1, 11 to D2 and 14 to D3. There were 192 mandatory master-only fallbacks (60 development plus 132 held) and 128 eligible pseudo-instrument contexts; 39 of 128 were promoted and 89 of 128 had no candidate pass. All 100 pills held contexts lacked pseudo-instrument selection support and necessarily stayed D0-M; these are not 100 failed D3 transfer tests. Advancement required all G3 conditions: mean pseudo-domain BA gain ≥0.02; worst pseudo-domain change ≥−0.02; guard-unit BA change ≥−0.02; strictly positive paired gain in ≥60% of pseudo-domains; and collapse ≤5%. No global or cross-context held-winner selection occurred.

Source evidence totals 14,940: 14,904 new successes plus 36 reused pilot fits. Of the best checkpoints, 128/14,940 predicted fewer than two classes on source validation (32 D0-M, 31 D1, 30 D2, 35 D3). That collapse is a finite diagnostic, not an infrastructure failure or a retry cause. Best checkpoint epochs ranged 1–150; refits 30–118. Actual distinct refits were 1,995, with 1,995 source-only calibrations and 2,880 strategy-seed aliases (885 duplicate specifications avoided). New neural successes were 16,899 of 16,900 attempts, including one original source interruption; there were 0 recovery or refit failures. An owner-approved exact recovery after a host-memory interruption preserved completed work and settings. New successful optimizer updates were 3,149,052; the charged upper figure 3,149,852 is not exact, with an observed lower figure of 3,149,120. Reused pilot updates, 6,880, are excluded.

Cumulative scientific execution through reporting and the independent publication audit was bounded at 20.48 h, including the 35.07 s audit, within the 48 h ceiling. This includes conservative charges for the interrupted run; it is not measured GPU-hours or elapsed calendar time. The approved outer GPU-memory ceiling was 8 GiB, while the frozen inner guard remained 4 GiB; refit peak allocation was 205,188,608 bytes. See [selection counts](release/tables/selection_counts.csv), [source diagnostics](release/tables/fit_summary.csv), [costs](release/tables/costs.json) and the [release record](release_manifest.json). The cost table stops at comparison; the release record also accounts for reporting and publication authentication. The [best-epoch figure](release/figures/diagnostic_bundle/figures/P05D_best_epoch_held_evaluation_inherited_selection_fit.html) shows checkpoint selection, not loss curves.

## Calibration

Expected calibration error (ECE) measures mismatch between model confidence and observed correctness. Negative log-likelihood (NLL) penalizes assigning low probability to the true class; Brier score measures squared probability error. Lower values are better. Macro-F1 averages the classwise balance between precision and recall.

Mean per-context M01 ECE for D0-M/D3 was CWA 0.373/0.370, pills 0.178/0.157 and surfaces 0.290/0.276. Pooled reliability ECE differs: CWA 0.173/0.162, pills 0.202/0.168, surfaces 0.084/0.095. Pooled surfaces ECE worsens even though mean-context ECE improves. No calibration was fitted to held outcomes. See the [reliability table](release/tables/reliability_summary.csv) and [confidence-versus-correctness curves](release/figures/diagnostic_bundle/figures/P05D_reliability_held_evaluation_M01.html). Their bins pool repeated context–spectrum appearances, not independent samples.

## What this does not establish, and next steps

No formal hypothesis test or new confidence interval is reported. Splits, seeds and prediction appearances are not independent chemicals. No causal disentanglement is shown, and no physical causation is attributed to substrate or instrument. No optimization setting was changed after outcomes.

Next steps: preserve the frozen results. Planned P11 grouped and domain-aware uncertainty work plus the remaining P06 synthesis are required before any definitive superiority claim. A P08 preprocessing comparison on matched splits needs separate bounded approval. A P14 metadata-only support and compute audit may be chosen, not automatic training. Classical methods remain essential, and no post hoc selection from held scores is permitted.

## Figures and artifacts

All accepted scientific files are copied unchanged into [release/](release/). Tables sit under `release/tables/`. All 35 figures have native TikZ, offline HTML, vector PDF and PNG versions. The [figure index](README.md) provides a reading order. Additional views include the [sample-level Random Forest comparison](release/figures/paired_bundle/figures/P05B_C_RANDOM_FOREST_M06.html), [Extra Trees comparison](release/figures/paired_bundle/figures/P05B_C_EXTRA_TREES_M01.html) and [sample-level matched-control comparison](release/figures/paired_bundle/figures/P05B_D0_M_M06.html). In the [source-versus-held scatter](release/figures/diagnostic_bundle/figures/P05D_source_vs_held_held_evaluation_M01.html), each dot is a context/strategy, not a physical sample. The release manifest retains the reporting receipt hash and 159 public file hashes: 14 tables, the cost JSON, 35 figures in four formats, 2 semantic CSVs and 2 figure manifests.
