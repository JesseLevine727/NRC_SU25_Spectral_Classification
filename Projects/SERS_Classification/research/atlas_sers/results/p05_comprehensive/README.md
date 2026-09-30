# NATO field-trial SERS: CNN benchmark figures

Short navigation README for the accepted P05 scientific bundle. These pages add no new
data or figures; they only make the existing 35 accepted figures easier to browse.

## Start here

- [index.html](index.html) — offline index linking all 35 figures as offline HTML, vector PDF, PNG and native TikZ (`.tex`).
- [P05_RESULTS.md](P05_RESULTS.md) — main report with interpretation and the key findings.
- [release_manifest.json](release_manifest.json) — release manifest.
- [release/tables/](release/tables/) — released tables.

## Viewing notes

GitHub may show the HTML **source** of a figure instead of rendering an interactive plot.
To see the interactive pages, download this folder and open [index.html](index.html) with the
`release/` directory beside it, then open the `.html` figure links; or open an individual
`.html` figure from `release/`. To read a figure directly on GitHub without downloading
anything, open its **PDF** or **PNG** link, which GitHub displays in the browser.

## Recommended starting plots (5)

These are a good first pass through the bundle; they are links to the figure HTML pages.

1. [Matched control D0-M, M01](release/figures/paired_bundle/figures/P05B_D0_M_M01.html).
   The new CNN against its matched control on individual spectra; the cleanest like-for-like comparison.
2. [Random Forest, M01](release/figures/paired_bundle/figures/P05B_C_RANDOM_FOREST_M01.html).
   The new CNN against a strong classical baseline on individual spectra.
3. [Random Forest, M06](release/figures/paired_bundle/figures/P05B_C_RANDOM_FOREST_M06.html).
   The same comparison when per-sample model probabilities are combined, not averaged raw spectra.
4. [Source validation versus held-instrument evaluation, M01](release/figures/diagnostic_bundle/figures/P05D_source_vs_held_held_evaluation_M01.html).
   How source-validation scores compare with the outer held-out ensemble; each dot is a
   context/strategy, not a physical sample, and the difference is not an unbiased transfer gap.
5. [Reliability, held evaluation, M01](release/figures/diagnostic_bundle/figures/P05D_reliability_held_evaluation_M01.html).
   Confidence versus correctness across repeated context–spectrum appearances;
   bins are not independent samples and pooled ECE differs from the mean per-context ECE.

Optional follow-up: [best-epoch selection in development](release/figures/diagnostic_bundle/figures/P05D_best_epoch_development_inherited_selection_fit.html).
Shows the fraction of source fits choosing each best checkpoint; these are not terminal training
epochs or loss curves.

## Contents

35 figures: 12 paired performance + 12 paired deltas + 4 source-versus-outer + 3 best-epoch +
4 reliability. Every figure has native TikZ, offline HTML, vector PDF and PNG; black serif text,
data unchanged. All paths are relative.

For the headline findings — the fixed-D3 average gain over the matched control, the
source-selected procedure, the non-uniform domain effects and the strong classical baselines —
see [P05_RESULTS.md](P05_RESULTS.md). This index deliberately does not restate those numbers.
