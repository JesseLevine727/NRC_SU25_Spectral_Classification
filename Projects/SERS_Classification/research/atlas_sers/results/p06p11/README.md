# Frozen-result uncertainty and synthesis

Start with the [chemist-readable report](../../reports/NATO_SERS_UNCERTAINTY_REPORT.md) and [four-figure browser](index.html). The [supervisor review](../../plan/delegation/P06P11_REVIEW.md) records numerical acceptance, retained software corrections, resources and publication checks. This package adds no training, predictions or preprocessing.

## Reading the results

- `release/tables/summary.csv`: all 17 fixed comparisons at M01 and M06; equal-domain effects, support and stability summaries.
- `release/tables/intervals.csv`: crossed, sample-only and instrument-only weighted intervals, plus unavailable original-hierarchy bounds and explicit reasons. All values are fractions, not percentages.
- `release/tables/domain_metrics.csv`: both methods' scores on exactly paired support; domains are station–instrument combinations.
- `release/tables/leave_one_out.csv`: descriptive deletions, not refits.
- `release/tables/sign_flip.csv`: exact descriptive symmetry sensitivities; primary unadjusted and the other 33 Holm-adjusted separately for each sign scheme. These are not confirmatory randomized tests.
- `release/tables/feasibility.csv`: undefined-draw and empty-cell accounting; no conditional-on-survival interval is substituted.
- `release/tables/tie_corrections.csv`: seven secondary count corrections at numerical tolerance 1e-12. Original scores and intervals are unchanged.
- `release/g4_decision.json`: four supported, zero failed and two unassessable criteria; promotion is false.

## Supplementary metrics

`release/metrics` separates full available method support from the primary common support. Macro-F1 uses each station's full three-class vocabulary; balanced accuracy uses each test context's represented true classes. Domain/model summaries average contexts/domains equally. NLL uses the established probability clip [1e-7, 1−1e-7]; Brier sums squared errors across all three class probabilities. Lower NLL/Brier is better.

Confusion counts and fixed-width reliability bins pool repeated prediction appearances by station, method and endpoint. They are not independent sample counts. Class sensitivity includes pooled, mean-context and mean-contributing-domain recall; unavailable class support remains missing. Mean-context equal-mass ECE is explicitly distinct from pooled equal-width ECE. No calibration is fitted by this analysis.

M01 predicts stored spectra individually. M06 combines saved class probabilities per physical master/context, not raw spectra. Full-support M01 has 2,785 appearances of 557 distinct spectra; M06 has 1,310 appearances of 262 master/domain units. Both derive from 69 physical masters. The primary selected pair uses 252/260 contexts, with 2,714 M01 and 1,268 M06 appearances.

## Reproduction and provenance

The [locked numerical specification](../../plan/P06P11_INFERENCE_PROTOCOL.md) governs the single approved stochastic execution. Re-rendering must use its existing authenticated private analysis, not run new random draws:

```bash
python -m atlas_sers.evaluation.p06p11_release \
  --analysis /absolute/private/frozen-analysis \
  --output /absolute/private/new-release
```

Run under the specified ten-minute, 2-GiB resident-memory and 1-GiB output guard; the supervisor's release record supplies measured use. The output must be a fresh directory outside the repository. The command verifies the original receipt and fourteen output hashes, derives deterministic supplementary tables, and renders all four figures. It requires the package's visualization dependencies, `pdflatex` and `pdftocairo`. It does not import a training or resampling engine.

The release manifest retains original input/code/output hashes separately from the release-generation code hashes. Its `requires_supervisor_review` state is intentional: generation cannot approve itself. The dated supervisor review supplies acceptance and the exact Git publication check supplies remote status. Published aggregates exclude private row predictions, identities, arrays and source paths.

Each figure has one semantic CSV, native PGFPlots source, offline HTML, vector PDF and 300-DPI PNG. Labels use black text and standard LaTeX/Times-compatible fonts at the owner's request. The flat, self-contained figure bundle is this release's layout convention; it preserves the existing semantic-parity and four-format requirements.

## Next boundary

The [P08 handoff](../../plan/P08_HANDOFF.md) is planning only. Minimal min–max scaling has not been established as optimal. The wider P06/P11 programme, missing selected-classical T1 retention comparison and unsupported neural task branches remain explicitly incomplete.
