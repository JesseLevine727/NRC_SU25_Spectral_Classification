# Universal preprocessing benchmark

The completed comparison uses MIN, smoothing (SG), and baseline correction
(arPLS), with five classifier strategies and the same 260 held-instrument test
contexts. This is P08-U1 only; adaptive preprocessing and robustness experiments
are separate stages.

Start with the [plain-language report](../../reports/NATO_SERS_UNIVERSAL_PREPROCESSING_REPORT.md)
and [interactive figure browser](release/figures/index.html). Download the
repository and open the HTML files locally. GitHub displays their source but
does not run the interactive figures.

The release contains 117 panels in native TikZ, vector PDF, PNG and offline
HTML, nine aggregate result tables, and a separately labelled family-deletion
supplement. Raw spectra, sample identities, row predictions, checkpoints,
operator details and workstation paths are excluded.

## Main finding

Baseline correction improves mean balanced accuracy across all five methods:
5.70–6.70 percentage points for individual spectra and 5.72–9.31 points for
combined model predictions per sample. Smoothing gives small, mixed changes.
Some domains worsen even when the average improves. These secondary results
neither select a deployment winner nor establish chemical/nuisance separation.

## Navigation

- [All exact model scores](release/tables/model_summary.csv)
- [Paired effects and conditional intervals](release/tables/contrast_summary.csv)
- [Individual domain scores](release/tables/domain_metrics.csv)
- [Family-deletion supplement](release/family_sensitivity/family_deletions.csv)
- [Release review and interpretation boundaries](release/RELEASE_REVIEW.md)
- [Requirement-to-evidence completion audit](../../plan/P08_U1_COMPLETION_AUDIT.md)

CSV accuracies and effects are proportions; multiply effects by 100 for
percentage points. Blank cells are undefined/unavailable, not zero. Counts
of repeated test appearances are not independent sample counts. M06 combines
model probabilities; it never averages spectra before classification.

The release manifest binds every released artifact. Build-manifest `published`
fields describe the pre-push sealing stage, not subsequent GitHub availability;
the repository commit and CI record establish transport/publication status.
