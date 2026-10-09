# P08-U1 release review — 2026-10-09

## Scientific evidence

All 195,202 unique SG/arPLS fit slots and 3,354 scalar calibrations completed.
The last execution phase reused 41,568 authenticated fits and completed 153,634
new fits. All 404,814 registered operations are complete; no workers or jobs
remain. The retained MIN reference is authenticated historical evidence, not
a newly selected or refitted baseline. Historical interruptions and recovery
overhead remain in the private ledger.

The final reader authenticated completed operation receipts, dependencies,
source-only selection/calibration and complete test roles. The numerical panel
contains 3,900 strategy/policy/context cells and 3,237 distinct saved endpoints;
selected/ordinary-CNN aliases are not counted as independent fitted models.
All 260 contexts and 13 held domains are present for every strategy and policy.

The source population is 598 spectra/69 physical masters/10 instruments. Held
evaluation covers 557 spectra, all 69 masters and all 10 instruments; 41 spectra
in four exploratory domains remain outside the primary held comparison.
The equal-context/equal-domain and pooled-four-fold estimands remain distinct.
All 60 aggregate score rows were independently checked against the domain
means; all 88 available contrast means were checked against their component
procedure scores. Unavailable future QC entries retain their registered slots.

The shared 10,000-draw positive-weight analysis preserves sample/instrument
relationships and conditions on saved fits. Original hierarchical feasibility,
domain/instrument sign sensitivities, Holm families, weakest-domain summaries,
deletions, class recalls, confusion and probability diagnostics are retained.
No held-score winner is promoted, no new model is trained, and G4 is not passed.

## Corrections found during review

1. The historical primary manifest lacked the derived instrument-family column.
   This left the original family-deletion maps empty, without changing fits,
   scores, weights, intervals or sign tests. The separately hashed
   `family_sensitivity/family_deletions.csv` completes all 352 descriptive
   deletion rows using the previously frozen P02 mapping. Original diagnostic
   evidence is retained unchanged. The portable loader now derives and checks
   that metadata; regression tests cover the actual pinned-load seam.
2. Native spectral pages had insufficient text clearance. Only their margins
   were rebuilt. All spectral values, semantic hashes, CSV and HTML remain
   byte-identical. All 17 corrected PDFs pass the page-boundary check.
3. Draft report review removed an incorrect claim of a combined SG/arPLS arm
   and corrected the affected Pendar-2 station to CWA. Neither was an experiment
   or an error in the result tables. The final prose was checked against the
   frozen pipelines and domain results.

## Figure and disclosure review

All 117 native sources were compiled; all PDFs contain vector plots, with no
embedded raster plot objects. Output hashes and page bounds were independently
verified. All 117 offline HTML pages passed actual headless-browser checks for
their applicable hover/focus, toggles, data tables/downloads, missingness and
absence of remote assets. The first browser harness run had 13 failed checks:
12 used an incorrect training-mark expectation and one transient focus check
did not recur. The corrected full run passed 117/117; the earlier receipt is
retained privately rather than relabelled as a success.

Direct visual inspection covered each of the six figure types in both native
and HTML forms, including a nonempty training panel and unavailable training
groups. It checked axes, category order, intervals, labels, legend encodings,
black Roman text and caption boundaries. Automated checks cover every panel;
this is not a claim of manual inspection of every pixel of all 117 panels.
Data traces use redundant line/marker encodings as well as color.

The public spectral view retains 49 domain/analyte cells. Only the 46 cells
with at least two masters expose aggregate curves (138 curves); the other three
are explicitly unavailable for display. Curves first average stored views
within master, then masters equally, with no aggregate renormalization.
The first trace is MIN, not raw counts. These are not averaged classifier inputs.

All released table/semantic schemas were checked for forbidden row, master,
operator and source-path fields. Browser payloads and downloads contain only
approved aggregates. No individual trace, checkpoint, private log or raw monitor
history is released. The 51 domain/action preservation groups use 1,794 retained
P01 records, summarized into 561 metric rows; these are observed-reference
diagnostics, not clean chemical truth.

Training summaries cover 6,402 expected neural fits; 6,366 have authenticated
monitor histories. The 36 historical pilot histories are absent, not fabricated.
The 24 recipe/policy/stage groups include eight structurally empty neural
calibration groups. Bands show descriptive 10th–90th percentiles across fits;
the contributing-fit count falls with stopping. No new hypothesis is inferred
from these curves.

## Software verification

DeepSeek V4.1 Flash authored the analysis/rendering components through the
tool-denied OpenCode workflow. The supervisor independently reviewed, corrected,
tested and applied them. The public package promotes 47 new Python files;
private launchers, process permits, raw logs and workstation-specific guards
are not promoted.

The staged full suite had 6,632 passes, 11 launch-environment failures and seven
CUDA-disabled skips. A corrected real-file/Git-root launch reran all three
affected modules: 41 passes, covering every failed node. The isolated analysis
runner passed 16 tests. After the final metadata/layout corrections, all 422
new-package tests and 40 subtests passed. Sparse synthetic fixtures deliberately
emit classification warnings; they are not additional field-data experiments.
Package Ruff and public-boundary validation are separate release checks.

The final report's preservation check retained all 113 numeric tokens and both
bracket-reference tokens from the supervisor-corrected draft. Earlier rounding
is explicitly labelled in the report; exact values remain in the CSV tables.
Lenarizer guided claim boundaries and direct prose, not numerical changes.

## Publication boundary

Release copies of the figure/table manifests add the completed review and bind
their original private build-manifest hashes. Scientific data are unchanged.
`published: false` in a build manifest records the pre-push sealing state;
publication is established by the main-branch commit and its actual CI result.
The release-wide manifest binds all local outputs, the report and reviewed code.
Later P08 stages remain unexecuted and require their own bounded goals.
