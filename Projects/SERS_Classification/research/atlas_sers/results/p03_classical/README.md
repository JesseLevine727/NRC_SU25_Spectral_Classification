# P03 classical benchmark public release

This directory is the curated public aggregate release for the completed NATO
SERS P03 classical benchmark.

## Included

- `P03_CLASSICAL_RESULTS.md`: authoritative aggregate report.
- `tables/`: aggregate endpoint coverage, domain summaries, model-selection
  stability, spectrum-versus-master comparisons, controls, calibration,
  confusion, and compute-cost summaries.
- `../../plan/figures/`: F12, F13, and F38–F43 as frozen CSV, native TikZ,
  vector PDF, PNG review copy, and standalone HTML.

## Excluded

- individual-spectrum and individual-master predictions;
- outer-fold membership and observation UIDs;
- fit caches and serialized models;
- calibration-record JSONL;
- complete fit and failure ledgers; and
- redundant protected-run state.

The exclusions keep the maintained release interpretable and reasonably sized;
they do not alter any reported denominator. The report preserves unavailable
endpoints and terminal failures explicitly.

## Frozen execution identity

- P03 run: `P03-513a0f9686c37cbc0d682645`
- Population: 598 spectra from 69 physical masters.
- Primary representation: `R_MIN_400_1800`.
- Primary preprocessing: `PP-U-MIN`.
- Source-selected comparator: `C-SELECTED`.

The original `atlas-sers-*` strings and `atlas_sers` import path remain legacy
reproducibility identifiers; the study is publicly identified as NATO SERS.
