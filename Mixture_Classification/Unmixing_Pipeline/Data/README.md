# Data Layout

This directory makes the active `Unmixing_Pipeline` self-contained.

## Contents

- `reference/reference_v2.csv`
  - original pure-compound reference library used by the active classical pipeline
- `reference/mixtures_dataset.csv`
  - original real binary-mixture evaluation set
- `pt2/Mixtures.txt`
  - manifest for the pt2 mixture compositions
- `pt2/<compound-or-mixture>/txt/*.txt`
  - pt2 pure and real-mixture spectra used by the active classical pipeline

## Why This Exists

The earlier experiment scripts reached into:

- `Notebooks/`
- `../Jesse dataset pt2/`

That made the active pipeline harder to reason about and harder to move or version cleanly.

The shared loaders in `Scripts/unmixing_common.py` now read from this local data layout instead.
