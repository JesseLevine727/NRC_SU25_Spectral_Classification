# SERS Classification

This directory groups the SERS-focused classification and exploration work.

## Main area

- [Workspace](Workspace)
  Mixed notebook-and-data workspace containing SERS references, cleaned CSVs, txt exports, clustering notebooks, and Siamese one-shot experiments.

## Current Substrate-Agnostic Work

- [docs/SERS_SUBSTRATE_AGNOSTIC_AUDIT.md](docs/SERS_SUBSTRATE_AGNOSTIC_AUDIT.md)
  Documents why the original notebook results were chemical-substrate pair classification and summarizes the current leave-one-substrate-out substrate-agnostic baselines.
- [docs/AGNP_FAILURE_DIAGNOSTIC.md](docs/AGNP_FAILURE_DIAGNOSTIC.md)
  Details the current AgNP failure mode after canonicalizing `bt -> benzenethiol`.
- [Workspace/substrate_agnostic](Workspace/substrate_agnostic)
  Cleaned result tree for the current substrate-agnostic story: best model, diagnostics, sweeps, baselines, and archived comparison runs.
- `scripts/sers_siamese_substrate_agnostic.py`
  CUDA-first Siamese/triplet training script for substrate-held-out chemical classification.
- `scripts/sers_kshot_substrate_agnostic.py`
  CUDA-first formal K-shot version of the grouped substrate-held-out Siamese evaluation. It samples only `K` spectra per held-in chemical-substrate-family cell and generates matching clustering/geometry diagnostics.
- `scripts/sers_agnp_deep_dive.py`
  Generates average spectra, PCA, UMAP, t-SNE, prototype-distance, and raw-file diagnostics in `Workspace/substrate_agnostic/diagnostics/agnp_failure/`.

## Typical contents in `Workspace`

- raw and cleaned SERS CSVs
- Raman reference CSVs used for comparison
- notebook-driven exploratory analysis
- per-platform folders such as `Ag`, `AgNP`, `Au`, `AuNP`, `PICO`, and `pSERS`
- `substrate_agnostic/` for current substrate-agnostic outputs and diagnostics

Current organization:

- `Workspace/data/processed/`
  Processed SERS CSVs, including `consolidated_SERS.csv`.
- `Workspace/data/raw_by_substrate/`
  Raw per-substrate SERS text exports grouped by substrate.
- `Workspace/data/raw_curated/`
  Curated raw SERS folder hierarchy used by the diagnostics.
- `Workspace/data/reference/` and `Workspace/references/`
  Raman/reference CSVs and raw reference exports.
- `Workspace/notebooks/`
  Legacy exploratory notebooks, including the original Siamese pair-classification notebooks.

## Related datasets

Most of the raw and derived SERS material also traces back to:

- [Data/Jesse_Dataset](../../Data/Jesse_Dataset)
- [Data/Jesse_Dataset_Update](../../Data/Jesse_Dataset_Update)
