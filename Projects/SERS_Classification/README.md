# SERS Classification

This directory groups the SERS-focused classification and exploration work.

## Main area

- [Workspace](Workspace)
  Mixed notebook-and-data workspace containing SERS references, cleaned CSVs, txt exports, clustering notebooks, and Siamese one-shot experiments.

## Current substrate-agnostic work

- [SERS_SUBSTRATE_AGNOSTIC_AUDIT.md](SERS_SUBSTRATE_AGNOSTIC_AUDIT.md)
  Documents why the original notebook results were chemical-substrate pair classification and summarizes the current leave-one-substrate-out substrate-agnostic baselines.
- [AGNP_FAILURE_DIAGNOSTIC.md](AGNP_FAILURE_DIAGNOSTIC.md)
  Details the current AgNP failure mode after canonicalizing `bt -> benzenethiol`.
- `sers_siamese_substrate_agnostic.py`
  CUDA-first Siamese/triplet training script for substrate-held-out chemical classification.
- `sers_agnp_deep_dive.py`
  Generates the average spectra, PCA, prototype-distance, and raw-file diagnostics in `Workspace/agnp_diagnostics/`.

## Typical contents in `Workspace`

- raw and cleaned SERS CSVs
- Raman reference CSVs used for comparison
- notebook-driven exploratory analysis
- per-platform folders such as `Ag`, `AgNP`, `Au`, `AuNP`, `PICO`, and `pSERS`

## Related datasets

Most of the raw and derived SERS material also traces back to:

- [Data/Jesse_Dataset](../../Data/Jesse_Dataset)
- [Data/Jesse_Dataset_Update](../../Data/Jesse_Dataset_Update)
