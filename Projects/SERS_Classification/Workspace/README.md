# Workspace Layout

This workspace is organized so source material is separated from the current substrate-agnostic story.

- `data/processed/`
  Processed CSV datasets used by scripts, including `consolidated_SERS.csv`.
- `data/raw_by_substrate/`
  Raw SERS text files grouped by substrate folder: `Ag`, `AgNP`, `Au`, `AuNP`, `PICO`, and `pSERS`.
- `data/raw_curated/`
  Curated raw SERS hierarchy grouped by chemical and substrate.
- `data/raw_exports/`
  Older exported raw SERS text collections.
- `data/reference/`
  Processed Raman/reference CSVs.
- `references/`
  Raw Raman/reference source folders.
- `notebooks/`
  Legacy exploratory notebooks, including original Siamese chemical-substrate pair work.
- `substrate_agnostic/`
  Current substrate-agnostic model outputs, diagnostics, sweeps, and archived comparison runs.
- `unknown_labeled/`
  Labeled unknown spectra retained separately from the current substrate-agnostic pipeline.
