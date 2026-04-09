# Data Index

This directory holds the top-level dataset collections shared across the repository.

## Main datasets

- [Jesse_Dataset](Jesse_Dataset)
  Original Raman dataset bundle with reference CSVs, unknown samples, and per-compound folders.

- [Jesse_Dataset_v2](Jesse_Dataset_v2)
  Later dataset bundle containing mixtures, pure Raman, pure Raman txt exports, SERS, and unknowns.

- [Jesse_Dataset_Update](Jesse_Dataset_Update)
  Updated Raman and SERS folders used in later exploratory work.

- [Jesse_Dataset_PT2](Jesse_Dataset_PT2)
  PT2 collection with added pure compounds plus `Mix 13` to `Mix 22` real mixture folders and `Mixtures.txt`.

- [Feb26_Spectra](Feb26_Spectra)
  Additional Raman spectra collected on February 26.

- [Test_Data](Test_Data)
  Query/reference CSV splits used in earlier standard Raman experiments.

## Important note

The active mixture unmixing workflow is intentionally self-contained and uses its own localized copy of required inputs under [Mixture_Classification/Unmixing_Pipeline/Data](../Mixture_Classification/Unmixing_Pipeline/Data). Use the top-level dataset folders for historical work, dataset provenance, and cross-project exploration.
