# Unmixing Results

This directory contains the active saved outputs for mixture-classification experiments.

## Recommended entry points

- [best_models_full_new_dataset_comparison](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Unmixing_Pipeline/Results/best_models_full_new_dataset_comparison)
  Start here for the main mixture-only full 17-class comparison across legacy Siamese, clean classical, and clean deep models.

- [pair_nnls_replicate_dictionary](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Unmixing_Pipeline/Results/pair_nnls_replicate_dictionary)
  Main clean classical benchmark.

- [deep_similarity_supervision](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Unmixing_Pipeline/Results/deep_similarity_supervision)
  Main clean deep benchmark.

- [best_models_existing_real_comparison](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Mixture_Classification/Unmixing_Pipeline/Results/best_models_existing_real_comparison)
  Full 17-class comparison on the original real-mixture dataset only.

## Naming convention

- `pair_*`, `exhaustive_*`, `full_library_*`, `cardinality_*`
  Classical unmixing and support-selection experiments.

- `deep_*`
  Library-constrained deep mixture models.

- `best_models_*` and `classical_vs_deep_comparison`
  Comparison artifacts and final tables.

- `*_fallback`
  Diagnostic engineering variants, not the main clean benchmarks.
