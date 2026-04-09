# Mixture Results Index

This file is the quickest way to find the important mixture-classification results.

## Active results

All active experiment outputs live under [Unmixing_Pipeline/Results](Unmixing_Pipeline/Results).

## Most important result folders

- [pair_nnls_replicate_dictionary](Unmixing_Pipeline/Results/pair_nnls_replicate_dictionary)
  Clean classical binary pair-NNLS benchmark with replicate-aware references.

- [deep_similarity_supervision](Unmixing_Pipeline/Results/deep_similarity_supervision)
  Best clean deep mixture model without pair-specific hand tuning.

- [mixture_only_full_new_dataset_model_comparison.pdf](Unmixing_Pipeline/mixture_only_full_new_dataset_model_comparison.pdf)
  Mixture-only full 17-class comparison PDF across legacy Siamese, the replicate-dictionary pair NNLS baseline, and the similarity-supervised coefficient regressor.

- [best_models_existing_real_comparison](Unmixing_Pipeline/Results/best_models_existing_real_comparison)
  Full 17-class comparison on the original real-mixture set.

- [classical_vs_deep_comparison](Unmixing_Pipeline/Results/classical_vs_deep_comparison)
  Consolidated classical-versus-deep comparison tables.

## Diagnostic and historical result folders

- [pair_nnls_baseline_fallback](Unmixing_Pipeline/Results/pair_nnls_baseline_fallback)
  Diagnostic classical reranking variant.

- [pair_nnls_family_fallback](Unmixing_Pipeline/Results/pair_nnls_family_fallback)
  Diagnostic family-specific classical ceiling, not the main clean benchmark.

- [deep_binary_variant_suite](Unmixing_Pipeline/Results/deep_binary_variant_suite)
  CNN and replicate-decoder deep variants.

- [deep_hybrid_pair_rerank](Unmixing_Pipeline/Results/deep_hybrid_pair_rerank)
  Global deep-plus-classical fusion experiment.

## Legacy Siamese outputs

The old mixture Siamese notebooks and their saved outputs live under [Legacy_Siamese_Pipeline/Notebooks](Legacy_Siamese_Pipeline/Notebooks).
