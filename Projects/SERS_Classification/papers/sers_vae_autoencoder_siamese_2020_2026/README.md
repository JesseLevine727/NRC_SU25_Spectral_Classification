# SERS VAE, Autoencoder, Siamese Literature Review (2020-2026)

Created: 2026-06-26

This folder collects recent papers relevant to a substrate-agnostic SERS classification pipeline built around VAE/autoencoder representation learning, domain adaptation, and Siamese/metric learning.

## Files

- [01_disentangled_representation_learning_for_spectroscopy.md](01_disentangled_representation_learning_for_spectroscopy.md): 10 VAE/autoencoder/generative-representation papers for Raman, SERS, IR, and cross-modality spectroscopy.
- [02_domain_adaptation_and_generalization_for_sers_raman.md](02_domain_adaptation_and_generalization_for_sers_raman.md): 10 papers on Raman/SERS domain adaptation, instrument transfer, contrastive transfer, and substrate/instrument variability.
- [03_siamese_metric_learning_for_spectral_identification.md](03_siamese_metric_learning_for_spectral_identification.md): 10 Siamese, pseudo-Siamese, contrastive, and deep metric learning papers for spectral matching and molecular identification.
- [04_sers_deep_learning_autoencoders_siamese.md](04_sers_deep_learning_autoencoders_siamese.md): 10 SERS-specific deep learning papers, emphasizing autoencoders, VAEs/adversarial autoencoders, and Siamese networks.

## Selection Rules

- Date window: 2020 through the current literature visible as of 2026-06-26.
- Priority: direct SERS/Raman papers first; adjacent IR/materials spectroscopy papers were included only when they directly demonstrate transferable VAE/autoencoder disentanglement, physical-prior latent structure, or cross-domain spectral representation learning.
- Focus: VAE, autoencoders, denoising autoencoders, masked autoencoders, adversarial autoencoders, Siamese networks, pseudo-Siamese networks, contrastive learning, and deep metric learning.
- Data details: sample counts are stated only when available in the abstract or accessible full-text metadata. Otherwise the note says "not specified in accessible abstract".

## Fast Reading Order

For a substrate-agnostic SERS + VAE/Siamese project, read these first:

1. Shuai et al. 2024, DMSGL-VAE: closest paper to explicit source-domain decoupling in Raman spectra.
2. Zaki et al. 2025, explainable SERS bioquantification: clean denoising-autoencoder plus CNN/Vision Transformer pipeline.
3. Guo and Bocklitz 2026, Siamese networks against replicate variability: directly attacks replicate/domain variation.
4. Bao et al. 2024, Siamese Raman under inter-instrument variation: low-shot cross-instrument Raman classification.
5. Zhang et al. 2026, RSCDM: modern unsupervised domain adaptation for batch, strain, and instrument shifts.
6. Ju et al. 2023, CaPSim SERS-to-Raman-library matching: practical answer to substrate-specific SERS variability.
7. He et al. 2022, Raman VAE tumor subtype detection: compact VAE latent features for classification.
8. Kazemzadeh et al. 2024, interpretable deep autoencoder for SERS mixtures/EVs: closest to mixture factor analysis.
9. Li et al. 2022, contrastive Raman spectrum matching: one-reference spectral matching without heavy preprocessing.
10. Gao et al. 2025, KAN-AAE synthetic SERS spectra: adversarial autoencoder augmentation for scarce SERS cancer data.

## Caveats

- "Disentangled VAE for SERS" remains a sparse exact phrase. The strongest design evidence comes from adjacent Raman/IR representation-learning papers plus SERS autoencoder and Siamese papers.
- Many biomedical Raman/SERS papers report high accuracy on limited or private data. For model design, prioritize papers that explicitly evaluate held-out domains, instruments, replicates, or blinded samples.
- Before formal citation, re-check publisher pages for final volume/page metadata and any corrections.
