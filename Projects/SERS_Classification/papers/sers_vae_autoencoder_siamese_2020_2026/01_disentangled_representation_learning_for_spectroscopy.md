# 01. Disentangled Representation Learning for Spectroscopy

These papers are the closest references for factorizing spectra into reusable latent representations. The direct SERS literature is still thin, so this section includes adjacent Raman/IR work where the architecture or loss design is directly transferable to substrate-agnostic SERS.

## 1. Shuai et al. (2024/2025), DMSGL-VAE for multi-source Raman IgAN diagnosis

- Source: [DOI](https://doi.org/10.1016/j.artmed.2024.103053), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39701016/)
- Architecture: multi-source VAE with a decoupling module; latent variables are split into global shared representation and local source-specific representation.
- Goal: fuse serum and urine Raman spectra for immunoglobulin A nephropathy diagnosis while avoiding direct multi-source fusion that can amplify noise.
- Data: serum and urine Raman source domains for IgAN diagnosis; sample count is not specified in the accessible abstract. The VAE reconstruction step doubled the available sample size.
- Process: segment spectra, extract statistical features, encode with VAE, decouple global/local factors, constrain with cross-source reconstruction loss and decoupling loss, classify using fused features, then interpret peaks with SHAP.
- Result: reported test AUC of 0.9958; interpreted shared spectral fingerprints included proteins, hydroxybutyrate, and guanine.
- Why it matters: this is the best architectural neighbor for a SERS model that separates molecular identity from substrate/environment effects.

## 2. Zhu and Tadesse (2025), SpectroGen physical-prior VAE for cross-modality spectra

- Source: [DOI](https://doi.org/10.1016/j.matt.2025.102434), [Matter abstract](https://www.cell.com/matter/abstract/S2590-2385%2825%2900477-1), [MIT News](https://news.mit.edu/2025/checking-quality-materials-just-got-easier-new-ai-tool-1014), [HKU page](https://ece.hku.hk/20260102-1/)
- Architecture: physical-prior VAE that represents spectral peaks with Gaussian/Lorentzian/Voigt-like line-shape priors before generating another modality.
- Goal: generate high-resolution spectra in another spectroscopy modality, such as Raman, IR, or XRD, from a single measured modality.
- Data: public materials/mineral spectral datasets; public descriptions report more than 6,000 mineral samples available and several hundred used for model training/evaluation.
- Process: convert spectral signatures into distribution-based physical representations, encode them in a VAE latent space, and decode into target-modality spectra.
- Result: public abstract reports 99% correlation to ground truth and RMSE of 0.01 a.u.
- Why it matters: it suggests a practical way to bake peak-shape priors into a SERS VAE, reducing the chance that the latent space learns arbitrary nuisance variation.

## 3. Paidi and Maheshwari (2025), RamanMAE masked autoencoder

- Source: [DOI](https://doi.org/10.1021/acs.analchem.5c05656), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41269768/)
- Architecture: masked autoencoder trained as a spectral language model on large Raman spectral datasets.
- Goal: learn biologically meaningful spectral representations that can transfer to downstream molecular imaging tasks with limited labels.
- Data: large Raman spectral datasets from biological applications; exact spectrum count is not specified in the accessible abstract.
- Process: mask spectral patches, reconstruct them with the decoder, use the latent representation for downstream tasks, and use the decoder as a smoothing/noise-reduction tool for spectral maps.
- Result: strong masked-patch reconstruction; learned latent representations captured biological composition and transferred between biological applications.
- Why it matters: masked reconstruction is a strong pretraining option for small SERS datasets before supervised or Siamese fine-tuning.

## 4. Yao et al. (2025), DiffRaman with VQ-VAE and latent diffusion

- Source: [DOI](https://doi.org/10.1016/j.aca.2025.344372), [PubMed](https://pubmed.ncbi.nlm.nih.gov/40903108/)
- Architecture: conditional latent denoising diffusion model built on a Vector Quantized VAE encoder/decoder.
- Goal: generate realistic bacterial Raman spectra under limited-data conditions and improve bacterial identification.
- Data: bacterial Raman spectra; exact spectrum count is not specified in the accessible abstract.
- Process: transform 1D spectra into 2D representations, compress with VQ-VAE, model the latent space with conditional DDPM, then decode generated latent samples back into spectra.
- Result: generated spectra mimicked real spectra and improved diagnostic model performance in data-scarce settings.
- Why it matters: VQ-VAE plus diffusion is a modern augmentation path for low-shot SERS where ordinary oversampling is too weak.

## 5. Liu et al. (2023), Raman pathogenic bacteria classification with VAE + LSTM

- Source: [DOI](https://doi.org/10.1002/jbio.202200270), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36519533/)
- Architecture: VAE for spectral generation/denoising plus LSTM classifier.
- Goal: reduce the number of single-cell Raman spectra needed for pathogen classification.
- Data: Raman signals of pathogens; the study reduced required collected spectra from 1000 to 200 for the reported setup.
- Process: train VAE on pathogen spectra, generate additional high-SNR spectra, combine generated and real spectra, then train the LSTM classifier.
- Result: average pathogen classification accuracy reached 96.9%.
- Why it matters: directly supports VAE augmentation when culturing or measuring enough spectra is expensive.

## 6. He et al. (2022), Raman VAE for tumor subtype detection

- Source: [DOI](https://doi.org/10.1021/acsomega.1c07263), [PubMed](https://pubmed.ncbi.nlm.nih.gov/35382336/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8973095/)
- Architecture: VAE for dimensionality reduction and denoising, followed by Gaussian naive Bayes.
- Goal: classify cancer subtypes from complex Raman spectra at cellular and tissue levels.
- Data: Raman spectra from three non-small-cell lung cancer cell subtypes and two kidney cancer tissue subtypes; exact spectrum count should be checked in the full text.
- Process: encode high-dimensional Raman spectra into a 2D VAE latent space, then classify tumor subtype from latent coordinates.
- Result: VAE latent features substantially outperformed classification from original spectra.
- Why it matters: simple evidence that VAE latents can improve downstream spectral classifiers even without a complex classifier.

## 7. Kazemzadeh et al. (2024), interpretable deep autoencoder for Raman/SERS mixtures

- Source: [DOI](https://doi.org/10.1364/BOE.522376), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39022543/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11249694/)
- Architecture: tunable interpretable deep autoencoder.
- Goal: analyze complex Raman/SERS datasets including synthetic data, chemical mixtures, a milling reaction, and extracellular vesicle mixtures.
- Data: several small and heterogeneous Raman/SERS datasets; exact sample counts vary by experiment and should be taken from the full text.
- Process: learn compressed representations, compare with PCA/UMAP, interpolate/fill unknown gaps in spectral datasets, and quantify mixture ratios.
- Result: handled small datasets, generalized across gaps, and quantified relative EV mixture ratios.
- Why it matters: directly relevant to disentangling mixture composition and nuisance variation in SERS.

## 8. Guo et al. (2022), CVDE convolutional VAE deep embedding clustering for Raman spectra

- Source: [DOI](https://doi.org/10.1039/d2ay01184k), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36169059/)
- Architecture: convolutional variational autoencoder deep embedding clustering method, replacing fully connected VAE-GMM layers with convolution and pooling layers.
- Goal: unsupervised clustering of Raman spectra without labels.
- Data: MNIST benchmark plus two Raman datasets: soybean oil spectra with small spectral differences and drug spectra with small sample size.
- Process: learn VAE latent embeddings, cluster in the latent space, and use Grad-CAM to visualize spectral features driving clusters.
- Result: clustering accuracies of 94.48% on MNIST, 90.43% on soybean oil Raman, and 98.70% on drug Raman.
- Why it matters: useful for discovering latent chemical/substrate clusters before supervised SERS model building.

## 9. Grossutti et al. (2022), beta-VAE for IR spectroscopy representation learning

- Source: [DOI](https://doi.org/10.1021/acs.jpclett.2c01328), [PubMed](https://pubmed.ncbi.nlm.nih.gov/35726872/)
- Architecture: beta-variational autoencoder for disentangled latent representations.
- Goal: extract independent generative factors from complex IR spectra of cross-linked polyethylene pipe.
- Data: database of PEX-a pipe IR spectra plus hyperspectral IR imagery of a crack in a pipe wall.
- Process: train beta-VAE, compare latent factors with PCA, and map learned representations spatially across a crack hyperspectrum.
- Result: beta-VAE outperformed PCA and learned interpretable, independent representations of spectral variance.
- Why it matters: a strong model pattern for separating global molecular identity from physical aging/degradation or substrate effects.

## 10. Grossutti et al. (2023), beta-VAE generative modeling of IR hyperspectral images

- Source: [DOI](https://doi.org/10.1021/acsami.3c02564), [PubMed](https://pubmed.ncbi.nlm.nih.gov/37097086/)
- Architecture: beta-VAE deep generative model for hyperspectral IR images.
- Goal: learn disentangled factors of aging, degradation, and cracking in PEX-a pipe.
- Data: high-resolution hyperspectral IR images from unused virgin, used in-service, and cracked PEX-a pipe cross-sections.
- Process: train beta-VAE on spectra, identify physicochemical latent factors, and map those factors back onto spatial IR images.
- Result: identified three distinct physicochemical factors associated with aging/degradation and visualized them spatially.
- Why it matters: shows how latent factors can become interpretable spatial maps, which is relevant for SERS maps and substrate heterogeneity.
