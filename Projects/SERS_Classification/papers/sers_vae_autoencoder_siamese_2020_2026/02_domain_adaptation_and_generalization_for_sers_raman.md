# 02. Domain Adaptation and Generalization for SERS/Raman

These papers are the main "surface-agnostic" neighbors. They attack domain shifts from instruments, batches, strains, biological replicates, source datasets, and SERS substrate variability.

## 1. Zhang et al. (2026), RSCDM for batch, strain, and instrument variation

- Source: [DOI](https://doi.org/10.1021/acs.analchem.5c07113), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41842761/)
- Architecture: Raman Spectral Classification Discrepancy Model (RSCDM), an unsupervised domain adaptation framework using classifier-output discrepancy plus adversarial feature alignment.
- Goal: improve Raman-based bacterial pathogen identification under batch, strain, and instrument domain shifts.
- Data: seven bacterial species across batches and strains on a commercial spectrometer; six clinical isolates, excluded from training, acquired with a home-built spectrometer.
- Process: detect target samples far from the source feature distribution through disagreement between task-specific classifiers, then adversarially align source/target features. A later fine-tuning step uses reference strains measured on both instruments.
- Result: accuracy improved from 81.6% to 95.4% on commercial-spectrometer batch/strain shifts and from 77.5% to 91.3% on clinical isolates measured by the home-built spectrometer. Fine-tuning boosted clinical-isolate accuracy to 99.3%.
- Why it matters: this is one of the clearest recent templates for handling real spectral domain shift without needing a full retraining set.

## 2. Li et al. (2026), SFAN cross-instrument glioma calibration transfer

- Source: [DOI](https://doi.org/10.1039/d6ay00651e), [PubMed](https://pubmed.ncbi.nlm.nih.gov/42300701/)
- Architecture: Subdomain Feature Alignment Network (SFAN) using Local Maximum Mean Discrepancy (LMMD), soft-hard label weighting, and a two-stage migration strategy.
- Goal: transfer glioma Raman classifiers from a master instrument to a slave instrument without requiring paired transfer samples measured on both devices.
- Data: human glioma Raman dataset; sample count is not specified in the accessible abstract.
- Process: align class-conditional feature distributions rather than mapping spectra directly; use hard labels to guide direction and soft labels to preserve class probability structure.
- Result: outperformed conventional calibration-transfer approaches on the glioma dataset.
- Why it matters: LMMD-style feature alignment is highly relevant to transferring SERS models across substrates or spectrometers.

## 3. Lai et al. (2025), LoRA-CT calibration transfer for deep Raman models

- Source: [DOI](https://doi.org/10.1021/acs.analchem.5c01846), [PubMed](https://pubmed.ncbi.nlm.nih.gov/40922652/)
- Architecture: Low-Rank Adaptation calibration transfer (LoRA-CT) for parameter-efficient fine-tuning of deep Raman models.
- Goal: make deep learning-enhanced Raman models portable across spectrometers with very few transfer samples.
- Data: three datasets, including solvent mixtures and blended oils; detailed sample counts are not specified in the accessible abstract.
- Process: freeze most model weights and train low-rank adaptation modules for each target spectrometer.
- Result: LoRA-CT reduced trainable parameters by about 600x compared with full fine-tuning. On methanol mixtures it achieved R2 = 0.952 and RMSE = 0.072, better than piecewise direct standardization and full fine-tuning baselines.
- Why it matters: this is a practical route for adapting a SERS classifier to a new substrate/instrument while preserving a shared base model.

## 4. Zhang et al. (2025), CDAN-PL for autoimmune Raman diagnosis

- Source: [DOI](https://doi.org/10.3390/s25196186), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41095007/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526587/)
- Architecture: pseudo-label-based Conditional Domain Adversarial Network (CDAN-PL) with spectral data-adaptive feature extraction.
- Goal: perform label-free unsupervised transfer diagnosis of autoimmune diseases from Raman spectra.
- Data: Raman spectra for autoimmune disease diagnosis; exact sample count is not specified in the accessible abstract.
- Process: generate pseudo-labels for unlabeled target-domain spectra, apply conditional adversarial alignment, and learn spectral features that generalize across homologous and non-homologous transfer tasks.
- Result: average homologous-transfer accuracy of 92.3% and non-homologous-transfer accuracy of 90.05%, outperforming baseline UDA models.
- Why it matters: CDAN-style adversarial conditioning is a strong candidate for enforcing substrate-invariant SERS features.

## 5. Liu et al. (2024), MURDA multisource Raman domain adaptation

- Source: [DOI](https://doi.org/10.1021/acs.analchem.4c01581), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39301586/)
- Architecture: Multisource Unsupervised Raman Spectroscopy Domain Adaptation Model with Reconstructed Target Domains (MURDA), plus a Double-Branch Multiscale Convolutional Self-Attention feature extractor.
- Goal: transfer knowledge from several labeled disease-source domains to an unlabeled target disease domain.
- Data: three serum Raman datasets for autoimmune diseases and a validation experiment on the public RRUFF Raman dataset.
- Process: combine multisource domain adaptation with reconstructed target domains, learn multiscale/self-attention spectral features, and analyze important decision peaks.
- Result: reported target-domain accuracies of 73.6%, 83.4%, and 82.9%, improving over no-adaptation source-only tasks by 15.1%, 36%, and 21.6%.
- Why it matters: the multisource setup maps well to training on several known SERS substrates and adapting to a new unknown substrate.

## 6. Grajales et al. (2025), fine-tuning vs test-time adaptation for prostate Raman

- Source: [DOI](https://doi.org/10.1117/1.BIOS.2.3.032706), [PubMed](https://pubmed.ncbi.nlm.nih.gov/42028257/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13052485/)
- Architecture: ResNet-style 1D CNN with multi-organ pretraining, efficient fine-tuning, and test-time adaptation.
- Goal: deploy real-time Raman prostate cancer confirmation with limited prospective data.
- Data: retrospective Raman data from brain, breast, and prostate tissue from 202 patients, plus pre-trained bacterial models; prospective ex vivo prostate data from 10 patients with two to five biopsies each.
- Process: pretrain on broader Raman domains, fine-tune on prostate data, and compare to test-time adaptation when target labels are unavailable.
- Result: fine-tuned model achieved AUC 0.76, accuracy 0.79, sensitivity 0.83, and specificity 0.72; test-time adaptation improved predictions without labels.
- Why it matters: shows how deployment-time adaptation can help when target-domain SERS labels are scarce.

## 7. Wang et al. (2024), transfer contrastive learning for skin-cancer Raman

- Source: [DOI](https://doi.org/10.1109/JBHI.2024.3451950), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39208055/)
- Architecture: Transfer Contrastive Learning Paradigm (TCLP), combining transfer learning with contrastive augmentation.
- Goal: improve skin cancer tissue classification from noisy, scarce Raman spectra.
- Data: Raman skin-cancer tissue data; exact sample count is not specified in the accessible abstract. Pretraining uses Raman data from related domains collected on different equipment/tasks.
- Process: pretrain on related Raman spectra, use contrastive learning to make features robust to noisy augmentations, then classify target skin-cancer spectra.
- Result: outperformed deep learning baselines in reported experiments.
- Why it matters: contrastive pretraining is a strong alternative or complement to VAE pretraining for substrate-invariant SERS features.

## 8. Ho et al. (2022), clinical bacterial Raman identification with deep transfer learning

- Source: [DOI](https://doi.org/10.1021/acs.analchem.2c03391), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36214808/)
- Architecture: ResNet-based deep transfer learning with data augmentation.
- Goal: identify bacterial pathogens directly from clinical isolates at single-cell Raman resolution.
- Data: eight pathogenic bacterial species; blinded validation included cultured and noncultured isolates from clinical sources such as blood, urine, pus, and sputum. Exact sample counts are not specified in the accessible abstract.
- Process: augment spectra, train/transfer a ResNet model, and validate robustness on blinded clinical datasets.
- Result: reported 99.99% classification accuracy and high performance on blinded datasets.
- Why it matters: a useful example of transfer learning under real clinical heterogeneity, though it is not explicitly a SERS substrate-domain paper.

## 9. Bao et al. (2024/2025), Siamese Raman classification with inter-instrument variation

- Source: [DOI](https://doi.org/10.1016/j.saa.2024.125207), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39369591/)
- Architecture: modular Siamese neural network with multiple projection layers and pluggable spectral encoders.
- Goal: classify Raman spectra when instruments and spectral resolutions differ.
- Data: bacterial Raman datasets created by the authors; the model was trained with only 10 spectra per category in the reported low-shot setup.
- Process: encode spectra, map feature distances into spectral similarities, and compare similarity sets rather than classifying raw spectra directly.
- Result: best encoder variant exceeded 90% classification accuracy and supported fusion training/prediction across different spectral resolutions.
- Why it matters: metric-learning with instrument-aware projection layers is a direct match for cross-substrate SERS.

## 10. Ju et al. (2023), CaPSim SERS-to-Raman library matching

- Source: [DOI](https://doi.org/10.1021/acsnano.3c05510), [PubMed](https://pubmed.ncbi.nlm.nih.gov/37910670/)
- Architecture: machine-learning spectral matching metric called Characteristic Peak Similarity (CaPSim).
- Goal: identify unknown SERS spectra using a standard Raman spectral library despite substrate-specific SERS variability.
- Data: SERS spectra collected from variable nanoparticle/nanostructured metallic substrates and compared against standard Raman library spectra; exact count is not specified in the accessible abstract.
- Process: focus similarity scoring on characteristic peaks while tolerating nuisance variables from substrate-specific enhancement patterns.
- Result: CaPSim substantially outperformed existing spectral matching algorithms.
- Why it matters: this is the clearest substrate-variability paper in the set and should inform any SERS "identity vs substrate" latent-space design.
