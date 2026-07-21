# 04. SERS Deep Learning Papers Focused on Autoencoders, VAEs, and Siamese Networks

These papers are the most directly useful for SERS workflows. They cover label-free SERS, biofluids, pathogens, biomarkers, aerosols, mixtures, and substrate-enabled detection, with emphasis on autoencoders, denoising autoencoders, adversarial autoencoders, and Siamese networks.

## 1. Qiu et al. (2025), autoencoder-SVM differential SERS for parathyroidectomy assessment

- Source: [DOI](https://doi.org/10.1021/acs.nanolett.5c02299), [PubMed](https://pubmed.ncbi.nlm.nih.gov/40586615/)
- Architecture: autoencoder feature extraction plus SVM classifier on label-free differential SERS spectra.
- Goal: rapidly assess complete vs partial parathyroid gland resection intraoperatively.
- Data: 2 uL untreated plasma per measurement; internal test set and independent validation cohort reported as n = 144 and 33 spectra.
- Process: compare postoperative vs preoperative plasma spectra, use differential SERS to reduce individual variability, encode features with an autoencoder, and classify with SVM.
- Result: 16 minute workflow; 95.8% accuracy on the internal test set and 79% accuracy on the independent validation cohort.
- Why it matters: practical example of using differential spectra to remove patient-specific nuisance variation, analogous to subtracting substrate/background effects.

## 2. Zaki et al. (2025), explainable SERS bioquantification with denoising autoencoder

- Source: [DOI](https://doi.org/10.1021/acssensors.5c01058), [PubMed](https://pubmed.ncbi.nlm.nih.gov/40892429/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481578/)
- Architecture: denoising autoencoder for spectral enhancement, CNNs and Vision Transformers for quantification, and CRIME for context-aware explainability.
- Goal: quantify serotonin in urine from SERS spectra and expose spectral contexts that drive predictions.
- Data: 682 SERS spectra measured in a micromolar serotonin range using cucurbit[8]uril chemical spacers.
- Process: preprocess spectra, denoise with an autoencoder, train CNN/ViT quantifiers, and explain model contexts with CRIME.
- Result: CNN with three-parameter logistic output on denoised spectra achieved MAE = 0.15 uM and mean percentage error = 4.67%.
- Why it matters: one of the cleanest recent SERS pipelines combining denoising, deep prediction, and explainability.

## 3. Ciloglu et al. (2022), autoencoder-SVM for colistin-resistant Klebsiella pneumoniae

- Source: [DOI](https://doi.org/10.1016/j.aca.2022.340094), [PubMed](https://pubmed.ncbi.nlm.nih.gov/35934394/)
- Architecture: autoencoder nonlinear feature extractor plus SVM classifier; compared with PCA-SVM.
- Goal: distinguish colistin-resistant and colistin-susceptible Klebsiella pneumoniae using label-free SERS.
- Data: 16 K. pneumoniae strains incubated in tryptic soy broth for 4 hours.
- Process: collect SERS spectra, extract nonlinear features with an autoencoder, classify with SVM, and compare against linear PCA features.
- Result: autoencoder-SVM achieved 94% accuracy, 94.2% sensitivity, 93.8% specificity, and AUC 0.98.
- Why it matters: direct evidence that nonlinear autoencoder features can beat PCA features for resistance classification from SERS.

## 4. Hwang et al. (2022), Au-TiO2 SERS face mask with ablation-assisted autoencoder

- Source: [DOI](https://doi.org/10.1021/acsami.2c16446), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36448483/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718102/)
- Architecture: ablation-assisted autoencoder for quantitative SERS analysis.
- Goal: detect SARS-CoV-2 in respiratory aerosols using a SERS face-mask substrate.
- Data: spike proteins in artificial respiratory aerosols at 100 pM; SARS-CoV-2 lysates from 10^1 to 10^4 pfu/mL, comparable to PCR cycle thresholds 19 to 29.
- Process: use Au-TiO2 nanocomposite SERS mask to preconcentrate aerosols, measure SERS spectra, remove nondiscriminant features with ablation, and quantify with autoencoder-aided deep learning.
- Result: Au-TiO2 improved SERS signal intensity by 47% over simple Au nanoislands; quantitative assay accuracy exceeded 98%.
- Why it matters: shows how substrate engineering and autoencoder feature selection can be coupled.

## 5. Ciloglu et al. (2021), stacked-autoencoder DNN for MRSA/MSSA SERS

- Source: [DOI](https://doi.org/10.1038/s41598-021-97882-4), [PubMed](https://pubmed.ncbi.nlm.nih.gov/34531449/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8446005/)
- Architecture: stacked autoencoder-based deep neural network.
- Goal: identify methicillin-resistant and methicillin-sensitive Staphylococcus aureus from label-free SERS spectra.
- Data: MRSA and MSSA bacteria measured by SERS; exact spectrum count should be checked in the full text.
- Process: feed raw SERS spectra to SAE-DNN, compare with traditional classifiers, and evaluate with statistical analysis.
- Result: 97.66% accuracy and AUC 0.99, outperforming traditional classifiers.
- Why it matters: early but still highly relevant example of autoencoder feature learning from raw SERS spectra.

## 6. Gao et al. (2025), KAN-AAE synthetic SERS spectra for cancer identification

- Source: [DOI](https://doi.org/10.1016/j.saa.2025.125696), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39798513/)
- Architecture: Kolmogorov-Arnold Networks combined with Adversarial Autoencoder (KAN-AAE) for synthetic SERS generation.
- Goal: improve classification of scarce serum SERS data for cancer identification.
- Data: serum samples from four cancer types, two other disease groups, and healthy individuals; exact sample count is not specified in the accessible abstract.
- Process: train KAN-AAE on real serum SERS spectra, generate synthetic spectra, combine synthetic and real spectra, and train multiple classifiers.
- Result: synthetic augmentation improved classifier accuracy by about 1% to 3%; KAN classifier reached 95.62%.
- Why it matters: directly supports adversarial-autoencoder augmentation for scarce biomedical SERS.

## 7. Kazemzadeh et al. (2024), interpretable deep autoencoder for SERS mixtures and EVs

- Source: [DOI](https://doi.org/10.1364/BOE.522376), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39022543/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11249694/)
- Architecture: tunable interpretable deep autoencoder.
- Goal: analyze chemical mixtures, reaction dynamics, and extracellular-vesicle mixtures from Raman/SERS data.
- Data: synthetic datasets, chemical mixtures, chemical milling reaction data, and extracellular vesicle mixtures; exact counts vary by experiment.
- Process: compress spectra into interpretable representations, compare with PCA/UMAP, interpolate missing regions of spectral datasets, and estimate mixture ratios.
- Result: handled small datasets and quantified relative ratios of cell-line-derived EVs to fetal-bovine-serum-derived EVs.
- Why it matters: valuable template for SERS mixture disentanglement and latent-space interpretability.

## 8. Das et al. (2023), SERS nanowire chip plus Siamese networks for bacteria and AMR

- Source: [DOI](https://doi.org/10.1021/acsami.3c00612), [PubMed](https://pubmed.ncbi.nlm.nih.gov/37158639/)
- Architecture: Siamese neural networks for species-level and resistant/susceptible strain classification.
- Goal: identify wild-type and antibiotic-resistant bacteria using a reproducible SERS nanowire chip.
- Data: SERS chips detected R6G down to 10^-12 M and bacteria down to 100 CFU/mL. The model identified 12 bacterial species and differentiated resistant vs susceptible E. coli; synthetic urine tests used 10^3 CFU/mL E. coli.
- Process: fabricate silver nanoparticle-loaded silicon nanowire SERS chips, acquire spectra, train Siamese models for species and AMR classification, and test direct detection in synthetic urine.
- Result: demonstrated low-limit bacterial detection and Siamese-based classification across clinically relevant species/strains.
- Why it matters: direct SERS + Siamese evidence for substrate-enabled pathogen identification and AMR classification.

## 9. Thrift et al. (2020), SERS deep learning for rapid antimicrobial susceptibility testing

- Source: [DOI](https://doi.org/10.1021/acsnano.0c05693), [PubMed](https://pubmed.ncbi.nlm.nih.gov/33095005/)
- Architecture: deep neural network and unsupervised Bayesian Gaussian mixture analysis over SERS spectral latent structure.
- Goal: rapidly determine antimicrobial susceptibility from bacterial lysate metabolic profiles after antibiotic exposure.
- Data: SERS spectra from Escherichia coli and Pseudomonas aeruginosa after antibiotic exposure; exact spectrum counts should be checked in the full text.
- Process: use controlled-nanogap SERS sensors, expose bacteria to antibiotics, acquire spectra after short exposure windows, and classify susceptible vs resistant metabolic response.
- Result: DNNs discriminated antibiotic response in 10 minutes after exposure; a culture-free dataset enabled 30 minute treatment selection, and Bayesian Gaussian mixture analysis achieved 99.3% susceptible/resistant discrimination.
- Why it matters: foundational SERS + deep learning AST paper and useful for thinking about latent metabolic response spaces.

## 10. Rashidi et al. (2025), SERS Siamese CNN for multiplex naphthenic acid profiling

- Source: [DOI](https://doi.org/10.1021/acs.analchem.5c04463), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41369273/)
- Architecture: random forest and ridge regression baselines plus Siamese convolutional neural network for multilabel mixture identification.
- Goal: detect and quantify multiple naphthenic acid types in water without extraction or separation.
- Data: eight naphthenic acid types across classical linear, cyclic, and heteroatom-containing groups; detection limits 10^-4 to 10^-5 M.
- Process: use uniform Ag nanoparticles plus cationic surfactant to enhance acid-particle interactions, train classifiers/regressors on transformed spectra, and compare mixture fingerprints to individual references with a Siamese CNN.
- Result: single-acid random forest accuracy 86.3%; ridge regression average R2 about 99.5% for concentration prediction; Siamese CNN reached about 95% overall identification accuracy and about 95% averaged F1.
- Why it matters: strong SERS example of Siamese comparison for complex mixture fingerprints.
