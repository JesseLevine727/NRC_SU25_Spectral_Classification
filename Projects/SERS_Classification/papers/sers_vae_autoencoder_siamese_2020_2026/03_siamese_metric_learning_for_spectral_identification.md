# 03. Siamese and Metric Learning for Spectral Identification

These papers support learning a distance/similarity function over spectra instead of only learning a closed-set classifier. That matters for SERS because substrate changes can move spectra while preserving molecular identity.

## 1. Guo and Bocklitz (2026), Siamese networks against replicate variability

- Source: [DOI](https://doi.org/10.1016/j.talanta.2026.129422), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41579739/)
- Architecture: Siamese neural network compared with model transfer methods such as score movement and extended multiplicative scatter correction.
- Goal: improve Raman model generalization when biological replicates differ between train and test data.
- Data: Raman spectra from four bacterial species with nine biological replicates per species; generalization also tested on a second mouse tissue Raman dataset.
- Process: train similarity-based embeddings and incorporate variability into the loss function rather than requiring target test data for adjustment.
- Result: Siamese network outperformed conventional transfer methods, especially with larger training datasets.
- Why it matters: directly relevant to replicate and substrate variability, and it does not require target-domain spectra at prediction time.

## 2. Bao et al. (2024/2025), inter-instrument Siamese Raman classification

- Source: [DOI](https://doi.org/10.1016/j.saa.2024.125207), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39369591/)
- Architecture: modular Siamese neural network with projection layers and swappable spectral encoder modules.
- Goal: classify biological Raman spectra under inter-instrument and resolution variation.
- Data: author-created bacterial Raman datasets; low-shot experiments used 10 spectra per category.
- Process: extract features from spectra, convert feature distances to similarities, and classify by comparing similarity sets.
- Result: best model exceeded 90% accuracy and enabled training/prediction across different spectral resolutions.
- Why it matters: a strong design reference for a SERS Siamese branch where each substrate/instrument has slightly different spectral sampling.

## 3. Contreras et al. (2024), Siamese networks for clinically relevant bacteria

- Source: [DOI](https://doi.org/10.3390/molecules29051061), [PubMed](https://pubmed.ncbi.nlm.nih.gov/38474573/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934697/)
- Architecture: two Siamese CNN variants compared with classical ML, shallow CNNs, and deeper CNNs.
- Goal: classify clinically relevant bacteria from Raman spectra while reducing the retraining cost for new classes.
- Data: Raman spectral datasets of bacteria; exact spectrum counts should be checked in the full text.
- Process: train twin CNN branches with a distance/similarity head, then compare sensitivity, training time, prediction time, and parameter count across models.
- Result: Siamese-model2 achieved mean sensitivity of 83.61 +/- 4.73 and 73% prediction accuracy in limited/unbalanced data conditions.
- Why it matters: useful tradeoff paper for choosing Siamese models when SERS classes are imbalanced or under-sampled.

## 4. Mou et al. (2024), pseudo-Siamese CNN for antibiotics in human milk

- Source: [DOI](https://doi.org/10.1016/j.fochx.2024.101507), [PubMed](https://pubmed.ncbi.nlm.nih.gov/38855098/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11157215/)
- Architecture: CNN for recognition plus Independent Component Analysis and pseudo-Siamese CNN for mixed-antibiotic ratio quantification.
- Goal: classify and quantify trace doxycycline and tetracycline residues in human milk using label-free SERS.
- Data: constructed human-milk system containing doxycycline and/or tetracycline; exact sample count is not specified in the accessible abstract.
- Process: classify single/mixed antibiotic spectra, then use ICA and pseudo-Siamese CNN to quantify component ratios in mixtures.
- Result: CNN recognition accuracy reached 98.85% under the best hyperparameter combination.
- Why it matters: strong example of pseudo-Siamese learning for mixture decomposition, which is relevant to overlapping SERS peaks.

## 5. Fan et al. (2023), DeepRaman pseudo-Siamese component identification

- Source: [DOI](https://doi.org/10.1021/acs.analchem.2c03853), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36908216/)
- Architecture: pseudo-Siamese neural network with spatial pyramid pooling for input-shape flexibility.
- Goal: identify components in Raman spectra despite mixture interference, noise, baseline shifts, and spectrometer differences.
- Data: 41,564 augmented Raman spectra from two databases: pharmaceutical materials and S.T. Japan. Six additional measured datasets from different instruments were used for evaluation.
- Process: train a comparison model that can handle variable spectral shapes and evaluate on different instruments, complexity levels, low-content components, SERS data, and Raman imaging data.
- Result: test accuracy 96.29%, true positive rate 98.40%, and true negative rate 94.36%; outperformed hit quality index and other deep learning models.
- Why it matters: a mature pseudo-Siamese recipe for matching unknown spectra to references without retraining per dataset.

## 6. Tian et al. (2023), Siamese-like blood species Raman similarity

- Source: [DOI](https://doi.org/10.1002/jbio.202200377), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36906736/)
- Architecture: Siamese-like neural network for Raman spectral similarity measurement.
- Goal: identify blood species for customs inspection, forensics, and wildlife protection.
- Data: Raman spectra from blood of 22 species.
- Process: train similarity-based classifier, test on known species spectra excluded from training, and update the model with new species without full retraining.
- Result: average test accuracy above 99.20%; worked better than alternatives under small-dataset settings.
- Why it matters: supports using similarity learning when new analytes/classes will be added over time.

## 7. Cai et al. (2023), deep metric learning plus GADF for handheld Raman

- Source: [DOI](https://doi.org/10.1016/j.saa.2023.123085), [PubMed](https://pubmed.ncbi.nlm.nih.gov/37454497/)
- Architecture: deep metric learning network trained on Gramian Angular Difference Field images generated from Raman spectra.
- Goal: make Raman identification robust across portable/handheld devices and large class counts.
- Data: 450 mineral classes from the RRUFF database for training; evaluated on 260 mineral classes, eight pathogenic bacteria classes, and 350 chemical samples across 32 classes on a handheld device.
- Process: convert spectra with different resolutions into same-resolution GADF images, learn nonlinear inter-class distances, and deploy on embedded hardware.
- Result: 98.05% accuracy on 260 mineral classes, 90.13% on noisy eight-class pathogenic bacteria, and 99.14% on 32-class handheld chemical identification.
- Why it matters: demonstrates metric learning as a scalable alternative to closed-set classifiers when class count grows.

## 8. Li et al. (2022), contrastive representation learning for Raman spectrum matching

- Source: [DOI](https://doi.org/10.1039/d2an00403h), [PubMed](https://pubmed.ncbi.nlm.nih.gov/35474361/)
- Architecture: contrastive representation learning for spectrum identification.
- Goal: identify spectra by matching to a reference database without preprocessing and with as little as one reference spectrum per analyte.
- Data: two Raman spectral datasets and one single-component SERS dataset.
- Process: learn contrastive embeddings, compare candidate/reference spectra, and optionally use conformal prediction to increase candidate set size.
- Result: improved or matched state-of-the-art analyte identification accuracy.
- Why it matters: one-reference-per-analyte matching is directly aligned with practical unknown SERS library search.

## 9. Skvortsova et al. (2021/2022), SERS Siamese detection of beta-lactam resistance gene fragment

- Source: [DOI](https://doi.org/10.1016/j.aca.2021.339373), [PubMed](https://pubmed.ncbi.nlm.nih.gov/35057931/)
- Architecture: Siamese neural network combined with robust statistics and Bayesian decision theory.
- Goal: detect a specific oligonucleotide sequence corresponding to a blaNDM-1 beta-lactam antibiotic resistance gene fragment.
- Data: DNA-targeted SERS on plasmonic gold grating functionalized with capture oligonucleotides; target detection down to 3 x 10^-12 M against 10^-10 M similar non-target background.
- Process: collect SERS spectra from functionalized substrates, compare spectra with a Siamese model, then use decision theory to control confidence/error and determine required spectra/sample count.
- Result: target detected with at least 99% confidence at picomolar levels.
- Why it matters: strong example of Siamese SERS under complex background and explicit confidence control.

## 10. Park et al. (2021), dynamic one-shot pseudo-Siamese Raman target detection

- Source: [DOI](https://doi.org/10.1039/d1an01352a), [PubMed](https://pubmed.ncbi.nlm.nih.gov/34676386/)
- Architecture: pseudo-Siamese network for one-shot detection and classification.
- Goal: detect and classify biological/chemical defense targets from Raman spectra without preprocessing and without retraining for untrained target classes.
- Data: Raman spectra measured with a Raman spectrometer; exact sample count is not specified in the accessible abstract.
- Process: compare observed spectra to reference/target spectra using a pseudo-Siamese model rather than relying on conventional preprocessing plus GLRT/ICA/NMF-style detection.
- Result: demonstrated one-shot detection/classification and no-preprocessing operation.
- Why it matters: useful for field SERS workflows where only one or a few reference spectra may exist for a new compound.
