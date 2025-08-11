# Spectral Classification Project

This repository contains implementations of various machine learning and deep learning methods for the **one-shot classification of chemical spectra**. The goal is to identify the **chemical identity** of an unknown **Raman spectrum** by finding the best match from a reference library of known spectra.

The project explores and compares **four distinct approaches**, inspired by the methods described in *[Halas et al., ACS Nano, 2023](https://doi.org/10.1021/acsnano.3c05510)*.


---

## Core Concept: Spectral Identification

The fundamental problem is to take a **high-dimensional spectrum** from an unknown sample and accurately identify it by comparing it against a **library of known spectra**. Each method in this repository tackles this problem through a different **feature extraction** and **similarity matching** strategy.

# Spectral Classification Project

This repository tracks the evolution of one-shot spectral classification for Raman and SERS data.  
The work progresses from implementing the 2023 Halas “CaPSim” method to Siamese-network
approaches for mixtures and SERS spectra, with extensive clustering analyses.

## Project Evolution

1. **CaPSim (Halas et al., ACS Nano 2023)**  
   - Implementation of Characteristic Peak Similarity for single Raman spectra  
   - Code: `Scripts/CaPSim.py`  
   - Notebook: `Notebooks/CaPE_CaPSIM_Notebook.ipynb`

2. **CaPSim + k-NN**  
   - Adds a k-nearest-neighbor classifier on CaPSim features  
   - Code: `Scripts/CaPSim_kNN.py`  
   - Notebook: `Notebooks/CaPSim_with_kNN.ipynb`

3. **Siamese Network for Raman One‑Shot Learning**  
   - 1D convolutional Siamese model replacing hand‑engineered features  
   - Code: `Scripts/SiameseNetwork.py`  
   - Notebook: `Notebooks/Siamese_Network_OneShot.ipynb`

4. **SERS Single‑Spectrum Classification**  
   - Adapts the Siamese approach to SERS signals  
   - Data & notebook: `SERS_to_ Raman/Siamese_Network_OneShot.ipynb`

5. **Raman Mixture Classification (Siamese + MLP)**  
   - Embeddings from a Siamese network fed to an MLP classifier  
   - Notebook: `Mixture_Classification/Notebooks/Siamese_MLP_3.ipynb`

## Clustering & Exploratory Analyses
- `Notebooks/PCA_Centroid_Correlation_Notebook.ipynb` – PCA visualization and centroid correlations  
- `Mixture_Classification/Notebooks/PCA_tSNE_UMAP_HDBSCAN.ipynb` – PCA, t‑SNE, UMAP, and HDBSCAN clustering for standard Raman single spectra and Mixtures
- `SERS_to_Raman` - PCA, t‑SNE, UMAP, and HDBSCAN clustering for SERS data  

## Repository Structure
- `Scripts/` – Core Python implementations (CaPSim, CaPSim_kNN, PCA, Siamese)
- `Notebooks/` – Development notebooks for single-spectrum models
- `SERS_to_ Raman/` – SERS datasets and notebooks
- `Mixture_Classification/` – Mixture datasets, Siamese + MLP workflow, and clustering studies
- `Papers/` – Related literature and presentations

## Reference
- N. J. Halas et al., “Identifying Surface-Enhanced Raman Spectra with a Raman Library Using Machine Learning,” ACS Nano, 2023.


