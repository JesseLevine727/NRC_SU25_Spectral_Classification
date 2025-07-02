# Spectral Classification Project

This repository contains implementations of various machine learning and deep learning methods for the **one-shot classification of chemical spectra**. The goal is to identify the **chemical identity** of an unknown **Raman spectrum** by finding the best match from a reference library of known spectra.

The project explores and compares **four distinct approaches**, inspired by the methods described in *[Halas et al., ACS Nano, 2023](https://doi.org/10.1021/acsnano.3c05510)*.


---

## Core Concept: Spectral Identification

The fundamental problem is to take a **high-dimensional spectrum** from an unknown sample and accurately identify it by comparing it against a **library of known spectra**. Each method in this repository tackles this problem through a different **feature extraction** and **similarity matching** strategy.

---

## Included Implementations

This repository includes four primary methods for spectral classification:

- `CaPSim` (Characteristic Peak Similarity)
- `CaPSim` with `k-NN` Classifier
- `PCA` (Principal Component Analysis) Classifier
- `Siamese Network` for One-Shot Learning

All methods assume the data is provided in **CSV format**, with one file for the **reference library** and one for the **query spectra**. The **last column** in each file should be named **`Label`** and contain the chemical identifier.

---

### 1. CaPSim (Characteristic Peak Similarity)
**Script:** `CaPSim.py`  
**Approach:** Local peak-based dot-product similarity

**Concept:**  
This method directly implements the approach described in the Halas et al. paper. It assumes that key information in a spectrum is concentrated in a few discrete **characteristic peaks (CPs)**.

**Workflow:**
- **Preprocessing:** Baseline correction and L2 normalization
- **Feature Definition:** Identify most frequent or intense peaks per class → class-specific CPs
- **Feature Extraction:** Extract maximum intensity around CPs for each query
- **Similarity Matching:** Compute mean dot-product (CaPSim score); assign query to class with highest score

---

### 2. CaPSim with k-NN Classifier
**Script:** `CaPSim_kNN.py`  
**Approach:** Local peak-based features with supervised classification

**Concept:**  
An extension of CaPSim that uses the same CP-based features but replaces dot-product similarity with a learned `k-Nearest Neighbors` classifier.

**Workflow:**
- **Preprocessing:** Same as baseline CaPSim
- **Feature Definition:** Global CP set = union of all class CPs
- **Feature Extraction:** Fixed-length feature vectors for all reference spectra
- **Training:** Train a `k-NN` model using cosine distance
- **Classification:** Predict query identity using trained model

---

### 3. PCA (Principal Component Analysis) Classifier
**Script:** `PCA_Classification.py`  
**Approach:** Global shape-based classification in PCA space

**Concept:**  
A benchmark method that uses global spectral shape via PCA. It reduces the dimensionality of the spectra and uses **class centroids** in PCA space for classification.

**Workflow:**
- **Preprocessing:** Baseline correction and normalization
- **Feature Extraction:** Fit PCA on reference spectra → project all spectra
- **Centroid Calculation:** Compute class average in PCA space
- **Classification:** Classify query by nearest centroid (Euclidean distance)

---

### 4. Siamese Network for One-Shot Learning
**Script:** `SiameseNetwork.py`  
**Approach:** Deep learning with contrastive similarity learning

**Concept:**  
A deep learning model that learns a mapping from spectra into a **low-dimensional embedding space**, where similar spectra are close together.

**Workflow:**
- **Training:** Train a 1D convolutional Siamese network using spectrum pairs and contrastive loss
- **Embedding:** Use network to embed all reference/query spectra
- **Classification:** Query is classified via nearest-neighbor search in embedding space

---

## Exploratory Data Analysis
**Notebook:** `PCA_Centroid_Correlation_Notebook.ipynb`

This notebook provides tools for exploratory analysis:
- **PCA visualization:** Plot processed spectra in PCA space to assess class separability
- **Centroid Correlation:** Compute a heatmap of pairwise correlations between class centroids
