# NATO SERS raw-data restart: initial exploration report

> Superseded by `FINAL_REPORT.md`, which incorporates the completed preprocessing, identifiability, quality-sensitivity, classical, CNN, acquisition-shift, and literature analyses.

Status: active analysis, not a final protocol or confirmatory model comparison.  
Protocol seed: `20260805`. Source: `/home/elfo/Documents/NRC/2026July21/NATO SERS Data`.

## 1. What was restarted

This analysis reparsed the source workbooks and readable vendor exports rather than importing the previous 598-row dataset or its preprocessing decisions. The mirror under the project-level `2026July21` directory is byte-identical to the authoritative source and is not additional data. Every raw file was inventoried and hashed.

The archive has 7,148 files (1,397,661,278 bytes): 2,516 SPC, 1,420 TXT, 1,341 PDF, 934 CSV, 933 PRB, two XLSX, and two CEZ files. There are 2,883 exact-content duplicate groups involving 5,769 file references, largely because `Z Converted` mirrors native/readable exports.

The source note states that four SERS sensor types were used and that `na` means no SERS sensor, i.e. ordinary Raman. Therefore `na` rows are retained in the full manifest but are not silently relabeled as SERS.

## 2. Observation tiers and why 721 becomes 598

| Tier | Count | Meaning |
|---|---:|---|
| Expanded recording-log observations | 1,172 | One row per logged scan after expanding multi-scan cells |
| Named-SERS observations | 721 | A SERS sensor is explicitly named |
| Ambiguous SERS-like rows | 3 | `normal=N`, but no sensor name; kept separate |
| All parseable named-SERS | 626 | Named SERS, matched source, numeric spectrum passes basic QC |
| Attributable SERS | 615 | Also maps to a master sample and target analyte |
| Unique attributable SERS | 598 | Also removes conflicting/repeated references to the same source spectrum |
| Notes-clear sensitivity tier | 500 | Also removes severe-quality and low-signal/noise notes |

Thus, 123 of the 721 named-SERS log entries are not in the primary 598-row analysis tier. The largest loss occurs because the instrument data or scan filename is unavailable. A further 11 parseable rows cannot be attributed to a target, and 17 attributable references are removed to avoid source reuse/conflicting metadata. The 98 spectra excluded only by the notes-clear tier are not discarded globally: they remain available for sensitivity analysis.

The 598 spectra represent 69 physical master samples, seven target classes, ten instruments, four sensor families, and three trial stations.

## 3. Native axes and the common interval

| Instrument family | Native/effective support | Typical grid |
|---|---|---:|
| Agilent | nominal 200–2,000; first ~150 points are constant padding, effective lower edge 350–353 | 1 cm⁻¹ |
| Mira | 400–2,300 | 1 cm⁻¹ |
| Pendar | 275–1,849 | 1 cm⁻¹ |
| RMX | approximately −93–2,907/2,915; scan-specific calibration | ~1.37 cm⁻¹ |

The strict all-instrument measured intersection is 400–1,849 cm⁻¹. The lower bound is 400—not 100—because Mira has no coordinates below 400. Extending an all-instrument matrix to 100 would fabricate Mira, Pendar, and Agilent values by extrapolation. Lower regions can still be studied in instrument-restricted analyses, explicitly as a different dataset.

The 400–1,849 common grid contains 1,450 points at 1 cm⁻¹. Every selected source spectrum covers every grid coordinate; validation confirms no extrapolation.

## 4. Dataset balance and confounding

Row counts range from 54 ethyl-paraoxon spectra to 120 benzyl-fentanyl spectra. Physical-master counts range from seven ethyl-paraoxon masters to 15 benzyl-fentanyl masters. A master has a median of nine spectra from five instruments and two sensor families, making this a useful repeated-measures acquisition-shift dataset.

The design is not factorial:

- Station is strongly linked to chemistry. CWA contains only 4-nitrophenol, ethanol, and ethyl paraoxon; pills contains 4-ANPP, benzyl fentanyl, and blanks; surfaces contains 4-ANPP, acetaminophen, and benzyl fentanyl.
- Several instruments observe only three target classes; only Agilent-3, Pendar-2, and Pendar-3 observe all seven.
- Sensor family is also imbalanced: pSERS has 324 rows, H-kit 153, NRC SERS 98, and GaN/polymer 23.
- A classifier using station alone reaches balanced accuracy 0.429 although seven-class chance is 0.143. Acquisition metadata does not need the spectrum to predict substantial label information.

Any random row split would leak physical samples and overstate generalization. All model results below keep entire physical masters together.

## 5. Raw unsupervised structure

Raw channel-standardized PCA is almost one-dimensional: PC1 explains 93.68% and PC2 3.48%; only two components are needed for 95% variance. This is chiefly global baseline/intensity structure.

Per-spectrum min–max scaling makes the geometry less degenerate: PC1 explains 42.31%, PC2 15.26%, and 46 components are required for 95% variance.

At the spectrum-row level:

| Method | Silhouette | Chemistry NMI | Instrument NMI | Sensor-family NMI |
|---|---:|---:|---:|---:|
| K-means, best exploratory k=10 | 0.353 | 0.256 | 0.482 | 0.304 |
| HDBSCAN | 0.505 on non-noise rows | 0.314 | 0.399 | 0.343 |

HDBSCAN marks 36.5% of rows as noise. These values show structure, but the clusters align more with acquisition system than chemistry.

When each spectrum is min–max scaled and then averaged within the 69 physical masters, K-means selects k=5 by silhouette (0.388). Its chemistry NMI rises to 0.668, but station NMI is also 0.585. Chemical information is present, but station/acquisition and chemistry cannot be cleanly separated by these data alone.

## 6. Preprocessing comparison

Eleven deterministic candidates were compared on measured common support: min–max, SNV, vector normalization, isolated-spike handling, Savitzky–Golay smoothing, rubber-band, AsLS, arPLS, combinations, and a first derivative. Selection was not based on a single embedding.

Key exploratory results:

| Candidate | Target balanced accuracy | Instrument balanced accuracy | Peak-location recall | Baseline-span proxy |
|---|---:|---:|---:|---:|
| Min–max only | 0.608 | 0.742 | 1.000 | 0.347 |
| Despike + Savitzky–Golay + min–max | **0.628** | 0.759 | 0.991 | 0.373 |
| AsLS + min–max | 0.560 | 0.704 | 0.955 | 0.155 |
| arPLS + smoothing + min–max | 0.564 | 0.699 | 0.948 | **0.128** |
| First derivative + SNV | 0.449 | 0.655 | 0.558 | 0.250 |

These are diagnostic five-fold master-grouped Ridge scores, not final test estimates. Baseline correction removes measurable instrument structure, but it also removes useful target signal. Therefore low instrument predictability alone is not a success criterion.

The effect is instrument-dependent. Across instruments, the worst prominent-peak recall is 0.980 for despike + smoothing, about 0.870 for arPLS + smoothing, 0.862 for rubber-band, and 0.202 for the derivative. Mira-2/Mira-3 and RMX-1 are among the systems most altered by aggressive baseline correction. Rubber-band creates visibly unstable edge behavior on representative Agilent, Mira, and Pendar spectra and is rejected as a universal transform.

The range sensitivity is small relative to fold variation. Despike + smoothing scores 0.628 on 400–1,849, 0.626 on 400–1,800, 0.616 on 450–1,800, and 0.608 on 500–1,800. There is no empirical reason to discard 400–449 globally. The conservative 400–1,800 range remains a reasonable primary interval because it avoids the extreme Pendar endpoint; 400–1,849 should be a prespecified sensitivity analysis.

Provisional preprocessing recommendation:

1. Interpolate only within native measured support onto 400–1,800 cm⁻¹ at 1 cm⁻¹.
2. Apply conservative isolated-impulse handling.
3. Apply an 11-point, third-order Savitzky–Golay smoother as the primary candidate, with no-smoothing as a required ablation.
4. Apply per-spectrum min–max scaling to put all spectra on the same [0,1] scale.
5. Do **not** apply baseline correction or derivatives universally. Test arPLS + min–max as a sensitivity branch and report instrument-stratified damage/preservation.

This remains provisional until repeated comparisons and a locked analysis protocol are finalized.

## 7. Fresh classical and CNN baselines

Across 3 × 5 repeated master-grouped folds:

- RBF SVM with smoothed/min–max spectra: spectrum balanced accuracy 0.698 ± 0.046; master-aggregated 0.878 ± 0.092.
- Random Forest with smoothed/min–max spectra: 0.688 ± 0.032; master-aggregated 0.924 ± 0.074.
- Extra Trees with arPLS/smoothed/min–max spectra: 0.685 ± 0.048; master-aggregated 0.948 ± 0.067.
- Seven-class prior baseline: 0.143 balanced accuracy.

Master aggregation is a different deployment task from classifying one spectrum. Its high score benefits from many spectra and multiple acquisition systems per physical sample. It must not be reported as single-spectrum performance.

A location-preserving 277,479-parameter 1D CNN, with inner-master early stopping and the same outer master folds, reaches 0.612 ± 0.104 per spectrum and 0.731 ± 0.168 per master. On the same five split assignments, Random Forest reaches 0.692 per spectrum and 0.905 per master. The bounded CNN therefore does not justify replacing classical baselines on the ordinary closed-set task.

Acquisition shift is the real failure mode. With both the instrument and the test physical masters absent from training, mean balanced accuracy across the ten held-out instruments is approximately 0.47–0.48 for both RBF SVM and Random Forest. Mira-3 transfer is 0.00 for all tested combinations. Held-out sensor-family transfer is generally near chance; smoothed Random Forest averages 0.348 across four families, while other combinations are around 0.23–0.24.

## 8. Project decision

### Recommended primary project

**Acquisition-shift-aware SERS identification using paired multi-instrument views of physical field samples.**

The distinctive asset is not dataset size; it is the repeated measurement of the same physical masters across instruments and sensor families. This supports a rigorous comparison of:

- classical SVM/Random Forest/Extra Trees;
- a properly bounded 1D CNN;
- supervised contrastive or domain-adversarial representations using cross-instrument views;
- evaluation that jointly excludes the test instrument/sensor and test master;
- calibration/rejection when the acquisition domain is unsupported.

The central research question should be whether paired acquisition views improve transfer without erasing chemically useful peaks—not whether a deep model can maximize an ordinary random-split score. Recent Raman work treats cross-system transfer as a practical barrier, and current SERS work explicitly identifies substrate-dependent variability and cross-device standardization as open problems. Relevant starting points include the 2024 multi-centre Raman transfer study, the 2023 self-supervised SERS inter-laboratory study, 2024 transfer-contrastive Raman classification, and 2025 LoRA calibration transfer.

### Fallback project

**Transparent classical closed-set SERS classification with master-level aggregation and uncertainty.**

This is feasible and likely strong, but less novel. It should compare single-spectrum and multi-view master decisions, report class/domain-stratified uncertainty, and emphasize leakage-safe grouping.

### Lower-priority options

- Open-set identification is scientifically useful, but only seven classes are present and unknown-class splits are confounded with station. It is better as a secondary stress test, not the primary claim.
- A VAE may be used for representation or anomaly-score ablation, but this archive has no clean target spectra paired with corrupted SERS measurements, no ground-truth nuisance decomposition, and incompletely crossed factors. It cannot validate “denoising” or chemical/nuisance disentanglement.
- A larger generic CNN/Transformer is not supported by 69 independent physical samples unless external pretraining is introduced and evaluated without leakage.

## 9. Interactive artifacts

- `index.html`: 35 raw-data, PCA, UMAP, t-SNE, K-means, HDBSCAN, and physical-master plots.
- `preprocessing/index.html`: candidate scorecards, range sensitivity, one representative overlay per instrument, and four instrument-stratified preservation heatmaps.
- `benchmarks/index.html`: closed-set, metadata-only, leave-instrument-out, and leave-sensor-out results.
- `benchmarks/cnn/training_history.html`: inner-validation learning curves for the bounded CNN.

All Plotly files are standalone HTML documents with embedded JavaScript, zoom, hover, and export controls. The raw audit passed all 22 structural validation checks, and 60 initial artifacts were hashed before the downstream analyses were added.

## 10. Work still required before final conclusions

1. Audit remaining instrument-report metadata and quantify saturation/quality flags by instrument and tier.
2. Add cluster stability/resampling and candidate-preprocessed PCA/UMAP views.
3. Lock the preprocessing protocol and rerun confirmatory repeated comparisons without using the same score to both choose and claim superiority.
4. Build paired-view contrastive/domain baselines and compare them under the strict unseen-domain + unseen-master protocol.
5. Add uncertainty, calibration, classwise confusion, and bootstrap intervals at the physical-master level.
6. Complete the primary-literature matrix, project-option scoring, artifact validation, and final hashes.

No final “best model” or publication claim should be made until these steps are complete.
