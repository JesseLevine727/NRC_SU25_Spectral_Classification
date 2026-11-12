# NATO field-trial SERS raw-data restart and project decision

Date: 2026-08-05  
Authoritative source: `/home/elfo/Documents/NRC/2026July21/NATO SERS Data`  
Primary analysis workspace: `Workspace/nato_sers_raw_reexploration_v1`

## Executive conclusion

The archive supports a defensible machine-learning study, but not the original disentangled-VAE claim.

The best primary project is:

> **Station-conditioned chemical identification under instrument and SERS-sensor acquisition shift, using repeated multi-instrument measurements of physical field samples.**

The scientific asset is the repeated-measures structure: 69 physical masters were measured by up to six instruments and up to three sensor families. The main question is whether a representation can retain chemistry while transferring to an unseen instrument—not whether a high-capacity network can maximize an ordinary closed-set score.

The strongest fallback is a transparent classical classifier on three within-station tasks. PCA–LDA, RBF SVM, Random Forest, and Extra Trees are already strong. A bounded 1D CNN is less accurate and substantially less stable, so classical ML must be the default comparator.

A VAE may remain an exploratory representation or anomaly-score baseline. This archive cannot verify signal denoising or chemical/nuisance disentanglement because it lacks paired clean targets, factor-complete interventions, and independently known nuisance components.

## 1. Source archive and provenance

The authoritative archive contains 7,148 files totaling 1,397,661,278 bytes:

| Extension | Files | Principal role |
|---|---:|---|
| SPC | 2,516 | Native binary spectra and converted copies |
| TXT | 1,420 | RMX processed/dark spectra, metadata, and converted copies |
| PDF | 1,341 | Mira/Pendar instrument reports and converted copies |
| CSV | 934 | Agilent/Mira/Pendar readable numeric exports |
| PRB | 933 | Pendar probe bundles and converted copies |
| XLSX | 2 | Recording log and master sample list |
| CEZ | 2 | Agilent session archives |

There are 2,883 exact-content duplicate groups involving 5,769 file references, primarily because `Z Converted` mirrors native/readable exports. The project-level `2026July21/NATO SERS Data` directory is a duplicate of the authoritative source and contributes no additional observations.

Every file path, role, byte size, and SHA-256 hash is recorded in `tables/raw_file_inventory.csv`.

### Source documentation read

- `Notes.txt`
- All sheets of `01_Raman Team recordings_06Mar2024.xlsx`
- All sheets of `MASTER SAMPLE LIST FOR NATO SERS CDT_MARCH 2024.xlsx`
- Every matched Mira/Pendar PDF report used to extract acquisition metadata
- Every matched RMX TXT header used to extract acquisition metadata
- All readable numeric exports indexed by the parser

The recording workbook contains three session sheets with 277, 396, and 502 XML rows, plus Raman-system and team sheets. The master workbook contains pills, surfaces, and CWA sheets and an overview sheet. Sheet names, dimensions, and headers are preserved in `deep_audit/tables/workbook_sheet_audit.csv`.

## 2. Meaning of `na` and SERS identification

The source note explicitly defines four SERS sensor families and states that `na` means no SERS sensor—ordinary Raman—not missing SERS metadata.

Normalized families are:

- pSERS / Metrohm silver
- NRC Canadian SERS
- H-SERS / H-kit
- GaN/polymer variants: AgGaN, AuGaN, AgPol, and AuPol

Only rows with an explicitly named SERS substrate are considered named SERS. Three rows have `Normal Raman = N` but no named substrate. They are preserved as ambiguous and excluded from named-SERS analyses rather than silently relabeled. Eighteen selected rows name a SERS substrate while the normal-Raman flag says `Y`; this source inconsistency is retained as `sensor_flag_conflict`.

## 3. How 721 named-SERS rows become 598 spectra

| Step | Count | Difference | Reason |
|---|---:|---:|---|
| Expanded log observations | 1,172 | — | Multi-scan cells expanded into observations |
| Explicitly named SERS | 721 | — | Sensor family named in log |
| Matched, numeric-QC-pass SERS | 626 | −95 | 58 instrument datasets unavailable; 37 scan files not logged |
| Attributable to master and target | 615 | −11 | Master sample/target cannot be resolved |
| Unique attributable sources | 598 | −17 | Reused source spectra or conflicting source-to-sample metadata |
| Notes-clear sensitivity tier | 500 | −98 | Severe-quality and/or low-signal/noise notes |

The 598 tier is not an arbitrary quality filter. It is the largest tier that is numeric, chemically attributable, and free of repeated/conflicting source references. The 500-row tier is retained only for sensitivity analyses.

Across all indexed sources, 1,373 spectra load successfully; 1,367 are nonconstant numeric spectra. Five Agilent-3 sources and one Pendar-3 source are constant and excluded by numeric QC.

## 4. What metadata exists

The selected manifest has 112 columns, including 44 experimental/acquisition fields audited separately.

### Complete or nearly complete

- Master sample, target, station, session, team, operator, instrument, sensor family/variant, sheet, sample matrix
- Recorded time for 571/598 spectra
- Instrument result for 566/598
- Comments for 198/598
- Logged target-detected flag for 380/598

### Master-derived chemistry/context

- Sample matrix: 208 pill, 208 CWA solution, 182 field surface coupon
- Carrier geometry: 119 Hexa-L, 63 Square-P, otherwise absent
- Nominal concentration: 108 at 3 mM, 38 at 4 mM, 31 at 5%, otherwise not given
- Seven normalized target classes

### Vendor acquisition metadata

- Instrument serial: 486/598; eight serials
- Mira integration time, automatic integration, averages, laser power, smart-tip type, firmware, and system-suitability status: 170/598
- Pendar software version, duration, start date/time: 214/598
- RMX timestamp, CCD bias/gain/read noise, wavenumber correction, scan count, exposure, scan mode, replacement list, and detected peaks: 112/598

Mira-1's vendor reports state that the most recent system-suitability test failed for all 23 selected Mira-1 spectra. This instrument is retained in the primary tier and removed only in a declared sensitivity analysis.

No reliable common wavelength field is present across all vendors. Instrument identity/serial is therefore the defensible acquisition-domain label.

## 5. Native instrument characteristics

| Instrument family | Native support | Points/step | Important behavior |
|---|---|---|---|
| Agilent | nominal 200–2,000 cm⁻¹; effective lower edge 350–353 | 1,801 / 1 cm⁻¹ | Approximately 150 leading constant/padded points |
| Mira | 400–2,300 | 1,901 / 1 cm⁻¹ | Strong and variable curved backgrounds; automatic integration |
| Pendar | 275–1,849 | 1,575 / 1 cm⁻¹ | Lower amplitudes, some negative values, many manual poor-quality notes |
| RMX | approximately −93 to 2,907/2,915 | 2,048 / ~1.37 cm⁻¹ | Scan-specific calibration; vendor-processed spectra and separate dark SPC; high curvature/spike proxy |

The strict measured intersection is 400–1,849 cm⁻¹. A lower bound of 100 cm⁻¹ is impossible for an all-instrument dataset: Mira has no coordinates below 400, Pendar begins at 275, and Agilent's useful signal begins around 350. Extending to 100 would invent values by extrapolation.

The audit finds no broad repeated-maximum plateau consistent with detector saturation. Only two selected spectra have a repeated-maximum fraction above 0.001, and none exceeds 0.005. All selected axes are finite and strictly increasing.

### Background, noise, and processing differences

Median baseline-energy proxy:

- Agilent: 0.13–0.14
- Mira: 0.86–0.96
- Pendar: 0.24–0.26
- RMX: 0.02–0.04

This quantifies the visual observation that Mira commonly retains a large background while RMX appears strongly vendor-processed. The systems are not raw-intensity apples-to-apples before scaling.

Median curvature/spike proxy is about 0.041 for RMX, 0.011 for Mira-3, and at or below 0.002 for most Pendar spectra. Because RMX is already processed and its fine structure differs systematically, this proxy cannot be interpreted as literal cosmic-ray frequency. It is one reason blanket despiking is not automatically safe.

Pendar-2 and Pendar-3 have severe-quality notes on 53% and 40% of selected rows. These notes do not justify silent removal: the notes-clear sensitivity produces similar downstream conclusions.

## 6. Physical samples, repeats, and confounding

The 598 spectra represent:

- 69 physical masters
- seven target classes
- ten instruments
- four sensor families
- three stations
- 15 operator labels

A master has a median of nine spectra, five instruments, and two sensor families. There are 3,063 within-master spectrum pairs, including 2,473 cross-instrument pairs.

After light smoothing/min–max scaling, median cosine similarity is:

| Pair type | Median cosine |
|---|---:|
| Same physical master | 0.762 |
| Same target, different master | 0.753 |
| Different target | 0.669 |

Repeated views are more similar than different targets, but only marginally more similar than independent masters of the same target. A contrastive method should therefore use all same-target examples as positives; treating same-target/different-master rows as negatives would be incorrect.

### Station–chemistry confounding

- CWA contains only 4-nitrophenol, ethanol, and ethyl paraoxon.
- Pills contains only 4-ANPP, benzyl fentanyl, and blank.
- Surfaces contains only 4-ANPP, acetaminophen, and benzyl fentanyl.

Station alone obtains seven-class balanced accuracy 0.429 versus chance 0.143. The aggregate seven-class task is therefore not a clean chemistry experiment.

Within each station, acquisition metadata alone falls to approximately chance:

| Station | Metadata-only balanced accuracy | Three-class chance |
|---|---:|---:|
| CWA | 0.345 ± 0.043 | 0.333 |
| Pills | 0.335 ± 0.068 | 0.333 |
| Surfaces | 0.255 ± 0.058 | 0.333 |

This supports three station-conditioned chemistry tasks as the primary classification design.

## 7. Unsupervised structure

Raw channel-standardized PCA is dominated by global acquisition/background variation: PC1 explains 93.68%, PC2 3.48%, and only two components reach 95% variance.

After per-spectrum min–max scaling, PC1 explains 42.31%, PC2 15.26%, and 46 components are needed for 95% variance.

At the spectrum level:

| Method | Silhouette | Chemistry NMI | Instrument NMI | Sensor NMI |
|---|---:|---:|---:|---:|
| K-means, exploratory k=10 | 0.353 | 0.256 | 0.482 | 0.304 |
| HDBSCAN | 0.505 on non-noise | 0.314 | 0.399 | 0.343 |

HDBSCAN labels 36.5% as noise. The unsupervised geometry is more strongly associated with acquisition system than chemistry.

K-means k=7 is stable to 80% master subsampling under min–max scaling (median pairwise ARI 0.961), but stable nuisance clusters are not chemical clusters: instrument NMI is 0.460 versus chemistry NMI 0.227. arPLS reduces instrument NMI to 0.249 but increases sensor NMI to 0.397 and reduces stability to median ARI 0.734. Baseline removal changes which nuisance dominates; it does not establish domain invariance.

At the physical-master mean level, K-means selects k=5 with silhouette 0.388. Chemistry NMI increases to 0.668, but station NMI is 0.585. Chemical signal is present, yet station remains entangled.

## 8. Preprocessing evidence and final protocol

Twelve candidates were evaluated: min–max, SNV, vector and area normalization; conservative despiking; Savitzky–Golay smoothing; rubber-band, AsLS, and arPLS baselines; combined baseline/smoothing; and a first derivative.

Representative diagnostic results:

| Candidate | Chemistry balanced accuracy | Instrument balanced accuracy | Peak-location recall | K-means chemistry/instrument NMI |
|---|---:|---:|---:|---:|
| Min–max | 0.608 | 0.742 | 1.000 | 0.226 / 0.462 |
| Area | 0.517 | 0.654 | 1.000 | 0.251 / 0.466 |
| SNV | 0.519 | 0.671 | 1.000 | 0.333 / 0.502 |
| Despike + Savitzky–Golay + min–max | 0.628 | 0.759 | 0.991 | 0.230 / 0.453 |
| AsLS + min–max | 0.560 | 0.704 | 0.955 | 0.322 / 0.261 |
| arPLS + smoothing + min–max | 0.564 | 0.699 | 0.948 | 0.274 / 0.211 |
| First derivative + SNV | 0.449 | 0.655 | 0.558 | 0.435 / 0.295 |

Baseline correction removes instrument structure but also reduces supervised chemistry performance and modifies peak structure. Its worst instrument-level prominent-peak recall is about 0.87. Rubber-band has unstable edge behavior and worst-instrument recall 0.862. The derivative is destructive under the tested settings.

Smoothing gives small gains in aggregate folds, but the effect is model/task dependent. For RBF SVM the paired aggregate improvement is +0.0076 balanced accuracy; for Random Forest +0.0088 with a confidence interval crossing zero. Within-station differences are mostly near zero, and minimal processing is often better in cross-station and unseen-instrument transfer.

### Locked primary preprocessing

1. Interpolate only within measured support to 400–1,800 cm⁻¹ at 1 cm⁻¹.
2. Per-spectrum min–max scaling to `[0,1]`.
3. No universal despiking, smoothing, baseline correction, derivative, or peak alignment.

### Required sensitivities

- 400–1,849 support
- conservative despike + 11-point/order-3 Savitzky–Golay + min–max
- arPLS + min–max
- notes-clear and Mira-1-excluded tiers

The range analysis provides no reason to discard 400–449. Scores change only modestly between 400–1,849, 400–1,800, 450–1,800, and 500–1,800.

### Alignment decision

Cross-instrument same-master derivative correlation estimates give median absolute lag 2 cm⁻¹, 90th percentile 7 cm⁻¹, and only 45.9% within ±1 cm⁻¹. Instrument-pair signed medians are usually 0 or ±1, while within-pair variation is broad and often changes sign. There is no stable fixed offset that justifies per-spectrum correlation alignment. Per-spectrum alignment is excluded; small shift augmentation is allowed as a neural robustness experiment.

## 9. Classical and neural results

### Aggregate seven-class descriptive baseline

Across 3 × 5 master-grouped folds:

| Preprocessing/model | Spectrum balanced accuracy | Master balanced accuracy |
|---|---:|---:|
| Minimal min–max PCA–LDA | 0.702 ± 0.049 | 0.948 ± 0.080 |
| Minimal min–max RBF SVM | 0.690 ± 0.048 | 0.886 ± 0.093 |
| Minimal min–max Random Forest | 0.679 ± 0.042 | 0.924 ± 0.074 |
| arPLS PCA–LDA | 0.727 ± 0.048 | 0.948 ± 0.067 |
| Prior baseline | 0.143 | 0.143 |

These are continuity results, not the primary chemistry claim, because of station confounding.

### Primary within-station tasks

Best or representative spectrum-level results across 3 × 4 master-grouped folds:

| Station | Best classical result | Balanced accuracy |
|---|---|---:|
| CWA | Smoothed PCA–LDA | 0.663 ± 0.075 |
| Pills | Smoothed Extra Trees | 0.845 ± 0.036 |
| Surfaces | Smoothed PCA–LDA | 0.836 ± 0.059 |

Minimal min–max PCA–LDA obtains 0.648, 0.823, and 0.825 respectively. The preprocessing differences are much smaller than the differences between stations.

### Bounded 1D CNN

The location-preserving CNN has 277,479 parameters and uses inner-master early stopping.

| Station | CNN balanced accuracy | Strong classical comparator |
|---|---:|---:|
| CWA | 0.432 ± 0.125 | PCA–LDA 0.648–0.663 |
| Pills | 0.776 ± 0.029 | Extra Trees/Random Forest ~0.84–0.85 |
| Surfaces | 0.470 ± 0.228 | PCA–LDA 0.825–0.836 |

Several CNN folds stop after one or two epochs and collapse toward chance. With only 20–25 independent masters per station, deep classification from scratch is not reliable.

### Clean cross-station binary transfer

For 4-ANPP versus benzyl fentanyl:

- Pills → surfaces: spectrum balanced accuracy ranges 0.696–0.774 across classical methods; minimal PCA–LDA is 0.774.
- Surfaces → pills: 0.747–0.781 for PCA–LDA/SVM/tree methods; minimal RBF SVM is 0.781.

This is a useful secondary domain-generalization task with 26 masters. Directional asymmetry and small master count must be reported.

### Unseen instrument plus unseen master

Restricting to 13 instrument/station domains with all three station targets and at least 15 test masters:

| Preprocessing/model | Mean domain-balanced accuracy | Median | Worst |
|---|---:|---:|---:|
| Minimal RBF SVM | 0.686 | 0.734 | 0.333 |
| Minimal Random Forest | 0.667 | 0.677 | 0.333 |
| Smoothed RBF SVM | 0.677 | 0.710 | 0.333 |
| Smoothed Random Forest | 0.667 | 0.613 | 0.333 |

Performance varies dramatically by domain. Mira-3 on the pills task remains at three-class chance for all tested combinations. This is the acquisition-shift problem the proposed project should address.

### Quality-tier robustness

Removing the 98 note-flagged rows or all Mira-1 rows does not reverse model rankings or task conclusions. Aggregate seven-class SVM remains 0.70–0.72. Within-station results remain approximately:

- CWA: 0.60–0.65
- Pills: 0.80–0.84
- Surfaces: 0.78–0.81

The primary 598 tier is therefore retained, with both exclusions reported as sensitivities.

## 10. Why the VAE idea is not verifiable here

A disentangled VAE requires evidence that latent factors correspond to distinct generative causes. This archive does not supply:

- clean chemical spectra paired with noisy/background-contaminated SERS spectra;
- the same chemical crossed factorially with every instrument, sensor, station, batch, and concentration;
- known nuisance-only interventions;
- a target spectrum against which denoising error can be measured;
- enough independent masters to fit and validate a high-capacity generative model.

A VAE can reconstruct spectra or produce anomaly scores, but reconstruction is not denoising, and visually interpretable latent axes are not proof of causal disentanglement. A VAE anomaly model is scored 17/30 as an exploratory ablation; a disentangled denoising VAE is scored 10/30 and should not be the main claim.

## 11. Literature-guided project selection

The literature matrix is in `literature_matrix.csv`. The most relevant precedents are:

- Fornasaro et al.'s large-scale multi-instrument SERS interlaboratory study, establishing reproducibility as a central SERS issue.
- Park et al.'s self-supervised MVNet with leave-one-laboratory-out SERS validation.
- Blake et al.'s three-centre Raman study, which split at the patient level and explicitly compared PCA–LDA, SVM, and CNN transfer.
- Wang et al.'s transfer-contrastive Raman classifier.
- Song et al.'s LoRA calibration transfer across Raman instruments.
- The 2025 SERS-D2DNet cross-device standardization study.
- Raman/SERS open-set studies using Objectosphere and the 2026 GASF framework.

These studies make acquisition transfer and unknown rejection current topics, but they also expose the NATO archive's distinguishing feature: heterogeneous field systems measuring repeated physical samples rather than a large standardized spectral map.

Project options were scored on novelty, feasibility, sample adequacy, confounding control, evaluation validity, and publication value. The primary acquisition-shift project scores 28/30; classical within-station classification scores 24/30 as the fallback. Full scores and rationales are in `project_option_scores.csv`.

## 12. Recommended study

### Primary

**Station-conditioned SERS chemical identification under unseen-instrument acquisition shift.**

Compare:

1. Nearest-centroid/correlation matching
2. PCA–LDA or PLS-DA
3. RBF SVM
4. Random Forest and Extra Trees
5. Bounded 1D CNN
6. Multi-positive supervised contrastive encoder
7. Paired-view consistency learning
8. Domain-adversarial and CORAL controls

The primary endpoint is the unweighted mean single-spectrum balanced accuracy across the 13 adequately supported unseen-instrument/station domains. Test physical masters and the test instrument are both absent from training.

The key proposed method should weight same-target cross-instrument/sensor pairs as positives. Same-target spectra from different masters must not be negatives. The method must improve acquisition transfer while preserving chemistry and peaks; merely making instrument identity less predictable is insufficient.

### Secondary task

Bidirectional pills/surfaces transfer for 4-ANPP versus benzyl fentanyl.

### Fallback

Classical three-class station-specific classification with calibrated uncertainty and instrument-balanced multi-view master aggregation.

### Secondary stress test only

Within-station leave-one-chemical-out unknown rejection. Seven-class open-set claims are prohibited because chemical and station are coupled.

## 13. Claims and limitations

Supported claims concern only the measured field-trial targets, stations, instruments, and sensors. The study may report transfer to held instruments within this design.

It must not claim:

- generic SERS denoising;
- chemical/nuisance disentanglement;
- transfer to arbitrary instruments or chemicals;
- that a master-aggregated score is single-spectrum accuracy;
- that seven-class aggregate performance is chemistry-only;
- that station-held-out chemicals constitute valid open-set recognition.

The strongest future additions would be calibration standards, pure/control Raman and SERS spectra, factorial analyte × instrument × sensor × station measurements, independent substrate batches, additional masters, and genuinely external unknown chemicals.

## 14. Artifact guide

Open `ALL_RESULTS.html` first.

- `index.html`: 35 raw inventory, support, PCA, UMAP, t-SNE, K-means, HDBSCAN, and physical-master plots.
- `preprocessing/index.html`: 17 preprocessing and preservation plots.
- `deep_audit/index.html`: 54 all-spectrum, quality, replicate, alignment, stability, and candidate-embedding plots.
- `benchmarks/index.html`: aggregate classical, metadata, and acquisition-shift results.
- `identifiability/index.html`: within-station, cross-station, and station-conditioned instrument-shift results.
- `benchmarks/cnn/training_history.html`: aggregate CNN learning curves.
- `docs/NATO_SERS_ACQUISITION_SHIFT_PROTOCOL_V1.md`: locked task, preprocessing, split, metric, ablation, and claim protocol.
- `FINAL_VALIDATION_REPORT.json`: completion audit, including source rehashing and split reconstruction.
- `FINAL_ARTIFACT_HASHES.json`: SHA-256 freeze for the complete workspace and its scripts/configurations.

All Plotly pages are standalone HTML with embedded JavaScript, hover, zoom, and export controls.

## 15. Reproduction

From the project root and its existing `.venv`:

```bash
.venv/bin/python scripts/run_nato_sers_raw_reexploration.py
.venv/bin/python scripts/validate_nato_sers_raw_reexploration.py
.venv/bin/python scripts/run_nato_sers_preprocessing_exploration.py
.venv/bin/python scripts/summarize_nato_sers_preprocessing_by_instrument.py
.venv/bin/python scripts/run_nato_sers_deep_audit.py
.venv/bin/python scripts/run_nato_sers_classical_fresh.py
.venv/bin/python scripts/run_nato_sers_cnn_fresh.py
.venv/bin/python scripts/run_nato_sers_identifiability.py
.venv/bin/python scripts/run_nato_sers_cnn_station.py
.venv/bin/python scripts/run_nato_sers_quality_sensitivity.py
```

Final validation passed all 20 completion checks. It rehashed all 7,148 files in both the authoritative source and project mirror, reconstructed master-disjoint and unseen-instrument splits, validated all 117 registered standalone HTML plots, and froze 228 artifact hashes.
