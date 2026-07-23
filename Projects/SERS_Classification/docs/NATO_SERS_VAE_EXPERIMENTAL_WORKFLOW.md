# Evidence-Based Workflow for Generalizable SERS Classification with Autoencoders and Disentangled VAEs

Date: 2026-07-22  
Project: NATO field-trial SERS classification  
Primary objective: learn a chemical representation that remains useful when substrate, Raman instrument, vendor processing, baseline, acquisition conditions, and field noise change.

## 1. Executive decision

The work should proceed as a controlled comparison of increasingly structured representation-learning models:

1. classical derivative and nearest-prototype baselines;
2. the existing Siamese/triplet model;
3. a deterministic autoencoder;
4. a denoising autoencoder;
5. a standard VAE with a mixed latent space;
6. a semi-supervised/disentangled VAE;
7. a hybrid disentangled VAE that retains the useful cross-substrate metric objective from the Siamese work.

All NATO model inputs will use the same 400--1800 cm⁻¹ axis. The principal VAE representation will be independently min--max normalized to `[0,1]` for every spectrum. Original intensities, scaling parameters, estimated baselines, spike masks, and all provenance will remain available.

Preprocessing will remove only well-defined measurement artifacts. It will not attempt to erase all substrate-dependent spectral behavior, because changes in relative peak enhancement, peak visibility, binding, and broadening may contain real chemical--surface physics. Those remaining factors are the reason for learning an explicit nuisance representation.

The recommended final architecture is not a standalone β-VAE. It is a reconstruction-preserving, semi-supervised model with a chemical latent, a nuisance latent, chemical classification and cross-domain metric losses, and instrument/sensor conditioning or nuisance supervision.

## 2. Research questions and hypotheses

### 2.1 Primary question

Can a model preserve analyte information while reducing sensitivity to SERS sensor, Raman instrument, vendor processing, background, and acquisition noise?

### 2.2 Secondary questions

1. How much apparent invariance comes from preprocessing alone?
2. Does a reconstruction constraint prevent the representation collapse observed in the difficult AgNP/4NP Siamese case?
3. Does a standard VAE encode dominant instrument/background variation rather than chemical identity?
4. Does explicitly partitioning chemical and nuisance latents improve supported held-out-domain classification?
5. Does combining VAE reconstruction with cross-domain Siamese/contrastive supervision outperform either mechanism alone?
6. Can the model reconstruct or recover a stable spectrum after controlled noise, baseline, spike, scale, and shift perturbations?

### 2.3 Hypotheses

- **H1:** common-axis min--max scaling improves numerical comparability but does not remove instrument identity.
- **H2:** spike removal and appropriate baseline handling improve field robustness, but excessive baseline removal damages chemically relevant broad or weak features.
- **H3:** a denoising autoencoder is more directly effective against specified corruptions than a standard VAE.
- **H4:** a standard VAE reconstructs major nuisance variance unless given chemical and domain structure.
- **H5:** a chemical/nuisance VAE with reconstruction and cross-domain chemical supervision yields a better chemical-versus-domain trade-off than the existing Siamese embedding.
- **H6:** the controlled poster dataset can validate substrate disentanglement more cleanly than the NATO archive, while NATO provides the stronger multi-instrument field stress test.

## 3. Evidence used to construct this workflow

This protocol is based on four local evidence sources:

1. [`SERS-VAE.md`](../SERS-VAE.md), which proposes a standard VAE, a molecule/surface latent split, a latent swap test, and classification from the molecule latent;
2. the [CSCCE 2026 poster](cscce_2026_poster/cscce_2026_sers_poster.pdf) and associated Siamese results;
3. the [complete NATO archive audit](NATO_SERS_FIELD_TRIAL_AUDIT.md);
4. direct diagnostics calculated from the 598-spectrum NATO strict core.

### 3.1 What the poster established

The poster moved from chemical--substrate pair recognition to chemical-label prediction with leave-one-substrate-family-out testing.

- Stage 1 pair classification reached 98.76%, but the target still included substrate identity.
- The substrate-agnostic experiment used 275 spectra from 4NP, benzenethiol, and pyridine across Ag, Au, PICO, and p-SERS; 4NP was absent on Au.
- The best Siamese/triplet model reached 0.975 mean held-out-substrate accuracy.
- The best classical second-derivative nearest-centroid model reached 0.987.
- The raw-spectrum Siamese control reached only 0.440, demonstrating that preprocessing was essential.
- Chemical silhouette increased from 0.304 in derivative-input space to 0.797 in the Siamese embedding.
- Substrate silhouette decreased from 0.101 to -0.040, demonstrating genuine suppression of substrate organization.
- In the difficult Ag/AgNP 4NP case, the derivative representation retained usable chemical geometry, while the learned Siamese embedding pulled 4NP toward benzenethiol. Six of 25 held-out Ag 4NP spectra were misclassified in the poster-level fold, and the deeper AgNP diagnostic found complete class-level collapse in the learned prototype geometry.

These results show that Siamese metric learning can learn useful invariance. They also show that its objective does not require spectral preservation: it can improve global separation while distorting a difficult local case. This is the principal justification for adding a decoder and reconstruction constraint.

### 3.2 What the NATO archive established

The NATO archive is not a ready-made factorial dataset. It contains normal Raman and SERS measurements, duplicate conversions, missing system archives, calibration/unlogged scans, conflicting log references, several vendor formats, and unequal acquisition processing.

The reconstructed observation funnel is:

| Sequential rule | Remaining | Removed at step |
|---|---:|---:|
| Explicitly named SERS sensor | 721 | -- |
| Readable spectrum matched | 626 | 95 |
| Master sample and target resolved | 615 | 11 |
| Source reference unambiguous | 599 | 16 |
| One primary row per consistent source | 598 | 1 |
| Numeric spectrum valid | 598 | 0 |

There are 619 unique readable SERS source spectra. The 21 beyond the strict core can potentially support unsupervised reconstruction, but their labels or source assignments are not reliable enough for supervised evaluation. The primary labelled dataset is therefore 598 spectra. The conservative field-quality subset contains 500.

All ten instruments for which files exist remain in the strict core:

| Instrument | Core spectra |
|---|---:|
| Agilent-1 | 29 |
| Agilent-3 | 73 |
| Mira-1 | 23 |
| Mira-2 | 75 |
| Mira-3 | 72 |
| Pendar-1 | 46 |
| Pendar-2 | 98 |
| Pendar-3 | 70 |
| RMX-1 | 47 |
| RMX-2 | 65 |

Agilent-2 and PMCDS measurements cannot be reconstructed from the present archive because the required spectral files are absent.

### 3.3 How the common spectral interval was determined

The interval is the reliable intersection of the native axes, not an arbitrary Raman fingerprint convention:

| System | Reliable native coverage |
|---|---|
| Agilent | approximately 350--2000 cm⁻¹; aggregate CSV values below 350 are padded |
| Mira | exactly 400--2300 cm⁻¹ |
| Pendar | 275--1849 cm⁻¹ |
| RMX | approximately -93--2907/2915 cm⁻¹; axis varies slightly per scan |

The common lower bound is therefore 400 cm⁻¹. A 100 cm⁻¹ lower bound would contain no measurements for Mira, no measurements below 275 for Pendar, and no trustworthy Agilent measurements below approximately 350. Filling those regions by edge interpolation would manufacture an instrument identifier. The upper bound of 1800 cm⁻¹ stays safely inside Pendar's 1849 cm⁻¹ limit.

The poster dataset retains 330--1800 cm⁻¹ because it was measured on a common Horiba system and had a separate low-frequency artifact below approximately 300 cm⁻¹. The two datasets do not require identical lower cutoffs unless they are jointly modeled; a joint representation would use 400--1800 cm⁻¹.

### 3.4 How system spectral behavior was determined

System characteristics were measured from the files rather than assigned from vendor reputation:

- every readable axis, length, spacing, intensity range, and negative-value fraction was profiled;
- representative readable CSV/TXT spectra were checked against binary SPC files;
- RMX main and dark TXT blocks were checked against their respective SPC files;
- all readable Mira/Pendar reports were parsed for acquisition metadata;
- common lower-envelope and AsLS baseline diagnostics were calculated;
- representative spectra and baseline estimates were plotted by instrument;
- target and instrument classifiers were run on normalized representations with master-sample grouping.

The [instrument background figure](../Workspace/nato_sers_field_trial/figures/instrument_background_examples.png) shows the resulting behavior.

The native lower-envelope span relative to spectral span was approximately:

| Instrument family | Observed range |
|---|---:|
| Agilent | 0.106--0.120 |
| Mira | 0.708--0.899 |
| Pendar | 0.055--0.098 |
| RMX | 0.003--0.015 |

A common-grid candidate AsLS estimate instead assigned RMX baseline-span fractions of approximately 0.367--0.551. That disagreement is evidence that broad RMX structure cannot safely be declared either signal or background from one algorithm. Mira, in contrast, shows a strong curved background under both diagnostics.

### 3.5 What the initial scaling diagnostics showed

The following exploratory PCA/logistic results use master-sample-grouped cross-validation. They are screening diagnostics, not final performance estimates:

| Representation | Target balanced accuracy | Instrument balanced accuracy |
|---|---:|---:|
| Raw + SNV | 0.693 | 0.857 |
| Candidate AsLS + SNV | 0.704 | 0.892 |
| Raw + per-spectrum min--max | **0.729** | 0.872 |
| Raw + robust min--max | 0.717 | 0.886 |
| Candidate AsLS + min--max | 0.722 | 0.888 |

These results determined three decisions:

1. min--max belongs in the primary experiment because it gave the strongest preliminary target result and provides the requested common numerical scale;
2. min--max is not sufficient for domain invariance because it preserves baseline shape, noise texture, resolution, peak response, and vendor processing;
3. baseline correction cannot be assumed beneficial merely because spectra look flatter: the candidate AsLS branch increased instrument predictability.

Instrument prediction is also partly confounded by the experimental design: instruments measured different mixtures of targets and sensors. Domain leakage must therefore also be tested on target-balanced and same-master subsets.

## 4. Claims the present datasets can and cannot support

### 4.1 Poster dataset

The poster dataset is the cleaner substrate-methodology benchmark because it uses one Raman system and an almost crossed chemical × substrate design. It can test whether a representation removes substrate organization while preserving chemical identity.

It cannot yet establish universal SERS generalization because there are only three modeled chemicals, 4NP is missing on Au, and 25 map locations are not 25 independent preparations.

### 4.2 NATO dataset

NATO is the stronger field-domain stress test because it includes four normalized SERS families, ten Raman instruments, multiple matrices, and real quality failures.

It cannot by itself separate every nuisance factor because target, station, sensor, instrument, and physical sample are not fully crossed. For the strict domain-and-sample protocol:

- GaN/polymer has 23 supported test spectra across two classes;
- NRC/Canadian SERS has 78 supported test spectra across six classes;
- H-Kit leaves only one supported test observation after same-sample exclusion;
- p-SERS leaves zero supported target classes after same-sample exclusion.

Therefore, NATO cannot produce one defensible universal “unseen surface” number. It can produce controlled paired-domain tests, grouped new-sample tests, supported-class domain-and-sample tests, and detailed per-domain stress results.

## 5. Data products and immutable provenance

The source archive remains read-only. Every derived row retains its original log position, source scan, instrument, sensor text, normalized family, master sample, target, station, quality flags, and acquisition metadata.

The existing derived directory is [`Workspace/nato_sers_field_trial`](../Workspace/nato_sers_field_trial/README.md).

The canonical data layers are:

1. **Native source:** original readable vendor export;
2. **Common raw:** 400--1800 cm⁻¹, 1 cm⁻¹ spacing, 1,401 points;
3. **Artifact annotations:** spike mask, saturation mask, quality flags, baseline estimate, alignment shift;
4. **Scaled model representation:** independently min--max normalized spectrum;
5. **Alternative representation:** baseline-corrected and min--max normalized;
6. **Discriminative baselines:** first- and second-derivative branches;
7. **Scaling metadata:** original minimum, maximum, range, or robust quantiles.

No derived operation overwrites an earlier layer.

## 6. Preprocessing philosophy

“Same scale” and “same physical representation” are different goals.

Per-spectrum min--max guarantees the same numerical range. It removes offset and scale but retains baseline curvature, resolution, smoothing, peak-width response, and noise structure. These retained differences are exactly what the domain-aware model must address.

Preprocessing should remove:

- invalid arrays;
- fabricated or unsupported axis regions;
- isolated cosmic spikes;
- saturation artifacts when detectable;
- validated smooth fluorescence/background components;
- validated small calibration shifts;
- optionally, high-frequency noise when smoothing demonstrably preserves peaks.

Preprocessing should not blindly remove:

- substrate-dependent relative peak enhancement;
- chemically meaningful peak disappearance or appearance;
- broad chemical bands merely because they are broad;
- binding-induced shifts or broadening;
- all instrument variation by label-specific tuning.

The latter effects require representation learning, paired supervision, or additional experimental controls.

## 7. Mandatory preprocessing

### 7.1 Observation selection

- Include only explicitly named SERS sensors for the labelled dataset.
- Treat literal `na` as normal Raman without a SERS sensor.
- Derive target labels only from the master sample list.
- Do not use vendor `Results` or `Target Det` as ground truth.
- Exclude contradictory source reuse from supervised training and evaluation.
- Retain raw log inconsistencies as flags.

### 7.2 Common axis

- Crop/interpolate every NATO spectrum to 400--1800 cm⁻¹.
- Use the spectrum's own calibrated axis, especially for RMX.
- Reject nonmonotonic, nonfinite, or constant spectra.
- Never pad unsupported low-frequency regions for a common-domain model.

### 7.3 Spike and saturation handling

An isolated spike can become the maximum used by min--max and compress all genuine peaks toward zero. Spike handling therefore occurs before scaling.

Initial method:

1. calculate a robust second-difference or local-median residual;
2. estimate scale using median absolute deviation;
3. flag only extreme, isolated deviations spanning approximately one to three points;
4. replace flagged values by local interpolation or a local robust median;
5. preserve the original value and binary mask.

Thresholds are selected using injected-spike recovery and false removal of repeatable peaks, not appearance alone. Long plateaus are flagged as possible saturation and are not repaired automatically.

### 7.4 Per-spectrum min--max scaling

For the primary representation:

\[
x_{\mathrm{scaled}} = \frac{x-\min(x)}{\max(x)-\min(x)}.
\]

Rules:

- calculate minimum and maximum only within 400--1800 cm⁻¹;
- scale each spectrum independently;
- save the original minimum and range;
- apply scaling after spike and optional baseline processing;
- do not use a single dataset-global range, which would be dominated by Mira;
- do not fit feature-wise scaling on the full dataset.

A robust percentile variant using the 1st and 99th percentiles remains an ablation. It reduces spike sensitivity but was not superior in the first target diagnostic.

## 8. Experimental preprocessing

### 8.1 Baseline correction

Baseline correction is a model-selection experiment, not a mandatory cleanup step.

Candidate methods:

- no baseline correction;
- AsLS;
- arPLS;
- conservative rubber-band/convex-hull baseline.

Initial AsLS grid on the common axis:

```text
lambda: 10^4, 10^5, 10^6, 10^7
p:      0.001, 0.01
```

The grid is screened sequentially inside training folds. A larger exhaustive search is not justified by 598 spectra.

Every candidate saves both the estimated baseline and residual. All residuals are subsequently min--max normalized for the VAE comparison.

### 8.2 System evidence and initial expectations

| System | Evidence | Initial experiment |
|---|---|---|
| Mira | Strong curved fluorescence/background | Baseline correction is a high-priority candidate |
| Agilent | SORS export is already peak-like and near a low envelope | None versus mild correction; avoid double-processing |
| Pendar | Peak-like but often noisy and frequently field-flagged | Prioritize spikes/noise; use conservative baseline correction |
| RMX | Vendor main spectrum plus separate dark; broad shape is estimator-dependent | Corrected and uncorrected branches; do not subtract dark automatically |

These expectations guide diagnostics, not the primary generalization rule.

### 8.3 Domain-blind versus system-aware processing

Two claims must be separated:

1. **Domain-blind generalization:** one algorithm and parameter set is applied to every spectrum without knowing the instrument name. It may remove different amounts because the spectra differ, but its rule is identical.
2. **Known-system harmonization:** processing is selected from known vendor/system metadata. This is valid when deployment instruments are known, but it is not evidence of generalization to an unseen system.

The domain-blind pipeline is the headline experiment. The known-system pipeline is a secondary upper-bound comparison.

### 8.4 Smoothing and derivatives

No smoothing is used in the initial VAE representation. If noise diagnostics justify it, compare Savitzky--Golay windows of approximately 7, 11, and 15 cm⁻¹ with polynomial order 3. The chosen window must remain narrower than the narrowest repeatable chemical feature.

Derivative branches are retained because the poster established their importance:

- SNV + first derivative + row L2 for the Siamese baseline;
- SNV + second derivative + row L2 for the best classical baseline.

Derivatives will not initially be the sole VAE reconstruction target because they discard absolute spectral shape and amplify high-frequency noise. They may later enter a secondary encoder channel or a derivative reconstruction loss.

### 8.5 Alignment

Wavenumber alignment is added only if standards or same-master comparisons demonstrate repeatable system-level shifts.

- Prefer a single small correction per instrument/session estimated without target labels.
- Estimate corrections inside the training data.
- Report the distribution of applied shifts.
- Do not use flexible per-spectrum dynamic warping toward class templates.

Flexible label-informed warping would convert peak position into a leakage channel.

## 9. How preprocessing will be selected

No method is selected because its spectra look flatter or because its target accuracy alone is highest.

### 9.1 Sequential screening

The sequence is:

1. common-axis and numeric QC;
2. ordinary versus robust spike handling;
3. ordinary versus robust min--max;
4. no baseline versus conservative baseline candidates;
5. optional alignment;
6. optional smoothing;
7. derivative branches.

Only the best few candidates from one stage proceed to the next. This prevents an uninterpretable combinatorial search.

### 9.2 Selection measurements

For every candidate calculate:

- target balanced accuracy, macro F1, per-class recall, and calibration;
- instrument and sensor probe performance;
- target-conditional/domain-balanced probe performance;
- same-master cross-domain distance;
- different-target separation;
- chemical and domain silhouette scores;
- spectral angle and correlation between paired measurements;
- peak-position, width, and local-relative-intensity preservation;
- injected-corruption recovery;
- performance on strict-core and quality-pass data;
- sensitivity across multiple seeds and grouped folds.

### 9.3 Paired-master criterion

Repeated NATO measurements of one `master_sample_id` across instruments or sensors are the most direct harmonization evidence.

Define:

\[
D_{\mathrm{same}} = \text{distance between the same master sample across domains}
\]

and

\[
D_{\mathrm{different}} = \text{distance between different targets}.
\]

A useful transformation decreases `D_same` without collapsing `D_different`. The ratio or margin between them is reported per target and domain, rather than only as one pooled number.

### 9.4 Pareto selection

Preprocessing is selected from a Pareto frontier:

- maximize chemical performance and peak preservation;
- minimize domain leakage and same-master cross-domain distance;
- minimize corruption sensitivity;
- avoid avoidable complexity.

A method that produces slightly higher target accuracy by increasing instrument dependence is not automatically preferred. Likewise, a method that eliminates instrument prediction by erasing chemical information is rejected.

### 9.5 Leakage control

- Outer test domains and test master samples are never used to choose preprocessing.
- Baseline and smoothing parameters are chosen only in inner training folds.
- Per-spectrum min--max itself uses no other samples, but choosing min--max because of outer-test performance would still be leakage.
- All augmentations and corruption distributions are estimated from training data.

## 10. Dataset versions used in experiments

### 10.1 NATO-L598

The 598-spectrum strict labelled core. This is the primary supervised dataset.

### 10.2 NATO-Q500

The 500-spectrum conservative quality-pass subset. This is used for sensitivity analysis and initial clean-manifold learning.

### 10.3 NATO-U619

The 619 unique readable SERS source spectra, including 21 without safe supervised labels. This may be used only for unsupervised pretraining under a declared protocol. Unresolved observations cannot cross into evaluation in a way that exposes test-domain distributions unless the experiment is explicitly described as transductive.

### 10.4 Poster-275

The controlled three-chemical substrate dataset underlying the substrate-agnostic poster result. It is used to validate architecture and failure behavior, not as if it shared the NATO target set.

## 11. Split and evaluation protocols

### 11.1 Ordinary NATO model development

Use the existing five deterministic folds in `grouped_sample_cv_assignments.csv`.

- Group by `master_sample_id`.
- Keep all repeated spectra of a physical sample in one fold.
- Preserve all seven target classes per fold.
- Use nested inner folds for hyperparameter selection.

Random spectrum splits may be shown only as leakage diagnostics.

### 11.2 NATO domain-only transfer

Hold out one sensor family or Raman instrument. Other-domain measurements of the same master sample may remain in training.

Purpose: isolate domain-style transfer using naturally paired specimens.

Limitation: this is not new-specimen deployment.

### 11.3 NATO domain-and-sample transfer

Hold out one sensor/instrument and remove every master sample appearing in its test set from training.

Purpose: test simultaneous new-domain and new-specimen generalization.

Rules:

- evaluate only target classes still represented in training;
- list unsupported classes explicitly;
- never average incomparable domain folds into one universal number without class/support qualification.

### 11.4 Poster leave-one-substrate-family-out

Reproduce the existing Ag, Au, PICO, and p-SERS held-out folds. This is the primary controlled comparison between derivative, Siamese, AE, VAE, and disentangled models.

### 11.5 Quality stress test

Evaluate the 98 strict-core spectra excluded from NATO-Q500 as a separate field-quality stress cohort, stratified by instrument and failure note. Because poor-quality notes are strongly concentrated in Pendar-2/3, this is not an independent random test set.

## 12. Model ladder

All neural models use capacity-matched 1D convolutional encoders where practical. Larger networks are not justified until the model ladder demonstrates underfitting.

### M0: Classical baselines

- raw/min--max nearest centroid or linear classifier;
- first derivative + nearest centroid;
- second derivative + nearest centroid;
- linear SVM and cosine k-nearest neighbors where appropriate.

Purpose: establish whether deep learning improves on preprocessing and simple geometry.

### M1: Existing Siamese/triplet model

- same-chemical, preferably cross-domain positives;
- different-chemical, preferably same-domain negatives;
- nearest chemical prototypes at inference;
- leave-domain-out evaluation.

Purpose: retain the poster benchmark and measure what the metric objective changes.

### M2: Deterministic convolutional autoencoder

Input and target are the same minimally processed min--max spectrum.

Purpose: determine the effect of a reconstruction bottleneck without stochastic KL regularization.

### M3: Denoising autoencoder

Input is an artificially corrupted training spectrum; target is its pre-corruption version.

Training corruptions include controlled:

- smooth baseline slopes and low-order curves;
- intensity scaling and offset before final scaling;
- Gaussian or signal-dependent noise;
- isolated cosmic spikes;
- small wavenumber shifts;
- mild peak broadening.

Purpose: explicitly learn invariance to specified measurement nuisances. This is the direct noise-filtering baseline.

### M4: Standard VAE

The encoder emits one mixed latent `z`. Start with approximately 8--32 latent dimensions, selected in grouped inner validation.

Purpose: test compression, reconstruction, generative regularization, and whether an unsupervised latent naturally separates chemistry from domain.

Expected risk: the VAE may encode instrument and background because they explain large variance.

### M5: Semi-supervised VAE

Add a chemical classifier from `z` and a supervised classification loss.

Purpose: determine whether explicitly requesting target information is sufficient without a nuisance partition.

### M6: Two-block disentangled VAE

The encoder produces:

```text
z_chemical
z_nuisance
```

The decoder receives their concatenation, with optional observed instrument and sensor conditioning.

Start with approximately 8--16 dimensions per block. Split nuisance into sensor and instrument latents only if the two-block model is stable and supported by ablations.

### M7: Recommended hybrid VAE

The hybrid retains the strongest part of the Siamese work while adding reconstruction:

```text
                              ┌─> chemical classifier
                              │
spectrum ─> encoder ─> z_chemical ─> cross-domain metric loss
                  └─> z_nuisance ─> domain classifier

[z_chemical, z_nuisance, observed metadata] ─> decoder ─> spectrum
```

An initial objective is:

\[
\begin{aligned}
\mathcal{L} ={}&
\lambda_{rec}\mathcal{L}_{rec}
+ \beta\mathcal{L}_{KL}
+ \lambda_{chem}\mathcal{L}_{chem}
+ \lambda_{metric}\mathcal{L}_{metric} \\
&+ \lambda_{domain}\mathcal{L}_{domain,nuis}
+ \lambda_{ind}\mathcal{L}_{independence}
+ \lambda_{adv}\mathcal{L}_{domain,adv}.
\end{aligned}
\]

Components:

- `L_rec`: continuous-spectrum reconstruction;
- `L_KL`: VAE regularization for both latent blocks;
- `L_chem`: target classification from `z_chemical`;
- `L_metric`: supervised contrastive or triplet loss joining the same chemical across domains;
- `L_domain,nuis`: instrument/sensor prediction from `z_nuisance`;
- `L_independence`: cross-covariance or related dependence penalty between latent blocks;
- `L_domain,adv`: optional adversarial suppression of domain prediction from `z_chemical`.

The adversarial term is introduced last. Target and domain are confounded in NATO, so a strong adversary can erase genuine target information. It must use balanced or target-conditional batches and be justified by ablation.

## 13. Reconstruction and denoising objectives

Pointwise MSE alone encourages smooth average spectra and may underweight narrow peaks. Candidate reconstruction loss combines:

- MSE or smooth L1 on the normalized spectrum;
- spectral-angle loss;
- correlation loss;
- first-derivative reconstruction loss;
- optional peak-weighted error defined without test-label information.

The initial VAE uses a continuous decoder and continuous loss. Although `[0,1]` inputs permit a sigmoid output, binary cross-entropy is not assumed appropriate for continuous intensity measurements.

The denoising experiment has an objective clean target because corruption is injected after the training spectrum is selected. Real field spectra do not have clean counterparts, so claims about real-noise removal rely on stability, paired-domain consistency, and peak preservation rather than an unknowable “true clean” spectrum.

## 14. Experimental phases and gates

### Phase 0: Reproduce frozen baselines

**Data:** Poster-275 and NATO-L598.  
**Models:** M0 and M1.  
**Inputs:** frozen derivative pipelines plus NATO min--max branches.  
**Outputs:** exact split-level metrics, embeddings, prototypes, confusion matrices, and failure cases.

Gate: baseline results must reproduce within declared seed variability before VAE claims are compared.

### Phase 1: Select minimal preprocessing

**Data:** NATO-L598 with NATO-Q500 sensitivity.  
**Models:** simple linear/centroid classifiers and fixed-capacity autoencoder diagnostic.  
**Experiments:** spike handling, ordinary/robust min--max, baseline candidates, then optional smoothing/alignment.

Gate: retain a small Pareto set, expected to include at least:

```text
P0 = despiked + per-spectrum min--max
P1 = despiked + domain-blind baseline correction + per-spectrum min--max
P2 = SNV + first derivative + row L2
P3 = SNV + second derivative + row L2
```

### Phase 2: AE versus denoising AE

**Data:** start on Poster-275, then NATO-Q500/L598.  
**Models:** M2 and M3 with matched capacity.  
**Question:** does explicit corruption training recover chemical geometry and resist nuisances better than reconstruction alone?

Gate: the denoising model must improve corruption recovery without materially degrading uncorrupted chemical performance or peak preservation.

### Phase 3: Standard VAE baseline

**Models:** M4 across retained preprocessing branches.  
**Ablations:** latent dimensions, β schedule, deterministic AE comparison, reconstruction losses.

Gate: establish whether the mixed latent improves, matches, or damages target and domain geometry. Do not infer disentanglement from a t-SNE/UMAP visualization.

### Phase 4: Semi-supervised and two-block VAE

**Models:** M5 and M6.  
**Ablations:** chemical classification, nuisance supervision, metadata-conditioned decoder, independence penalty.

Gate: `z_chemical` must improve the chemical-versus-domain trade-off relative to M4, not merely target accuracy.

### Phase 5: Hybrid VAE plus metric learning

**Model:** M7.  
**Ablations:** contrastive versus triplet, cross-domain positive preference, same-master consistency, adversarial term, domain-balanced batches.

Gate: show whether reconstruction prevents the poster's AgNP/4NP-type local collapse while retaining the Siamese model's reduction in substrate clustering.

### Phase 6: Swap and counterfactual validation

For an observed paired master sample:

1. encode its spectrum in domain A;
2. retain `z_chemical` from A;
3. combine it with nuisance/domain information from B;
4. decode the counterfactual spectrum;
5. compare with the actual measurement of the same master sample in B.

Metrics include spectral angle, correlation, peak positions, local peak ratios, and classification agreement.

A plausible-looking spectrum for an unobserved target × sensor cell is not accepted as proof of disentanglement.

### Phase 7: Generalization stress tests

- Poster leave-one-substrate-family-out;
- NATO grouped new-master folds;
- NATO leave-one-instrument-out domain-only;
- NATO leave-one-sensor-out domain-only;
- supported NATO domain-and-sample tests;
- NATO-Q500 versus flagged-quality stress cohort;
- controlled synthetic corruptions at increasing severity.

## 15. Evaluation matrix

### 15.1 Classification

- balanced accuracy;
- macro F1 over supported true classes;
- per-class recall and precision;
- confusion matrices;
- calibration error and confidence distributions;
- bootstrap or repeated-seed uncertainty intervals.

### 15.2 Reconstruction

- normalized MSE/smooth L1;
- spectral angle;
- Pearson/Spearman correlation;
- derivative error;
- peak-location error;
- local peak-ratio error;
- error stratified by instrument, sensor, target, and quality flag.

### 15.3 Representation

- chemical silhouette;
- instrument and sensor silhouette;
- chemical-minus-domain silhouette;
- prototype margins;
- same-master cross-domain distances;
- linear chemical probe from each latent;
- linear instrument/sensor probes from each latent;
- target-conditional domain probes;
- domain-conditional chemical probes where supported.

### 15.4 Robustness

- prediction agreement before and after corruption;
- `z_chemical` drift versus corruption strength;
- reconstruction recovery versus known pre-corruption target;
- spike-removal false-positive and recovery rates;
- performance on field-flagged spectra;
- sensitivity to baseline algorithm and scaling choice.

### 15.5 Interpretability

- latent traversals;
- measured-counterpart swaps;
- decoder difference spectra;
- peak attribution or saliency stability;
- relation of nuisance latents to baseline burden, instrument, sensor, exposure, quality, and acquisition metadata.

Projection plots such as PCA, UMAP, and t-SNE remain qualitative diagnostics and never replace held-out metrics or prototype distances.

## 16. Required ablations

### Preprocessing ablations

- no spike correction versus spike correction;
- ordinary versus robust min--max;
- no baseline versus AsLS/arPLS/rubber band;
- raw/min--max versus first/second derivative;
- no smoothing versus selected Savitzky--Golay smoothing;
- domain-blind versus known-system preprocessing.

### Model ablations

- deterministic AE versus VAE;
- VAE versus denoising VAE;
- one latent versus chemical/nuisance split;
- no labels versus chemical classification;
- no metric loss versus triplet/contrastive loss;
- row-balanced versus domain-balanced sampling;
- no domain conditioning versus instrument/sensor-conditioned decoder;
- no independence term versus cross-covariance penalty;
- no adversary versus target-conditional adversary;
- core versus quality-pass training;
- latent sizes and β schedules;
- multiple independent seeds.

Every ablation uses the same outer split and, where possible, matched model capacity.

## 17. Decision rules

A representation is preferred only if it occupies a better chemical/domain/robustness trade-off.

Minimum evidence for advancement:

1. chemical performance is competitive with the best classical and Siamese baselines within uncertainty;
2. instrument/sensor predictability from `z_chemical` decreases relative to a standard VAE at comparable chemical performance;
3. same-master spectra become closer across domains without collapsing different targets;
4. injected-corruption stability improves;
5. peak preservation does not deteriorate materially;
6. no new localized collapse analogous to AgNP/4NP is hidden by the mean score;
7. conclusions survive core versus quality-pass sensitivity analysis;
8. all unsupported target/domain cells are disclosed.

There is no single fixed acceptable instrument-probe score because the NATO design itself is confounded. Results are compared on matched subsets and as Pareto curves against target performance.

## 18. Reproducibility requirements

Each run records:

- dataset manifest hash/version;
- observation IDs in train, validation, test, and exclusions;
- preprocessing configuration and fitted parameters;
- spectrum-level spike, baseline, shift, and scaling metadata;
- model architecture and parameter count;
- random seeds;
- optimizer, schedule, early stopping, and epochs;
- hardware and software environment;
- all fold-level and per-domain metrics;
- predictions, latent vectors, reconstructions, and uncertainty;
- failure examples and quality flags.

Required output structure:

```text
Workspace/nato_sers_vae/
  dataset_version/
  preprocessing_runs/
  baselines/
  autoencoders/
  vae/
  disentangled_vae/
  hybrid_vae/
  evaluations/
  figures/
  run_manifest.csv
```

No result is promoted from a single favorable seed or a random spectrum split.

## 19. Additional data required for stronger claims

The cleanest future experiment is a crossed design:

- every target and blank on every SERS sensor family;
- several independently prepared physical samples per target × sensor cell;
- each cell measured on multiple Raman instruments;
- randomized order, sessions, and operators;
- calibration standards and explicit raw/dark/reference exports;
- sufficient independent samples to hold out both a sensor and all associated physical specimens without eliminating a target class.

Additional spectra from the missing PMCDS and Agilent-2 systems should be recovered if possible, but file recovery alone does not fix the missing-cell design.

## 20. Immediate implementation order

1. Freeze and version the 598 and 500 manifests and existing split files.
2. Generate spike masks and ordinary/robust min--max arrays without overwriting the common raw matrix.
3. Implement the sequential preprocessing benchmark and paired-master diagnostics.
4. Reproduce the poster derivative, classical, and Siamese baselines from frozen splits.
5. Establish NATO classical and Siamese baselines using the retained preprocessing branches.
6. Implement capacity-matched deterministic and denoising autoencoders.
7. Implement the standard VAE and characterize its mixed latent.
8. Add chemical supervision and then the two-block chemical/nuisance architecture.
9. Add the cross-domain metric objective from the Siamese work.
10. Introduce domain conditioning, independence, and adversarial terms one at a time.
11. Run measured-counterpart swap tests and all supported domain stress tests.
12. Produce a final comparison centered on chemical performance, domain leakage, reconstruction/peak preservation, corruption robustness, and localized failures.

## 21. Final methodological position

The Siamese work should not be discarded. It already demonstrates that metric learning can reduce substrate organization. Its limitation is that the embedding is free to discard or warp spectral information as long as relative distances satisfy the training objective.

A standard VAE adds reconstruction, but it does not automatically distinguish signal from noise and may preferentially encode dominant background or instrument variation. A β-VAE alone does not assign chemical meaning to a latent block.

The scientifically strongest next step is therefore:

> a min--max-scaled, reconstruction-preserving denoising/disentangled VAE whose chemical latent is supervised by chemical labels and cross-domain metric learning, whose nuisance latent is allowed to explain sensor/instrument/background effects, and whose claims are tested with grouped specimens, held-out domains, domain probes, controlled corruptions, peak preservation, and real measured counterpart swaps.

This approach directly joins the demonstrated strength of the Siamese model with the missing spectral-preservation and factorization objectives.
