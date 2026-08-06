# ATLAS field-trial SERS research master plan v1

**Plan date:** 2026-08-05
**Plan status:** execution-ready analysis plan; no definitive experiments are authorized by this document itself
**Primary data workspace:** private path supplied through `ATLAS_PRIVATE_ROOT`
**Plan workspace:** `research/atlas_sers/plan`
**Primary scientific theme:** chemical identification under unseen-instrument acquisition shift
**Secondary scientific theme:** narrow, station-conditioned unknown-chemical rejection
**Independent experimental unit:** physical `master_sample_id`

## 0. Master decision

The definitive research question is:

> **Can an acquisition-aware representation trained on heterogeneous field-trial SERS spectra identify supported chemicals on an instrument absent from training, and on physical samples absent from training, more reliably than rigorously tuned classical chemometric and machine-learning methods?**

The proposed contribution is not a generic classifier, generic CNN, VAE denoiser, or claim of semantic nuisance disentanglement. It is a leakage-controlled comparison of classical and deep methods under measured acquisition shift, with repeated cross-instrument observations of physical field samples used as structured supervision.

The primary method candidate is a compact, location-preserving one-dimensional residual encoder trained with:

1. a station-conditioned chemical classification loss;
2. multi-positive supervised contrastive learning;
3. additional emphasis on same-chemical observations acquired by different instruments or SERS sensors; and
4. paired-view consistency for the same physical master measured by different acquisition systems.

Conditional CORAL and conditional domain-adversarial learning are controls, not assumed improvements. A larger ordinary CNN is not the default response to the pilot failures.

The strongest fallback study is a classical, station-conditioned acquisition-shift benchmark with calibrated uncertainty, instrument-balanced master aggregation, and a fully documented negative result if deep learning adds no reliable value.

## 1. Evidence already observed and its consequences

### 1.1 Fresh raw-data restart

The independently reconstructed archive contains:

- 7,148 files and 1,397,661,278 bytes;
- 1,172 expanded recording-log observations;
- 721 rows with an explicitly named SERS substrate;
- 626 numeric-QC-pass named-SERS observations;
- 615 observations attributable to a physical master and target;
- 598 unique attributable SERS spectra;
- 500 notes-clear sensitivity spectra;
- 69 physical masters, seven targets, ten instruments, four sensor families, and three stations.

The archive has a strict measured all-instrument intersection of 400–1,849 cm⁻¹. The primary analysis range is 400–1,800 cm⁻¹ to avoid terminal-edge behavior. A lower bound of 100 cm⁻¹ is prohibited because it would require fabricated extrapolation for Mira and other systems.

Raw spectral structure is acquisition dominated. Raw channel-standardized PCA places 93.68% of variance in PC1. After per-spectrum min–max scaling, PC1 falls to 42.31%, and 46 PCs are needed for 95% variance. Stable K-means structure remains more associated with instrument than chemistry: instrument NMI approximately 0.460 versus target NMI approximately 0.227.

Minimal min–max preprocessing is the primary protocol. Smoothing gives small and task-dependent gains, while baseline corrections reduce instrument association but can expose sensor association and alter peak structure. No preprocessing candidate establishes a chemistry-only representation.

### 1.2 Fresh predictive pilot

Current master-grouped within-station classical balanced accuracies are approximately:

| Station | Strong classical pilot | Balanced accuracy |
|---|---|---:|
| CWA | smoothed PCA–LDA | 0.663 |
| Pills | smoothed Extra Trees | 0.845 |
| Surfaces | smoothed PCA–LDA | 0.836 |

A bounded 277,479-parameter CNN obtained approximately 0.432, 0.776, and 0.470 respectively. The current CNN is therefore a negative control, not evidence that all deep approaches fail.

Across the 13 adequately supported unseen-instrument/station domains, minimal-preprocessing RBF SVM currently obtains mean balanced accuracy 0.686, median 0.734, and worst-domain 0.333. This is the reference value to beat, but it is already observed and cannot be treated as a prospectively hidden endpoint.

### 1.3 Earlier broad open-world pilot

An earlier study trained classical, Siamese, supervised-contrastive, VAE, structured-VAE, residual-CNN, Objectosphere, CORAL, MMD, and GroupDRO controls. Its residual CNN passed zero of six promotion gates, and source-only aggregate held-instrument CNN performance was poor. The study also found that VAE reconstruction error was much better at detecting field-quality shift than unseen chemistry.

These results are disclosed pilot knowledge. The definitive study must not:

- describe itself as a prospective preregistration;
- retune methods against the same held-domain outcomes and then report naive confidence intervals;
- repeat the full broad CNN objective grid without a new mechanistic rationale;
- reinterpret VAE reconstruction as denoising or chemical/nuisance separation.

The fresh study differs by conditioning chemistry tasks on station, defining 13 support-qualified acquisition domains, using the repeated-master structure directly, and comparing all methods under a single common evaluation harness.

## 2. Claim hierarchy

Every result and figure must carry one of these scope labels.

### P — primary

Station-conditioned, zero-shot unseen-instrument chemical identification with unseen physical masters. The primary endpoint is the unweighted mean single-spectrum balanced accuracy over 13 support-qualified instrument/station domains.

### S — secondary

- within-station master-grouped chemical classification;
- bidirectional pills/surfaces transfer for 4-ANPP versus benzyl fentanyl;
- calibration, selective prediction, and instrument-balanced master aggregation;
- explicitly declared target-instrument adaptation regimes;
- narrow within-station unknown-chemical rejection.

### E — exploratory

- UMAP and t-SNE geometry;
- individual peak attributions;
- low-support held-instrument domains;
- candidate architectures eliminated during development;
- qualitative case studies and individual-spectrum narratives.

### X — prohibited claim

The study must not claim:

- recovery of a unique clean or chemical-only spectrum;
- generic denoising without paired clean targets;
- causal chemical/nuisance disentanglement;
- generalization to arbitrary chemicals, concentrations, substrates, sensors, or instruments;
- that row-level spectra are independent samples;
- that station-confounded seven-class accuracy is chemistry-only evidence;
- that master-aggregated performance is single-spectrum performance;
- that a target instrument is unseen if any of its spectra informed training, normalization, selection, thresholding, or stopping;
- that a station-held-out chemical is a valid general open-set test.

## 3. Immutable data contract

### 3.1 Authoritative population

The primary population is `tier_unique_attributable_sers`, containing 598 rows. Each row must map to exactly one:

- `observation_uid`;
- unique `(instrument, source_scan_id)` source;
- physical `master_sample_id`;
- normalized `target_analyte`;
- station;
- instrument;
- sensor family and, where available, sensor variant;
- native-axis source and immutable source hash.

The notes-clear population of 500 spectra and Mira-1-excluded population of 575 spectra are sensitivity populations. They may not replace the primary population after model outcomes are inspected.

### 3.2 Target classes by station

| Station | Target classes | Spectra | Masters |
|---|---|---:|---:|
| CWA | 4-nitrophenol, ethanol, ethyl paraoxon | 208 | 24 |
| Pills | 4-ANPP, benzyl fentanyl, blank | 208 | 20 |
| Surfaces | 4-ANPP, acetaminophen, benzyl fentanyl | 182 | 25 |

The aggregate seven-class task is descriptive only because station predicts target above chance.

### 3.3 Representations

Every experiment must identify one immutable representation ID.

| ID | Range | Operation | Scope |
|---|---|---|---|
| `R_MIN_400_1800` | 400–1,800 | measured-support interpolation, per-spectrum min–max | primary |
| `R_MIN_400_1849` | 400–1,849 | measured-support interpolation, per-spectrum min–max | range sensitivity |
| `R_SG_400_1800` | 400–1,800 | conservative impulse replacement, SG(11,3), min–max | smoothing sensitivity |
| `R_ARPLS_400_1800` | 400–1,800 | arPLS, min–max | baseline sensitivity |
| `R_SNV_400_1800` | 400–1,800 | interpolation, SNV | normalization sensitivity |
| `R_VECTOR_400_1800` | 400–1,800 | interpolation, L2 vector norm | normalization sensitivity |
| `R_AREA_400_1800` | 400–1,800 | interpolation, nonnegative integrated-area norm | normalization sensitivity |
| `R_D1_400_1800` | 400–1,800 | first SG derivative, SNV | negative/destructive control |

Population-fitted operations—including PCA, learned standardization, feature selection, class centroids, calibration, and domain alignment—must be fit inside the training partition. Row-local interpolation and row-local scaling may be performed before splitting because they use no other observation.

### 3.4 Native-axis preservation

No common-axis array replaces the native source. Each prediction must be traceable to native coordinates and intensity hash. Peak-attribution figures must show both the common-grid model input and the corresponding native spectrum.

## 4. Scientific estimands and hypotheses

### 4.1 Primary estimand

For method \(m\), station \(s\), and support-qualified held instrument \(d\), let \(BA_{m,s,d}\) be balanced accuracy after pooling the out-of-fold predictions in which the test spectrum belongs to a held master and instrument \(d\) is absent from training. The primary estimand is:

\[
\theta_m = \frac{1}{13}\sum_{(s,d)\in\mathcal D_{13}} BA_{m,s,d}.
\]

The primary comparison is:

\[
\Delta = \theta_{\text{acquisition-aware deep}}-
\theta_{\text{development-selected classical}}.
\]

Domains receive equal weight. Row count may not weight the primary endpoint.

### 4.2 Primary hypothesis H1

The acquisition-aware deep pipeline improves mean domain-balanced accuracy by at least 0.03 over the classical selection pipeline while satisfying chemistry-retention and worst-domain safety gates.

This is an estimation problem first. Report \(\Delta\), its interval, the 13 paired domain differences, and leave-one-domain-out sensitivity. A binary p-value does not replace these quantities.

### 4.3 Supporting hypotheses

| ID | Hypothesis | Endpoint | Required interpretation |
|---|---|---|---|
| H2 | acquisition shift materially degrades ordinary validation | ordinary within-station BA minus T3 BA | quantifies the shift gap |
| H3 | paired-view supervision improves cross-instrument consistency | paired master prediction agreement and embedding distance, conditioned on correctness | cannot be interpreted alone |
| H4 | acquisition-aware learning reduces residual domain information without erasing chemistry | target-adjusted instrument/sensor probe increment and chemistry BA | both sides required |
| H5 | aggressive preprocessing is not universally superior | paired T3 differences across preprocessing branches | primary preprocessing remains fixed |
| H6 | calibrated uncertainty supports useful abstention | risk–coverage curve, NLL, Brier, ECE | thresholds use development-known data only |
| H7 | limited target-instrument data changes the achievable operating point | zero-shot versus UDA versus paired calibration versus supervised few-shot | information regimes never pooled |
| H8 | unknown rejection remains fragile under field stress | station-conditioned open-set AUROC, AUPR, OSCR | secondary, narrow claim only |

## 5. Task and information-access regimes

### T0 — aggregate descriptive continuity

Seven-class master-grouped classification across all stations. Used only to compare with historical work and demonstrate station confounding. It is not used for primary method selection.

### T1 — within-station closed-set chemistry

Three separate three-class tasks. Outer unit is physical master. This estimates chemistry separability without requiring transfer to a completely absent instrument.

### T2 — bidirectional cross-station binary transfer

4-ANPP versus benzyl fentanyl, pills→surfaces and surfaces→pills. Hyperparameters and calibration are selected only within the training station.

### T3-ZS — zero-shot unseen instrument plus unseen master

Primary regime. For every outer master fold and held instrument:

- training uses outer-training masters only;
- every spectrum from the held instrument is removed from training, validation, preprocessing fitting, calibration, and early stopping;
- testing uses outer-test masters measured by the held instrument;
- no target-instrument spectrum is available during fitting;
- all source-domain hyperparameter selection uses only training masters and training instruments.

### T3-UDA — unlabeled target-instrument adaptation

Secondary. A target-instrument adaptation-master set is disjoint from evaluation masters. Target labels, cross-instrument master links, and evaluation-master spectra remain hidden. Only target intensities and their instrument membership may be used. Results are explicitly labelled transductive/unsupervised adaptation and never compared as if zero-shot.

### T3-PC — paired calibration

Secondary operational scenario. A disjoint calibration-master set supplies target-instrument spectra and their known links to labelled source-instrument views. Target-view chemical labels are not directly supplied, but the paired source view makes chemistry inferable. This is paired calibration, not UDA.

### T3-FS — supervised few-shot calibration

Secondary learning curve at \(k\in\{1,2,3,5\}\) labelled target-instrument masters per class when support permits. Calibration and evaluation masters are disjoint. Repeated draws are frozen. Report performance versus labelled masters, not spectra.

### T4 — station-conditioned unknown rejection

Hold one nonblank chemical out within a station. The unknown chemical is absent from classifier fitting, score selection, calibration, and thresholding. Scores and score-selection policy are fixed across held chemicals. This is a stress test because only two known classes remain in each station task.

## 6. Primary domain registry

The 13 primary domains contain all three station targets and at least 15 held-instrument test masters across pooled outer folds:

| Station | Held instruments |
|---|---|
| CWA | Mira-2, Pendar-2, RMX-1 |
| Pills | Agilent-1, Agilent-3, Mira-3, Pendar-1, Pendar-3 |
| Surfaces | Agilent-3, Mira-1, Pendar-2, Pendar-3, RMX-2 |

Four additional domains remain exploratory and visible:

- CWA/Agilent-3: three classes, eight masters;
- CWA/Pendar-3: three classes, eight masters;
- Pills/Pendar-2: two classes, four masters;
- Surfaces/Mira-2: two classes, two masters.

Eligibility is frozen from metadata/support, never from model performance.

## 7. Master execution map

| Phase | Sub-plan | Depends on | Terminal gate |
|---|---|---|---|
| P00 | governance and artifact contract | completed restart | immutable registries and hashes |
| P01 | data and representation freeze | P00 | exact UID, source, axis, tier validation |
| P02 | split and information-regime freeze | P01 | zero leakage in reconstructed partitions |
| P03 | classical nested benchmark | P02 | calibrated row/master/domain predictions for every candidate |
| P04 | compact deep ERM baseline | P02 | stable training and fair classical comparison |
| P05 | acquisition-aware development | P04 | source-only development winner frozen |
| P06 | definitive T1/T2/T3-ZS evaluation | P03–P05 | complete paired predictions and primary inference |
| P07 | adaptation and calibration regimes | P06 | UDA/PC/FS labelled separately |
| P08 | robustness and preprocessing sensitivities | P06 | registered perturbation and tier results |
| P09 | narrow open-set evaluation | P06 | fixed-score known-only thresholds and held-class results |
| P10 | explainability, consistency, and error taxonomy | P06–P09 | attribution sanity and domain failure audit |
| P11 | statistics and decision gates | P06–P10 | intervals, paired contrasts, promotion verdict |
| P12 | figures, manuscript tables, and reproducibility | all | TikZ/HTML parity and final validation |

No phase may consume locked outputs from a downstream phase.

## 8. Sub-plan P00 — governance and artifact contract

### Objective

Make every scientific choice machine-readable before new definitive fits.

### Required actions

1. Record repository commit, environment lock, operating system, Python, CUDA, GPU, BLAS, and dependency versions.
2. Hash authoritative manifests, arrays, configuration files, split registries, and source parser.
3. Assign every experiment a stable `experiment_id` before execution.
4. Assign every output row a `run_id` determined from experiment, outer split, held domain, representation, hyperparameters, seed, and code hash.
5. Write protected-state hashes before model fitting.
6. Make execution idempotent: an existing successful `run_id` is verified and skipped; an incomplete run is quarantined rather than silently overwritten.
7. Store every deviation in `deviations.csv` with timestamp, rationale, affected runs, and whether it occurred before or after outcome access.

### Terminal checks

- All configuration files parse.
- Experiment IDs are unique.
- Every primary/secondary/exploratory label is populated.
- No output location points into the raw archive.
- Input hashes agree with the completed restart.

## 9. Sub-plan P01 — data and representation freeze

### Objective

Create analysis bundles that are reversible to native source and cannot silently mix tiers or preprocessing.

### Outputs

- `data/primary_manifest.csv`
- `data/notes_clear_manifest.csv`
- `data/mira1_excluded_manifest.csv`
- `data/native_source_registry.csv`
- `data/representation_registry.csv`
- one array bundle per representation;
- row-level QC and native/common-axis hashes.

### Validation

- Exactly 598 primary rows, 500 notes-clear rows, and 575 Mira-1-excluded rows.
- Exactly 69 primary masters.
- No duplicated `(instrument, source_scan_id)` in primary data.
- All axes finite and strictly increasing.
- Every common-grid coordinate lies inside every selected spectrum’s measured effective support.
- Every normalized row satisfies its declared invariant within tolerance.
- Candidate transformations reproduce the frozen preprocessing exploration within numeric tolerance.

No spectra are removed because a future model predicts them poorly.

## 10. Sub-plan P02 — split and information-regime freeze

### 10.1 Canonical outer splits

Use five declared repeat seeds: `20260805`, `20260817`, `20260829`, `20260910`, and `20260922`. For each station and repeat, construct four stratified group folds over target and `master_sample_id`. The smallest class/master cell determines four folds.

Each master appears in exactly one outer test fold per repeat. All rows from a master inherit its fold.

### 10.2 T3-ZS derivation

For held instrument \(d\) and outer fold \(o\):

- `train_source`: station rows with outer fold not \(o\), instrument not \(d\);
- `test_target`: station rows with outer fold \(o\), instrument equal to \(d\);
- `excluded_train_target`: training-master rows from instrument \(d\);
- `excluded_test_source`: test-master rows from instruments other than \(d\).

The excluded groups are preserved with reason codes. They are not deleted from provenance.

### 10.3 Inner model-selection splits

Within `train_source`, construct master-grouped inner folds. For T3-ZS, candidate ranking uses pseudo-domain validation where a supported source instrument is held out in turn. The inner objective is lexicographic:

1. highest unweighted mean supported pseudo-domain balanced accuracy;
2. highest worst supported pseudo-domain balanced accuracy;
3. highest mean macro-F1;
4. lowest model complexity;
5. stable declared candidate order.

If fewer than two supported pseudo-domains exist, fall back to master-grouped inner balanced accuracy and record `selection_fallback=master_cv`.

### 10.4 Leakage assertions

For every run:

- training, validation, calibration, and test master sets are pairwise disjoint where their roles require disjointness;
- the held instrument is absent from all T3-ZS fitting rows;
- test labels never determine preprocessing, hyperparameters, epochs, scores, thresholds, or representation;
- target-instrument rows are absent from T3-ZS population fitting;
- repetitions and neural seeds are technical repeats, not independent sample units.

The split validator must reconstruct partitions from metadata and compare UID sets exactly.

## 11. Sub-plan P03 — definitive classical benchmark

### 11.1 Candidate families

All models use class-balanced fitting where supported and are tuned only inside the training data.

1. **Prior dummy:** empirical class prior and uniform prior.
2. **Correlation/spectral-angle matching:** training-master prototypes; cosine, Pearson, and spectral-angle distances.
3. **Nearest centroid:** Euclidean and cosine; shrinkage `None`, 0.01, 0.1, 0.5.
4. **PCA–LDA:** PCA components `{5,10,20,40,0.95 variance}` bounded by inner sample rank; LDA solvers `svd` and shrinkage `lsqr/auto`.
5. **PLS-DA:** components `{2,4,8,12,16}` bounded by rank; class-balanced multinomial logistic head.
6. **Elastic-net multinomial logistic regression:** `C={0.001,0.01,0.1,1,10,100}`, `l1_ratio={0,0.25,0.5,0.75,1}`.
7. **RBF SVM:** `C={0.01,0.1,1,10,100,1000}` and `gamma={scale,0.0001,0.001,0.01,0.1,1}`; probabilities derived only after cross-fitted calibration.
8. **Random Forest:** 1,000 trees; `max_features={sqrt,0.1,0.3,0.5}`, `min_samples_leaf={1,2,4,8}`, class weight balanced.
9. **Extra Trees:** same forest grid with bootstrap disabled.
10. **CORAL plus classical classifier:** source-only pseudo-domain alignment followed by PCA–LDA or RBF SVM; no held-target covariance in T3-ZS.

Gradient boosting is exploratory only and is included only if its installed implementation and full grid are frozen before T3 access.

### 11.2 Classical selection pipeline

The primary classical comparator is an algorithmic selection pipeline, not the retrospectively best test model. In every outer run, the inner source-only objective selects one candidate and hyperparameter set. The selected model is refit on all allowable source-training rows.

Also report fixed PCA–LDA, fixed RBF SVM, Random Forest, and Extra Trees to make selection behavior transparent.

### 11.3 Calibration

Generate master-grouped cross-fitted training scores. Fit a single scalar temperature for multiclass scores by minimizing known-development NLL. For classifiers without logits, use clipped log probabilities before temperature fitting. Calibration uses no outer-test or held-instrument rows.

### 11.4 Required outputs

- inner candidate scores and selections;
- fitted hyperparameter registries;
- spectrum-level probabilities and predictions;
- instrument-balanced master probabilities;
- per-class, per-fold, per-station, and per-domain metrics;
- calibration curves and scores;
- training time, prediction time, peak memory, and serialized model size.

### 11.5 Negative controls

- permute target labels at the master level inside each station;
- replace spectra with acquisition metadata only;
- replace spectra with station/target priors;
- verify that target performance collapses toward chance under master permutation.

## 12. Sub-plan P04 — compact ordinary deep baseline

### 12.1 Architecture contract

Use a compact location-preserving 1-D residual encoder on 1,401 channels:

- input: `(batch,1,1401)`;
- stem: Conv1d `1→24`, kernel 11, stride 1, padding 5; GroupNorm; GELU;
- stage 1: two residual blocks, 24 channels, kernel 7, dilations 1 and 2;
- transition 1: Conv1d `24→48`, kernel 5, stride 2;
- stage 2: two residual blocks, 48 channels, kernel 7, dilations 1 and 2;
- transition 2: Conv1d `48→64`, kernel 5, stride 2;
- stage 3: two residual blocks, 64 channels, kernel 5, dilations 1 and 2;
- adaptive average pooling to 16 ordered bins, preserving coarse peak location;
- projection `1024→96→64` with GELU and dropout 0.2;
- linear class head from the 64-dimensional embedding.

No batch normalization is allowed. Parameter count must be reported and remain below 250,000. Any architectural correction made to satisfy dimensions must occur before outcome-bearing evaluation and be entered in the deviations log.

### 12.2 Optimization

- AdamW;
- learning rate candidates `{3e-4,1e-3}`;
- weight decay `{1e-5,1e-4,1e-3}`;
- batch size 48 or the largest deterministic size not exceeding 64;
- class-balanced cross-entropy;
- maximum 200 epochs;
- minimum 30 epochs before stopping;
- validation every epoch;
- patience 20 epochs;
- gradient clipping at norm 5;
- three training seeds: `20260805`, `20260817`, `20260829`;
- checkpoint chosen by source-only validation balanced accuracy, then NLL, then earliest epoch.

Learning curves must demonstrate whether a fold failed through underfitting, collapse, overfitting, or optimization instability. A failed fold is retained, not silently restarted until favorable.

### 12.3 Augmentation contract

Primary neural augmentation is restricted to physically modest training-only perturbations:

- wavenumber translation uniformly in ±2 cm⁻¹;
- multiplicative intensity factor 0.9–1.1 before row scaling;
- additive linear baseline span up to 5% of the row range;
- Gaussian noise calibrated to training-fold first-difference noise quantiles.

Augmentation parameters are logged per example. ±5 cm⁻¹ shift is a robustness sensitivity, not the primary training policy. Peak deletion, arbitrary warping, mixup across different chemicals, and target-instrument-derived noise are prohibited.

## 13. Sub-plan P05 — acquisition-aware deep development

### 13.1 Batch construction

Each batch is sampled at the master level and must contain:

- at least two station-local target classes;
- at least two masters per represented target where feasible;
- cross-instrument views of at least one master or target where feasible;
- no duplicate row counted as an independent batch item.

Sampling weights prevent masters with more recorded spectra from dominating.

### 13.2 Multi-positive supervised contrastive loss

For anchor \(i\), positives share station and target. Positive weights are:

- 1.0 for same-target/same-instrument different-master pairs;
- 1.5 for same-target/different-instrument pairs;
- 2.0 for same-master/different-instrument pairs;
- an additional factor 1.25 for different-sensor pairs, capped at total weight 2.5.

Different-target spectra within the same station are negatives. Same-target different-master rows are never negatives. Cross-station pairs are excluded from the primary contrastive denominator because station and chemistry support are not factorially crossed.

Candidate temperatures are `{0.05,0.1,0.2}`. The contrastive projection head is used only for the loss; classification uses the pre-projection 64-dimensional embedding.

### 13.3 Paired-view consistency

For training-master cross-instrument pairs, minimize:

1. symmetric KL divergence between class probabilities; and
2. cosine distance between normalized embeddings.

Consistency loss is evaluated beside correctness. Agreement between consistently wrong predictions is not a success.

### 13.4 Conditional CORAL control

Align class-conditional first and second embedding moments across supported source instruments only. A class/domain cell must contain at least two batch or memory-bank observations. Missing cells contribute no penalty. Held-instrument statistics are prohibited in T3-ZS.

### 13.5 Conditional domain-adversarial control

A gradient-reversal instrument head receives the detached/conditioned combination of embedding and chemical probabilities. It predicts only source instruments observed during training. The gradient-reversal coefficient follows a frozen sigmoid schedule from 0 to its selected maximum. Domain confusion is diagnostic; chemistry performance remains the optimization and promotion requirement.

### 13.6 Development candidate ladder

Candidates are added sequentially:

| ID | Loss |
|---|---|
| D0 | cross-entropy ERM |
| D1 | ERM + supervised contrastive |
| D2 | ERM + paired prediction/embedding consistency |
| D3 | ERM + supervised contrastive + paired consistency |
| D4 | D3 + conditional CORAL |
| D5 | D3 + conditional domain adversary |

Loss-weight grids are deliberately small:

- `lambda_supcon={0.1,0.3,1.0}`;
- `lambda_pair={0.1,0.3,1.0}`;
- `lambda_coral={0.01,0.1}`;
- `lambda_domain={0.01,0.1}`.

Candidate development uses source-only inner pseudo-domain validation. D4 and D5 cannot both advance unless their source-only performance is indistinguishable and compute permits a declared secondary comparison.

### 13.7 Deep advancement rule

One acquisition-aware candidate is frozen before definitive outer aggregation. It advances over D0 only if, across source-only development tasks:

- mean pseudo-domain BA improves by at least 0.02;
- worst pseudo-domain BA does not decrease by more than 0.02;
- ordinary within-source BA does not decrease by more than 0.02;
- at least 60% of pseudo-domains improve;
- no more than 5% of runs collapse to chance or a single predicted class.

If no candidate passes, D3 remains a named mechanistic control but is not described as the expected winner.

## 14. Sub-plan P06 — definitive T1/T2/T3-ZS evaluation

### 14.1 Models carried forward

- prior dummy;
- fixed PCA–LDA;
- fixed RBF SVM;
- fixed Random Forest;
- fixed Extra Trees;
- development-selected classical pipeline;
- compact D0 ERM network;
- one frozen acquisition-aware deep candidate;
- D3 as a fixed mechanistic ablation if it is not the selected candidate.

No additional model is added after inspecting primary paired domain differences.

### 14.2 Prediction levels

**Spectrum:** every spectrum receives one out-of-fold probability vector.
**Instrument-view:** repeated spectra from one master/instrument are averaged.
**Master:** instrument-view probabilities are averaged with equal weight per instrument.

Master aggregation may not let an instrument with more repeats dominate. Report the number of spectra and instruments contributing to every master prediction.

### 14.3 Primary metrics

- single-spectrum balanced accuracy per domain;
- unweighted mean over 13 domains;
- median, IQR, and worst-domain BA;
- macro-F1 and per-class sensitivity;
- paired method difference for every domain.

### 14.4 Secondary predictive metrics

- instrument-balanced master BA;
- multiclass NLL;
- multiclass Brier score;
- ECE with equal-mass bins and bootstrap uncertainty;
- accuracy and confusion matrices;
- prediction entropy;
- model size and inference latency.

## 15. Sub-plan P07 — adaptation and target-calibration regimes

Zero-shot conclusions are frozen before adaptation work.

### 15.1 UDA

Use disjoint target-instrument adaptation masters with labels and pair identities hidden. Compare source-only D0/D3 with CORAL, entropy minimization, and feature-statistic adaptation. Adaptation sample sizes are `{3,5,10}` masters when support permits. Evaluate on masters absent from adaptation.

### 15.2 Paired calibration

Permit target spectra from calibration masters and their paired source views. Compare:

- prototype correction;
- paired embedding consistency fine-tuning;
- piecewise direct standardization only if enough paired spectra and rank exist;
- low-rank adapter modules attached to the frozen encoder.

### 15.3 Supervised few-shot

Use `k={1,2,3,5}` labelled target masters per class. All draws are master-stratified, repeated, and frozen. Compare full fine-tuning, head-only tuning, and low-rank adapters. Never report few-shot results as zero-shot transfer.

## 16. Sub-plan P08 — robustness and preprocessing sensitivities

### 16.1 Prespecified input perturbations

Apply only at test time, without retraining:

- wavenumber shifts: −5 to +5 cm⁻¹ in 1 cm⁻¹ increments;
- additive baseline slope: −10% to +10% of row range;
- broad quadratic background: 0–10% of row range;
- Gaussian noise at training-noise quantiles 0.50, 0.75, 0.90, and 0.95;
- isolated impulse contamination at 0, 1, 3, and 5 locations;
- intensity clipping at upper quantiles 1.00, 0.999, 0.995, and 0.99.

Robustness area under the degradation curve is reported. Perturbations are not claimed to reproduce a specific physical instrument unless supported by source measurements.

### 16.2 Preprocessing branches

Rerun the fixed classical and frozen neural pipelines under `R_SG_400_1800`, `R_ARPLS_400_1800`, and `R_MIN_400_1849`. Broader normalization branches are evaluated for the fixed classical champion only unless compute is explicitly expanded before results.

### 16.3 Population branches

Repeat primary comparisons on notes-clear and Mira-1-excluded tiers using regenerated master-group splits. Preserve the same eligibility logic; report domains that become unsupported.

## 17. Sub-plan P09 — narrow open-set evaluation

### 17.1 Held-unknown tasks

- CWA: hold each of its three chemicals in turn;
- Pills: hold 4-ANPP and benzyl fentanyl in turn; blank remains known;
- Surfaces: hold each of its three chemicals in turn.

This gives eight station/held-chemical tasks. Chemical overlap between pills and surfaces is reported; task rows are not treated as eight universally independent chemicals.

### 17.2 Frozen anomaly scores

- one minus calibrated maximum probability;
- predictive entropy;
- energy from raw neural logits;
- minimum class-conditional Mahalanobis distance;
- k-nearest-training-master embedding distance;
- inductive conformal nonconformity.

Score direction is always larger = more unknown. The score-selection policy is fixed across all held chemicals and cannot use true held-unknown outcomes.

### 17.3 Thresholds

Use master-grouped, known-only cross-fitted development scores. Thresholds target known coverages `{0.95,0.90,0.80,0.70,0.50}` using a declared quantile method. Unknown spectra cannot select thresholds.

### 17.4 Metrics

- unknown AUROC and AUPR;
- OSCR and OSCR-AUC;
- FPR at 95% known recall;
- realized known coverage;
- accepted-known balanced accuracy;
- unknown rejection at each known-only threshold.

Open-set figures must show each held chemical/task, not only the mean.

## 18. Sub-plan P10 — representation and failure analysis

### 18.1 Required probes

Fit training-only linear probes on frozen embeddings for:

- target;
- instrument;
- sensor family;
- station;
- master identity as an overfitting diagnostic;
- quality tier.

Instrument and sensor probes require a chemistry-label-only null reflecting the observed factor-support table. Report increment over that null, not raw domain predictability alone.

### 18.2 Paired-view diagnostics

For same-master pairs:

- cross-instrument cosine distance;
- cross-sensor cosine distance;
- prediction agreement;
- agreement conditional on either/both predictions being correct;
- class-probability Jensen–Shannon divergence;
- top-k retrieval of paired masters and same-target masters.

### 18.3 Attribution

Use at least two methods for the final neural model:

- integrated gradients with a training-only baseline policy;
- occlusion over fixed 10 cm⁻¹ windows.

Perform attribution sanity tests by randomizing the classification head and, separately, all encoder weights. Attribution must materially change under randomization. Compare highlighted regions with peak-preservation diagnostics; do not assign chemical bonds without external references.

### 18.4 Error taxonomy

Every persistent failure is categorized by:

- station/target;
- held instrument and serial;
- sensor family/variant;
- master;
- quality notes and system-suitability status;
- baseline/noise/spike proxies;
- native support and acquisition metadata;
- confident wrong, uncertain wrong, rejected known, or accepted unknown.

Case studies are selected by prespecified rules—largest paired method disagreement, worst calibrated error, and representative median—not subjective visual interest.

## 19. Sub-plan P11 — statistical inference

### 19.1 Independent units

- physical masters for row/master performance uncertainty;
- 13 instrument/station domains for the primary transfer contrast;
- held station/chemical tasks for open-set summaries;
- training seeds are averaged before inference;
- fold repetitions are repeated measurements, not independent samples.

### 19.2 Primary interval

Use a paired hierarchical bootstrap with 10,000 replicates:

1. sample the 13 eligible domains with replacement;
2. within each sampled domain, resample physical masters within target class;
3. carry all spectra and paired method predictions for a sampled master together;
4. recompute each domain BA and the unweighted mean difference.

Report percentile and BCa intervals where numerically stable. Also report:

- paired domain differences;
- leave-one-domain-out estimates;
- leave-one-instrument-identity-out estimates because Agilent-3 and other identities may occur in multiple stations;
- exact or Monte Carlo paired sign-flip test as a descriptive sensitivity.

### 19.3 Multiplicity

There is one primary comparison and no adjustment for that contrast. Apply Holm correction separately within each prespecified secondary family: T1 models, preprocessing branches, robustness perturbations, adaptation regimes, and open-set scores. Exploratory results receive intervals without confirmatory significance language.

### 19.4 Missing/undefined outcomes

Unsupported class/domain metrics remain explicit missing values with reason codes. They are never replaced with zero or omitted without count. Collapse to one predicted class remains a valid poor result.

## 20. Promotion and publication decision gates

### G0 — integrity

All hashes, UIDs, axes, tiers, and split-leakage checks pass. Failure blocks every model result.

### G1 — classical benchmark validity

Permutation performance is near chance; all primary candidates produce complete out-of-fold predictions; calibration is fitted without test data; no master leakage exists.

### G2 — deep training validity

At least 95% of planned runs finish without NaN/Inf; all failed/collapsed runs remain in denominators; parameter and epoch budgets are obeyed; learning curves are archived.

### G3 — acquisition-aware development advancement

The P05 source-only advancement rule passes. Otherwise the acquisition-aware model remains a mechanistic control.

### G4 — primary superiority

Promote the acquisition-aware model as superior only if all hold:

- mean T3-ZS domain BA difference versus selected classical is at least +0.03;
- 95% hierarchical-bootstrap interval lower bound exceeds 0;
- at least 8 of 13 domains improve;
- worst-domain BA is not more than 0.03 below the classical worst domain;
- T1 chemistry BA is noninferior within −0.02;
- peak preservation remains inside the primary input contract.

If the point improvement is at least +0.03 but the interval crosses zero, label the result promising but inconclusive. If G4 fails, the deep method is not promoted.

### G5 — invariance interpretation

A domain-invariance claim requires reduced target-adjusted instrument/sensor probe increment plus noninferior chemistry and improved T3 transfer. Reduced probe accuracy alone fails this gate.

### G6 — publication route

- **Route A:** acquisition-aware deep method passes G4; primary methods paper.
- **Route B:** deep method fails G4, but classical benchmark is stable and shift effects are heterogeneous; rigorous benchmark/negative-result paper.
- **Route C:** both families are unstable; dataset/measurement and identifiability paper emphasizing failure modes and required future acquisition design.

Open-set performance cannot rescue a failed acquisition-shift primary analysis.

## 21. Figure and visualization contract

### 21.1 Mandatory paired outputs

Every quantitative plot or scientific schematic has:

1. editable native TikZ/PGFPlots source: `figures/tikz/<figure_id>.tex`;
2. compiled vector PDF for verification: `figures/pdf/<figure_id>.pdf`;
3. standalone self-contained HTML: `figures/html/<figure_id>.html`;
4. frozen plotting data: `figures/data/<figure_id>.csv` or `.json`;
5. a figure-manifest row containing data hash, filters, metrics, units, caption, and generation command.

The TikZ source must draw the data using TikZ/PGFPlots primitives. Embedding PNG, JPEG, PDF, SVG, or screenshots inside TikZ is prohibited. External frozen CSV data tables are permitted and preferred for large quantitative figures.

The HTML must embed its JavaScript and data. CDN-dependent Plotly pages are prohibited. Hover text must include observation/master/domain IDs appropriate to the aggregation level.

### 21.2 Semantic parity

TikZ and HTML counterparts must use identical:

- population and filters;
- aggregation unit;
- model/preprocessing labels;
- point estimates and intervals;
- axis limits, transforms, and units;
- color/category mapping;
- caption claim scope.

Interactive HTML may reveal additional hover fields or toggle traces but may not silently use a different analysis.

### 21.3 Style

- Okabe–Ito palette;
- redundant color plus shape/line-style encoding;
- grayscale interpretable;
- minimum 8 pt text and 0.5 pt strokes at IEEE double-column size;
- direct labels where practical;
- panel labels in bold capitals;
- no rainbow scale;
- diverging scales centered at a scientifically meaningful null;
- uncertainty displayed, not hidden behind bars;
- sample size and independent unit in caption.

### 21.4 Required figure families

The machine-readable figure registry defines the complete list. At minimum it covers:

- provenance and observation flow;
- factor-support and confounding matrices;
- native-axis coverage and instrument spectra;
- preprocessing preservation and structure changes;
- split/information-regime diagrams;
- classical and deep selection diagnostics;
- per-domain performance and paired differences;
- confusion, calibration, and risk–coverage;
- learning curves and architecture/loss diagrams;
- embedding target/domain trade-offs;
- paired-master consistency and retrieval;
- robustness degradation curves;
- preprocessing and quality-tier sensitivities;
- cross-station transfer;
- open-set ROC/OSCR and held-task forests;
- decision gates and final evidence map.

UMAP and t-SNE are exploratory visuals only. They are never quantitative proof of separation or invariance.

## 22. Compute and orchestration plan

### 22.1 Staged budget

| Stage | Approximate fits | Purpose |
|---|---:|---|
| data/split validation | 0 | fail fast before compute |
| classical inner/outer | 5,000–15,000 lightweight fits | nested model/hyperparameter comparison |
| compact D0 development | 100–250 neural fits | architecture and optimizer validity |
| D1–D5 source-only development | 300–700 neural fits | loss selection without held target |
| definitive D0 + selected deep | about 624 neural fits | 13 domains × 4 folds × 3 seeds × 2 models × 2 representative repeat sets, with remaining repeats added after stability gate |
| full five-repeat confirmation | up to about 1,560 additional fits | only after pipeline stability and resource audit |
| sensitivities/adaptation/open set | gated, separately budgeted | cannot delay primary completion |

Exact counts must be computed from the frozen registries before launch. A dry run lists run IDs, estimated wall time, disk, and GPU hours without fitting.

### 22.2 Execution behavior

- shard only by independent run ID;
- atomic write to temporary run directory followed by rename;
- checkpoint models and optimizers;
- save stdout/stderr and structured event logs;
- heartbeat and failure reason for every run;
- never overwrite completed outputs with a different code/config hash;
- aggregate only after the expected run registry is complete or missing runs have declared terminal reasons.

### 22.3 Early termination

Stop a run for NaN/Inf loss, irrecoverable I/O corruption, or resource failure. Do not stop because test performance is poor. Development candidates may be pruned only by prespecified source-only rules.

## 23. Artifact schemas

Every row-level prediction must include:

- protocol version and code/config/input hashes;
- experiment/run IDs;
- task and information regime;
- primary/secondary/exploratory scope;
- station, held domain, outer repeat/fold, seed;
- representation, model, hyperparameter hash, selected epoch;
- observation UID, source ID, master ID, instrument, sensor, target, quality tier;
- split role and every exclusion reason;
- true label, predicted label, ordered class vocabulary and probabilities;
- raw logits where defined;
- anomaly scores where defined;
- inference time and failure status.

Aggregate tables must always name their independent unit and aggregation function.

## 24. Reproducibility and validation

The final execution validator must check:

1. source and representation hashes;
2. expected population/tier counts;
3. master, held-instrument, held-chemical, calibration, and adaptation leakage;
4. complete expected run IDs;
5. unique prediction keys;
6. probability simplex and finite metrics;
7. recomputed metrics from row predictions;
8. seed averaging before inference;
9. undefined-support reason codes;
10. paired TikZ/HTML/data artifacts for every registered figure;
11. successful TikZ compilation without raster inclusion;
12. HTML self-containment without CDN references;
13. TikZ/HTML semantic-data hash parity;
14. captions, units, independent sample counts, and scope labels;
15. environment, runtime, and resource logs;
16. final artifact hashes.

## 25. Completion definition for the eventual research program

The research program is complete only when:

- all primary P00–P06 and P11–P12 gates pass;
- every primary run is complete or has a transparent terminal failure;
- classical and deep models share identical data/split/metric contracts;
- all 13 domains and four low-support exploratory domains remain visible;
- every primary claim is supported by row-level reproducible outputs;
- every planned quantitative figure has native TikZ and standalone HTML counterparts;
- promotion gates yield an unambiguous Route A, B, or C decision;
- prohibited claims are absent from abstracts, captions, and conclusions;
- validation and artifact hashes pass from a clean rebuild.

## 26. Immediate next action after plan approval

Do not begin with another model. Begin by implementing P00–P02:

1. freeze the experiment, metric, figure, and artifact registries;
2. materialize the five-repeat master split registry;
3. derive and validate the 13 T3-ZS domain partitions;
4. build the dry-run run registry and compute estimate;
5. obtain explicit approval of the frozen evaluation contract;
6. then execute the definitive classical benchmark before any new deep model.

This ordering establishes the strongest comparator and ensures the deep investigation answers a scientific question rather than becoming an unconstrained architecture search.

## 27. Phase-specific acceptance checklist

These are operational definitions of done. A phase does not advance because its principal script finished; it advances only when its evidence package passes.

### P00 acceptance

- Protocol, phase, task, experiment, metric, model, figure, and artifact registries are mutually consistent.
- Environment and repository state are recorded.
- Protected input hashes match the completed restart.
- The deviations log exists even when empty.
- A dry-run command can enumerate work without training.

### P01 acceptance

- Primary and sensitivity manifests contain their exact declared rows.
- Every representation has axis, transformation, row-order, and source hashes.
- Native-source reversibility is demonstrated for a random and boundary-case audit sample.
- No interpolation extrapolates outside measured effective support.
- Transformation invariants and peak-preservation summaries pass.

### P02 acceptance

- Every master appears exactly once as outer test per repeat and never crosses train/test within a fold.
- Every T3-ZS training role excludes the held instrument.
- Primary-domain eligibility reproduces 13 domains from metadata alone.
- Adaptation, paired-calibration, few-shot, and evaluation masters are disjoint according to regime.
- All excluded rows remain in the registry with reason codes.

### P03 acceptance

- Every classical candidate completes nested selection on the same folds.
- Population-fitted transformations occur only inside training data.
- Cross-fitted calibration contains no outer-test prediction.
- Master permutation approaches chance and metadata-only confounding is reported.
- Row, instrument-view, master, class, fold, station, and domain tables reconcile exactly.

### P04 acceptance

- The implemented architecture matches the tensor and parameter contract.
- Histories, checkpoints, selected epochs, failures, runtimes, and resource usage exist for every run ID.
- No test metric selects architecture, optimizer, augmentation, or epoch.
- Collapse and numerical-failure rates satisfy or transparently fail G2.

### P05 acceptance

- Positive/negative pair construction passes unit tests, including the prohibition on same-target negatives.
- Pair weighting and loss values are recorded per batch or reproducibly reconstructable.
- D1–D5 are ranked only on source pseudo-domains.
- The advancement decision and all failed candidates remain archived.
- Exactly one advancing acquisition-aware configuration is frozen, or the declared no-advance outcome is recorded.

### P06 acceptance

- Classical and deep predictions use identical test UIDs and aggregation rules.
- All 13 primary and four exploratory domains are present.
- Neural seeds are averaged before domain inference.
- Single-spectrum and multi-view master results are labelled separately.
- The primary paired effect table can be recomputed entirely from saved row predictions.

### P07 acceptance

- Every result is visibly labelled zero-shot, UDA, paired calibration, or supervised few-shot.
- Calibration/adaptation and evaluation masters are disjoint.
- Learning-curve x axes count physical masters per class, not spectra.
- Target-information access is summarized beside every metric.

### P08 acceptance

- Every perturbation is generated from the frozen unperturbed test row.
- No perturbed result changes a model or hyperparameter.
- Degradation curves include zero perturbation and reconcile with primary metrics.
- Preprocessing and tier branches report domains lost through insufficient support.

### P09 acceptance

- All eight held station/chemical tasks are present.
- Held unknowns never enter fitting, score selection, calibration, or thresholding.
- Thresholds reproduce known-development coverage rules.
- AUROC, AUPR, OSCR, FPR95, and risk–coverage are shown per held task.

### P10 acceptance

- Domain probes include chemistry-only nulls.
- Pair agreement is conditioned on correctness.
- Retrieval gives same-master and same-target results separately.
- Attribution changes under randomized-head and randomized-encoder controls.
- Error cases are selected by declared ranking rules rather than subjective preference.

### P11 acceptance

- Bootstrap sampling preserves method pairing and master grouping.
- Domain-weighted and row-weighted summaries are never confused.
- Leave-one-domain and leave-one-instrument-identity sensitivities are complete.
- G4 and G5 inputs are machine-generated from frozen evidence.
- Route A, B, or C is selected without informal exception.

### P12 acceptance

- Every completed figure has one frozen plot table, native TikZ, compiled vector PDF, and standalone HTML.
- TikZ and HTML point estimates, intervals, axes, and labels match.
- Accessibility and final-size reviews pass.
- Manuscript tables reconcile with machine metrics.
- Clean rebuild validation and artifact hashes pass.

## 28. Intended manuscript evidence package

The eventual main manuscript should remain compact even though the supplement is comprehensive.

### Main-text evidence

1. Dataset/design figure: observation flow, factor support, and T3 split logic.
2. Preprocessing evidence: why minimal min–max is primary and what nuisance structure remains.
3. Classical versus deep T1/T3 performance across all 13 domains.
4. Paired primary effect forest with hierarchical interval and worst-domain behavior.
5. Mechanistic evidence: paired-view consistency plus target-adjusted domain probes.
6. Calibration/operating figure or robustness figure, selected by the declared publication route.

### Supplementary evidence

- complete raw/native-axis and quality audit;
- every classical hyperparameter/selection outcome;
- every deep learning curve and failure;
- preprocessing, range, quality-tier, and perturbation branches;
- cross-station transfer and adaptation regimes;
- complete open-set task results;
- confusion matrices, per-class recall, and error taxonomy;
- attribution sanity checks;
- run, environment, compute, validation, and hash registries.

### Abstract-level claim template

The abstract must state the number of physical masters, instruments, eligible domains, split independence, and whether the primary deep promotion gate passed. It reports the paired effect and uncertainty rather than only the winning score. If Route B or C is selected, the negative result is stated directly; it is not buried behind the best isolated task or master-aggregated number.
