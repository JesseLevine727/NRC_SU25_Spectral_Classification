# NATO field-trial SERS acquisition-shift protocol v1

Status: prespecified research protocol derived from the fresh raw-data audit. It replaces earlier assumptions about the 598-row dataset, preprocessing, and VAE suitability.

## 1. Research question

Can chemical identity be recovered from heterogeneous field-trial SERS spectra when the acquisition instrument is not represented in training, without relying on station identity, duplicated physical samples, or destructive universal preprocessing?

The study is not framed as denoising. It evaluates chemical classification and representation robustness under observed acquisition shifts.

## 2. Data and independent unit

The authoritative source is `/home/elfo/Documents/NRC/2026July21/NATO SERS Data`. The project mirror is a duplicate, not additional data.

The independent experimental unit is the physical `master_sample_id`, not a spectrum row. All spectra belonging to a test master remain absent from training and validation. Source spectra referenced by multiple log rows are represented once in the unique tier.

### Preserved tiers

1. `tier_all_parseable_sers` (626): named SERS, matched numeric source, numeric QC pass.
2. `tier_attributable_sers` (615): additionally maps to a target analyte.
3. `tier_unique_attributable_sers` (598): additionally resolves reused/conflicting source references. This is the primary tier.
4. `tier_notes_clear_sers` (500): additionally excludes severe-quality and low-signal/noise notes. This is a sensitivity tier, not a preferred cleaned dataset.
5. Mira-1 system-suitability sensitivity: exclude all 23 Mira-1 spectra whose vendor reports state that the last system-suitability test failed. Do not remove these silently from the primary tier.

## 3. Task hierarchy

### T0 — aggregate seven-class continuity baseline

Seven targets across all 69 masters and three stations. Report only as a descriptive continuity baseline because station alone obtains balanced accuracy 0.429 versus seven-class chance 0.143.

### T1 — primary chemistry-identifiable, within-station tasks

Run three independent three-class tasks:

| Station | Targets | Spectra | Masters |
|---|---|---:|---:|
| CWA | 4-nitrophenol, ethanol, ethyl paraoxon | 208 | 24 |
| Pills | 4-ANPP, benzyl fentanyl, blank | 208 | 20 |
| Surfaces | 4-ANPP, acetaminophen, benzyl fentanyl | 182 | 25 |

Acquisition metadata alone is approximately chance within these tasks. These are the primary closed-domain chemistry evaluations.

### T2 — cross-station binary transfer

Use only 4-ANPP and benzyl fentanyl, the two targets observed at both pills and surfaces. Train on one station and test on the other, then reverse. The 26 physical masters are station-disjoint. No target-station shortcut can solve this binary task.

### T3 — primary unseen-instrument evaluation

Within each T1 station, hold out one instrument entirely. In each outer master fold:

1. Training contains only training masters and excludes every spectrum from the held instrument.
2. Testing contains only held-out masters measured by the held instrument.
3. Validation is drawn only from the training masters and training instruments.

The primary domain summary includes only instrument/station cells containing all three targets and at least 15 test masters. Smaller cells remain visible as explicitly low-support exploratory results. This currently yields 13 adequately supported instrument/station domains.

### T4 — secondary unknown rejection

Within each station, hold one of three chemicals out as unknown and train on the other two. This is a stress test only. With 4–11 masters per unknown class, it cannot support a broad open-world claim. Aggregate seven-class leave-one-chemical-out is prohibited because chemical identity and station are coupled.

## 4. Spectral support and preprocessing

### Primary protocol

1. Load the readable vendor spectrum and retain its native axis for audit.
2. Confirm finite intensity, increasing axis, and nonconstant signal.
3. Interpolate linearly, without extrapolation, onto 400–1,800 cm⁻¹ at 1 cm⁻¹.
4. Scale each spectrum independently to `[0,1]` using its own minimum and maximum.
5. Apply no blanket baseline subtraction, smoothing, derivative, or per-spectrum peak alignment.

The lower bound is 400 because Mira begins at 400. An all-instrument 100 cm⁻¹ lower bound would create unmeasured values. The upper bound 1,800 avoids the terminal Pendar edge. The empirically complete 400–1,849 interval is a required sensitivity branch.

### Required preprocessing ablations

- Conservative isolated-impulse replacement + 11-point, third-order Savitzky–Golay smoothing + min–max.
- arPLS baseline correction + min–max.
- SNV, vector, and non-negative integrated-area normalization.
- First Savitzky–Golay derivative + SNV as a negative/control branch.
- 400–1,849, 450–1,800, and 500–1,800 range sensitivities.

Minimal min–max is primary because smoothing improvements are small and task-dependent, while baseline/derivative methods measurably alter peaks. The smoothed candidate is expected to be competitive and must be reported, not silently selected after viewing a test result.

### Alignment decision

No per-spectrum correlation alignment is allowed in the primary protocol. Cross-instrument repeated-master pairs have median absolute estimated lag 2 cm⁻¹ but a 90th percentile of 7 cm⁻¹, broad signed distributions, and weakly stable pair-specific directions. These estimates mix calibration, resolution, background, and SERS hotspot variation. Shifting each test spectrum to maximize similarity could manufacture agreement and leak a reference choice.

Peak-shift augmentation of ±2 cm⁻¹ may be used during neural training, with ±5 cm⁻¹ as a robustness sensitivity. A fixed instrument correction becomes permissible only if learned from training-fold calibration standards or other training-only references and applied unchanged to validation/test spectra.

## 5. Splitting and model selection

- Freeze outer master partitions before model development.
- T1 uses 4-fold stratified group cross-validation repeated five times; four folds are required because the smallest station/class cell has four masters.
- T2 uses directional station transfer. Hyperparameters are chosen using master-grouped validation inside the training station only.
- T3 uses the frozen T1 master folds plus complete instrument exclusion described above.
- Any transform fitted from multiple spectra—channel standardization, PCA, learned baseline, feature selection, calibration, or domain transformation—is fitted inside the training fold.
- Per-spectrum min–max and deterministic native-to-common interpolation are row-local and require no population fitting.
- Model/preprocessing selection uses inner validation or fixed prespecified settings. Test-domain results cannot choose preprocessing or hyperparameters.

## 6. Baselines and proposed methods

### Mandatory baselines

- Stratified prior/dummy.
- Metadata-only logistic regression.
- Nearest-centroid or correlation spectral matching.
- PCA-LDA or PLS-DA conventional chemometric baseline.
- RBF SVM.
- Random Forest and Extra Trees.
- Location-preserving bounded 1D CNN with inner-master early stopping.

### Proposed acquisition-aware methods

1. **Multi-positive supervised contrastive encoder.** Positives share the target within the same station; cross-instrument and cross-sensor positives receive additional weight. Spectra of the same target but different masters must never be treated as negatives.
2. **Paired-view consistency loss.** When the same master is observed across instruments, encourage prediction/embedding agreement while retaining the supervised target loss.
3. **Domain-adversarial control.** Predict chemistry while discouraging a training-domain head from recovering instrument/sensor identity. Domain confusion alone is never an optimization endpoint.
4. **CORAL or feature-moment alignment control.** Training-domain-only alignment provides a low-complexity comparison with adversarial learning.

The proposed method succeeds only if chemistry performance improves or remains stable in T3 while instrument/sensor predictability decreases. Erasing instrument information at the expense of chemistry is a failure.

## 7. Endpoints and statistics

### Primary endpoint

Unweighted mean spectrum-level balanced accuracy across adequately supported T3 instrument/station domains. Every eligible domain contributes equally regardless of row count.

### Secondary endpoints

- Macro-F1 and per-class sensitivity for T1–T3.
- T2 balanced accuracy, AUROC, log loss, and Brier score.
- Expected calibration error and reliability plots where probabilities are available.
- Domain-level worst case, median, and interquartile range.
- Linear-probe instrument and sensor predictability from learned embeddings.
- Cross-instrument same-master retrieval recall and cosine similarity.
- Open-set AUROC, AUPR, false-positive rate at 95% known recall, and OSCR for T4.

Uncertainty is obtained by bootstrapping physical masters, not rows, with at least 2,000 stratified bootstrap replicates. Paired method differences reuse identical outer partitions and master bootstrap samples. Report effect sizes and 95% intervals; do not interpret overlapping fold rows as independent observations.

### Multi-view master result

Master-level performance is secondary. Average logits within each instrument first, then average instruments so an instrument with more repeated scans cannot dominate. Report how many instruments and spectra were available per master. Never label this as single-spectrum performance.

## 8. Required ablations and controls

- Primary versus notes-clear versus Mira-1-excluded tiers.
- Minimal versus smoothed versus arPLS preprocessing.
- 400–1,800 versus 400–1,849 support.
- Single-spectrum versus instrument-balanced master aggregation.
- Ordinary master-grouped versus unseen-instrument + unseen-master validation.
- Within-station versus confounded aggregate seven-class task.
- Contrastive loss, paired consistency, domain adversary, and augmentation removed one at a time.
- Instrument/sensor/target probes on embeddings.
- Peak-shift, baseline-slope, intensity-scale, and noise stress tests applied to test spectra without retraining.

## 9. Permitted and prohibited claims

### Permitted if supported

- The method improves chemical identification under the tested instruments/sensors and field-trial task definitions.
- Paired multi-instrument views improve representation or prediction consistency.
- Particular preprocessing choices preserve or damage measured peak structure differently by instrument.
- The method rejects held-out classes under the narrow T4 protocol.

### Prohibited with this archive alone

- The model denoises spectra in the signal-recovery sense.
- A VAE or any model separates “true chemical signal” from substrate/instrument nuisance.
- Latent dimensions correspond to causal chemical and nuisance factors.
- Performance generalizes to arbitrary chemicals, concentrations, substrates, instruments, or field deployments.
- Master-aggregated performance is single-spectrum performance.
- Aggregate seven-class accuracy is chemistry-only evidence.
- A held-out chemical from a held-out station is valid open-set chemical recognition.

## 10. Additional data that would unlock stronger claims

- Pure/reference Raman and SERS spectra for every analyte under controlled conditions.
- Paired clean/corrupted or high-SNR/low-SNR measurements for denoising validation.
- A factorial design crossing every analyte with station, instrument, sensor, concentration, and day.
- Calibration-standard spectra from every instrument for fixed wavenumber/response transfer.
- Multiple independently prepared substrates and batches, not merely repeated scans.
- More physical masters per class and truly novel external chemicals for open-set evaluation.
- An external trial collected on later dates and unseen operators.

## 11. Decision rule

The primary paper proceeds as an acquisition-shift study if at least one acquisition-aware method improves the T3 primary endpoint over the strongest classical and bounded-CNN baselines without materially degrading T1 chemistry performance or peak-preservation diagnostics. If not, publishable value remains in the negative result and rigorous benchmark, with the classical within-station classifier as the fallback deliverable.
