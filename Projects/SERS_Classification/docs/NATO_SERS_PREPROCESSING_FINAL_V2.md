# Final NATO SERS preprocessing protocol v2

Date frozen: 2026-07-23  
Dataset version: `nato-sers-preprocessing-v2`  
Status: **closed for downstream model comparison**

## 1. Final decision

The NATO SERS preprocessing exploration is complete. The three frozen model
representations are:

1. `minimal_minmax`: conservative despiking followed by independent
   per-spectrum min--max scaling;
2. `arpls_minmax`: the same despiking, one domain-blind arPLS baseline
   estimate, baseline subtraction, and independent per-spectrum min--max
   scaling;
3. `derivative_1`: row SNV, Savitzky--Golay first derivative, and row-L2
   normalization, retained unchanged as the poster/Siamese discriminative
   control.

The final decisions are:

- no general smoothing;
- no noise-gated or instrument-specific smoothing;
- no additional spectral alignment;
- no target-informed or flexible spectral warping;
- no further preprocessing selection after v2.

This is deliberately a multi-view dataset. `minimal_minmax` preserves
background and broad structure; `arpls_minmax` tests a common baseline-removed
view; `derivative_1` tests the prior discriminative representation. The VAE
experiments should determine which information is chemical and which is
nuisance rather than pretending that one preprocessing operation has already
solved that problem.

The closed bundle is
[`Workspace/nato_sers_field_trial/preprocessing_v2`](../Workspace/nato_sers_field_trial/preprocessing_v2/README.md).

## 2. Dataset contract

| Cohort | Spectra | Role |
|---|---:|---|
| Strict SERS core | 598 | Primary labelled dataset |
| Quality pass | 500 | Prespecified sensitivity analysis |
| Field-quality stress | 98 | Confirmatory stress test only |

The 500 and 98 rows are disjoint and exhaust the 598-row core. All three
cohorts retain the same observation IDs and master-sample assignments. The
stress cohort was not used to select preprocessing.

All spectra use the empirically supported common interval:

- minimum: 400 cm⁻¹;
- maximum: 1800 cm⁻¹;
- spacing: 1 cm⁻¹;
- length: 1,401 values.

The 400 cm⁻¹ lower bound is the native lower limit of Mira and remains the
only non-fabricated lower bound shared by every included system. Extending to
100 cm⁻¹ would require extrapolated values for every Mira scan, values below
Pendar's approximately 275 cm⁻¹ limit, and values below Agilent's trustworthy
approximately 350 cm⁻¹ region. Such padding would encode instrument identity.

The core includes every attributable, valid SERS spectrum available from the
ten represented instruments. The reduction from 721 named-SERS log rows to
598 spectra is an attribution and validity funnel, not a quality filter:

| Sequential rule | Remaining |
|---|---:|
| Explicitly named SERS sensor | 721 |
| Readable source spectrum matched | 626 |
| Master sample and target resolved | 615 |
| Contradictory source reuse removed | 599 |
| Consistent repeated reference deduplicated | 598 |
| Numeric spectrum valid | 598 |

Literal `na` sensor entries do not identify a SERS sensor and remain outside
this dataset. The 95 named-SERS rows without source spectra and ambiguous or
contradictory rows cannot be recovered by preprocessing.

## 3. Exact final preprocessing

The operation is identical for every instrument. No transform reads the
instrument, sensor, target, station, or quality label.

### 3.1 Common source layer

1. Parse the native vendor spectrum and its own calibrated wavenumber axis.
2. Restrict to measured support.
3. Interpolate to 400--1800 cm⁻¹ at 1 cm⁻¹.
4. Preserve this as `raw_common_grid`; never overwrite it.

No extra alignment is performed after common-grid interpolation.

### 3.2 Artifact annotations

On each common-grid spectrum, an isolated-point candidate is detected when:

- prominence is at least 10% of that row's full intensity span; and
- half-prominence width is no more than 1.25 grid points.

Only detected points in the derived branch are linearly reconstructed from
unflagged neighbours. The raw row and `spike_mask` remain available. This is
a conservative transient detector, not a claim that every flagged value is a
cosmic ray.

A maximum-level run of at least three points, within `1e-6` of the row span,
is marked as possible saturation. Saturation is recorded, not automatically
repaired.

### 3.3 Reconstructive view A: `minimal_minmax`

For despiked row \(x\):

```text
minimal_minmax = (x - min(x)) / (max(x) - min(x))
```

Scaling is fitted independently to each spectrum over the full 400--1800
cm⁻¹ interval. Every nonconstant row is therefore exactly on `[0,1]`. This
removes absolute offset and scale but intentionally retains baseline shape,
resolution, peak-width response, and noise texture.

### 3.4 Reconstructive view B: `arpls_minmax`

1. Estimate an arPLS baseline from the despiked row with:
   `lambda=1e6`, at most 30 iterations, and relative weight tolerance `1e-3`.
2. Subtract that estimated baseline.
3. Apply independent row min--max scaling to `[0,1]`.

The same arPLS parameters apply to all instruments. The fitted baseline is
stored separately. This branch is not declared more physically correct than
the minimal branch: the archive showed that baseline algorithms can treat
broad RMX structure differently, so the two views must remain separate.

### 3.5 Discriminative control: `derivative_1`

1. Apply row standard-normal-variate normalization.
2. Apply a Savitzky--Golay derivative with window 17, polynomial order 3, and
   derivative order 1.
3. Normalize each derivative row to unit L2 norm.

This branch is signed and is not min--max scaled. Mapping it to `[0,1]` would
change the physical meaning of zero slope and would no longer reproduce the
poster/Siamese control. It is intended for a classifier, encoder control,
auxiliary channel, or derivative-domain loss rather than a sigmoid
reconstruction target.

## 4. Why the spectra are on a common scale but not forced into one view

Per-spectrum min--max scaling answers the numerical question: each
reconstructive spectrum has the same `[0,1]` range. It avoids Mira's native
intensity scale dominating systems whose values are orders of magnitude
smaller.

It does not make the measurement processes identical. After scaling,
instrument and sensor remain partly predictable because the spectra still
contain:

- vendor background removal or retention;
- fluorescence and baseline curvature;
- spectral resolution and peak width;
- structured and high-frequency noise;
- wavelength calibration differences;
- sensor-dependent enhancement and peak visibility;
- target/instrument/sensor imbalance in the field design.

Trying to erase all of these effects during preprocessing would risk deleting
real analyte--surface physics. The downstream disentangled VAE is specifically
meant to model this chemical-versus-nuisance distinction.

## 5. How the bounded smoothing study was determined

The candidate study was declared before classifier screening:

- method: Savitzky--Golay smoothing, polynomial order 3;
- windows: 7, 11, and 15 points, equal to 7, 11, and 15 cm⁻¹;
- families: the minimal and arPLS residual branches;
- order: smoothing after despiking, and after baseline subtraction for arPLS,
  but before row min--max scaling;
- unchanged control: `derivative_1`;
- headline processing had to remain domain-blind;
- the outer test and flagged stress cohort could not select a method.

The windows were intentionally bounded. They cover mild to moderate
high-frequency smoothing without creating an open-ended parameter search on
only 598 spectra.

### 5.1 Preservation gates

A smoother had to satisfy every gate:

| Gate | Required |
|---|---:|
| Repeatable-peak weighted recall | ≥ 0.98 |
| Median matched-peak displacement | ≤ 1 cm⁻¹ |
| Median absolute relative peak-width change | ≤ 0.15 |
| Median absolute peak-prominence change | ≤ 0.10 |
| Median row correlation to unsmoothed family | ≥ 0.98 |
| Clean target balanced-accuracy loss, core and quality | ≤ 0.01 |
| Target-adjusted instrument leakage increase | ≤ 0.02 |
| Target-adjusted sensor leakage increase | ≤ 0.02 |
| Same-master cross-instrument distance increase | ≤ 0.02 |

It also had to show at least one benefit:

- Gaussian-noise target balanced accuracy improved by at least 0.01;
- noisy-versus-clean prediction agreement improved by at least 0.02; or
- the target-blind robust second-difference noise score fell by at least 10%.

### 5.2 Peak-definition amendment

An implementation sanity check initially counted every detected local maximum
as a feature that smoothing had to preserve. Dense high-frequency Pendar
maxima were consequently treated as chemical peaks, which contradicts the
purpose of a denoising test.

Before any target classifier, domain probe, corruption classifier, outer-test,
or flagged-stress result was calculated, the peak rule was amended:

- all-detected-peak recall remains a diagnostic, not a hard gate;
- a repeatable reference peak must have prominence at least 0.15;
- another observation of the same master sample must support it within
  3 cm⁻¹;
- at least half of the other same-master observations must support it; and
- support must include a different instrument.

The amendment and its timing are recorded in
[`predeclared_protocol.json`](../Workspace/nato_sers_field_trial/preprocessing_v2/predeclared_protocol.json).
It prevents a denoiser from being rewarded for preserving nonrepeatable
high-frequency maxima.

## 6. Leakage-safe selection

The frozen five folds group by `master_sample_id`. All observations of one
physical master sample remain in one fold, even when measured with different
instruments or sensors.

For each candidate and for both the 598-row core and 500-row quality subset:

1. one fold was sealed as outer test;
2. each of the other four folds was used once as inner validation;
3. PCA, logistic regression, centroids, and domain probes were fitted only on
   training rows;
4. preprocessing gates and selection used only the 20 inner-validation
   results per candidate and subset;
5. the five outer results were retained as confirmation only.

The 98-row stress evaluation trained on quality-pass development rows and
tested flagged rows from the held-out master-sample fold. It was also
confirmatory only.

There are nine representations. The audit contains:

```text
9 candidates × 2 subsets × (20 inner + 5 outer) = 450 rows
9 candidates × 5 flagged-stress outer folds       = 45 rows
total                                              = 495 rows
```

The screening classifier is class-balanced logistic regression after up to
32 whitened PCA components. A nearest-centroid score checks representation
geometry. Instrument and sensor leakage are reported relative to target-only
null probes and balanced over observed target × domain cells. Same-master,
cross-instrument distance provides complementary paired evidence.

Deterministic corruptions are independently generated per observation ID:
scale/offset, curved baseline, Gaussian noise at 3% of row span, two isolated
spikes at 80% of row span, ±3 cm⁻¹ shift, and their composite.

## 7. Smoothing results

The values below are nested-inner means except the clearly labelled stress
column. Deltas compare a smoother with its unsmoothed family.

| Candidate | Clean target | Quality target | Noise-target Δ | Noise-agreement Δ | Repeatable-peak recall | HF reduction | Flagged stress, confirmatory | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `minimal_minmax` | 0.697 | 0.739 | -- | -- | 1.000 | 0.000 | 0.443 | retained control |
| `minimal_sg7_minmax` | 0.695 | 0.741 | -0.003 | +0.000 | 0.971 | 0.325 | 0.480 | no |
| `minimal_sg11_minmax` | 0.693 | 0.744 | -0.002 | -0.010 | 0.903 | 0.627 | 0.468 | no |
| `minimal_sg15_minmax` | 0.691 | 0.745 | -0.005 | -0.003 | 0.811 | 0.746 | 0.424 | no |
| `arpls_minmax` | 0.707 | 0.742 | -- | -- | 1.000 | 0.000 | 0.407 | retained control |
| `arpls_sg7_minmax` | 0.705 | 0.748 | -0.003 | -0.005 | 0.961 | 0.319 | 0.438 | no |
| `arpls_sg11_minmax` | 0.709 | 0.753 | +0.001 | -0.008 | 0.863 | 0.616 | 0.478 | no |
| `arpls_sg15_minmax` | 0.707 | 0.754 | -0.009 | -0.006 | 0.758 | 0.734 | 0.468 | no |
| `derivative_1` | 0.701 | 0.739 | -- | -- | -- | -- | 0.339 | retained control |

Every smoother reduced the high-frequency score, but none improved noisy
target accuracy by 0.01 or prediction agreement by 0.02. More importantly,
every smoother failed the 0.98 repeatable-peak recall gate. The 15-point
arPLS smoother also exceeded the peak-displacement limit.

Some smoothing candidates improved accuracy on the flagged cohort, especially
the 11-point arPLS candidate. That result cannot overturn the decision:
the cohort was declared confirmatory, and the candidate failed chemical
feature preservation and both classifier-based synthetic-noise benefit gates.
Selecting it from the stress result would be leakage.

See:

- [smoothing selection plot](../Workspace/nato_sers_field_trial/preprocessing_v2/figures/smoothing_selection.png);
- [instrument preservation plot](../Workspace/nato_sers_field_trial/preprocessing_v2/figures/smoothing_instrument_preservation.png);
- [complete objective table](../Workspace/nato_sers_field_trial/preprocessing_v2/smoothing_selection_objectives.csv).

## 8. Effect of smoothing by instrument

The mildest seven-point smoother already fell below the global preservation
gate:

| Instrument | Minimal SG7 recall | arPLS SG7 recall |
|---|---:|---:|
| Agilent-1 | 0.981 | 0.976 |
| Agilent-3 | 1.000 | 1.000 |
| Mira-1 | 1.000 | 0.995 |
| Mira-2 | 0.995 | 0.997 |
| Mira-3 | 1.000 | 0.994 |
| Pendar-1 | 0.977 | 0.964 |
| Pendar-2 | 0.959 | 0.967 |
| Pendar-3 | 0.962 | 0.966 |
| RMX-1 | 0.995 | 1.000 |
| RMX-2 | 0.957 | 0.915 |

Longer windows increasingly removed repeatable structure from Pendar and
RMX-2. At 15 points, repeatable-peak recall reached 0.702/0.680 for Pendar-2
and 0.781/0.631 for RMX-2 in the minimal/arPLS families. Mira and RMX-1 were
often less affected. A global window therefore does not preserve the systems
equally.

This is why visual noisiness alone is not a sufficient smoothing criterion.
The instruments have different resolution, background handling, and peak
width. A window that looks benign on Mira can erase narrow repeatable features
from Pendar or RMX-2.

## 9. Why conditional smoothing was rejected

The predeclared trigger to consider one label-blind noise gate fired because
smoothing strongly reduced the target-blind high-frequency score while
failing preservation inconsistently across systems.

The policy was considered but not implemented:

- no uniform smoother preserved enough repeatable peaks;
- no uniform smoother met either classifier-based synthetic-noise benefit;
- the flagged cohort could not tune a gate;
- a fold-fitted threshold would produce different input spectra in different
  train/test runs rather than one immutable dataset;
- a threshold fitted on all 598 spectra would expose outer-test distribution
  information;
- the high-frequency score can measure native instrument resolution as well
  as unwanted noise.

Instrument-specific smoothing would be possible as a separate known-device
deployment study, but it is not apples-to-apples evidence for generalization
to a new instrument and is not part of the headline dataset.

For denoising, the correct next move is training-time corruption in a
denoising AE/VAE while reconstructing the frozen clean representation. That
tests an explicit noise-removal objective without permanently erasing features
from every stored spectrum.

## 10. Alignment audit and decision

The archive contains 17 named standard/calibration spectra, but they cover
only five instruments:

| Instrument | Files | Anchor range near 1000 cm⁻¹ |
|---|---:|---:|
| Mira-1 | 3 | 1003 |
| Mira-2 | 3 | 1004 |
| Mira-3 | 5 | 1003 |
| Pendar-2 | 4 | 1002--1003 |
| Pendar-3 | 2 | 1002--1003 |

No comparable named standard was available for Agilent-1, Agilent-3,
Pendar-1, RMX-1, or RMX-2. A correction for only half the instruments would
be partial and system-aware.

As a secondary diagnostic, 2,473 same-master cross-instrument pairs were
cross-correlated in the first-derivative representation over integer lags
from -5 to +5 cm⁻¹. Only 6 of 31 observed instrument-pair summaries passed all
the count, class, concentration, IQR, shift-magnitude, and quality-direction
gates. More fundamentally, same-master spectra still confound calibration
with sensor response, peak visibility, chemistry, and resolution. They cannot
replace a shared standard.

The alignment operation is therefore `none`: retain the native calibrated
axis and the frozen common-grid interpolation. Flexible per-spectrum or
target-informed warping remains prohibited.

See:

- [alignment evidence plot](../Workspace/nato_sers_field_trial/preprocessing_v2/figures/alignment_evidence.png);
- [standard evidence](../Workspace/nato_sers_field_trial/preprocessing_v2/alignment_standard_evidence.csv);
- [pair evidence](../Workspace/nato_sers_field_trial/preprocessing_v2/alignment_pairwise_evidence.csv);
- [machine-readable decision](../Workspace/nato_sers_field_trial/preprocessing_v2/alignment_decision.json).

## 11. Downstream use

Use
[`final_model_inputs_core.npz`](../Workspace/nato_sers_field_trial/preprocessing_v2/final_model_inputs_core.npz)
for the primary model comparison. It contains:

```text
axis_cm1
observation_uid
minimal_minmax
arpls_minmax
derivative_1
```

Use the corresponding quality and field-stress files only for their declared
roles. Join metadata by `observation_uid`; never assume an independently
sorted manifest has the same row order without checking the IDs.

Recommended first model sequence:

1. classifier and existing Siamese controls on all three representations;
2. deterministic AE on `minimal_minmax` and `arpls_minmax`;
3. denoising AE using synthetic corruptions as inputs and the frozen clean
   row as target;
4. standard VAE on each reconstructive view;
5. chemical/nuisance disentangled VAE with reconstruction, chemical
   classification, nuisance/domain supervision or conditioning, and
   cross-domain metric loss;
6. latent domain probes, same-master geometry, latent swaps, reconstruction
   fidelity, and held-out supported-domain evaluation.

Model and hyperparameter selection must continue to use the frozen
master-sample-grouped nested splits. Preprocessing must not be retuned after
seeing downstream outer-test results.

## 12. Auditability and reproduction

The bundle preserves:

- common-grid raw and despiked spectra;
- spike and saturation masks;
- arPLS baseline estimates;
- every evaluated smoothing candidate;
- the three selected final inputs;
- observation and cohort manifests;
- nested grouped splits and domain partitions;
- all 495 benchmark rows;
- spectrum- and instrument-level preservation evidence;
- alignment standards and same-master pair evidence;
- the predeclared protocol and amendment;
- v1 control hashes, v2 artifact hashes, software versions, and input hashes.

Rebuild and validate from the repository root:

```bash
.venv/bin/python scripts/finalize_nato_sers_preprocessing_v2.py
.venv/bin/python scripts/validate_nato_sers_preprocessing_v2.py
```

The validator independently checks cohort membership, axes, normalization,
finite values, exact v1 controls, exact SG calculations, selected archives,
nested split grouping, benchmark structure, smoothing and alignment
decisions, input hashes, and every artifact hash.

An independent clean rebuild was also completed on 2026-07-23 in a separate
temporary directory. It passed the validator and matched the canonical bundle
as follows:

- identical 33-file inventory, including the hash catalog;
- all 6 NPZ archives semantically exact;
- all 63 NPZ arrays identical in key, dtype, shape, and value;
- all 25 deterministic non-container files byte-for-byte identical;
- version records identical except for the expected `created_utc` timestamp.

Compressed NPZ container bytes and their catalogued hashes are permitted to
differ because ZIP member timestamps can change even when every stored array
is exact. Reproducibility is therefore asserted on array content, while each
individual bundle remains protected by its own SHA-256 catalog.

## 13. Closure rule

Preprocessing is now frozen. Downstream model failure is not by itself
permission to search more smoothing windows, alignment rules, baseline
parameters, or instrument-specific transforms. Reopening preprocessing
requires a new version, a concrete failure mechanism, new predeclared gates,
and a new nested study. Version v2 remains immutable as the comparison
baseline.
