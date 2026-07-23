# NATO SERS preprocessing baseline v1

## Decision

The leakage-safe preprocessing baseline is frozen as
`nato-sers-preprocessing-v1`. The downstream AE, denoising-AE, VAE,
disentangled-VAE, and Siamese-hybrid experiments must begin with the same
observation IDs, axis, master-sample folds, and representations in
[`preprocessing_v1`](../Workspace/nato_sers_field_trial/preprocessing_v1/README.md).

The three authorized model inputs are:

1. `minimal_minmax`: conservative despiking followed by independent
   per-spectrum min--max scaling;
2. `arpls_minmax`: the same despiking, one domain-blind arPLS baseline rule,
   and independent per-spectrum min--max scaling;
3. `derivative_1`: the prior-work SNV + Savitzky--Golay first derivative +
   row-L2 representation, retained as a discriminative control.

`minimal_minmax` and `arpls_minmax` are the reconstructive inputs. Every row
in either representation is exactly on `[0,1]`. `derivative_1` is signed and
has unit L2 norm by design; forcing it onto `[0,1]` would destroy the meaning
of zero slope and would no longer reproduce the poster/Siamese preprocessing.
It should initially be used as a classifier, encoder-control, auxiliary
channel, or derivative loss rather than the sole sigmoid-decoder target.

This freeze does **not** claim that preprocessing has removed instrument or
sensor information. It has not. The remaining predictability is the reason
to test representation learning and explicit chemical/nuisance separation.

## Why the labelled dataset contains 598 spectra rather than 721

The number 721 counts field-log observations with an explicitly named SERS
sensor. It is not the number of usable, uniquely labelled spectral files.
The deterministic inclusion funnel is:

| Sequential rule | Remaining |
|---|---:|
| Explicitly named SERS sensor | 721 |
| Readable source spectrum matched | 626 |
| Master sample and target resolved | 615 |
| Contradictory reuse of a source scan removed | 599 |
| One primary row retained per consistent repeated reference | 598 |
| Numeric spectrum valid | 598 |

The 95 unmatched entries have no recoverable spectrum in the supplied
archive: 56 PMCDS measurements, one Agilent-2 entry, one unrecognized system,
31 RMX-1 rows without a logged filename, four RMX-2 rows without a filename,
and two Mira-3 rows without a filename. Nineteen named-SERS rows participate
in contradictory source assignments and are excluded rather than assigned a
guessed target. One remaining consistent repeated reference is deduplicated.

All ten instruments for which valid, attributable SERS spectra exist are in
the 598-row core. Literal `na` in the sensor field means that no SERS sensor
was identified; those observations are normal Raman and are intentionally
outside this SERS dataset.

There are 619 distinct readable SERS source spectra in total. The additional
21 may later be used in an explicitly unlabelled reconstruction experiment,
but their target or source assignment is not reliable enough for supervised
selection or evaluation.

The 500-row quality subset is not a different experiment assembled by
cherry-picking model results. The 598-row core contains 88 spectra with severe
field-note flags and 23 with low-signal/noise flags, with 13 spectra in both
groups. Removing their union leaves `598 - (88 + 23 - 13) = 500`. The core is
the primary labelled dataset; the quality subset is a prespecified sensitivity
analysis.

## Why the common range begins at 400 cm⁻¹

The lower bound was determined from the native axes, not from a generic Raman
convention:

- Mira begins at exactly 400 cm⁻¹;
- Pendar begins near 275 cm⁻¹;
- the trustworthy Agilent region begins near 350 cm⁻¹;
- RMX varies slightly scan by scan and is interpolated using its own calibrated
  axis.

Using 100 cm⁻¹ would therefore require fabricated values for every Mira
spectrum, values below 275 cm⁻¹ for Pendar, and questionable values below
about 350 cm⁻¹ for Agilent. Padding or edge extrapolation would create a nearly
perfect instrument marker. The literal upper-axis intersection reaches only
about 1849 cm⁻¹ because of Pendar, so 1800 cm⁻¹ avoids edge behavior. The
canonical NATO axis is therefore 400--1800 cm⁻¹ in 1 cm⁻¹ steps: 1,401 values
per spectrum.

The poster data can retain its own 330--1800 cm⁻¹ range when evaluated alone.
If NATO and poster spectra ever enter one shared encoder, their common input
must use 400--1800 cm⁻¹ or explicitly model missing regions; it must not invent
the NATO low-frequency values.

## How instrument characteristics were determined

The system characterization came from four independent sources:

1. parsing every vendor format and recording its native wavenumber axis,
   length, intensity range, and duplicate/reference behavior;
2. reading the field workbooks and notes, including severe and low-signal
   annotations;
3. plotting representative spectra and lower-envelope/baseline estimates by
   instrument;
4. measuring grouped target and instrument predictability after candidate
   transforms.

This established the following working interpretation:

| System | Observed behavior | Processing implication |
|---|---|---|
| Mira | Large curved fluorescence/background; raw levels around 10,000--19,000 | Corrected and uncorrected branches are both necessary |
| Agilent | Vendor SORS export is comparatively peak-like and close to a low envelope; raw levels around 138--142 | Avoid assuming another aggressive correction is required |
| Pendar | Peak-like, with frequent field-quality/noise flags; raw levels around 12--50 | Spike/noise sensitivity matters more than visual flattening |
| RMX | Vendor main spectrum plus a separately stored dark; raw median near 0.09; broad shape is baseline-estimator-dependent | Do not subtract the dark a second time; retain corrected and uncorrected branches |

Those observations explain why a single raw intensity matrix is not
apples-to-apples. They do not license target-dependent or instrument-specific
cleanup. The headline generalization pipeline applies the same mathematical
rule and parameters to every instrument. A rule can remove a different
estimated baseline from different spectra because the spectra differ, but it
does not inspect the instrument label.

Known-system harmonization can be evaluated later as a secondary deployment
scenario. It must be labelled as such and cannot be used as evidence of
generalization to an unseen system.

## Frozen preprocessing operations

No operation overwrites the common-grid source spectrum. Raw spectra,
despiked spectra, masks, fitted baselines, and scaling parameters are separate
artifacts.

### Artifact detection

Candidate isolated spikes are local maxima with:

- prominence at least 10% of that spectrum's full span; and
- half-prominence width no greater than 1.25 grid points.

Only flagged points in the derived branch are replaced by interpolation over
unflagged neighbors. The binary mask and prominence fraction are retained.

A possible saturation plateau is a run of at least three values at the
spectrum maximum within a relative tolerance of `1e-6` of spectral span.
Plateaus are flagged, not automatically repaired.

On the observed core, the conservative detector found six candidate isolated
points in five RMX-2 spectra and no numerical maximum plateau. These are
candidate transients, not proven cosmic rays. On deterministic synthetic
spike injection it detected 871 of 1,196 inserted points (recall 0.728,
precision 1.000 among its flags). In the composite corruption it detected 791
of 1,196 (recall 0.647, precision 0.979). This behavior is deliberately
conservative: uncertain points remain in the raw layer rather than being
silently erased.

### Scaling and representation definitions

All intensity candidates first use the derived despiked spectrum.

| Representation | Exact definition |
|---|---|
| `minimal_minmax` | `(x - min(x)) / (max(x) - min(x))`, independently per row over 400--1800 cm⁻¹ |
| `robust_minmax` | scale by row percentiles 1 and 99, then clip to `[0,1]` |
| `asls_minmax` | AsLS baseline with `lambda=1e6`, `p=0.001`, 12 iterations; subtract and row-min--max |
| `arpls_minmax` | arPLS baseline with `lambda=1e6`, at most 30 iterations, relative weight tolerance `1e-3`; subtract and row-min--max |
| `derivative_1` | row SNV, Savitzky--Golay window 17, polynomial order 3, derivative 1, row L2 |
| `derivative_2` | row SNV, Savitzky--Golay window 17, polynomial order 3, derivative 2, row L2 |

Ordinary min--max was required because the instruments do not share an
intensity scale. A dataset-global range would leave Mira dominating the
numerics, while feature-wise normalization fitted on all observations would
introduce fold dependence. Independent row scaling removes offset and scale
without using another sample.

Min--max does not remove baseline curvature, resolution, smoothing, peak
width, calibration drift, or structured noise. “Same numerical range” is not
“same physical measurement process.” This distinction is central to the VAE
question.

The baseline settings were fixed a priori from the conservative diagnostic
starting point and applied domain-blindly. They were not selected separately
for Mira, Pendar, Agilent, or RMX, and no outer-test score tuned them. A larger
baseline/smoothing/alignment grid is intentionally outside v1; adding it now
would turn a small 598-spectrum experiment into an underpowered search. It may
become a separate, nested v2 experiment if v1 demonstrates a specific failure.

## Leakage-safe evaluation design

Every observation has a persistent `observation_uid` and a
`master_sample_id`. All scans of one physical master sample stay in one of five
frozen folds, regardless of instrument, sensor, session, or scan number. Each
fold contains all seven target classes.

For each outer fold:

1. that master-sample fold is sealed as outer test;
2. each of the other four folds serves once as inner validation;
3. PCA and every classifier are fit only on the corresponding training rows;
4. preprocessing selection uses only the 20 inner-validation results per
   representation and subset;
5. the five outer-test results are confirmatory and never change the freeze.

With six representations and two dataset subsets, this yields
`6 × 2 × (20 inner + 5 outer) = 300` auditable evaluation rows.

The fixed screening models are deliberately simple:

- PCA with at most 32 whitened components, fitted inside the split;
- class-balanced logistic regression for the primary target score;
- nearest centroid on the representation itself as a geometry-sensitive
  secondary score.

The goal is to choose inputs, not to claim the best classifier.

### Domain leakage is target-adjusted

Instrument and sensor probes are trained with inverse-frequency weights for
each observed target × domain cell. Scores average accuracy across those cells
rather than letting a large target/domain combination dominate. A separate
target-only classifier estimates how much domain can be predicted from the
confounded target distribution alone. The reported leakage objective is:

`spectral domain-probe score - target-only null score`.

This does not magically repair missing target × sensor cells. It is a more
honest diagnostic than ordinary instrument accuracy, and the same-master
geometry test supplies complementary paired evidence.

### Paired geometry and peak preservation

Correlation distance is measured between test spectra. The primary paired
quantity is the distance between the same `master_sample_id` measured on
different instruments. The separation margin is different-target mean
distance minus same-master cross-instrument mean distance. Lower same-master
distance and higher margin are preferred together.

For intensity candidates, peaks in `minimal_minmax` are detected with
prominence 0.05 and minimum spacing 5 cm⁻¹. A transformed peak is recovered
when it can be paired within 5 cm⁻¹. A corrected candidate must have at least
0.90 mean peak recall and at least 0.90 median row correlation to the minimal
representation before it can be frozen.

### Corruption stress tests

Deterministic perturbations are injected into raw test spectra and the entire
preprocessing function is rerun:

| Stressor | Magnitude |
|---|---|
| Scale/offset | multiply by 1.7 and add 15% of original span |
| Smooth baseline | curved component with amplitude 30% of span |
| Gaussian noise | standard deviation 3% of span |
| Isolated spikes | two points, each increased by 80% of span |
| Wavenumber shift | ±3 grid points |
| Composite | all five operations |

These are controlled sensitivity tests, not a claim that their distribution
perfectly models field corruption. Metrics include target balanced accuracy,
agreement with clean predictions, and clean/corrupt shape correlation.

## Inner-validation evidence used for selection

Higher target, composite-corruption, separation-margin, peak-recall, and
correlation values are better. Lower leakage increments and same-master
distance are better.

| Representation | Core target | QC target | Composite target | Instrument leakage Δ | Sensor leakage Δ | Same-master distance | Separation margin | Peak recall | Correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `minimal_minmax` | 0.697 | 0.739 | 0.664 | 0.671 | 0.442 | 0.760 | 0.113 | 1.000 | 1.000 |
| `robust_minmax` | 0.696 | 0.730 | 0.685 | 0.682 | 0.442 | 0.768 | 0.106 | 0.996 | 0.999 |
| `asls_minmax` | 0.714 | 0.755 | 0.654 | 0.661 | 0.446 | 0.700 | 0.148 | 0.961 | 0.984 |
| `arpls_minmax` | 0.707 | 0.742 | 0.653 | 0.611 | 0.446 | 0.675 | 0.186 | 0.937 | 0.944 |
| `derivative_1` | 0.701 | 0.739 | 0.550 | 0.640 | 0.433 | 0.684 | 0.163 | N/A | N/A |
| `derivative_2` | 0.646 | 0.711 | 0.452 | 0.527 | 0.409 | 0.707 | 0.155 | N/A | N/A |

All six candidates occupy part of the broad seven-objective Pareto surface.
That is not surprising: derivative 2 suppresses domain predictability but
loses target/corruption performance, while minimal processing preserves the
original shape and corruption stability but also preserves domain variation.
A one-number winner would conceal those scientific roles.

The selection rule therefore freezes one candidate per necessary role:

- `minimal_minmax` is retained as the reconstructive causal control. Without
  it, an apparent benefit of a VAE could actually be the baseline algorithm.
- `arpls_minmax` is the corrected reconstructive branch. Relative to AsLS it
  gives up 0.007 inner core target accuracy but reduces instrument leakage
  from 0.661 to 0.611, lowers same-master cross-instrument distance from 0.700
  to 0.675, and increases the separation margin from 0.148 to 0.186. It still
  passes both peak-preservation gates.
- `derivative_1` is the prior-work discriminative branch. It materially
  exceeds derivative 2 in core target accuracy (0.701 versus 0.646), composite
  corruption performance (0.550 versus 0.452), and same-master distance
  (0.684 versus 0.707).

`robust_minmax` is not frozen because its small composite-corruption advantage
does not offset weaker target/QC performance, greater instrument leakage, and
worse paired geometry relative to ordinary min--max. It remains in the full
candidate archive for audit. `asls_minmax` remains an important ablation
because it has the highest inner target scores, but arPLS has the better
generalization trade-off. `derivative_2` remains the classical baseline but is
not the primary derivative input.

The equal-weight utility used to break role-specific Pareto ties min--max
normalizes each objective across the six candidates. It is a transparent
screening convention, not a statistical proof that one preprocessing method
is universally optimal.

## Sealed outer-test confirmation

Outer target balanced accuracy was not used for selection:

| Representation | Strict core mean ± SD | Quality-pass mean ± SD |
|---|---:|---:|
| `minimal_minmax` | 0.722 ± 0.059 | 0.748 ± 0.036 |
| `robust_minmax` | 0.721 ± 0.062 | 0.738 ± 0.040 |
| `asls_minmax` | 0.732 ± 0.054 | 0.759 ± 0.043 |
| `arpls_minmax` | 0.729 ± 0.041 | 0.765 ± 0.048 |
| `derivative_1` | 0.716 ± 0.060 | 0.745 ± 0.053 |
| `derivative_2` | 0.664 ± 0.053 | 0.716 ± 0.073 |

Quality-pass performance is generally higher, but the ordering and trade-offs
do not collapse. This supports keeping NATO-L598 as the primary field dataset
and NATO-Q500 as a sensitivity analysis rather than training only on cleaner
rows and overstating deployment performance.

The remaining domain signal is substantial. Even after target-only adjustment,
the selected arPLS inner leakage increments are about 0.611 for instrument and
0.446 for sensor. Preprocessing alone is therefore not an invariance solution.
It only supplies comparable, auditable inputs on which invariance can be
measured.

## Frozen artifact contract

The principal files are:

| File | Contract |
|---|---|
| `frozen_model_inputs_core.npz` | Axis, 598 ordered UIDs, and only the three selected arrays |
| `frozen_model_inputs_quality.npz` | Exact 500-row UID-aligned subset and the same three arrays |
| `candidate_spectra_core.npz` | Raw, despiked, masks, baselines, and all six screening candidates |
| `core_preprocessing_manifest.csv` | Provenance, labels, folds, quality flags, artifact counts, and scaling metadata |
| `nested_group_cv_assignments.csv` | Explicit outer/development and inner-fold roles |
| `domain_evaluation_partitions_core.csv` | Frozen instrument/sensor transfer partitions |
| `benchmark_fold_metrics.csv` | All 300 fold-level measurements |
| `selection_objectives.csv` | Exact inner-selection table |
| `frozen_selection.json` | Machine-readable objectives, gates, Pareto sets, utilities, and selections |
| `dataset_version.json` | Configuration, software versions, source hashes, and selected names |
| `artifact_hashes.json` | SHA-256 hashes for every generated artifact |

The validator checks the 598/500 counts, row order, exact axis, all array
shapes and finite values, row-wise `[0,1]` bounds, derivative unit norms,
quality-subset identity, raw-source equality, mask counts, master-sample split
isolation, target coverage, 300-row benchmark structure, metric bounds,
selected-only archive identity, injection evidence, source hashes, and all
bundle hashes.

The canonical bundle passed all checks with 21 verified artifact hashes. A
second clean build in a separate directory also passed independently and
reproduced 38 stored NPZ arrays exactly, reproduced 16 other artifacts
byte-for-byte, and produced an identical version record after removing only
the expected creation timestamp. The temporary replica was then moved to the
system trash.

Rebuild and validate from the repository root:

```bash
.venv/bin/python scripts/freeze_nato_sers_preprocessing.py
.venv/bin/python scripts/validate_nato_sers_preprocessing_freeze.py
```

## What happens next

The preprocessing question is now frozen for the first model comparison. The
next goal should implement a common evaluation harness, then run models in an
incremental sequence:

1. fixed PCA/logistic and nearest-centroid baselines from this freeze;
2. the existing Siamese/triplet method on `derivative_1`;
3. deterministic AE on `minimal_minmax` and `arpls_minmax`;
4. denoising AE using the frozen corruption operators;
5. standard VAE;
6. β/TC/disentanglement ablations;
7. chemical/nuisance VAE with instrument/sensor nuisance supervision and
   leakage probes on the chemical latent;
8. reconstruction + cross-domain metric hybrid, motivated by the poster's
   strong global performance and its local Ag/AgNP 4NP collapse.

Every model must use the same master-sample folds, report strict-core and
quality-pass sensitivity, keep outer-test folds sealed, and compare target
utility against instrument/sensor leakage and paired-master geometry. A model
is better only if it preserves chemical information while reducing nuisance
dependence; a visually clean reconstruction or lower domain accuracy alone is
not sufficient.

The complete upstream evidence and broader experiment sequence are in
[`NATO_SERS_FIELD_TRIAL_AUDIT.md`](NATO_SERS_FIELD_TRIAL_AUDIT.md) and
[`NATO_SERS_VAE_EXPERIMENTAL_WORKFLOW.md`](NATO_SERS_VAE_EXPERIMENTAL_WORKFLOW.md).
