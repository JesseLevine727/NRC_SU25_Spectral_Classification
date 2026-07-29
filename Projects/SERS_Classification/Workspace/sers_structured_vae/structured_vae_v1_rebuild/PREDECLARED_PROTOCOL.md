# NATO SERS structured/disentangled-VAE protocol v1

Status: predeclared on 2026-07-28 after the metadata/identifiability audit and
before any structured-model execution.

## Scientific question

Can explicit chemical and nuisance structure remove instrument, sensor,
substrate, baseline, intensity, and corruption variation from a chemical
representation without erasing analyte information, repeatable Raman/SERS
peaks, or reconstructable spectral shape?

The completed adequacy study established that the 100-epoch standard VAE was
undertrained for reconstruction, but that a converged mixed latent retained
instrument structure and increased same-master cross-instrument distance.
This protocol tests mechanisms that tell the model which variation is
chemical and which is nuisance.

## Immutable boundary

The following are frozen:

- 598 strict-core, 500 quality-pass, and 98 field-stress spectra;
- the 400–1800 cm⁻¹, 1 cm⁻¹, 1,401-point axis;
- `master_sample_id` grouping and existing nested/outer assignments;
- `arpls_minmax` as primary and `minimal_minmax` as mandatory sensitivity;
- preprocessing-v2 membership and quality decisions;
- the 8→16, two-max-pool backbone, spectral-composite loss, z64 total latent
  budget, Adam settings, and 500-epoch schedule;
- all frozen PCA/logistic, Siamese, AE, DAE, VAE-100, and VAE-500 comparators.

Architecture, objective, and weight selection use only the 20 grouped strict
arPLS inner folds. Outer, field-stress, confirmatory corruption, held-out
instrument/sensor, and poster outcomes remain locked until selection closes.
Because the outer cohorts were observed in prior studies, their final use is
confirmatory rather than human-blind.

## Identifiability audit

The audit found 69 master samples, of which 67 span multiple instruments and
39 span multiple sensor families. There are 2,473 cross-instrument and 1,171
cross-sensor observation pairs. Real-pair consistency and cross-reconstruction
are therefore possible.

The design is nevertheless strongly observational and confounded. Only 44/70
analyte×instrument and 17/28 analyte×sensor cells are supported. Sensor family
has bias-corrected Cramér's V 0.542 with analyte; instrument has V 0.382.
Unconditional domain adversaries could erase analyte information and appear
invariant. All nuisance objectives and probes must therefore be target-aware
and analyte-domain cell-balanced. Unsupported cells are never silently treated
as errors or successes.

There is no defensible independent preparation identifier. Acquisition
metadata are incomplete and instrument-specific. Preparation, concentration,
and acquisition invariance can be described only where support exists.

## Claim vocabulary

- **Structured:** architectural or objective roles are assigned to different
  latent partitions.
- **Nuisance-suppressed:** the chemical partition passes registered nuisance
  reduction, chemistry retention, geometry, spectral, posterior, and stability
  gates.
- **Disentangled:** nuisance-suppressed plus consistent evidence from
  partition-specific probes, dependence diagnostics, real-pair behavior,
  swaps/cross-reconstruction, negative controls, and unseen domains.
- **Unsuccessful:** no candidate passes the required gates.

Low nuisance predictability caused by chemical collapse is failure, not
disentanglement.

## Stage 0 — exact identity control

`mixed_z64_zero_structure` invokes the exact frozen adequacy implementation:
model, loss arithmetic, beta schedule, folds, seeds, optimizer, and
checkpoints. Selection cannot begin unless histories match within `1e-12` and
checkpoint tensors match exactly.

## Stage 1 — partition and supervision controls

The bounded controls are:

1. z48 chemical + z16 nuisance, unsupervised;
2. z32 + z32, unsupervised capacity allocation;
3. z48 + z16 with a chemical-classification head;
4. z48 + z16 with chemical and nuisance heads plus instrument/sensor
   conditioned decoding.

These distinguish semantic gains from merely splitting dimensions, adding
chemical supervision, adding parameters, or giving the decoder domain labels.
The total stochastic budget remains 64. Encoder–decoder skips are prohibited.

## Stage 2 — mechanisms

Mechanisms open sequentially rather than as a Cartesian grid:

1. target-conditioned, cell-balanced instrument adversary with weights
   0.00125, 0.0025, and 0.005;
2. same-master chemical consistency with weights 0.005, 0.01, and 0.02,
   accompanied by real-partner cross-reconstruction at weight 0.25;
3. sensor adversary at 0.00125 or 0.0025 only if instrument suppression first
   preserves chemistry;
4. cross-partition covariance penalty at 0.001 or 0.005 only if maximum
   canonical correlation exceeds 0.50;
5. a combination only from individually eligible, directionally compatible
   mechanisms.

The adversary receives the analyte label as a fixed conditioning variable and
uses analyte-domain cell weights. Its gradient reversal ramps from zero during
epochs 101–200 and is fixed afterward. Paired losses use only real partners
inside the current grouped training partition.

## Negative controls

After an inner candidate is frozen and before locked confirmation:

- nuisance labels are permuted within analyte strata;
- partner identities are permuted within analyte strata;
- chemical labels are permuted by master-sample group;
- structural weights set to zero retain the exact identity control.

Chemical permutation must produce balanced accuracy no greater than 0.25.
Nuisance and pair permutations must remove mechanism-specific gains.

## Measurements and gates

Every partition and their union are evaluated for:

- supported-class balanced accuracy and macro F1;
- target-adjusted instrument and sensor probe increments;
- same-master distance, different-analyte distance, and separation margin;
- cross-covariance and maximum canonical correlation;
- KL, active units, KL-active dimensions, and posterior variability;
- reconstruction MSE, Smooth L1, spectral angle, derivative error, and
  correlation;
- repeatable-peak recall, shift, width, and prominence;
- corruption agreement, reconstruction recovery, and latent drift;
- paired consistency and real cross-reconstruction.

Relative to VAE-500, the chemical latent must retain BA ≥0.662119 and macro F1
≥0.650692; reduce instrument probe to ≤0.542584 and same-master distance to
≤0.727530; retain separation margin ≥0.224031, correlation ≥0.914933, and peak
recall ≥0.447539. A sensor claim additionally requires sensor probe ≤0.398991.
At least 75% of folds must improve instrument nuisance and at least 75% must
preserve chemical accuracy. Both partitions must remain variationally active
and all outputs finite.

Convergence requires median final-50 validation-objective improvement below
0.5% and fewer than 25% of folds improving by at least 1%. Diagnostic runs do
not stop early.

When no candidate passes every applicable gate, the decision order is:
convergence, all-gate status, gate count, Pareto nondominance, registered
utility, smaller parameter count, then lexical identifier. Gates are never
weakened after observing results.

## Confirmation and terminal decision

The selected arPLS candidate alone undergoes quality confirmation and minimal
sensitivity, followed by one locked evaluation on grouped outer cohorts,
field stress, registered corruptions, instrument/sensor domain transfer,
same-master geometry, granular failure tables, and descriptive poster
transfer. Chemical, nuisance, combined, and conditioned modes are reported
separately.

The experiment ends by classifying the evidence as disentangled,
nuisance-suppressed, structured-only, or unsuccessful. A model is frozen for
downstream classification only when the registered evidence supports its claim
level.
