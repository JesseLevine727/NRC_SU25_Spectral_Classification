# Structured/disentangled VAE study on the NATO SERS field trial

## Executive conclusion

**Terminal evidence class: unsuccessful.** The proposed structured VAE did
not learn a defensible chemical/nuisance disentanglement and did not become a
general instrument, sensor, substrate or field-noise filter.

There is a real partial effect: the selected dependence penalty reduced the
inner target-adjusted instrument probe from `0.572` for the zero-structure
partition control to `0.543`, and reduced cross-covariance from `0.0655` to
`0.0248`. That effect did not survive all required tests. Same-master geometry
remained `0.783`, only 50% of folds preserved chemistry, maximum canonical
correlation remained `0.990`, and locked instrument leakage rose to `0.584`.

The chemical/nuisance split is not semantic: the nuisance partition alone
classifies strict-core analytes at BA `0.586`,
while chemical–nuisance maximum canonical correlation is
`0.994`. The union
latent behaves like the original mixed VAE, whereas the chemical partition
does not outperform it.

## Question and claim rules

The study asked whether a fixed-capacity z64 VAE, split into z48 chemical and
z16 nuisance variables, could preserve analyte signal while removing
instrument, sensor/substrate, baseline and noise variation. The registered
terminal classes were:

1. **Disentangled:** chemical and nuisance semantics, low dependence, nuisance
   removal, preserved chemistry and swap evidence all pass.
2. **Nuisance-suppressed:** chemistry is preserved and nuisance is reduced,
   but full semantic factorization is not supported.
3. **Structured-only:** partitions are operationally useful but nuisance
   suppression is not established.
4. **Unsuccessful:** no registered candidate passes every applicable gate.

No inner candidate passed every gate, so locked results could characterize but
could not rescue the claim.

## Data and preprocessing held fixed

- Strict core: 598 spectra, 69 master samples, 7 analytes, 10 instruments and
  4 sensor families.
- Quality-pass subset: 500 spectra.
- Field-quality stress cohort: 98 spectra.
- Common axis: 400–1800 cm⁻¹ at 1 cm⁻¹ spacing (1,401 values).
- Primary view: despiking/alignment, arPLS baseline correction and per-spectrum
  min–max scaling.
- Sensitivity view: the same axis, alignment and min–max scaling with minimal
  baseline removal.
- All split decisions are grouped by `master_sample_id`.

The metadata audit found only 44/70 analyte×instrument cells and 17/28
analyte×sensor cells supported. Analyte–sensor Cramér's V was 0.542 and no
independent preparation ID was available. These facts cap causal
identifiability: a low domain probe cannot by itself prove removal of physical
instrument or substrate effects.

## Model and training

The encoder has two 1-D convolution/max-pooling blocks with 8 and 16 channels.
It produces separate posterior means/log-variances for z48 chemical and z16
nuisance partitions. The decoder receives their concatenation; optional fixed
instrument/sensor conditioning inputs are present in every control so parameter
comparisons remain fair. Registered heads include chemical classification,
nuisance instrument/sensor classification and target-conditioned gradient
reversal adversaries.

Every authoritative run used:

- spectral-composite reconstruction loss;
- β=0.25 with the frozen four-phase warm-up;
- Adam, learning rate 0.001, weight decay 1e-5;
- batch size 64, gradient clipping at 5;
- exactly 500 epochs with checkpoints at 100/300/400/500;
- total latent capacity fixed at 64 and no encoder–decoder skips.

Selected parameter count was `1,169,412`
versus `1,082,353` for the mixed VAE
(`8.0%` more). The study ran
`404` authoritative fits and
`202,000` optimizer epochs;
diagnostic smoke tests are excluded.

## Identity and convergence

The exact standard-VAE identity control reproduced all 20 grouped-inner
histories, checkpoints and optimizer states with maximum numeric difference
zero. Selection was blocked until this passed.

The selected structured model converged under the registered rule: median
final-50 improvement was
`0.157%`
and only
`15%`
of folds improved at least 1%. Locked fits were more variable, but their median
final-50 change was
`-0.060%`
and half worsened. More epochs might move a minority of fits, but undertraining
does not explain near-unit partition dependence or held-out domain collapse.

## Inner mechanism search

| Branch winner | Gates | Chemical BA | Instrument probe | Same-master distance | Separation margin | Max CCA |
|---|---:|---:|---:|---:|---:|---:|
| Controls | 13/17 | 0.668 | 0.572 | 0.781 | 0.247 | 0.997 |
| Instrument Adversary | 14/17 | 0.669 | 0.580 | 0.685 | 0.326 | 0.997 |
| Pair | 12/17 | 0.655 | 0.456 | 0.008 | 0.002 | 0.997 |
| Dependence | 15/17 | 0.667 | 0.543 | 0.783 | 0.244 | 0.990 |

The instrument adversary improved same-master geometry and separation but did
not reduce the independent instrument probe consistently. Pair alignment
reduced same-master distance to `0.008`, but separation margin collapsed to
`0.002`: the model aligned both same- and different-analyte spectra. The
dependence penalty was selected by the fixed hierarchy at 15/17 gates; it
failed same-master geometry and fold-wise chemistry preservation.

Sensor adversaries remained closed because no instrument-adversarial candidate
was eligible. Combinations remained closed because no two individual
mechanisms were eligible. No post-hoc weights were added.

## Negative controls

Grouped chemical-label permutation reduced mean BA to `0.162`; every fold was
below 0.25 and the maximum was `0.218`. Nuisance-label and partner permutations
were non-applicable to the frozen dependence-only objective because all
nuisance-label, adversarial, pair and cross-reconstruction weights were zero.
They are recorded as non-applicable rather than passed.

## Preprocessing sensitivity

Quality-pass arPLS reached chemical BA `0.719`, instrument probe `0.540` and
same-master distance `0.743`. Strict minimal reached BA `0.628`, probe `0.644`
and distance `0.863`. Quality minimal recovered BA to `0.699`, but leakage
remained `0.631` and distance `0.834`.

Minimal preprocessing preserves more peaks, but leaves substantially more
instrument/background structure. Min–max scaling correctly places all spectra
on a common amplitude range; it cannot remove background curvature or
system-response shape. The evidence supports one common arPLS primary view plus
minimal sensitivity—not ad hoc instrument-specific preprocessing chosen after
outcome inspection.

## Locked grouped-outer results

| Model/view | Strict core | Quality pass | Field stress |
|---|---:|---:|---:|
| PCA/logistic | 0.725 | 0.771 | 0.447 |
| Linear SVM | 0.678 | 0.724 | 0.472 |
| Standard VAE-500 | 0.706 | 0.744 | 0.369 |
| Structured VAE—chemical | 0.681 | 0.728 | 0.327 |
| Structured VAE—union | 0.704 | 0.748 | 0.374 |
| Siamese | 0.632 | 0.677 | 0.370 |

For the primary arPLS chemical partition:

- strict BA `0.681`, instrument probe
  `0.584`, same-master distance
  `0.774`;
- quality BA `0.728`, instrument probe
  `0.591`;
- field-stress BA `0.327` and reconstruction
  correlation `0.393`.

The field-stress result is the main operational failure. Composite corruption
at severity 1 reduces strict chemical BA from `0.681` to `0.596` and clean/
corrupted agreement to `0.696`. Minimal spectra preserve more repeatable peaks
but do not solve field stress.

## Locked real-pair swaps

All 30 outer scenario/representation combinations have inspectable latent-swap
bundles. Each swap decodes the source chemical mean with a deterministic,
real same-master/different-instrument partner's nuisance mean and domain
labels. For arPLS, mean fold-level swap-to-partner median correlation is
`0.516` on strict,
`0.634` on quality, and
`0.306` on field stress,
compared with unmodeled source-to-partner correlations of
`0.217`,
`0.333`, and
`0.120`. These are
descriptive swap reconstructions, not semantic validation: the nuisance block
retains analyte information and the partitions remain almost canonically
collinear. `locked_outer_swap_metrics.csv` and `swaps/` retain every metric and
spectrum pair, including the one stress fold with no valid real partner.


## Held-out instrument and sensor transfer

Domain-only instrument BA averages
`0.613` on strict data and
`0.633` on quality data, with
large between-instrument ranges. Sensor-family BA averages only
`0.384` and
`0.344`.

Domain-plus-sample scores require caution: held-out domains may contain analytes
not represented in the remaining training partition. Some apparent 1.0 scores
are based on 17–21 supported spectra, while some sensor-family tests have zero
supported analytes. Tables retain supported/unsupported counts and do not use
those cells as evidence of generalization.

## Poster transfer

The architecture transfers descriptively to the poster data:
leave-one-substrate-family-out chemical BA is
`0.753` for arPLS and
`0.778` for minimal spectra. This is not NATO
label transfer—the poster analytes differ—and it is not disentanglement:
nuisance chemistry remains high and partition CCA is approximately 0.9995.

## Did the idea work?

**Not as intended.** A structured VAE can allocate capacity into named blocks,
but naming the blocks and penalizing covariance does not make the factorization
identifiable. The union representation remains competitive, so the architecture
can encode spectra. The failure is the semantic allocation:

- reconstruction can place chemistry and nuisance in either partition;
- a covariance penalty removes only batchwise linear covariance, not shared
  predictive structure;
- adversarial loss is unstable under analyte×instrument confounding;
- pair consistency can erase analyte separation while looking invariant;
- field-stress spectra are far outside the quality training distribution.

This does not show that disentangled SERS models are impossible. It shows that
this bounded formulation and this confounded field-trial dataset do not support
the claim.

## Reproducibility verification

An independent rebuild was executed in a previously nonexistent output
directory. All `7,139` cross-build checks passed:
canonical scientific tables and decision JSON were exact; frozen-input
SHA-256 digests matched after normalizing the intentionally different output
directory; every embedding and reconstruction array was exact; and every
checkpoint and optimizer tensor was exact. `rebuild_validation.json` contains
the complete machine-readable audit.


## Recommended next study

1. Treat acquisition as a balanced-design problem: collect the same analytes,
   preparations and concentrations across every instrument/sensor, and record
   independent preparation/batch IDs.
2. Add explicit class-preserving negatives to pair alignment: same-master
   positives plus different-analyte cross-instrument margin/contrastive terms.
3. Replace simple covariance with stronger dependence control (HSIC, total
   correlation or conditional mutual-information surrogates), while evaluating
   external probes.
4. Consider an instrument-aware physical front end or calibration layer before
   the shared chemical encoder; do not require one latent model to discover all
   baseline physics unsupervised.
5. Build a field-stress rejection/QC pathway. The current model should not
   classify spectra it cannot reconstruct or place near the training domain.
6. Keep PCA/logistic and the mixed VAE as mandatory comparators. A future model
   must beat them on grouped outer, field stress and held-out sensor transfer,
   not only on reconstruction.

## Artifact guide

- `terminal_decision.json`: final evidence class and headline numbers.
- `failure_attribution.json`: undertraining, capacity, objective and data-shift
  attribution.
- `inner_stage_winners.csv` / `inner_gate_matrix.csv`: all selection decisions.
- `locked_outer_*`, `locked_domain_*`, `locked_poster_*`: complete locked
  metrics, predictions, reconstruction, corruption, histories and registries.
- `locked_outer_swap_metrics.csv` / `swaps/`: real same-master latent swaps and
  the underlying source, partner and decoded spectra.
- `per_analyte_failures.csv`, `per_instrument_failures.csv`,
  `per_sensor_failures.csv`, `per_domain_failures.csv`: granular failures.
- `comparator_summary.csv`: matched outer-fold comparators.
- `figures/`: PDF and 600-DPI PNG figures.
- `validation_report.json` / `rebuild_validation.json`: within-build and exact
  independent-rebuild audits.
- `artifact_hashes.json`, `environment.json`, `reproduction_commands.sh`:
  rebuild provenance.
