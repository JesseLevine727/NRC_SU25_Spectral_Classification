# Structured/disentangled VAE metadata and identifiability audit

Protocol family: `sers-structured-vae-v1`  
Status: completed before structured-model preregistration or execution.

## Frozen population

- Spectra: 598
- Master samples: 69
- Analytes: 7
- Instruments: 10
- Sensor families: 4
- Every master sample has exactly one analyte: True

## Pair structure

- Masters measured on at least two instruments: 67/69
- Masters spanning at least two sensor families: 39/69
- Cross-instrument observation pairs: 2473
- Cross-sensor observation pairs: 1171
- Cross-instrument pairs using the same sensor family: 1568
- Cross-instrument pairs also changing sensor family: 905
- Median instruments per master: 5.0

Real pairs are sufficient for same-master consistency and carefully defined
cross-reconstruction. They are not randomized interventions: instrument,
sensor, matrix, scenario, and acquisition choices remain observational.

## Support and confounding

- Target×instrument supported cells:
  44/70
  (0.629)
- Target×sensor supported cells:
  17/28
  (0.607)
- Training-fold cross-instrument pairs range:
  1343–
  1606
- Validation-fold cross-instrument pairs range:
  430–
  572

Analyte, instrument, and sensor are substantially confounded. Consequently:

1. unconditional instrument/sensor adversaries are scientifically unsafe;
2. a low nuisance-probe score is not evidence of disentanglement if analyte
   information also falls;
3. objectives and probes must be target-adjusted and cell-balanced;
4. unsupported analyte-domain cells cannot be inferred from training results;
5. held-out-domain results must retain supported/unsupported flags.

## Metadata usability

- Instrument and sensor-family labels are complete and usable as primary
  nuisance variables.
- `master_sample_id` is complete and usable for grouping and real-pair
  consistency, but it is not an independent preparation/batch identifier.
- Acquisition metadata are fragmented by instrument family. The
  12 audited acquisition fields have coverage ranging
  from 0.187 to
  0.358. They may be used for
  descriptive probes within supported instrument families, not as a universal
  nuisance target.
- Nominal concentration is recorded for
  177/598
  spectra and has only
  3 levels. It is a
  partial chemical covariate, not a global supervised factor.
- No defensible independent preparation ID is available. Session, paper sheet,
  scenario, team, and date can be reported as proxies but cannot establish
  preparation invariance.

## Mechanisms justified by this audit

1. A fixed-capacity chemical/nuisance partition.
2. Target-adjusted, cell-balanced instrument adversarial suppression.
3. Sensor-family adversarial suppression only as a secondary, strongly
   confounded objective.
4. Instrument- and/or sensor-conditioned decoding.
5. Same-master cross-instrument consistency using only pairs contained within
   the current grouped training partition.
6. Cross-reconstruction or latent swapping only where source and reference
   share `master_sample_id`; never manufacture a chemical ground truth across
   unrelated samples.
7. Dependence penalties as diagnostics/regularizers, not independent proof of
   semantic identifiability.

## Required negative controls

- Structural-loss weights set to zero must reproduce the frozen standard VAE.
- Permuted nuisance labels within analyte strata must remove any genuine
  adversarial/conditioning advantage.
- Permuted pair assignments within analyte strata must destroy real
  same-master consistency gains.
- Chemical-label permutation must drive supported-class performance toward
  chance.
- Every nuisance-suppression result must be paired with chemical retention,
  partition activity, and reconstruction/peak checks.

## Identifiability conclusion

The data can support a conservative claim of a **structured** or
**nuisance-suppressed chemical representation** if all registered evidence is
consistent. The observational, confounded design cannot by itself establish
unique causal factor recovery. The term **disentangled** must be reserved for
convergent evidence from partition-specific probes, real-pair behavior,
dependence diagnostics, negative controls, spectral preservation, and unseen
domain confirmation.
