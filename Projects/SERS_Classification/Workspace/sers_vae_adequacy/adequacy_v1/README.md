# SERS VAE adequacy v1

This immutable bundle determines whether the original NATO SERS standard-VAE
failures were caused by undertraining, backbone/loss/capacity choices, or domain
shift. Selection used only master-sample-grouped nested inner validation.
Outer, field-stress, instrument/sensor, and poster outcomes were locked until
the configuration was frozen.

## Outcome

The original 100-epoch cap was premature. Constant-LR training converged at
epoch 500 and raised strict-core arPLS reconstruction correlation from
`0.906072` to
`0.948112` and repeatable-peak
recall from `0.419394` to
`0.479627`. It did not materially raise strict
chemical classification.

- selected: `base_maxpool__z64__spectral_composite__beta0p25__constant_lr__e500`;
- parameters: `1,082,353`;
- inner gates: `7/9`;
- failed gates: `gate_instrument_probe, gate_same_master_distance`;
- strict-core arPLS balanced accuracy: `0.706236`;
- quality-pass arPLS balanced accuracy: `0.744493`;
- field-stress arPLS balanced accuracy: `0.368545`;
- strict leave-instrument-and-sample accuracy: `0.596374`;
- strict leave-sensor-family-and-sample accuracy: `0.584317`;
- descriptive poster arPLS accuracy: `0.787778`.

The experiment executed 512 distinct 500-epoch training runs (256,000
optimizer epochs): 260 grouped-inner selection runs and 252 locked
confirmatory runs. See `compute_accounting.json` and
`parameter_and_compute_accounting.csv`.

## Decision

The converged standard-VAE backbone is scientifically adequate as a reconstruction-capable mixed-latent comparator and initialization for the next study, but it is not adequate as an instrument- and substrate-invariant representation. It remains below the frozen PCA/logistic clean benchmark, fails the registered instrument and same-master gates, and does not solve field or sensor-family shift.

The frozen backbone is the starting point for a separately registered
structured/disentangled-VAE goal. No chemical/nuisance latent partitioning was
performed here.

See `DECISION_REGISTRY.md`, `final_decisions.json`,
`failure_attribution.json`, the comparator/uncertainty tables, and `figures/`.
