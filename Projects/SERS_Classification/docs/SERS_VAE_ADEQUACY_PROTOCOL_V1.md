# NATO SERS VAE training and architecture adequacy protocol v1

Status: predeclared before any adequacy-study model execution on 2026-07-24.

## Purpose

The first standard VAE was a controlled, capacity-matched comparator, not an
exhaustive VAE optimization. Its inner validation loss was still improving at
the 100-epoch boundary and it failed reconstruction-correlation,
repeatable-peak, and same-master geometry gates. This protocol distinguishes
five explanations before a structured latent is attempted:

1. the 100-epoch boundary prevented convergence;
2. two pooling operations or shallow feature extraction lost peak geometry;
3. the reconstruction objective underweighted peak shape;
4. latent capacity or KL pressure retained the wrong information;
5. the field and domain failures reflect unsupported distribution shift.

The frozen preprocessing-v2 data, group assignments, representations, and
prior comparators are never modified.

## Leakage boundary

Architecture, epoch, loss, latent, KL, and optimizer decisions use only the 20
`master_sample_id`-grouped nested inner folds. `arpls_minmax` is primary.
`minimal_minmax` is a mandatory sensitivity that cannot alter primary
selection. Outer folds, the 98 field-quality-stress spectra, held-out-domain
results, and poster spectra are locked until one configuration is frozen.
Because those cohorts have been observed in earlier work, their final use is
confirmatory rather than human-blind.

## Sequential decision tree

### Stage 0 — audit

Record the original best-epoch distribution, epoch-90-to-100 ELBO trend,
cycle/early-stop interaction, training-set-to-parameter ratio, and gate
failures.

### Stage 1 — convergence isolation

Reproduce the original optimization through epoch 100. The four KL cycles
remain exactly 25 epochs; increasing the maximum epoch must not stretch them.
For epochs 101–300, beta remains 1. Evaluate epochs 100, 125, 150, 175, 200,
225, 250, 275, and 300 under:

- constant Adam learning rate `1e-3`;
- the same first 100 epochs followed by validation-plateau reductions.

Diagnostic runs do not stop early because all registered checkpoints are
required. A configuration is considered converged only when median validation
ELBO improvement over its final 50 epochs is below 0.5% and fewer than 25% of
folds improve by at least 1%. Lower ELBO is not accepted as scientific
improvement by itself.

Architecture ablation is skipped only if a converged checkpoint passes every
standard-VAE gate on strict-core and quality-confirmation inner data.

### Stage 2 — bounded adequacy ablation

If required, test these factors sequentially rather than as an unrestricted
Cartesian product:

1. current `base_maxpool`, parameter-matched `residual_multiscale`, and
   `single_pool_peak`;
2. current spectral composite versus a preregistered first/second-derivative
   and pooled-scale loss;
3. latent dimensions 32, 64, and 128;
4. beta targets 0.25, 1, and 4.

Each later factor uses the earlier frozen winner. Alternative backbones must
remain within 0.75–1.25 times the reference parameter count. Encoder–decoder
skip connections are prohibited because they could improve reconstruction by
bypassing the representation being evaluated.

## Scientific measurements

Every candidate checkpoint is judged on:

- chemical balanced accuracy and macro F1;
- reconstruction MSE, Smooth L1, correlation, spectral angle, and derivative
  error;
- repeatable-peak recall, shift, prominence, and width;
- controlled-corruption recovery, prediction agreement, and latent drift;
- KL magnitude, active units, and posterior sampling variability;
- target-adjusted instrument and sensor predictability;
- same-master versus different-target cross-instrument geometry.

The frozen clean-AE-relative gates from standard-VAE v1 remain the eligibility
standard. When no candidate passes all gates, gate count, registered utility,
parameter count, and lexical identifier define the least-failing backbone.

## Confirmation and terminal decision

After selection closes, evaluate the frozen model once on grouped outer
chemical cohorts, controlled corruption, field stress, instrument/sensor
domain-and-sample transfer, same-master geometry, and descriptive poster
transfer using three registered seeds. Compare it with the original VAE and
frozen PCA, Siamese, AE, and DAE results.

The final record attributes each failure to convergence, architecture,
objective, latent/KL choice, data/domain shift, or an unresolved interaction.
It freezes the exact backbone and claim limits for a later structured-latent
goal. No chemical/nuisance partitioning occurs in this protocol.
