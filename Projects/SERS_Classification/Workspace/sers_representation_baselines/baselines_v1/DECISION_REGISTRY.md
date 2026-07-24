# SERS baseline decision registry

Bundle: `sers-representation-baselines-v1`  
Selection: closed before outer, stress, poster, and domain evaluation.  
VAE models: prohibited in this bundle and not run.
Training: deterministic CUDA. Canonical final inference: deterministic CPU replay from verified checkpoints.

## Final decisions

1. **Classical reference:** PCA-logistic is the primary clean classification reference. It remains strongest on NATO strict core and quality pass.
2. **Siamese:** retain as a deterministic metric-learning control. It learns similarity structure but has no decoder and no explicit denoising objective.
3. **Clean AE:** retain as a compression/reconstruction diagnostic. It does not beat the classical clean reference and every clean AE failed the absolute repeatable-peak gate.
4. **DAE:** retain as a robustness comparator. It substantially improves held-out synthetic composite-corruption recovery, agreement, and latent stability, but does not consistently improve clean, real-stress, or unseen-domain classification.
5. **Primary VAE input:** `arpls_minmax`.
6. **Mandatory sensitivity input:** `minimal_minmax`, because it reconstructs clean spectra and repeatable peaks more faithfully.
7. **Frozen VAE starting architecture:** channels `(8, 16)`, latent dimension `64`, spectral-composite reconstruction loss; clean curriculum for standard VAE and `mixed_uniform` only for the denoising comparator.

## Critical limits

- The 98 field-quality-stress spectra remain difficult for every model family.
- Absolute peak preservation is unresolved; no AE passed the registered peak gate.
- Poster substrate holdout is descriptive map-location transfer, not independent-preparation validation.
- Domain-and-sample tests can contain unsupported analytes. They are predicted and listed, but excluded from supported-class balanced accuracy.
- Denoising gains on synthetic corruptions must not be generalized to arbitrary instrument/substrate nuisance.

## Evaluation-metric correction

The first final-evaluation pass incorrectly defined repeatable test peaks across analytes. Before interpretation, the final metric code was corrected to the frozen rule: peaks repeat within the same master sample across instruments. Checkpoints and predictions were unchanged and replayed from verified state hashes.
