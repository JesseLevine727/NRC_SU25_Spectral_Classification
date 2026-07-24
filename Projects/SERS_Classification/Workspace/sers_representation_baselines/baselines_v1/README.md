# SERS representation baselines v1

This immutable result bundle establishes classical, deterministic Siamese, clean autoencoder, and denoising-autoencoder baselines before any VAE experiment.

## Outcome

The baseline does **not** show that reconstruction alone improves classification. PCA-logistic remains the strongest clean NATO reference. The arPLS DAE is materially more robust to registered synthetic corruptions than the matched clean AE, but it does not consistently improve real field-quality or unseen-domain classification. Peak preservation remains below the predeclared gate.

The next standard-VAE goal should begin from `arpls_minmax`, channels `(8, 16)`, latent dimension `64`, and the spectral-composite reconstruction loss. `minimal_minmax` is a mandatory sensitivity control.

Training uses strict deterministic CUDA. Because separate-process CUDA convolution inference retained tiny floating variation near a handful of decision and peak thresholds, all canonical final metrics, predictions, embeddings, reconstructions, and corruption outputs are replayed on CPU from state-hash-verified checkpoints. The clean rebuild matches that canonical layer exactly.

## Main records

- `predeclared_protocol.json`: frozen design and gates.
- `selected_configurations.json`: inner-only closed selection.
- `DECISION_REGISTRY.md`: final decisions and claim limits.
- `outer_fold_metrics.csv`: sealed NATO outer results.
- `domain_transfer_metrics.csv`: 56 domain scenarios.
- `poster_metrics.csv`: descriptive substrate-family transfer.
- `per_spectrum_predictions.csv`: all predictions and support flags.
- `corruption_metrics.csv`: corruption-by-severity results.
- `reconstruction_metrics.csv`: per-spectrum reconstruction evidence.
- `uncertainty_summary.csv`: fold/domain uncertainty and unit definitions.
- `artifact_hashes.json`: SHA-256 catalog.
- `validation_report.json`: automated integrity audit.

## Reproduce

Run selection, final evaluation, finalization, and validation with the project `.venv` and the commands recorded in `reproduction_commands.sh`.
