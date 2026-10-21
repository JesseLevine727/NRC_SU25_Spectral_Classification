# Paper 2 pilot: phase 3 (VAE model ladder)

Phases 1-2 are done (`PHASE1_SUMMARY.md`, `PHASE2_RESULTS.md`). Phase 3 asks whether VAEs
separate the chemical (analyte) from the substrate better than the phase 2 baselines, and which
ingredient is needed. Every cell is one map, so results are indicative only; report them that way.

## Hard rules (same as phase 1-2, plus GPU)

- Work only inside this folder. Read `outputs/*.npz` (regenerate with `load.py` if missing). Never
  touch raw data. No pip installs, no network, no git commands.
- Python: `/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python`
  (PyTorch 2.11 with CUDA; use the GPU).
- **Independent unit = one map.** Test maps never enter training, validation or early stopping.
  Early stopping may use a random 10% of *spots from training maps only*.
- Fixed configuration, no hyperparameter search. Seeds 0, 1, 2. Report failures (e.g. posterior
  collapse, NaNs) instead of hiding them.
- Budget: aim for under 3 hours total. If the projected runtime exceeds that, drop to 2 seeds and
  say so in the results.

## Preprocessing

Per-spectrum min-max (primary). arPLS + min-max (as in phase 2) as a sensitivity for E1 and E4 only.

## The four models (PyTorch; at most ~50k trainable weights each, excluding the physics
## decoder's spectral templates; report every parameter count)

Common: 1-D convolutional encoder, Adam (lr 1e-3, weight decay 1e-4), batch 64, at most 300
epochs with patience 30, KL warm-up over the first 50 epochs, reconstruction loss = mean squared
error.

1. **VAE:** 8 latent dimensions, KL weight 1.
2. **beta-VAE:** same architecture, KL weight 4 (unsupervised disentanglement).
3. **Conditional VAE with label-partitioned latents:** latent split into a chemical block (4 dims),
   a substrate block (4 dims) and, for block B, a laser block (2 dims). A small classification head
   on each block predicts its own label (cross-entropy, weight 1). The decoder sees only the latents.
4. **Physics-structured VAE with an additive decoder:**
   reconstruction = r_laser(shift) * ( G[a,s] * A_a(shift) + B_s(shift) ) + offset, where
   - A_a = non-negative learned spectrum per analyte, B_s = non-negative learned spectrum per
     substrate (softplus);
   - G[a,s] = learned positive scalar enhancement per analyte x substrate (the interaction term);
   - r_laser = smooth response per laser (at most 8 polynomial coefficients; fixed to 1 in block A);
   - the encoder outputs analyte probabilities p_a, substrate probabilities q_s (supervised with
     cross-entropy during training), a positive overall gain and an offset; mix A and B with p and q.

## Experiments (compare every number with the phase 2 baseline in the same table)

- **E1 Chemical part.** Fit each model on all maps of a block (3 seeds). Chemical part = A_a for
  model 4; for model 3, decode the analyte's mean chemical block with the substrate block set to
  its overall mean. Correlate with two targets: (a) the pure Raman reference (as phase 2), and
  (b) the **SERS signature** = the analyte effect of the phase 2 additive model (mean over
  substrates of the analyte's cell mean minus each substrate's mean over analytes). Note in the
  results that target (b) comes from a baseline. Models 1-2 have no chemical part: report n/a.
- **E2 Separation.** Map-grouped cross-validated logistic-regression probes: predict analyte and
  substrate from each latent block (models 3-4: chemical block / p vs substrate block / q; models
  1-2: the full latent, plus the mutual information gap (MIG) for analyte and substrate).
  Swap test for models 3-4: decode chemical block of spectrum i with substrate block of spectrum j
  (different substrate); classify the decoded spectrum's analyte and substrate with random forests
  trained on real spectra; report how often it reads as analyte i on substrate j.
- **E3 Unseen combinations** (models 3-4 only). Same held-out cells as phase 2: retrain without
  the cell, decode that analyte on that substrate (model 4: use the analyte's mean G over seen
  cells), and report correlation and RMSE against the real held-out cell mean.
- **E4 Unseen substrate / laser.** Same folds as phase 2 (A: leave-one-substrate-out; B:
  leave-one-substrate-out and leave-one-laser-out). Classify the analyte: models 1-2 with
  logistic regression on the latent mean fitted on training maps; model 3 with its chemical head;
  model 4 with p. Report balanced accuracy per spectrum and per map.

## Deliverables

`vae_models.py`, `run_phase3.py`, `outputs/phase3_*.csv`, a few figures in `figures/` (model 4's
learned analyte spectra vs references; swap-test example; E4 comparison), and `PHASE3_RESULTS.md`
(one page max): one table per experiment with the phase 2 baseline alongside, parameter counts,
runtime, failures, and two or three plain sentences on which rung of the ladder (if any) separates
chemical from substrate. Stop after phase 3 and print the path of `PHASE3_RESULTS.md`.
