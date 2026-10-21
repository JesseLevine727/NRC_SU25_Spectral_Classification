# Paper 2 pilot: phase 3b (re-run with corrected reconstruction loss)

## Why

In phase 3 the reconstruction loss was the squared error averaged over ~2,200 spectral points,
which made the KL term dominate: the plain VAE and beta-VAE collapsed in every fold (0 active
units) and the conditional VAE partly collapsed. Phase 3b repeats phase 3 with the standard
VAE scaling so the model ladder is a fair comparison. Nothing else changes.

## The one change

For all four models use

    loss = SUM over spectral points of squared error  +  beta * KL  +  w_ce * cross-entropy

with the KL summed over latent dimensions and every term averaged over the batch; beta = 1 (VAE,
conditional VAE, physics-structured VAE if it has a KL term) and beta = 4 (beta-VAE); keep the
50-epoch KL warm-up. Set **w_ce = number of spectral points** so the supervised heads keep exactly
the weight relative to reconstruction that they had in phase 3. (Equivalently: keep the phase 3
loss and divide only the KL term by the number of points.)

## Everything else stays identical

Same data, preprocessing (min-max primary; arPLS for E1 and E4), architectures, parameter counts,
optimiser, epochs, patience, early stopping on 10% of training-map spots, seeds 0/1/2, folds and
experiments E1-E4 as in `PHASE3_BRIEF.md`. Same hard rules: work only inside this folder, never
touch raw data, no pip installs, no network, no git; GPU allowed.

Implement the change as an option in the existing code (e.g. `--recon sum --tag phase3b`) so
phase 3 stays reproducible. Do not overwrite any phase 3 output.

## Added diagnostic

For every fit report the number of **active latent units** (latent dimensions whose posterior
mean varies across spectra with variance > 0.01). If collapse persists, report it plainly.

## Deliverables

`outputs/phase3b_*.csv`, `figures/phase3b_*.png` and `PHASE3B_RESULTS.md` (one page max): for
E1-E4, a table with the phase 2 baseline, phase 3 and phase 3b side by side; active units per
model; runtime; and two or three plain sentences on whether the fixed loss changes which rung of
the ladder separates chemical from substrate. Stop and print the path of `PHASE3B_RESULTS.md`.
