# SERS VAE adequacy protocol v1 — amendment 1

Status: predeclared on 2026-07-24 after stage 1 aggregation and before any
amended model execution.

Stage 1 did not establish convergence at epoch 300. The selected constant-LR
model improved median validation ELBO by 1.099% from epoch 250 to 300, and 55%
of grouped folds improved by at least 1%. Comparing architectures at that
boundary would confound optimization duration with backbone adequacy.

The convergence study is therefore extended to epoch 500. Epochs 1–100 retain
the original four fixed 25-epoch cycles and beta remains 1 thereafter. Two
policies are registered:

- constant learning rate `1e-3`;
- the identical path through epoch 300 followed by learning rate `1e-4`.

Metrics are evaluated at epochs 100, 150, 200, 250, 300, 350, 400, 450, and
500 on the same 20 strict-core `arpls_minmax` grouped inner folds.

Before the amended execution, the KL arithmetic is grouped exactly as in
standard-VAE v1 and downstream probe seeds are fixed across checkpoints. The
superseded first execution showed at most `1.35e-9` history drift and is not
used as authoritative selection evidence. No data, split, gate, candidate
architecture, locked-cohort boundary, or scientific claim was changed.
