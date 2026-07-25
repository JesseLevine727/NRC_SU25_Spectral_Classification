# SERS VAE adequacy protocol v1 — amendment 2

Status: predeclared on 2026-07-24 after convergence stage 1b and before any
stage-2 model execution.

At epoch 500, constant learning rate satisfies the registered convergence
criterion. It repairs clean correlation but continues to fail repeatable-peak
and same-master geometry gates. This opens the already registered bounded
ablation.

All stage-2 candidates train for 500 epochs on the same 20 strict-core
`arpls_minmax` grouped inner folds, using paired fold seeds. The search remains
sequential: architecture, loss, latent capacity, then KL strength. Earlier
identical candidates are reused. A stage prefers converged candidates, then
passes-all-gates, gate count, registered utility, smaller parameter count, and
lexical identifier.

The top two final KL candidates proceed to quality-pass `arpls_minmax`
confirmation. Quality balanced accuracy may be at most 0.05 below strict-core.
The final arPLS winner alone runs strict and quality `minimal_minmax`
sensitivity; minimal results cannot change selection. Locked cohorts remain
untouched.
