# SERS standard VAE v1

This immutable bundle evaluates a one-block, unsupervised standard VAE after
the frozen deterministic baselines and before any structured/disentangled VAE.

## Outcome

The selected four-cycle KL schedule produced a noncollapsed posterior but did
not pass all predeclared advancement gates. It failed clean correlation,
repeatable-peak preservation, and same-master cross-instrument geometry.

- strict-core arPLS balanced accuracy: `0.706081`;
- quality-pass arPLS balanced accuracy: `0.728353`;
- field-stress arPLS balanced accuracy: `0.342487`;
- descriptive poster transfer: `0.927222`;
- strict leave-instrument-and-sample transfer: `0.557490`.

The model remains a required mixed-latent comparator. It is not evidence of
disentanglement.

See `DECISION_REGISTRY.md`, `final_decisions.json`, and the tables/figures in
this directory for the complete result.
