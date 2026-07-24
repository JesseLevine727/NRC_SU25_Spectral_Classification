# Predeclared standard-VAE protocol v1

Date declared: 2026-07-24  
Status: frozen before standard-VAE training  
Machine-readable protocol:
[`configs/sers_standard_vae_v1.json`](../configs/sers_standard_vae_v1.json)

## Question

This phase asks what stochastic variational regularization contributes beyond
the frozen PCA-logistic, Siamese, clean-AE, and DAE baselines. It deliberately
uses one mixed latent vector. It does not claim, impose, or test a
chemical/nuisance latent split.

The phase must finish before any β-VAE, semi-supervised VAE, two-block VAE,
domain adversary, domain-conditioned decoder, or hybrid metric/reconstruction
objective is introduced.

## Immutable starting point

The NATO preprocessing-v2 and deterministic-baselines-v1 bundles remain
read-only. The primary VAE uses:

- `arpls_minmax`;
- Conv1D channels `8 → 16`, kernels 7 and 5;
- 64 diagonal-Gaussian latent variables;
- the matched deterministic decoder;
- spectral-composite reconstruction loss;
- one stochastic posterior sample per training observation.

`minimal_minmax` is a mandatory nonselective preprocessing sensitivity.

The earlier workflow suggested screening 8--32 latent variables before the
deterministic capacity study existed. The completed inner-only AE/DAE study
supersedes that provisional range and froze 64 variables for the matched
standard-VAE comparison. Latent dimension and decoder capacity are therefore
not reopened here.

## Standard ELBO definition

The posterior is diagonal Gaussian and the prior is an isotropic unit
Gaussian. Training minimizes

```text
spectral-composite reconstruction
+ beta(epoch) × KL(q(z|x) || N(0,I)) / 1401
```

The reconstruction components are means over the 1,401 measured spectral
positions. KL is first summed over the 64 latent variables per observation and
then divided by 1,401, putting both ELBO contributions on a per-measured-feature
scale. The final β is always 1.0.

Only three optimization schedules may be compared:

1. β = 1 from the first epoch;
2. linear warm-up to β = 1 at epoch 20;
3. four 25-epoch cycles, rising during the first half of each cycle.

This is a schedule comparison, not a β-VAE search.

## Selection

Only the 20 master-sample-grouped NATO inner folds may select the schedule.
All three schedules are evaluated on strict-core `arpls_minmax` with seed 1729.
At most two schedules proceed to the quality-pass inner sensitivity. The
result is frozen before outer, field-stress, domain, corruption, or poster
results are produced.

After selection, the same schedule and compact architecture are evaluated on
`minimal_minmax`; that branch cannot alter the primary selection.

Eligibility requires:

- clean median row correlation within 0.03 of the frozen arPLS AE;
- repeatable-peak recall within 0.05 of the frozen arPLS AE;
- chemical-probe balanced accuracy within 0.03 of the frozen arPLS AE;
- target-adjusted instrument leakage no more than 0.02 above the AE;
- same-master cross-instrument distance no more than 0.05 above the AE;
- at least four active dimensions by `Var(mu) > 0.01`;
- at least two dimensions with mean KL above 0.01;
- finite nonzero KL;
- quality-pass chemical performance no more than 0.05 below strict core.

If no schedule passes, the highest-utility model remains a diagnostic
standard-VAE comparator and is explicitly marked ineligible.

## Evaluation

Posterior means are the deterministic representation for probes,
classification, geometry, corruptions, and canonical reconstructions.
Posterior sampling is retained for training and separate stochastic diagnostics.

Final evaluation uses seeds 1729, 2718, and 3141 and includes:

- sealed NATO strict-core and quality-pass outer folds;
- quality-development to the 98 flagged spectra;
- all registered corruptions at severities 0.5, 1.0, and 1.5;
- strict domain-and-sample and descriptive domain-only instrument/sensor tests;
- unchanged poster leave-substrate-family-out transfer;
- reconstruction, peak, variational-health, chemical-probe, domain-probe, and
  same-master geometry measurements.

Projection plots are qualitative only.

## Interpretation boundary

A useful standard VAE is not evidence of disentanglement. This phase may show
whether a mixed Gaussian latent reconstructs, remains chemically useful,
collapses, encodes instruments, or becomes robust. Only the subsequent
registered structured-latent experiment can test whether chemical and nuisance
information can be assigned to different latent components.

The final bundle requires an independent clean rebuild and exact canonical CPU
comparison before its decision is promoted.
