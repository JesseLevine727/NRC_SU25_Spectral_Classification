# Predeclared SERS representation-baselines protocol v1

Date declared: 2026-07-23  
Status: frozen before AE/DAE model screening  
Machine-readable protocol:
[`configs/sers_representation_baselines_v1.json`](../configs/sers_representation_baselines_v1.json)

## 1. Question

Before adding VAE stochasticity or a chemical/nuisance latent split, determine:

1. what the classical and Siamese controls achieve under reproducible,
   leakage-safe evaluation;
2. whether deterministic compression preserves useful chemistry;
3. whether explicit denoising improves robustness relative to a matched AE
   without erasing clean or repeatable spectral structure;
4. which input view and deterministic encoder/decoder configuration may be
   carried unchanged into the next standard-VAE experiment.

This phase will not train a VAE, β-VAE, adversarial encoder, disentangled
model, or hybrid VAE.

## 2. Immutable inputs

NATO preprocessing-v2 is closed. Every program must independently validate
the bundle before reading it. Authorized NATO inputs are:

- `minimal_minmax`;
- `arpls_minmax`;
- `derivative_1`.

The 598-row strict core is primary, the 500-row quality subset is a
prespecified sensitivity analysis, and the complementary 98 rows are a
confirmatory field-quality stress cohort. Every NATO split groups by
`master_sample_id`.

Poster data are evaluated separately. The chemical-only matrix contains 275
map spectra from 4NP, benzenethiol, and pyridine after canonical label and
metal-family grouping. It has no physical-preparation identifier, so its
leave-one-substrate-family-out results are domain-transfer evidence, not
independent-preparation uncertainty.

The complete prior-work findings are in
[`SERS_BASELINE_PROVENANCE_AUDIT.md`](SERS_BASELINE_PROVENANCE_AUDIT.md).

## 3. Controls

### 3.1 Classical

For each frozen NATO representation:

- nearest centroid;
- PCA with at most 32 whitened components plus class-balanced logistic
  regression;
- class-balanced linear SVM.

Every fitted component uses training rows only.

The poster's second-derivative nearest-centroid value remains a historical
report-only reference. Second derivative is not reintroduced into NATO v2.

### 3.2 Siamese/triplet

The historical poster control is preserved exactly: two Conv1D blocks,
64-dimensional L2-normalized embedding, cross-domain-positive and
same-domain-negative triplets, margin 0.2, Adam at `1e-3`, batch 32, 100
epochs, and row-mean chemical prototypes.

The shared deterministic implementation retains those fixed settings but
replaces circular shifts with edge-filled shifts. Instrument is the NATO
triplet domain and substrate family is the poster triplet domain. These
settings are fixed controls, not selected from new outer results.

## 4. AE and DAE architecture search

Both models use the same encoder and decoder. The only difference is whether
the training input is clean or synthetically corrupted.

The bounded AE grid is:

| Factor | Values |
|---|---|
| Input view | `minimal_minmax`, `arpls_minmax` |
| Encoder channels | compact `8→16`, poster-matched `16→32` |
| Bottleneck | 16, 64 |
| Reconstruction loss | MSE, spectral composite |

This gives 16 configurations. Both encoders use kernels 7 and 5 with
twofold max pooling after each convolution. The symmetric decoder uses
deterministic nearest-neighbour upsampling followed by learned convolutions
and corrects its final length exactly to 1,401. Linear CUDA upsampling was
rejected during the pre-experiment determinism smoke test because its
backward pass is nondeterministic in the pinned PyTorch build. The output is
continuous intensity constrained to `[0,1]`.

The spectral-composite loss is:

```text
smooth-L1 + 0.1 × spectral-angle loss + 0.1 × first-difference loss
```

Training uses Adam at `1e-3`, weight decay `1e-5`, batch 64, gradient norm
limit 5, and a maximum of 100 epochs. Early stopping begins only after epoch
20, has patience 12, monitors the declared validation reconstruction loss,
and restores the best epoch.

## 5. Denoising curricula

After one AE architecture/loss is selected per intensity view, the matched
DAE compares:

1. Gaussian noise only;
2. uniform mixed corruptions;
3. progressive mixed corruptions, with severity increasing from 0.25 to 1.0.

The mixed set contains:

- scale and offset;
- a smooth low-order background;
- Gaussian noise;
- two isolated spikes;
- edge-filled spectral shifts;
- Gaussian peak broadening;
- a composite.

The full base severity is fixed in the JSON protocol. Corrupted training
inputs are generated only from training-clean rows after the fold is known.
The target is always the frozen uncorrupted representation. Validation and
test corruptions are deterministic held-out copies and never become training
examples.

## 6. Nested selection

Only NATO inner-validation folds select new model settings.

1. Evaluate all 16 AE candidates over the 20 strict-core inner folds using
   seed 1729.
2. Per input view, carry the top two eligible candidates to the corresponding
   quality-pass inner folds and select one.
3. With each selected AE architecture/loss fixed, evaluate all three DAE
   curricula on strict-core inner folds.
4. Carry at most two eligible curricula per view to quality-pass inner folds
   and select zero or one DAE.
5. Freeze the choices before any NATO outer, flagged-stress, poster, or
   domain-transfer neural result is calculated.

Candidate utilities combine chemical probe performance, reconstruction,
corruption recovery, repeatable-peak preservation, target-adjusted domain
leakage, and same-master cross-instrument distance using the declared JSON
weights.

## 7. Eligibility

An AE must preserve clean spectra and chemistry, not merely minimize pixel
error. Required gates include:

- median clean row correlation at least 0.95;
- repeatable-peak recall at least 0.90;
- chemical-probe balanced-accuracy loss no larger than 0.05 relative to the
  same input view;
- no more than 0.02 additional target-adjusted instrument leakage;
- no more than 0.02 additional same-master cross-instrument correlation
  distance;
- no more than 0.05 chemical-probe loss when moving from core to quality
  sensitivity.

A DAE is compared with its matched AE. It must preserve clean correlation,
peaks, and chemical performance within the declared tolerances and improve at
least one of:

- mixed-corruption recovery error by 5% relative;
- corrupted prediction agreement by 0.02;
- corrupted latent drift by 5% relative.

If no DAE passes, no DAE advances. If no AE passes, the least-complex AE is
retained only as a compression diagnostic and is marked ineligible for VAE
advancement.

## 8. Final evaluation

After selection is frozen, neural models use seeds 1729, 2718, and 3141.

NATO evaluation includes:

- five sealed strict-core outer folds;
- a separately fitted quality-pass sensitivity run;
- quality-development to flagged-stress transfer;
- supported leave-instrument-out and leave-sensor-family-out tests;
- domain-only and domain-and-sample variants;
- controlled corruptions at 0.5, 1.0, and 1.5 severity;
- target-adjusted instrument/sensor probes;
- same-master cross-instrument geometry;
- reconstruction and repeatable-peak fidelity.

The NATO-selected AE/DAE settings are then transferred unchanged to poster
leave-one-substrate-family-out evaluation. The silver-family and original
AgNP-source 4NP rows receive a dedicated failure analysis.

## 9. Determinism and artifacts

The new harness fixes cuBLAS workspace behavior, enables deterministic
PyTorch algorithms, disables cuDNN benchmarking and TF32, uses zero-worker
seeded loaders, and derives every run seed from its complete run identity.
Fold results must therefore be independent of execution order.

The final bundle must include:

- the frozen protocol and input hashes;
- complete run and seed assignments;
- all search histories and decisions;
- selected checkpoints;
- predictions, embeddings, and reconstructions;
- fold-, domain-, corruption-, and spectrum-level metrics;
- uncertainty summaries and localized failures;
- publication-quality figures;
- a decision registry and dataset version;
- artifact hashes and an independent validator;
- a clean exact rerun comparison.

Outer and stress results can confirm or refute the frozen decision, but may
not retroactively select another preprocessing or model configuration.
