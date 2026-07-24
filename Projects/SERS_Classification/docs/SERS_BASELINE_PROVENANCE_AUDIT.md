# SERS baseline provenance and reproducibility audit

Date: 2026-07-23  
Status: completed prerequisite audit for the representation-baselines goal

## 1. Scope

This audit establishes what the existing poster, classical, and Siamese
artifacts actually prove before deterministic AE and denoising-AE baselines
are added. It distinguishes:

1. the original chemical--substrate pair classifier;
2. the corrected chemical-only, leave-one-substrate-family-out analysis;
3. reproducible source artifacts versus missing model state;
4. historical model selection from the new leakage-safe nested experiment.

Nothing in this audit modifies the source poster dataset or the frozen NATO
preprocessing bundle.

## 2. Frozen NATO entry condition

The independent preprocessing-v2 validator passed at the start of this goal:

| Check | Result |
|---|---:|
| Strict labelled core | 598 |
| Quality-pass sensitivity cohort | 500 |
| Field-quality stress cohort | 98 |
| Candidate representations | 9 |
| Authorized representations | `minimal_minmax`, `arpls_minmax`, `derivative_1` |
| Fold-level preprocessing benchmarks | 495 |
| Verified v1 hashes | 21 |
| Verified v2 hashes | 32 |
| Smoothing | rejected |
| Additional alignment | rejected |

The baseline-model work must verify these hashes at every entry point and may
not alter inclusion, cohorts, splits, common axis, despiking, baseline
processing, scaling, smoothing, or alignment.

## 3. Poster dataset provenance

The current consolidated source is
`Workspace/data/processed/consolidated_SERS.csv`.

| Property | Value |
|---|---|
| SHA-256 | `a77ccf4dc389affa80bfedcaa70d8511fcf10ff4f4e4dcd9bbcb1329ce52c49b` |
| Rows | 503 |
| Spectral values per row | 1,024 |
| Native consolidated axis | 178--2445 cm⁻¹ |
| Median spacing | approximately 2.216 cm⁻¹ |
| Metadata columns | `Label`, `Substrate` |
| Exact duplicate spectral rows | 0 |
| Missing spectral values | 0 |

The dataset-construction notebook interpolated each source TXT spectrum to
`np.linspace(178, 2445, 1024)`, assigned its chemical label from the filename,
and concatenated substrate-specific files. It then created
`consolidated_SERS_avg.csv` by averaging every chemical × substrate cell.

After canonicalizing `bt -> benzenethiol`, grouping `AgNP -> Ag` and
`AuNP -> Au`, and excluding the one-domain DMF class, the chemical-only poster
dataset contains 275 spectra:

| Chemical | Ag | Au | PICO | pSERS | Total |
|---|---:|---:|---:|---:|---:|
| 4NP | 25 | 0 | 25 | 25 | 75 |
| Benzenethiol | 25 | 25 | 25 | 25 | 100 |
| Pyridine | 25 | 25 | 25 | 25 | 100 |

### Independence limitation

The consolidated file has no preparation, map, session, or physical-sample
identifier. The source tree indicates that each 25-row cell is a set of map
locations from one corresponding map/acquisition collection. Therefore:

- leave-one-substrate-family-out is useful domain-transfer evidence;
- the 25 rows in a cell must not be described as 25 independent
  preparations;
- no preparation-grouped split can be reconstructed from the consolidated
  metadata;
- uncertainty across row locations does not represent preparation-to-
  preparation uncertainty.

The NATO data, by contrast, have explicit `master_sample_id` groups and remain
the primary leakage-safe new-sample benchmark.

## 4. Original notebook result: pair classification

The original notebooks:

- construct `Class = Label + "__" + Substrate`;
- make one stratified random row split with 20% training and 80% query rows;
- train on chemical--substrate pair labels;
- use an averaged spectrum from every complete pair as a reference;
- use ALS baseline subtraction (`lambda=1e4`, `p=0.01`, 10 iterations) and
  row-L2 normalization;
- train a two-layer Conv1D encoder with contrastive loss for 100 epochs;
- classify a query by its nearest averaged pair reference.

The saved full-axis notebook reports 98.76% top-1 accuracy over 403 query
spectra. This proves that chemical--substrate pair signatures are highly
separable under a random within-pair row split. It does **not** prove
substrate-agnostic chemical classification or new-preparation
generalization.

The 330 and 400 cm⁻¹ notebook variants report 92.80% and 94.29%,
respectively. These notebook outputs have no saved checkpoint or run manifest,
and the notebooks do not set NumPy or PyTorch seeds.

## 5. Corrected poster control: chemical-only transfer

The corrected scripts predict only the canonical chemical label and use the
grouped substrate family solely as the held-out domain.

### 5.1 Fixed input and model

- Axis: 330.906--1797.919 cm⁻¹ from the consolidated grid.
- Input: SNV → Savitzky--Golay first derivative, window 17/order 3 →
  row-L2.
- Encoder:
  `Conv1d(1,16,k=7)` → ReLU → max-pool →
  `Conv1d(16,32,k=5)` → ReLU → max-pool →
  flatten → dense 64 → ReLU → row-L2.
- Objective: triplet margin loss, margin 0.2.
- Positive: same chemical on another substrate preferred.
- Negative: different chemical on the anchor substrate preferred.
- Training: Adam, learning rate `1e-3`, batch 32, 100 epochs.
- Augmentation: additive Gaussian noise with standard deviation 0.01 and
  circular `np.roll` shift from -2 to +2 spectral indices.
- Inference: nearest row-mean chemical prototype in the embedding.
- Historical seed argument: 42.

The saved result reports:

| Held-out family | Balanced accuracy | Standard macro F1 |
|---|---:|---:|
| Ag | 0.920 | 0.919 |
| Au | 0.980 | 0.660 |
| PICO | 1.000 | 1.000 |
| pSERS | 1.000 | 1.000 |
| Unweighted fold mean | 0.975 | 0.895 |

The low Au macro F1 despite 0.98 balanced accuracy is not a contradiction.
One Au benzenethiol row is predicted as absent class 4NP; scikit-learn's
default macro F1 includes the union of true and predicted labels, whereas
balanced accuracy averages recall over the two true Au classes. Future
reports must store both supported-true-class macro F1 and union-label macro
F1 explicitly.

### 5.2 Historical selection limitation

The saved 0.975 configuration was chosen from an 18-cell sweep:

- three feature representations;
- three metric losses;
- two prototype definitions;
- one seed.

Every candidate was ranked using the same four held-out substrate-family
folds later reported as the result. There was no nested development layer and
no independent outer domain. Therefore 0.975 is a useful historical
benchmark, but it is optimistically selected and must not be presented as a
new sealed estimate.

The new experiment will:

- preserve this exact configuration as a fixed historical control;
- select new AE/DAE settings only in inner development folds;
- use the NATO grouped master-sample folds for sealed new-sample estimates;
- avoid choosing any method from the 98-row NATO stress cohort.

## 6. Classical control

The existing classical script is deterministic and was reproduced exactly
under the current environment. Its best historical poster result is:

| Representation | Classifier | Mean balanced accuracy | Mean macro F1 |
|---|---|---:|---:|
| second derivative | nearest centroid | 0.987 | 0.987 |
| first derivative | nearest centroid | 0.973 | 0.973 |

The second derivative is retained only as a poster-specific historical
reference. It is not an authorized NATO preprocessing-v2 input. NATO model
comparisons are restricted to the three frozen v2 representations.

## 7. Checkpoint and run-state audit

No `.pt`, `.pth`, `.ckpt`, `.keras`, `.h5`, `.joblib`, or `.pkl` checkpoint
was found anywhere under `Workspace/substrate_agnostic`.

The available evidence consists of:

- training/evaluation scripts;
- result and confusion CSV files;
- saved notebook outputs;
- geometry tables and figures;
- one K-shot JSON configuration;
- a poster asset manifest with GPU and selected training settings.

Consequences:

- the historical encoder weights cannot be reloaded;
- historical embeddings cannot be independently traced to a checkpoint;
- the saved CSVs can be verified as artifacts but not regenerated from the
  exact original model state;
- the new baseline bundle must save checkpoints, configs, histories,
  predictions, latents, input hashes, and environment information together.

## 8. Same-seed reproducibility test

The historical fixed Siamese command was executed twice under the current
environment:

```text
PyTorch 2.11.0+cu130
NVIDIA GeForce RTX 5080
feature=derivative_1
loss=triplet
prototype=row_mean
margin=0.2
epochs=100
seed=42
```

Results:

| Artifact/run | Mean balanced accuracy | Ag | Au | PICO | pSERS |
|---|---:|---:|---:|---:|---:|
| Saved historical CSV | 0.975 | 0.920 | 0.980 | 1.000 | 1.000 |
| Fresh rerun 1 | 0.962 | 0.920 | 0.980 | 0.947 | 1.000 |
| Fresh rerun 2 | 0.972 | 0.907 | 0.980 | 1.000 | 1.000 |

The two fresh seed-42 result frames are not equal. The existing `set_seed`
function seeds Python, NumPy, and PyTorch, but it does not enable deterministic
algorithms, fix cuBLAS workspace behavior, disable cuDNN benchmarking/TF32, or
give every fold its own derived seed. It also seeds once before all folds, so
results depend on fold execution order.

The historical value is therefore not exactly reproducible from seed alone.
The new harness must:

- set `CUBLAS_WORKSPACE_CONFIG` before CUDA initialization;
- enable PyTorch deterministic algorithms;
- disable cuDNN benchmarking and TF32;
- use explicit per-run and per-fold derived seeds;
- use a seeded `DataLoader` generator with zero workers;
- save the complete environment and checkpoint;
- perform an independent rerun comparison on predictions, histories,
  embeddings, reconstructions, and model state.

## 9. Additional methodological issues to correct

1. Historical `np.roll` augmentation wraps the spectrum edge. The new
   corruption code must use edge filling and must never wrap 1800 cm⁻¹ content
   to the low-frequency edge.
2. The historical Siamese script records only final training loss. The new
   harness needs full epoch histories and inner-validation selection.
3. The historical script seeds once before the fold loop. New folds need
   caller-stable, order-independent seeds.
4. The poster sweep uses one training seed. New neural comparisons require
   multiple declared seeds.
5. The historical classifier has no clean reconstruction constraint. Its
   silver-family 4NP confusion remains the localized metric-learning failure
   case for AE/DAE comparisons.
6. Projection plots are qualitative. Claims must be based on held-out
   classification, probes, same-master geometry, fidelity, and corruption
   metrics.
7. Poster and NATO axes and independence structures differ. They must not be
   pooled into one training matrix or one headline accuracy.

## 10. Immutable source hashes

| Artifact | SHA-256 |
|---|---|
| `scripts/sers_siamese_substrate_agnostic.py` | `d13e0b0ae37b079dc9f184733a85fb5487add621955f432e6d4bb26b59fddc08` |
| `scripts/sers_substrate_agnostic_detection.py` | `04319fe40adf2ae5f0f82edb112be166424b0caaa86ce698cfae3572b31b690f` |
| `scripts/run_siamese_sers_sweep.py` | `63023ce009a718966b042d8d6c1345cc58abac9cc1e08273872c43001ea92753` |
| Original full-axis notebook | `b180246bdb2d25aac6e7ad3951adba4dad6ae1295c6f46961a02b5fe8accfc6c` |
| 330 cm⁻¹ notebook | `ecd0e1ccdca9ffb661f2693e3895be7d2be5a2d49d4a68e7194d22995d59b937` |
| 400 cm⁻¹ notebook | `42559d7791cca8cf70a4b7cf5057a073774b017b7a74ffb9edd486f03fb5db2e` |
| Saved grouped Siamese result | `2151314a816f02451b29af24bd1a325d45983878a126ba80504480314d608ebb` |
| Saved grouped classical result | `b5c2d35807dad96ec98c8ce4afd5f1a5b1e1a15dfd6525ac2d82ee9fbb41510e` |
| Poster averaged reference CSV | `7e987618f5a2c77d87dae7552c0c69046a8b3b0cd98c5a776a458f755447847e` |

These hashes identify the audited historical state. The new harness will
copy or reference them without overwriting the legacy artifacts.

## 11. Audit decision

The prior work is sufficient to define faithful historical controls, but not
to serve as the final reproducible baseline bundle.

The new experiment must preserve:

- the chemical-only grouped poster interpretation;
- the exact first-derivative Siamese control;
- nearest-prototype inference and the silver-family 4NP failure audit;
- the deterministic classical results.

It must add:

- strict run-level determinism;
- nested selection for all new choices;
- NATO master-sample grouping;
- matched AE/DAE encoders;
- clean and corrupted reconstruction evaluation;
- target-adjusted domain probes and same-master geometry;
- saved checkpoints and exact replayable artifacts.
