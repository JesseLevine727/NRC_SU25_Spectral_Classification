# Raman Mixture Classification Pivot

## Why This Directory Exists

This directory is a clean restart for Raman mixture classification under the correct problem framing:

- Task: identify mixtures of known library compounds
- Primary objective: recover compound support from a reference library
- Secondary objective: estimate relative contribution / confidence
- Important constraint: do not assume the exact mixture pair in advance

The previous Siamese + FFT + MLP pipeline was useful as an exploratory representation-learning path, but it is no longer the main direction.

## What We Learned From The Previous Pipeline

### Original pipeline

1. Start from `reference_v2.csv`
2. Generate synthetic pairwise mixtures from reference single-compound spectra
3. Train Raman and FFT Siamese towers on synthetic mixtures
4. Embed synthetic mixtures
5. Train an MLP on synthetic embeddings for multilabel presence prediction
6. Apply the model to real mixtures with thresholding / calibration

### Main conclusions

- The pipeline is closed-set with respect to chemical identity
- It can handle unknown combinations of known compounds better than unknown compounds
- When new compounds were not added to the reference, failure was expected because the model had no output class for them
- After adding pt2 pure compounds to the reference and retraining, performance improved materially on pt2 real mixtures
- The remaining dominant failure mode was extra false labels, not total failure to detect the new compounds
- This suggests the system is not aligned with the true task as cleanly as a sparse unmixing model would be

### Why the previous framing is weak

- The Siamese objective learns mixture-pair identity geometry, not necessarily additive chemistry
- The MLP is trained mostly on synthetic mixtures, so synthetic-to-real domain gap is baked in
- Thresholding is compensating for a representation / classifier mismatch
- There is no explicit unknown / reject mechanism

## New Framing

We will treat mixture classification as sparse library matching / unmixing:

`x ~= A c + B b + e`

- `x`: observed mixture spectrum
- `A`: library / dictionary of known pure spectra
- `c >= 0`: nonnegative chemical coefficients
- `B b`: optional nuisance terms such as baseline or background atoms
- `e`: residual

This is a better match to the scientific task than pair classification in embedding space.

Cardinality should be framed carefully:

- in principle, we do not assume we know the number of components in advance
- in practice, the current dataset consists of binary mixtures
- so the main inference rule can still be binary-constrained if that gives the best real-data performance

That means the scientific framing is "unknown support over a known library," while the current operational framing is "use a strong binary-support solver because the observed mixtures are binary."

## Data Layout

The active classical pipeline is intended to be self-contained inside `Unmixing_Pipeline/`.

Runtime data now lives under:

- `Data/reference/reference_v2.csv`
- `Data/reference/mixtures_dataset.csv`
- `Data/pt2/Mixtures.txt`
- `Data/pt2/<compound-or-mixture>/txt/*.txt`

The active scripts should use the shared loaders in `Scripts/unmixing_common.py` rather than reaching into `Notebooks/` or sibling dataset folders directly.

## Problem Definition

### Current task definition

The task in this directory is:

- input: a Raman spectrum from a mixture
- library: a reference set of known pure-compound spectra
- output: which library compounds are present in the mixture

Important assumptions:

- the exact mixture composition is not known in advance
- the true compounds are assumed to come from the library
- the broader intended problem does not assume we know the number of mixture components at inference time
- however, for the current dataset, binary-constrained inference is a deliberate and reasonable modeling choice because all labeled mixtures are binary
- coefficients must be nonnegative

So the current baseline framing is:

`x ~= a_i c_i + a_j c_j`

where:

- `x` is the observed mixture spectrum
- `a_i, a_j` are candidate library atoms
- `c_i, c_j >= 0` are mixture coefficients

The exhaustive pair NNLS solver tries every pair in the library and chooses the pair with the best reconstruction.

### Binary Assumption At Inference

The current best-performing model in this directory is a binary-mixture solver.

That means:

- it searches over all pairs of library compounds
- it does not know which pair is correct ahead of time
- but it does assume there are exactly `2` compounds in the mixture

This should be interpreted carefully:

- scientifically, it is not the most general formulation because the number of components is not assumed known a priori
- operationally, it is well matched to the current data because all available real mixtures are binary
- methodologically, it is the benchmark to beat before adding more flexible cardinality handling

This distinction matters:

- if the deployment problem remains binary mixtures of known compounds, the current solver may already be the right inference rule
- if future data includes ternary or more complex mixtures, the binary solver becomes an anchor rather than the final model

The current plan is therefore:

- keep the binary solver as the main benchmark and likely deployment baseline for the current dataset
- continue testing whether relaxing the cardinality assumption actually improves real-data performance
- only promote an open-cardinality solver if it beats the binary-constrained model on the mixtures that matter

### Benchmark Policy

This directory needs a clean distinction between:

- benchmark methods:
  - general methods intended to stand on their own without pair- or family-specific overrides
- diagnostic variants:
  - localized engineering rules used to probe whether the remaining errors are structural or just decision-boundary artifacts

For scientific comparison, the primary benchmark should remain a general method.

In the current repo state, that means:

- primary clean benchmark:
  - replicate-aware binary pair NNLS with one constant nuisance baseline atom
- diagnostic engineering variants:
  - selective low-baseline fallback
  - family-specific near-tie fallback for `1-dodecanethiol + meoh`

Those diagnostic variants are useful because they show the remaining error surface is highly localized.
They should not be treated as the main scientific method unless we explicitly decide that deployment-specific engineering is more important than methodological cleanliness.

### What is in the reference set

There are two reference definitions that matter.

Original reference:

- source: `Notebooks/reference_v2.csv`
- pure-compound library only
- 12 compounds:
  - `1,9-nonanedithiol`
  - `1-dodecanethiol`
  - `1-undecanethiol`
  - `6-mercapto-1-hexanol`
  - `benzene`
  - `benzenethiol`
  - `dmmp`
  - `etoh`
  - `meoh`
  - `n,n-dimethylformamide`
  - `pyridine`
  - `tris(2-ethylhexyl) phosphate`

Expanded reference used in the new unmixing experiments:

- built from the original reference plus pt2 pure spectra
- saved in experiment result directories as `reference_v2_plus_pt2.csv`
- 17 compounds total
- added pt2 compounds:
  - `acetonitrile`
  - `dichloromethane`
  - `diethylamine`
  - `n-hexane`
  - `toluene`

Important distinction:

- the reference contains pure spectra only
- the mixture datasets are evaluated against the reference
- the solver is free to search across all reference compounds
- the solver does not know the correct pair ahead of time

### Closed-set versus open-set

In the current unmixing track, the task is:

- closed-set over compound identity
- open over which combination of known compounds appears

This means:

- if a compound is not in the reference, the solver cannot identify it correctly
- if a compound is in the reference, the solver can search for it without knowing which pair is present beforehand

This is the scientifically correct framing for the current experiments.

## Core Principles For The New Track

- Start with the strongest non-deep baseline first
- Preserve linearity where possible; do not let preprocessing destroy the unmixing assumption
- Use real held-out mixtures, not just synthetic held-out mixtures, to decide whether a method is good
- Track false positives aggressively
- Add explicit reject / OOD logic once residuals are available
- Only move to deep models if they beat a strong sparse baseline on real data

## Experiment Order

### Phase 0: Infrastructure

1. Build a clean data-loading layer for:
   - original reference
   - expanded reference including pt2 pure compounds
   - real mixture datasets
   - pt2 mixture metadata
2. Standardize interpolation onto a common wavenumber axis
3. Record all experiment outputs in this directory only

### Phase 1: Strong non-deep baselines

1. Exhaustive 2-sparse NNLS over all library pairs
   - assume binary mixtures
   - solve NNLS for every candidate pair
   - rank pairs by residual
   - return top pair and residual
2. Exhaustive 3-sparse extension if needed
3. Non-negative elastic net over the full library
   - encourage sparse coefficient vectors
   - compare support recovery vs NNLS
4. Add residual-based reject / abstain rule
   - high residual means poor library fit or unknown chemistry

### Phase 2: Better physical modeling

1. Add nuisance baseline atoms / polynomial baseline terms
2. Compare:
   - raw interpolated spectra
   - baseline-corrected spectra
   - joint model with baseline atoms
3. Test whether preserving more of the raw intensity structure improves unmixing

### Phase 3: Better library usage

1. Use replicate spectra directly as dictionary atoms rather than class averages only
2. Aggregate atom-level solutions back to compound-level predictions
3. If needed, add group sparsity by compound

### Phase 4: Deep models only if justified

1. Learned coefficient regressor with nonnegative outputs
2. Deep unfolding / learned sparse solver
3. Library-constrained autoencoder
4. Denoiser + classical unmixing

These are only worth pursuing if they outperform the best non-deep baseline on real mixtures.

Deep work has now started in this directory, but the framing remains constrained:

- keep the classical binary anchor frozen as the main non-deep benchmark
- do not revive the old Siamese pair-identity pipeline as the primary path
- prefer library-constrained deep models that predict compound coefficients or support directly
- evaluate with binary top-2 support recovery as the main operational metric because the current labeled mixtures are binary

## Classical Exit Criteria

Deep methods should not be explored just because they are available. They become justified only after the classical track has been pushed to a credible ceiling.

For this directory, the classical track should be considered mature enough to justify a deep pivot only when most of the following are true:

- the binary replicate-aware NNLS anchor has been finalized and calibrated as the main operational model
- binary-first inference with reject / abstain logic has been evaluated on both real-mixture sets and both pure-spectrum sets
- additional classical changes stop improving real-mixture exact support recovery in a meaningful way
- remaining errors look structural rather than threshold- or dictionary-quality-related
- there is a concrete reason to expect a learned model to help:
  - stronger domain shift
  - nonlinear background or interference
  - larger and more variable training data
  - future non-binary mixtures that make the binary solver insufficient

Operational decision rule:

- if the binary-constrained classical model remains the best real-data method, keep it as the benchmark and likely deployment baseline
- if a more flexible classical model beats it cleanly, promote that classical model first
- only move to deep methods once the best classical model has clearly plateaued and the expected gain is no longer from better calibration, better support selection, or better dictionary construction

Current decision:

- the clean classical benchmark is already very strong
- the remaining improvements after that benchmark came mostly from localized fallback logic
- so this is the point where a deep pivot becomes reasonable if we want a more general method than those localized engineering fixes

Current deep status:

- first deep baseline implemented:
  - `Scripts/run_deep_binary_coefficient_regressor.py`
- best result so far:
  - deep MLP coefficient-regressor family, `baseline_corrected`
  - existing real mixtures: exact `~0.950` to `0.953`
  - pt2 real mixtures: exact `1.000`
- interpretation:
  - this already beats the frozen clean classical benchmark
  - it does not beat the localized diagnostic classical ceiling
  - the remaining deep errors are still concentrated in the `1-dodecanethiol + meoh` family
- follow-up general variants tried:
  - `Scripts/run_deep_binary_variant_suite.py`
  - tested:
    - `cnn_encoder`
    - `replicate_decoder`
  - neither beat the first deep baseline on `existing_real`
  - both stayed perfect on `pt2_real`
  - both worsened the `1-dodecanethiol` versus `1-undecanethiol` confusion
- follow-up generic supervision experiment:
  - `Scripts/run_deep_similarity_supervision.py`
  - slightly improved the clean deep result:
    - existing real mixtures: exact `0.952`
    - pt2 real mixtures: exact `1.000`
  - did so without changing the inference assumptions
- follow-up global hybrid experiment:
  - `Scripts/run_deep_hybrid_pair_rerank.py`
  - combined deep compound shares with the frozen clean pair-NNLS residual using one global fusion weight
  - did not improve beyond its own deep backbone
  - so a simple global fusion rule is not enough by itself

## Evaluation Rules

Every experiment should report at least:

- exact support match
- micro precision / recall / F1
- average number of predicted labels
- false positives per sample
- residual reconstruction error
- per-mixture breakdown

Recommended dataset views:

- original real mixtures
- pt2 real mixtures
- mixtures containing only original-library compounds
- mixtures containing newly added compounds
- samples expected to fail library matching cleanly

## Current Benchmark State

The benchmark hierarchy should now be read as:

- strongest clean deep method so far:
  - baseline-corrected deep MLP coefficient-regressor family
- strongest clean classical method:
  - baseline-corrected replicate-aware binary pair NNLS with one constant nuisance baseline atom
- strongest engineering ceiling:
  - the localized fallback classical variants

This separation matters:

- if the goal is a clean learned method, the deep coefficient regressor is now the model to iterate on
- if the goal is a clean non-deep method, the replicate-aware binary NNLS anchor remains the classical reference
- if the goal is maximum accuracy on the current datasets regardless of hand-tuned local logic, the diagnostic fallback variants still win

## Immediate Next Deep Experiments

The next deep experiments should stay library-constrained:

1. Stress-test the deep MLP coefficient-regressor family across multiple random seeds to separate real gain from ordinary run-to-run variance
2. Continue improving generic supervision around chemically similar compounds rather than changing encoder type alone
3. Revisit hybrid inference only if it uses richer uncertainty or candidate-structure information than one global fusion scalar

## Current Status

- pt2 pure compounds have already been pulled into the repo
- the clean classical benchmark has been frozen and pushed
- several new deep experiments have now been run inside `Unmixing_Pipeline`
- the old synthetic Siamese + MLP framework is no longer the primary direction
- the active comparison is now:
  - clean deep coefficient-regressor family
  - clean classical binary NNLS anchor
  - diagnostic classical ceiling

## Working Rule For This Directory

Do not mix outputs from the old notebook pipeline into this directory.

Everything here should be:

- scriptable
- reproducible
- evaluated on real mixtures
- easy to compare against the baseline experiment sequence above
