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
- however, the best current baseline experiments do assume binary mixtures unless stated otherwise
- coefficients must be nonnegative

So the current baseline framing is:

`x ~= a_i c_i + a_j c_j`

where:

- `x` is the observed mixture spectrum
- `a_i, a_j` are candidate library atoms
- `c_i, c_j >= 0` are mixture coefficients

The exhaustive pair NNLS solver tries every pair in the library and chooses the pair with the best reconstruction.

### Important limitation of the current best model

The current best-performing model in this directory is a binary-mixture solver.

That means:

- it searches over all pairs of library compounds
- it does not know which pair is correct ahead of time
- but it does assume there are exactly `2` compounds in the mixture

So the current line of work should be interpreted as:

- a strong binary-mixture baseline
- not yet the final solution to the more general "unknown number of known-library compounds" problem

This distinction matters:

- if the real deployment problem is mostly binary mixtures, the current model is already well aligned
- if the real deployment problem may include ternary or more complex mixtures, the binary solver is only an anchor, not the end state

The current plan is therefore:

- keep strengthening the binary solver until improvements plateau
- use it as the benchmark to beat
- then relax the cardinality assumption in later experiments

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

These are optional and only worth pursuing if they outperform the best non-deep baseline on real mixtures.

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

## Immediate Next Experiment

The first implementation target is:

`binary exhaustive NNLS on the expanded reference library`

Why this goes first:

- it matches the current problem framing directly
- it does not require knowing the pair a priori
- with the current library size, exhaustive pair search is computationally cheap
- it gives a natural residual for reject / OOD logic
- it provides a serious baseline that any deep model must beat

## Current Status

- pt2 pure compounds have already been pulled into the repo
- an expanded-reference retraining experiment was completed under the old synthetic Siamese + MLP framework
- that result improved pt2 performance but still produced too many extra labels on several real mixtures
- we are now pivoting away from embedding-first classification toward sparse unmixing

## Working Rule For This Directory

Do not mix outputs from the old notebook pipeline into this directory.

Everything here should be:

- scriptable
- reproducible
- evaluated on real mixtures
- easy to compare against the baseline experiment sequence above
