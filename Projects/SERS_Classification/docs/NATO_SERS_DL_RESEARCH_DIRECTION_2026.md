# NATO SERS: defensible deep-learning research direction

Date: 2026-07-30

## Executive decision

The strongest use of this archive is **open-world, shift-aware chemical
identification under real SERS field conditions**:

> Can a model correctly classify supported chemicals, reject a chemical class
> absent from training, and abstain on unreliable field spectra or unfamiliar
> acquisition domains?

This is better matched to the data than another chemical/nuisance
disentanglement search. It uses the archive's most unusual assets: 69 physical
master samples observed repeatedly, 10 instruments, 4 SERS sensor families,
and a natural 98-spectrum low-quality field cohort.

The primary contribution should be a rigorous problem formulation and
master-group/domain evaluation, not a claim that one more neural architecture
has “removed noise.”

## What the NATO archive supports

- 598 SERS spectra from 69 physical master samples.
- Seven targets when blank is treated as an operational class: six chemical
  analytes plus blank.
- Ten instruments and four sensor families.
- 500 quality-pass spectra and 98 naturally occurring field-quality-stress
  spectra.
- Frozen 400–1800 cm⁻¹ common axis and three justified representations:
  arPLS/min–max, minimal/min–max, and first derivative.
- Sixty-seven of 69 masters were measured on more than one instrument. This
  supports same-master robustness and leakage-safe grouping.

This enables:

1. grouped closed-set chemical classification;
2. leave-one-chemical-out open-set evaluation;
3. held-instrument and held-sensor transfer;
4. selective prediction and quality/stress rejection;
5. same-master consistency analysis across instruments;
6. small-data comparisons between classical and compact deep methods.

## Why physical D-VAE disentanglement is not identifiable here

The archive contains measured analyte-on-SERS-system spectra and blanks, but
not the target variables needed to identify a physical additive decomposition.
In particular, there is no paired chemical-only reference for each field
measurement, no clean/noisy pair, no isolated instrument response, and no
complete preparation/batch identifier.

The design is also incomplete:

- only 44 of 70 possible analyte × instrument cells are occupied;
- only 17 of 28 analyte × sensor-family cells are occupied;
- analyte and sensor are associated (Cramér's V = 0.542);
- field stress is overwhelmingly associated with Pendar systems.

Consequently, chemical identity, sensor, instrument, preparation, and quality
do not cross independently. A decoder can place analyte information in either
named latent block and still reconstruct the observations. A low instrument
probe can also be produced by deleting useful analyte signal when instrument
and analyte are correlated.

This is not only a local implementation issue. Unsupervised disentanglement is
not identifiable without assumptions or supervision:
<https://proceedings.mlr.press/v97/locatello19a>.

The completed NATO structured-VAE study empirically confirms the limitation:

- chemical/nuisance maximum canonical correlation was 0.994;
- the nuisance block alone classified analytes at BA 0.586;
- locked chemical BA was 0.681/0.728/0.327 for strict/quality/stress;
- the union latent, not the chemical block, retained most predictive utility;
- instrument information remained predictable.

### What a VAE can still do

A VAE remains legitimate as:

- a reconstruction-capable representation baseline;
- an anomaly score using reconstruction and latent-density features;
- a synthetic-corruption inversion model, provided outputs are described as
  augmentation robustness rather than true clean spectra;
- a pretrained encoder comparator for downstream classification.

It should not be described as recovering “chemical-only signal” from this
archive. Returning to that claim requires balanced factor crossings and
physical or operational targets for what clean chemistry and nuisance mean.

## Why open-world and shift-aware learning is current and relevant

Recent Raman/SERS work is moving toward the exact operational weaknesses
visible here:

- Open-set Raman methods address the otherwise uncontrolled false positives
  produced when an unknown substance is forced into a known class:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9728505/>.
- A 2026 SERS study jointly considers baseline/noise correction and rejection
  of unknown substances:
  <https://opg.optica.org/ao/abstract.cfm?uri=ao-65-18-6135>.
- Monte-Carlo-dropout Raman classification has shown the value of withholding
  decisions on uncertain spectra:
  <https://pubmed.ncbi.nlm.nih.gov/39580162/>.
- Raman domain-adaptation work now explicitly targets instrument, batch, and
  sample shifts:
  <https://pubmed.ncbi.nlm.nih.gov/41842761/>.
- Parameter-efficient calibration transfer has been demonstrated across Raman
  spectrometers:
  <https://pubmed.ncbi.nlm.nih.gov/40922652/>.
- Masked/self-supervised Raman encoders are increasingly used when labelled
  spectra are scarce:
  <https://pubs.acs.org/doi/10.1021/acs.analchem.5c05656> and
  <https://www.nature.com/articles/s41598-024-56788-7>.
- A larger pretrained spectral component model reports zero-shot denoising and
  SERS background analysis across multiple datasets:
  <https://www.nature.com/articles/s42256-025-01027-5>.

The NATO archive should not try to reproduce the scale of a foundation model
with 598 spectra. It can provide a hard, realistic field-shift test bed for
compact or pretrained models.

## Recommended study

### Working title

**Open-world, shift-aware SERS identification across field instruments and
sensor families**

### Core hypotheses

1. A model selected with physical-master grouping will perform materially worse
   than a row-random split, revealing the true generalization problem.
2. Uncertainty/open-set objectives will reduce confident false identification
   of unseen chemicals relative to maximum-softmax prediction.
3. Field-quality stress and unseen chemical identity are different failure
   modes; a useful system needs both class novelty and acquisition-quality
   rejection.
4. Instrument-aware adaptation or robustness objectives may improve held-domain
   classification, but apparent invariance will not count if chemical
   discrimination deteriorates.
5. Compact models and classical baselines may match or beat large deep models
   at this sample size; the DL contribution must therefore be uncertainty,
   transfer, or robustness rather than capacity alone.

## Experimental protocol

### 1. Immutable data boundary

- Keep the 598-row SERS core and all preprocessing outputs frozen.
- Group every split by `master_sample_id`.
- Never select a model using the 98 field-stress spectra.
- Report row counts and independent master counts together.
- Report occupied analyte × domain support for every held-domain result.

### 2. Tasks

#### Task A — known-class classification

Use the existing five master-group outer folds. Select preprocessing and model
hyperparameters only in grouped inner folds. Report strict, quality-pass, and
locked stress results.

#### Task B — unseen-chemical rejection

Run six leave-one-chemical-out experiments, retaining blank as a known
operational class. In each experiment:

- remove one nonblank chemical completely from training and calibration;
- split the remaining known classes by physical master;
- test classification among known classes and rejection of the held chemical;
- choose thresholds without examples of the held chemical.

Average at the held-chemical level, not merely across spectra.

#### Task C — field-quality rejection

Fit only on quality-pass development spectra. Treat the 98 stress spectra as a
locked out-of-distribution cohort. Evaluate whether uncertainty distinguishes
quality pass from stress, then evaluate chemical classification at fixed
coverage.

#### Task D — held-domain transfer

Use both frozen protocols:

- `domain_only`: hold out an instrument/sensor but allow the same physical
  master on another domain;
- `domain_and_sample`: also remove all spectra from masters represented in the
  held domain.

The second is the stronger generalization claim. Unsupported analytes must
remain visible in predictions but be excluded from supported-class BA.

### 3. Mandatory models

#### Classical controls

- PCA/logistic regression;
- linear and RBF SVM;
- random forest;
- optionally PLS-DA as a conventional chemometric comparator.

#### Deep controls

- compact 1-D CNN with cross-entropy;
- existing Siamese control;
- existing supervised-contrastive successor;
- standard mixed VAE followed by the same frozen classifier;
- structured VAE as a negative mechanistic comparator, not as a rescued model.

#### Research models

Start with one compact 1-D residual encoder. Compare:

1. ordinary empirical-risk minimization;
2. supervised contrastive pretraining;
3. energy-based or Objectosphere-style open-set training;
4. MC dropout and a small deep ensemble for epistemic uncertainty;
5. one domain-robust objective such as CORAL/MMD or GroupDRO;
6. a separately labelled unsupervised-domain-adaptation protocol only when
   unlabeled target-instrument spectra are explicitly allowed.

Do not combine all mechanisms first. Select each mechanism separately, then
combine only mechanisms that pass their registered gate.

### 4. Pretraining

Training a transformer or masked autoencoder from scratch on 598 spectra is
not a persuasive foundation-model experiment. Two defensible options are:

- compact masked reconstruction as an ablation, with no claim of learning a
  universal spectral model;
- pretraining on public Raman/SERS spectra, followed by parameter-efficient
  NATO fine-tuning and a from-scratch control.

External pretraining must be checked for overlapping chemicals and acquisition
conditions. It must not import test labels or tune on NATO stress/domain
outcomes.

### 5. Metrics

Closed set:

- balanced accuracy, macro F1, accuracy;
- per-class recall and confusion matrices;
- NLL, multiclass Brier score, and ECE.

Open set and stress rejection:

- AUROC and AUPRC;
- FPR at 95% true-positive rate;
- open-set classification rate/OSCR;
- known-class BA at fixed unknown-rejection rates;
- risk–coverage and accuracy–coverage curves.

Domain robustness:

- mean and worst held-domain BA;
- supported/unsupported rows and classes;
- same-master prediction agreement across instruments;
- instrument/sensor probe accuracy conditional on analyte.

Uncertainty intervals must resample physical masters or use outer folds as the
independent units. Three neural seeds are technical repeats, not three new
datasets.

### 6. Essential ablations and controls

- all three frozen representations;
- primary common preprocessing versus minimal-preprocessing sensitivity;
- with/without corruption augmentation;
- with/without uncertainty/open-set objective;
- with/without domain objective;
- row weighting versus inverse-master-frequency weighting;
- grouped master-label permutation;
- seen versus unseen instrument and seen versus unseen master;
- analyte-domain support and label-domain association audit.

No preprocessing should be chosen separately for an instrument using its
locked classification result. Instrument-specific operations require a
prespecified physical calibration rule.

## Publication claim boundary

The archive can support:

> A rigorously grouped evaluation of chemical identification, rejection, and
> cross-domain reliability under heterogeneous SERS field acquisition.

It cannot by itself support:

> Recovery of a unique chemical-only Raman spectrum, causal removal of
> substrate/instrument effects, or semantic chemical/nuisance disentanglement.

The scientifically useful result may be that a simple model classifies
quality-pass spectra well while every method fails on particular instruments,
unknown chemicals, or stress spectra. Mapping those limits reliably is a
stronger field-deployment contribution than reporting an optimistic
row-random accuracy.

## Immediate next experiment after the RF addendum

Implement the open-world benchmark shell before training a new neural model:

1. freeze six leave-one-chemical-out partitions;
2. freeze master-group calibration/validation splits for known classes;
3. define unknown and stress thresholds using known development data only;
4. run PCA/logistic, RBF SVM, RF, existing contrastive encoder, and standard
   VAE anomaly scores;
5. quantify which failure mode remains after these controls;
6. only then train the compact open-set 1-D CNN and domain-robust ablations.

That sequence determines whether DL adds genuine rejection/transfer value
rather than merely another closed-set score.
