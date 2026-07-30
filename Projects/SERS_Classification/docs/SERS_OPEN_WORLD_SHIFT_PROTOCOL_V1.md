# NATO SERS open-world and field-shift protocol v1

## Registered question

Can a model classify supported chemicals, reject a chemical absent from
training, and abstain on low-quality or unfamiliar-domain spectra when every
split is independent at the physical-master level?

This protocol is frozen before inspecting any open-set score on a true held
chemical. It does not reopen preprocessing decisions or the locked outcomes of
earlier experiments.

## Independent units

- `master_sample_id` is the indivisible sample unit.
- Held chemical is the independent unit for open-set conclusions.
- Held instrument or sensor family is the independent unit for domain
  conclusions.
- Model seeds are technical repeats and must be averaged before uncertainty
  intervals are calculated.

The frozen population remains 598 strict-core SERS spectra from 69 masters:
500 quality-pass spectra and 98 field-quality-stress spectra.

## Six true-unknown tasks

Blank remains a known operational class. Each nonblank chemical is held out in
turn:

1. `4_ANPP`
2. `4_nitrophenol`
3. `acetaminophen`
4. `benzyl_fentanyl`
5. `ethanol`
6. `ethyl_paraoxon`

For held chemical \(H\) and known outer fold \(O\):

- known development: quality-pass, analyte not \(H\), fold not \(O\);
- known test: quality-pass, analyte not \(H\), fold \(O\);
- unknown quality: quality-pass, analyte \(H\);
- known stress: field stress, analyte not \(H\);
- unknown stress: field stress, analyte \(H\).

The held chemical is never a training, calibration, threshold, preprocessing,
augmentation, score-selection, or model-selection example. Reusing the held
chemical test rows under five independently trained known-fold models is a
technical model repeat; inference is made over the six held chemicals.

## Known-only calibration

Within each \(H,O\) task, the four development folds are cross-fitted. One fold
at a time supplies known calibration predictions and the other three train the
model. Temperatures and rejection thresholds use only these cross-fitted known
scores.

Thresholds are empirical score quantiles corresponding to retained-known
coverages 95%, 90%, 80%, 70%, and 50%, using the declared `higher` quantile
method. Neither true unknowns nor stress spectra select a threshold.

## Surrogate-unknown selection

Open-set scores and open-set objectives cannot be selected on the true held
chemical. Each inner development task therefore removes a second nonblank
chemical \(S\), distinct from \(H\), as a surrogate unknown:

- train: quality-pass folds other than \(O,I\), analyte neither \(H\) nor \(S\);
- known validation: fold \(I\), analyte neither \(H\) nor \(S\);
- surrogate-unknown validation: fold \(I\), analyte \(S\);
- all other \(S\) rows remain excluded from that fit.

Blank cannot be a surrogate unknown. Candidates are ranked by master-aware
surrogate OSCR-AUC, then known balanced accuracy, then unknown AUROC, and
finally declared candidate order. The entire true-held-\(H\) population remains
locked throughout.

## Score directions

All registered anomaly scores are oriented so larger means more unknown:

- one minus calibrated maximum probability;
- predictive entropy;
- neural energy from frozen raw logits;
- minimum class-conditional Mahalanobis distance;
- negative log latent density;
- reconstruction error;
- MC-dropout or ensemble mutual information.

Energy is not reconstructed from normalized probabilities because logits are
identifiable only up to an additive constant. Scores are used only where their
required raw outputs exist.

## Field quality and unknown chemistry

Quality failure and chemical novelty are evaluated separately and jointly:

- known quality;
- unknown quality;
- known stress;
- unknown stress.

This four-cell design determines whether a score detects unseen chemistry,
acquisition failure, or only their mixture. Stress spectra remain locked from
all selection.

## Domain evaluation

The preprocessing-v2 `domain_only` and `domain_and_sample` instrument and
sensor-family partitions are copied byte-for-byte. The latter removes every
training spectrum sharing a master with the held domain and is the stronger
generalization test.

Test analytes absent from domain training remain in row-level predictions.
They are excluded only from supported-class balanced accuracy and macro-F1.
Zero-supported-class scenarios remain explicit missing outcomes, not successes.

Domain generalization receives no held-domain observations during fitting.
Any later unsupervised domain-adaptation experiment must be separately labelled
and may use only explicitly declared unlabeled target-domain spectra.

## Claim boundary

The study can support classification, rejection, calibration, selective
prediction, and operational robustness claims. It cannot establish that a
model reconstructed a chemical-only spectrum, removed a physical substrate or
instrument response, denoised against an unobserved clean target, or learned
semantic chemical/nuisance factors.
