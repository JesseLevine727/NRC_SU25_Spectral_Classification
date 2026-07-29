# Supervised-contrastive SERS protocol v1

Date declared: 2026-07-29  
Status: frozen before the expanded classical outer results and before any
successor-model fit  
Machine-readable protocol:
[`configs/sers_supervised_contrastive_v1.json`](../configs/sers_supervised_contrastive_v1.json)

Protocol history: the architecture, objectives, search space, seeds,
ablations, and broad gates were declared before the expanded classical outer
results. Before any registered successor fit, selection was restricted to
the 500 quality-pass rows, and the rejection rule and two primary shift
endpoints were made exact using only the already audited cohort support
structure. No successor result existed at that clarification point.

## Scientific question

Can class-preserving supervised contrastive learning improve chemical
classification and calibrated abstention under new samples, instruments,
sensor families, and field-quality shift relative to both classical methods
and the existing triplet Siamese model?

The proposed representation is described as domain robust. It is not a
physically denoised Raman spectrum and is not evidence of chemical/nuisance
disentanglement.

## Why this is a controlled Siamese successor

The primary objective comparison reuses the exact legacy two-block
convolutional encoder. Cross-entropy, supervised contrastive, combined, and
domain-aware losses therefore differ in training objective while capacity
remains fixed. A separate adaptive-pooling compact encoder is evaluated only
after the representation and objective weights are selected in inner folds.

The full successor constructs each training batch around anchors with:

1. same-master, cross-instrument positives where supported;
2. otherwise same-analyte, cross-instrument positives;
3. different-analyte, same-instrument hard negatives where supported;
4. sensor-matched or unrestricted negatives as deterministic fallbacks.

The supervised contrastive term uses the class relationships across the
whole structured batch. A paired margin term additionally requires the
chosen hard negative to remain farther from the anchor than its positive.
This directly addresses the pair-collapse seen in the structured VAE.

## Selection and leakage control

All representation, loss-weight, architecture, and epoch choices are made
inside the existing nested master-group development folds with seed 1729,
using only the 500 quality-pass spectra. The 98 field-stress rows are absent
from every selection training and validation partition. The 598-row strict
cohort is retained as a locked all-spectrum sensitivity evaluation, not a
selector. Outer folds and domain holdouts likewise cannot select a
configuration. Final outer runs use three declared seeds.

For the held-domain analyses, the single global configuration is locked by
quality-only master-group CV before any held-domain outcome is computed. As
in the classical benchmark, the configuration search spans the archive's
available domain identities; each domain model itself is then trained with
the held instrument or sensor removed. These are therefore locked
leave-one-domain-out transfer tests, not evidence that model development was
performed before the existence of that instrument family. Truly external
instrument validation still requires a later acquisition.

Collapse gates require nontrivial embedding rank, more than one predicted
class, and a positive different-minus-same-analyte distance margin.
Instrument or sensor predictability is diagnostic only: suppressing a domain
probe is not useful if chemical accuracy or class geometry is damaged.

## Calibration and rejection

Temperature scaling uses cross-fitted development scores. Maximum
probability, energy, and class-conditional embedding Mahalanobis distance are
compared as rejection scores. For each outer fold, the rejection score is
selected using only cross-fitted development predictions: mean selective
accuracy is ranked over the registered coverages below 100%, with maximum
probability, energy, then Mahalanobis as the deterministic tie order. The
selected score is then applied to field-stress spectra without reselection.
Field-stress labels remain sealed and cannot train a quality head, choose a
score, or choose a threshold.

## Promotion gate

The successor must reproducibly beat the current Siamese control and provide
a material advantage over the locked classical champion on held-domain
and/or field-stress selective performance. A mean advantage below 0.02 is
treated as practically inconclusive, and strict or quality balanced accuracy
may not fall by more than 0.03. At least two of three final seeds must agree
in the claimed direction.

The two prespecified shift endpoints are:

1. the equal-weight balanced-accuracy difference over supported
   held-instrument plus new-master-sample scenarios, pooling strict and
   quality cohorts; and
2. the outer-fold mean selective-accuracy difference on field-stress spectra
   at requested coverage 0.8, using each model's locked rejection rule.

Either endpoint may satisfy the registered “held-domain and/or stress”
condition. Domain-only and held-sensor analyses remain required diagnostics,
but cannot independently promote the successor; their effective support is
too sparse for that role.
