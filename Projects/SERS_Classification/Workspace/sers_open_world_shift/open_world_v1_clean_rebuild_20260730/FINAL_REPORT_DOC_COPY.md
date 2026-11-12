# Open-world, shift-aware SERS identification across field instruments and sensor families

Date completed: 2026-07-30  
Dataset: `nato-sers-preprocessing-v2`  
Validation status: **pass**

## Executive verdict

The residual CNN is not promoted: it passed 0 of 6 preregistered gates.

The result is scientifically useful whether positive or negative because the
true unknown chemical, field-stress cohort, outer master fold, and held domain
never selected preprocessing, architecture, epoch, score, temperature, or
threshold. The experiment distinguishes three questions that are often
collapsed in Raman/SERS papers: known-class classification, rejection of
unseen chemistry, and detection of acquisition-quality failure.

The standard and structured VAE controls do **not** establish chemical/nuisance
disentanglement. Reconstruction error can detect acquisition shift without
being a reliable unseen-chemical score, and the structured chemical head must
beat its union-latent negative comparator before any mechanistic separation
claim is defensible. This archive has no paired clean chemical-only target and
its chemical, instrument, sensor, station, and master factors are incompletely
crossed. It therefore cannot verify recovery of a unique nuisance-free Raman
spectrum.

## Dataset and preprocessing contract

- 598 attributable, valid SERS spectra from 69 physical master samples;
- 500 quality-pass and 98 locked field-stress spectra;
- six nonblank chemicals plus operational blank;
- ten instruments and four sensor families;
- common measured support 400–1800 cm⁻¹ at 1 cm⁻¹ (1,401 points);
- `minimal_minmax`, `arpls_minmax`, and `derivative_1` frozen before this study;
- no general, conditional, or instrument-specific smoothing;
- no post hoc alignment or instrument-specific preprocessing.

The lower bound is 400 cm⁻¹ because it is the first measured coordinate shared
by every included system. Extending to 100 cm⁻¹ would fabricate Mira values,
extend below Pendar support, and use an untrustworthy padded Agilent region.
The reduction from 721 named-SERS log rows to 598 spectra reflects missing
files, unresolved labels, contradictory source reuse, and one consistent
duplicate—not a model-driven quality filter.

The archive is not factorially crossed:

| Subset | Domain factor | Levels | Observed analyte×domain cells | Possible cells | Missing cells | Minimum masters/cell |
|---|---|---|---|---|---|---|
| field_quality_stress | instrument | 9 | 21 | 63 | 42 | 1 |
| field_quality_stress | outer_fold | 5 | 32 | 35 | 3 | 1 |
| field_quality_stress | sensor_family | 4 | 16 | 28 | 12 | 1 |
| quality_pass | instrument | 10 | 42 | 70 | 28 | 1 |
| quality_pass | outer_fold | 5 | 35 | 35 | 0 | 1 |
| quality_pass | sensor_family | 4 | 16 | 28 | 12 | 1 |
| strict_core | instrument | 10 | 44 | 70 | 26 | 1 |
| strict_core | outer_fold | 5 | 35 | 35 | 0 | 1 |
| strict_core | sensor_family | 4 | 17 | 28 | 11 | 1 |

Consequently, instrument or sensor predictability in an embedding is not by
itself evidence of nuisance encoding: chemistry-domain support imbalance can
create the same pattern. The target-adjusted probes below compare spectral
embeddings with a chemistry-label-only null to expose, but cannot causally
remove, this confounding.

## Leakage firewall

`master_sample_id` is indivisible. Each of the six true-unknown experiments
removes one nonblank chemical completely. Candidate objective, representation,
score, and epoch are selected in 600 inner surrogate-unknown scenarios; the
true held chemical and all 98 stress rows remain locked. Temperatures and
thresholds come only from four-fold cross-fitted known-development
predictions. Rejection thresholds target known retention of 95%, 90%, 80%,
70%, and 50%. Domain tests use the frozen `domain_only` and stricter
`domain_and_sample` partitions and provide no target-domain rows during
training.

## Models actually trained

- nine sealed classical candidates spanning PCA/logistic regression, RBF SVM,
  and random forest with three frozen representations;
- a validated random-forest addendum;
- Siamese/triplet and supervised-contrastive controls;
- standard VAE and structured chemical/union VAE controls;
- a 142,366-parameter residual 1-D CNN with seven bounded objectives:
  derivative ERM, arPLS ERM, joint supervised contrastive, synthetic
  Objectosphere/outlier exposure, conditional CORAL, conditional MMD, and
  GroupDRO. Its stem uses an 11-point stride-2 convolution; three residual
  stages use 24/48/96 channels, 7-point kernels, GroupNorm, adaptive pooling,
  a 64-dimensional embedding, and dropout 0.2;
- three final neural seeds treated as technical repeats;
- MSP, entropy, energy, class Mahalanobis distance, and MC-dropout mutual
  information, with the score selected only on surrogate unknowns.

Training budgets were fixed before locked outcomes: Siamese 100 epochs,
supervised contrastive 58, standard VAE 97, and structured VAE 500. CNN
surrogate candidates used AdamW (learning rate 0.001, weight decay 0.0001),
batch size 96, a 75-epoch maximum, first validation at epoch 25, and
10-epoch early-stopping patience. Each final CNN then retrained for the rounded
median development-selected epoch for its held-chemical/fold task. Thus a poor
result cannot be dismissed as an undocumented “few epochs” run; epoch and
objective sensitivity are preserved in the histories and selection tables.

## Primary open-world results

Values are master-level AUROC. Model seeds and the five outer models are
averaged before the six held chemicals are treated as independent units.
Chemical-novelty intervals resample the six held chemicals. Acquisition-shift
uncertainty instead uses leave-one-physical-master-out jackknife estimates
over the 69 masters; held-chemical task repeats are not counted as independent
field specimens.

| Method | Unseen chemical, quality | Unseen chemical, stress | Field-stress detection |
|---|---|---|---|
| Selected classical | 0.658 | 0.524 | 0.832 |
| Siamese triplet | 0.635 | 0.595 | 0.667 |
| Supervised contrastive | 0.658 | 0.533 | 0.763 |
| Standard VAE | 0.663 | 0.458 | 0.974 |
| Structured VAE—chemical | 0.679 | 0.508 | 0.783 |
| Structured VAE—union | 0.680 | 0.528 | 0.722 |
| Selected residual CNN | 0.454 | 0.599 | 0.555 |

AUROC 0.5 is chance. Quality-open AUROC asks whether an unseen chemical scores
as more anomalous than known quality spectra. Stress-open AUROC asks the same
question after both groups are field-stressed. Field-quality AUROC instead
asks whether known stress spectra separate from known quality spectra; a high
value here does not imply chemical novelty detection.

### Selected CNN by held chemical

| Held chemical | Quality AUROC | Quality OSCR | Stress AUROC | Stress OSCR |
|---|---|---|---|---|
| 4_ANPP | 0.399 | 0.336 | 0.758 | 0.158 |
| 4_nitrophenol | 0.570 | 0.432 | 0.597 | 0.156 |
| acetaminophen | 0.437 | 0.387 | 0.713 | 0.230 |
| benzyl_fentanyl | 0.346 | 0.283 | 0.527 | 0.116 |
| ethanol | 0.593 | 0.499 | 0.455 | 0.133 |
| ethyl_paraoxon | 0.379 | 0.342 | 0.546 | 0.142 |

These six rows are the independent chemical units. A high mean with one or two
low chemicals is not universal rejection.

### What development-only selection chose

| CNN objective | Selected tasks (of 30) |
|---|---|
| synthetic_objectosphere | 8 |
| conditional_coral | 7 |
| erm_arpls | 6 |
| supervised_contrastive_joint | 5 |
| erm_derivative | 4 |

| Novelty score | Selected tasks (of 30) |
|---|---|
| minimum_class_mahalanobis_distance | 13 |
| energy | 10 |
| predictive_entropy | 5 |
| one_minus_calibrated_max_probability | 2 |

Selection frequency is a stability diagnostic, not evidence that the most
frequent candidate wins on locked outcomes.

## Known-class results

| Method | Quality balanced accuracy | Stress balanced accuracy |
|---|---|---|
| Selected classical | 0.767 | 0.408 |
| Siamese triplet | 0.693 | 0.418 |
| Supervised contrastive | 0.757 | 0.465 |
| Standard VAE | 0.726 | 0.399 |
| Structured VAE—chemical | 0.746 | 0.399 |
| Structured VAE—union | 0.772 | 0.403 |
| Selected residual CNN | 0.576 | 0.257 |

Balanced accuracy averages recall over supported known classes, so class
imbalance cannot inflate the number. Unsupported held-domain analytes remain
in row-level predictions and are excluded only from supported-class metrics.
NLL, multiclass Brier score, ten-bin ECE, macro-F1, accuracy, per-class recall,
and confusion matrices are preserved in the machine-readable result tables.

### Dedicated random-forest baseline

The separately nested random-forest addendum produced:

| Cohort | Balanced accuracy | 95% half-width |
|---|---|---|
| Strict core | 0.690 | 0.041 |
| Quality pass | 0.742 | 0.045 |
| Field stress | 0.387 | 0.153 |

These intervals use five master-group outer folds after averaging the three
forest seeds. The forest is competitive on quality spectra but degrades
strongly on locked field stress; it remains a classifier, not a denoiser or
factorized representation.

## Preregistered CNN promotion gates

The best frozen primary comparator for quality chemical rejection was
`Structured VAE—chemical` at AUROC
0.679.

| Gate | Observed | Criterion | Decision |
|---|---|---|---|
| quality_known_ba_drop | -0.191 | difference >= -0.05 | fail |
| quality_open_auroc_improvement | -0.225 | difference >= 0.03 | fail |
| quality_open_chemical_count | 0.000 | chemicals improved >= 4 of 6 | fail |
| stress_open_auroc | 0.599 | AUROC >= 0.60 | fail |
| field_quality_auroc | 0.555 | AUROC >= 0.90 | fail |
| stress_known_ba_vs_supcon | -0.208 | difference >= 0.00 | fail |

No gate is rescued by selecting a different locked score, seed, chemical, or
stress subset.

## Selective prediction at known-only thresholds

The table uses only the selected CNN score and master-level aggregation.
Thresholds were fixed from cross-fitted known development spectra. “Realized
coverage” can differ from the requested development coverage on the locked
test masters; unknown rejection is never used to choose a threshold.

| Open-set evaluation | Requested known coverage | Realized known-master coverage | Accepted known-master BA | Unknown-master rejection |
|---|---|---|---|---|
| field_quality | 0.500 | 0.440 | 0.825 | 0.596 |
| field_quality | 0.700 | 0.750 | 0.763 | 0.305 |
| field_quality | 0.800 | 0.891 | 0.753 | 0.207 |
| field_quality | 0.900 | 0.974 | 0.760 | 0.090 |
| field_quality | 0.950 | 0.994 | 0.758 | 0.043 |
| field_quality | 1.000 | 1.000 | 0.759 | 0.000 |
| quality_open | 0.500 | 0.440 | 0.825 | 0.498 |
| quality_open | 0.700 | 0.750 | 0.763 | 0.185 |
| quality_open | 0.800 | 0.891 | 0.753 | 0.082 |
| quality_open | 0.900 | 0.974 | 0.760 | 0.017 |
| quality_open | 0.950 | 0.994 | 0.758 | 0.004 |
| quality_open | 1.000 | 1.000 | 0.759 | 0.000 |
| stress_open | 0.500 | 0.404 | 0.225 | 0.709 |
| stress_open | 0.700 | 0.695 | 0.294 | 0.366 |
| stress_open | 0.800 | 0.793 | 0.285 | 0.218 |
| stress_open | 0.900 | 0.910 | 0.268 | 0.089 |
| stress_open | 0.950 | 0.957 | 0.261 | 0.047 |
| stress_open | 1.000 | 1.000 | 0.260 | 0.000 |

## Held-domain transfer

The primary domain endpoint is quality-pass `domain_and_sample` instrument
transfer. Seeds are averaged first; mean and worst balanced accuracy use the
ten held instruments as independent units.

| CNN objective | Mean held-instrument BA | Worst held-instrument BA | SD across instruments |
|---|---|---|---|
| erm_derivative | 0.335 | 0.000 | 0.220 |
| erm_arpls | 0.309 | 0.000 | 0.268 |
| supervised_contrastive_joint | 0.295 | 0.056 | 0.155 |
| synthetic_objectosphere | 0.364 | 0.000 | 0.231 |
| conditional_coral | 0.324 | 0.000 | 0.258 |
| conditional_mmd | 0.133 | 0.000 | 0.096 |
| group_dro | 0.269 | 0.000 | 0.188 |

The corresponding sensor-family endpoint is necessarily narrower:

| CNN objective | Held sensor families | Families with supported classes | Mean supported-family BA | Worst supported-family BA |
|---|---|---|---|---|
| erm_derivative | 4 | 2 | 0.434 | 0.210 |
| erm_arpls | 4 | 2 | 0.227 | 0.154 |
| supervised_contrastive_joint | 4 | 2 | 0.146 | 0.144 |
| synthetic_objectosphere | 4 | 2 | 0.243 | 0.163 |
| conditional_coral | 4 | 2 | 0.232 | 0.207 |
| conditional_mmd | 4 | 2 | 0.250 | 0.167 |
| group_dro | 4 | 2 | 0.214 | 0.156 |

Some `domain_and_sample` sensor tasks have no supported analyte left after
same-master removal. Their test rows and predictions remain archived, but an
accuracy statistic is undefined; the table therefore reports both the four
attempted families and the number contributing a supported-class metric.

Registered source-only robustness contrasts against derivative ERM were:

| Candidate | Reference | Paired instruments | Mean paired BA difference | Worst paired difference | Improved | Tied | Worsened |
|---|---|---|---|---|---|---|---|
| conditional_coral | erm_derivative | 10 | -0.010 | -0.167 | 4 | 1 | 5 |
| conditional_mmd | erm_derivative | 10 | -0.201 | -0.460 | 0 | 2 | 8 |
| erm_arpls | erm_derivative | 10 | -0.026 | -0.400 | 4 | 2 | 4 |
| group_dro | erm_derivative | 10 | -0.065 | -0.280 | 2 | 0 | 8 |
| supervised_contrastive_joint | erm_derivative | 10 | -0.040 | -0.182 | 4 | 1 | 5 |
| synthetic_objectosphere | erm_derivative | 10 | 0.030 | -0.129 | 5 | 1 | 4 |

`domain_only` permits the same master on another device; `domain_and_sample`
removes every source-domain spectrum sharing a master with the held device.
The latter is the defensible generalization test. CORAL, MMD, and GroupDRO are
reported as prespecified paired contrasts against derivative ERM, not selected
using target-domain accuracy. This is source-only domain generalization, not
unsupervised domain adaptation.

## Controls and what they mean

The master-label permutation balanced accuracy was
0.161
(95% held-chemical bootstrap interval
0.155–
0.168).
This checks that physical-master grouping and the training code cannot recover
real labels after the development mapping is destroyed.

The augmentation/preprocessing ablation evaluates derivative ERM with and
without the registered shift/scale/noise perturbations, plus fixed arPLS and
minimal ERM. These are locked sensitivities with development-only epochs; no
true-unknown score selects among them. MC dropout is recomputed from a common
50-pass stream at 10, 20, and 50 prefixes, so pass-count stability is measured
without changing the trained model.

| Fixed sensitivity | Quality BA | Stress BA | Quality-open MSP AUROC | Stress-open MSP AUROC | Field-quality MSP AUROC |
|---|---|---|---|---|---|
| erm_arpls_augmented | 0.526 | 0.207 | 0.594 | 0.542 | 0.300 |
| erm_derivative_augmented | 0.593 | 0.254 | 0.462 | 0.561 | 0.430 |
| erm_derivative_no_augmentation | 0.629 | 0.254 | 0.569 | 0.537 | 0.566 |
| erm_minimal_augmented | 0.527 | 0.224 | 0.541 | 0.588 | 0.353 |

The preprocessing rows are a sensitivity analysis, not a second model search.
“MSP” is the fixed one-minus-maximum-probability diagnostic for every row;
these values do not replace the surrogate-selected primary CNN score.

| MC-dropout passes | Quality-open MI AUROC | Stress-open MI AUROC | Field-quality MI AUROC |
|---|---|---|---|
| 10 | 0.515 | 0.509 | 0.597 |
| 20 | 0.536 | 0.494 | 0.628 |
| 50 | 0.551 | 0.517 | 0.645 |

Target-adjusted domain probes on locked quality-test embeddings were:

| Domain | Embedding probe cell-BA | Chemistry-only null cell-BA | Increment over null |
|---|---|---|---|
| instrument | 0.545 | 0.146 | 0.398 |
| sensor_family | 0.802 | 0.427 | 0.375 |

Positive increments mean domain remains predictable beyond the analyte-label
null. That is evidence of residual acquisition information, not proof that the
network failed to learn any useful invariant feature.

Same-master prediction consistency was:

| Partition | Cross-domain pair | Masters with pairs | Pairs | Prediction agreement |
|---|---|---|---|---|
| known_quality_test | instrument | 10.433 | 285.667 | 0.502 |
| known_quality_test | sensor_family | 6.600 | 137.367 | 0.346 |
| known_stress | instrument | 2.633 | 9.567 | 0.432 |
| known_stress | sensor_family | 2.067 | 6.800 | 0.429 |

These are diagnostic task-averages. Agreement can be high because predictions
are consistently correct or consistently wrong, so it is interpreted beside
balanced accuracy rather than as a standalone success criterion.

The score-direction control is exactly `1 − AUROC`. Known-only threshold
coverage is independently recomputed from saved calibration scores. Exact UID
audits show no master, held chemical, outer-test, stress, or held-domain
leakage.

## Did the VAE/disentanglement idea work?

Not as a demonstrated disentangling mechanism. A VAE can compress,
reconstruct, provide density or reconstruction anomaly scores, and sometimes
improve downstream prediction. None of those facts identifies a latent as
“chemical” or “nuisance.” Without paired clean spectra, interventions that
change one nuisance while holding chemistry fixed, or a sufficiently crossed
factorial design, latent labels are underdetermined. This is consistent with
the general identifiability result that unsupervised disentanglement requires
inductive bias or supervision
([Locatello et al., 2019](https://proceedings.mlr.press/v97/locatello19a.html)).

The NATO archive *does* support publishable questions about open-set
recognition, uncertainty failure modes, source-only instrument transfer,
preprocessing sensitivity, and whether a compact CNN adds value beyond
classical/metric controls. It does not support “chemical-only spectrum
recovery,” causal substrate removal, or denoising against an unobserved clean
target.

## Relation to current Raman/SERS research

- Objectosphere losses have already been applied to open-world Raman
  identification with explicit irrelevant classes
  ([Balytskyi et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9728505/)).
  Here, the actual held chemical is never used as an irrelevant/outlier class;
  synthetic outliers are a separate registered objective.
- A 2026 SERS study combined learned enhancement, GASF, and MSP rejection
  ([Wang et al.](https://opg.optica.org/ao/abstract.cfm?uri=ao-65-18-6135)).
  Its very high within-dataset open-set values do not answer this archive’s
  cross-instrument, physical-master, and field-stress question.
- MC dropout has improved accuracy on retained low-uncertainty Raman subsets
  ([Contreras and Bocklitz, 2024](https://pubmed.ncbi.nlm.nih.gov/39580162/)).
  This study additionally tests whether MI recognizes unseen chemistry versus
  acquisition failure and locks its threshold to known development data.
- RamanMAE demonstrates the promise of masked spectral pretraining on much
  larger biological Raman corpora
  ([Paidi and Maheshwari, 2025](https://pubs.acs.org/doi/10.1021/acs.analchem.5c05656)).
  Training a foundation model from 598 NATO rows would not be an equivalent
  claim.
- DSCF uses broad self-supervised pretraining for cross-task spectral
  component filtering
  ([Nature Machine Intelligence, 2025](https://www.nature.com/articles/s42256-025-01027-5)).
  That paired/pretrained evidence is materially stronger for denoising than
  an unpaired NATO reconstruction loss.
- Parameter-efficient calibration transfer with a few target-instrument
  examples is a different protocol from source-only generalization
  ([Lai et al., 2025](https://pubmed.ncbi.nlm.nih.gov/40922652/)). A future NATO
  LoRA/PDS experiment must explicitly declare its unlabeled or labeled target
  spectra.
- A VAE-based Raman study handled varying ranges and noise across two labs
  using deliberate corruption training
  ([Jensen et al., 2024](https://www.nature.com/articles/s41598-024-56788-7)).
  It supports reconstruction/robustness as an empirical task, not automatic
  semantic identifiability.

## Claim limits and next experiment

Allowed claims are grouped known-class classification, unseen-chemical
rejection, field-quality rejection, selective prediction, uncertainty
diagnostics, and held-domain robustness on this archive. Prohibited claims are
denoising, chemical-only recovery, causal nuisance removal, semantic
disentanglement, or external-instrument generalization.

The most valuable next data collection is a crossed reference panel: the same
chemical preparations and concentrations measured across instruments and
sensor lots, plus technical replicates and a clean/reference measurement or
controlled corruption target. If limited target-device spectra become
available, evaluate calibration transfer/unsupervised adaptation as a new
protocol, never merge it with the source-only numbers above.

## Reproducibility and artifacts

- Frozen protocol: `predeclared_protocol.json`
- Baselines: `baselines/`
- Neural controls: `neural_controls/`
- CNN registry and results: `cnn/`
- Validation: `cnn/validation_report.json` and
  `cnn/validation_checks.csv`
- Summary tables: `summaries/`
- Figures: `figures/`
- Exact rebuild comparison: `rebuild_comparison.json`

Every result is retained at row level with observation UID, physical master,
instrument, sensor family, task, seed, probability vector, anomaly scores,
and immutable state hash. Seeds are technical repeats, held chemicals are the
open-set units, held domains are the transfer units, and physical masters are
the field-quality units.
