# NATO SERS standard-VAE training and architecture adequacy — full results

Protocol: `sers-vae-adequacy-v1`  
Status: complete, leakage-controlled selection followed by locked confirmation  
Selected configuration: `base_maxpool__z64__spectral_composite__beta0p25__constant_lr__e500`

## Executive answer

The idea was partly right, in a scientifically useful way:

1. **The original VAE was undertrained for reconstruction.** At 100 epochs,
   validation loss was still improving in every evaluable strict-core run;
   19/20 runs had their best epoch at
   or after 95. Extending the exact optimization path to 500 epochs materially
   improved correlation and repeatable-peak recovery.
2. **Undertraining was not the main classification or invariance failure.**
   Strict-core arPLS balanced accuracy changed only from
   `0.706081`
   to
   `0.706236`.
   The converged latent still failed the target-adjusted instrument-predictability
   and same-master cross-instrument-distance gates.
3. **The tested ordinary mixed-latent VAE cannot simultaneously preserve narrow
   chemistry-bearing structure and remove acquisition nuisance.** Lower beta
   preserves spectra and peaks but retains more instrument structure; higher
   beta suppresses some instrument information but damages chemistry and
   spectral fidelity.
4. **The frozen standard VAE is adequate as a converged reconstruction-capable
   comparator and initialization, not as an instrument/substrate-invariant
   representation.** This is the precise justification for a separately
   preregistered structured/disentangled-latent study.

No chemical/nuisance latent partitioning, adversarial loss, conditioned
decoder, or supervised contrastive loss was tested here.

## What was held fixed

- Population: 598 strict-core spectra, including the 500 quality-pass spectra
  and retaining the 98 field-quality-stress spectra as a locked stress cohort.
- Axis: 400–1800 cm⁻¹ inclusive, 1 cm⁻¹ step, 1,401 points.
- Grouping: `master_sample_id`; related observations could not cross a
  selection split.
- Primary view: `arpls_minmax`.
- Mandatory sensitivity view: `minimal_minmax`.
- Locked comparators: PCA/logistic, Siamese, AE, DAE, and the original
  100-epoch beta-1 VAE.
- Selection boundary: architecture, epoch, loss, latent width, beta, and
  optimizer decisions used only 20 grouped inner folds. Outer folds, field
  stress, held-out domains, and poster results were unavailable to selection.

The outer data had been seen in earlier projects, so the final stage is
confirmatory but not a human-blind external test.

## What was actually trained

- 260 distinct grouped-inner selection
  runs.
- 252 distinct locked
  confirmatory runs.
- 512 total model fits, each executing
  500 optimizer epochs.
- 256,000 optimizer epochs in total.
- Parameter range across ablations:
  544,689–2,157,681.
- 22,002 spectrum-level prediction rows
  and 22,002 reconstruction rows.
- 252 final checkpoints, 260 selection caches with model/optimizer states, 282
  embeddings, and 282 reconstruction arrays.

The chosen model has 1,082,353 parameters but only about
354–363 training spectra per strict inner fold. This unfavorable
data-to-parameter ratio was treated as a reason to keep the ablation bounded,
not as permission for an unrestricted architecture search.

## Frozen model and training policy

- 1-D convolutional encoder with channels 8→16 and two max-pooling stages.
- Mirrored decoder; no encoder–decoder skip connections.
- 64-dimensional Gaussian mixed latent.
- Spectral-composite reconstruction loss: Smooth L1 + 0.1 spectral angle +
  0.1 first-derivative loss.
- Adam, learning rate 0.001, weight decay 0.00001, batch size 64, gradient clip
  norm 5.
- Four fixed 25-epoch KL cycles during epochs 1–100. After epoch 100, beta is
  held at 0.25 through epoch 500.
- Canonical inference on CPU after deterministic CUDA training.

## Metric glossary: what the numbers mean

- **Balanced accuracy (BA):** mean class recall over supported analyte classes;
  each class has equal weight despite unequal spectrum counts. Higher is
  better.
- **Macro F1:** unweighted mean class F1 over supported classes; it penalizes
  both missed examples and false positives. Higher is better.
- **Median row correlation:** median Pearson correlation between each input
  spectrum and its reconstruction. It tests shape preservation, not absolute
  amplitude. Higher is better.
- **Repeatable-peak recall:** fraction of prominent reference peaks that recur
  across instruments for the same master sample and are reconstructed within
  ±5 cm⁻¹. Higher is better.
- **Instrument probe increment:** target-adjusted instrument-classification
  score above a target-only null model. Lower means less instrument information
  remains after accounting for analyte. Zero is ideal.
- **Same-master distance:** mean correlation distance between spectra of the
  same `master_sample_id` measured on different instruments. Lower is better.
- **Cross-instrument separation margin:** different-analyte distance minus
  same-master distance. Positive and larger is better.
- **KL per observation:** unnormalized divergence of the approximate posterior
  from the prior. Near zero can indicate posterior collapse; very high values
  indicate weak regularization.
- **Active units:** latent dimensions whose posterior-mean variance exceeds
  0.01. It measures use, not disentanglement.
- **Prediction agreement:** fraction whose predicted class is unchanged after
  controlled corruption. Higher is better.
- **Latent cosine drift:** change in latent direction after corruption. Lower
  is better.

Correlation and peak metrics do not prove that a VAE has removed noise:
identity-like reconstruction can score highly. Instrument probes,
same-master geometry, corruptions, and held-out domains are necessary
complements.

## Stage 0 — why 100 epochs was suspect

For strict-core arPLS, the original runs had median best epoch
99, with
8/20 exactly at epoch 100 and
19/20 at or beyond epoch 95.
Validation loss improved from epoch 90 to 100 in
100% of the
19 evaluable runs, by a median
1.989%.
The fourth KL cycle reached beta 1 only at epoch
89, leaving
12 beta-1 epochs before
the cap—less than the original early-stopping patience of
15.

The reproduced first 100 epochs matched the original histories to a maximum
absolute difference of `6.821e-13`,
inside the preregistered `1e-12` tolerance. The extension therefore tested
training duration rather than silently changing the original trajectory.

## Stage 1 — convergence isolation

Convergence required both: median validation-ELBO improvement over the final
50 epochs below 0.5%, and fewer than 25% of grouped folds improving by at
least 1%.

| policy | epoch | inner BA | correlation | peak recall | converged | median final-50 improvement | folds improving ≥1% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant_lr | 100 | 0.684118 | 0.897568 | 0.410915 | not assessed | NA | NA |
| constant_lr | 300 | 0.709397 | 0.923413 | 0.435952 | False | 1.099% | 55.0% |
| constant_lr | 400 | 0.707681 | 0.925951 | 0.436691 | True | 0.060% | 15.0% |
| constant_lr | 500 | 0.711457 | 0.927692 | 0.437414 | True | 0.254% | 10.0% |
| step_lr_300 | 100 | 0.684118 | 0.897568 | 0.410915 | not assessed | NA | NA |
| step_lr_300 | 300 | 0.709397 | 0.923413 | 0.435952 | False | 1.099% | 55.0% |
| step_lr_300 | 400 | 0.710065 | 0.925340 | 0.435919 | True | 0.379% | 0.0% |
| step_lr_300 | 500 | 0.699570 | 0.925960 | 0.437595 | True | 0.089% | 0.0% |

At epoch 300, constant learning rate was not converged: median improvement was
1.099% and 55% of folds improved by at least 1%. At epoch 500 it was converged:
0.254% median improvement and 10% of folds above 1%. The step-down policy did
not improve the registered scientific utility, so constant 0.001 was frozen.

This establishes that 100 epochs was a real spectral-fidelity problem.
It does not establish that longer training creates invariance.

## Stage 2A — architecture

| architecture | parameters | converged | gates | inner BA | correlation | peak recall | instrument probe | same-master distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_maxpool | 1,082,353 | True | 7/9 | 0.711457 | 0.927692 | 0.437414 | 0.558053 | 0.762050 |
| residual_multiscale | 1,088,849 | False | 6/9 | 0.684678 | 0.927043 | 0.417620 | 0.552479 | 0.763955 |
| single_pool_peak | 1,081,705 | False | 6/9 | 0.710483 | 0.923977 | 0.445281 | 0.566448 | 0.749479 |

The residual/multiscale model did not converge and reduced BA and peak recall.
The one-pool model modestly raised peak recall but did not converge, did not
improve BA, and retained more instrument information. The original two-pool
backbone was therefore not supported as the primary cause of failure.

## Stage 2B — reconstruction loss

| loss | parameters | converged | gates | inner BA | correlation | peak recall | instrument probe | same-master distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| peak_multiscale | 1,082,353 | True | 7/9 | 0.708762 | 0.932659 | 0.437320 | 0.563916 | 0.764781 |
| spectral_composite | 1,082,353 | True | 7/9 | 0.711457 | 0.927692 | 0.437414 | 0.558053 | 0.762050 |

The peak/multiscale loss raised correlation from 0.927692 to 0.932659 but peak
recall was essentially unchanged (0.437320 versus 0.437414) and BA fell
slightly. Extra derivative and multiscale terms therefore did not solve the
peak or nuisance problem.

## Stage 2C — latent width

| latent dimensions | parameters | converged | gates | inner BA | correlation | peak recall | instrument probe | same-master distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 544,689 | False | 8/9 | 0.685271 | 0.927966 | 0.443760 | 0.528305 | 0.765301 |
| 64 | 1,082,353 | True | 7/9 | 0.711457 | 0.927692 | 0.437414 | 0.558053 | 0.762050 |
| 128 | 2,157,681 | False | 5/9 | 0.701633 | 0.925482 | 0.431247 | 0.552403 | 0.760038 |

The 32-dimensional latent had eight gates but failed convergence and lost
chemical accuracy. The 128-dimensional latent doubled parameters, failed
convergence, and improved neither chemistry nor preservation. Width alone did
not filter nuisance; z64 remained the only converged option.

## Stage 2D — KL strength and the key trade-off

| beta | parameters | converged | gates | inner BA | correlation | peak recall | instrument probe | same-master distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 1,082,353 | True | 7/9 | 0.692119 | 0.944933 | 0.497539 | 0.592584 | 0.777530 |
| 1.0 | 1,082,353 | True | 7/9 | 0.711457 | 0.927692 | 0.437414 | 0.558053 | 0.762050 |
| 4.0 | 1,082,353 | True | 6/9 | 0.675150 | 0.876078 | 0.407044 | 0.504643 | 0.746167 |

- Beta 0.25 best preserved shape and repeatable peaks and used about 33.35
  active dimensions, but retained the most nuisance structure.
- Beta 1 had the best strict inner BA, but lower spectral and peak fidelity.
- Beta 4 reduced the instrument probe and same-master distance, but damaged BA,
  correlation, and peaks.

This is the central result: a single unsupervised latent is being asked both
to reconstruct nuisance-rich spectra and to discard that nuisance. Stronger
prior pressure does not tell the model which variation is chemical.

The top two beta candidates both passed the quality-sensitivity rule:

| candidate | strict BA | quality BA | quality−strict | utility |
| --- | --- | --- | --- | --- |
| base_maxpool__z64__spectral_composite__beta0p25__constant_lr__e500 | 0.692119 | 0.726615 | +0.034496 | 0.641839 |
| base_maxpool__z64__spectral_composite__beta1p0__constant_lr__e500 | 0.711457 | 0.724986 | +0.013528 | 0.572832 |

Beta 0.25 was selected by the preregistered multi-objective utility, not by
outer outcomes. Although beta 1 had higher strict inner BA, beta 0.25 provided
the stronger registered preservation/chemistry compromise and slightly higher
quality BA.

## Eligibility-gate result

The selected strict-core arPLS model passed 7/9 gates. It passed clean
correlation, repeatable peaks, chemical probe, active units, KL dimensions, KL
range, and finite-output checks. It failed:

1. **Instrument probe:** `0.592584`.
   Too much target-adjusted instrument information remained.
2. **Same-master distance:** `0.777530`,
   versus raw `0.675259`.
   Encoding increased rather than decreased the cross-instrument distance of
   replicate master samples by `+0.102271`.

The model was therefore selected as the rigorously defined least-failing,
converged backbone—not declared fully adequate.

## Mandatory preprocessing sensitivity

`minimal_minmax` preserves the instrument-delivered shape before common-axis
scaling; `arpls_minmax` removes more baseline and is the primary separability
view. The two views answer different questions and neither replaces the other.

| subset | inner BA | correlation | peak recall | instrument probe | same-master distance | KL/observation | active units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| quality_pass | 0.700431 | 0.979403 | 0.653712 | 0.646790 | 0.827594 | 16.550128 | 14.050000 |
| strict_core | 0.684827 | 0.969833 | 0.599545 | 0.665032 | 0.856952 | 17.290811 | 34.700000 |

Minimal preprocessing improved correlation and peak recovery substantially,
but its instrument predictability and same-master geometry were worse. This
means scaling every spectrum to the same 0–1 range does not itself make
spectra instrument-invariant; baseline/shape differences remain encoded.
arPLS remains primary for classification, while minimal remains mandatory to
detect peak destruction or preprocessing-dependent conclusions.

## Locked grouped-outer confirmation

Each value below averages five grouped outer folds and three registered neural
seeds. “Old” is the original 100-epoch beta-1 VAE.

| cohort | view | new BA | old BA | new macro F1 | new correlation | old correlation | new peak recall | old peak recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_core | arpls_minmax | 0.706236 | 0.706081 | 0.696068 | 0.948112 | 0.906072 | 0.479627 | 0.419394 |
| strict_core | minimal_minmax | 0.688145 | 0.690578 | 0.677099 | 0.973147 | 0.943520 | 0.597607 | 0.525900 |
| quality_pass | arpls_minmax | 0.744493 | 0.728353 | 0.722244 | 0.965410 | 0.928625 | 0.522561 | 0.486694 |
| quality_pass | minimal_minmax | 0.729604 | 0.708792 | 0.700214 | 0.980856 | 0.955415 | 0.656627 | 0.614530 |
| field_quality_stress | arpls_minmax | 0.368545 | 0.342487 | 0.311407 | 0.394393 | 0.296196 | 0.180720 | 0.128730 |
| field_quality_stress | minimal_minmax | 0.396561 | 0.366667 | 0.332993 | 0.476164 | 0.426943 | 0.170621 | 0.119536 |

The strict arPLS changes from 100 to 500 epochs were:

- BA: +0.000155
  — effectively unchanged.
- Correlation: +0.042040.
- Repeatable-peak recall: +0.060233.

Quality-pass and field-stress BA improved, but field BA remained only 0.368545
on arPLS and 0.396561 on minimal. The model therefore became a better
reconstructor and somewhat better stress classifier without becoming a
strong field model.

The 95% t-interval summaries use the five outer folds as independent units:

| cohort | view | mean BA | 95% half-width | folds |
| --- | --- | --- | --- | --- |
| strict_core | arpls_minmax | 0.706236 | ±0.063701 | 5 |
| strict_core | minimal_minmax | 0.688145 | ±0.074657 | 5 |
| quality_pass | arpls_minmax | 0.744493 | ±0.062984 | 5 |
| quality_pass | minimal_minmax | 0.729604 | ±0.038614 | 5 |
| field_quality_stress | arpls_minmax | 0.368545 | ±0.179101 | 5 |
| field_quality_stress | minimal_minmax | 0.396561 | ±0.183518 | 5 |

These intervals are descriptive across only five folds and should not be read
as precise population intervals.

## Comparison with frozen model families

Grouped-outer balanced accuracy:

| cohort | model | BA |
| --- | --- | --- |
| strict_core | PCA-logistic | 0.724631 |
| strict_core | AE | 0.703339 |
| strict_core | DAE | 0.713070 |
| strict_core | Siamese | 0.632094 |
| strict_core | VAE-100 β=1 | 0.706081 |
| strict_core | VAE-500 β=0.25 | 0.706236 |
| quality_pass | PCA-logistic | 0.771336 |
| quality_pass | AE | 0.721476 |
| quality_pass | DAE | 0.713910 |
| quality_pass | Siamese | 0.676669 |
| quality_pass | VAE-100 β=1 | 0.728353 |
| quality_pass | VAE-500 β=0.25 | 0.744493 |
| field_quality_stress | PCA-logistic | 0.447143 |
| field_quality_stress | AE | 0.335159 |
| field_quality_stress | DAE | 0.356190 |
| field_quality_stress | Siamese | 0.370450 |
| field_quality_stress | VAE-100 β=1 | 0.342487 |
| field_quality_stress | VAE-500 β=0.25 | 0.368545 |

The new VAE did not beat PCA/logistic on strict, quality, or field cohorts. It
also did not produce a general classification advantage over AE/DAE. The
correct claim is therefore “converged VAE comparator,” not “best classifier.”
The Siamese model remains a useful metric-learning control and has excellent
same-master alignment, but it is not a reconstruction or denoising model.

## Controlled-corruption behavior

The next table averages all seven registered corruptions at each severity for
the strict cohort:

| view | severity | BA | prediction agreement | latent drift | correlation | peak recall |
| --- | --- | --- | --- | --- | --- | --- |
| arpls_minmax | 0.5 | 0.699674 | 0.936093 | 0.017797 | 0.944890 | 0.478268 |
| arpls_minmax | 1.0 | 0.686238 | 0.891791 | 0.050829 | 0.935125 | 0.472749 |
| arpls_minmax | 1.5 | 0.656771 | 0.842280 | 0.089012 | 0.913975 | 0.465875 |
| minimal_minmax | 0.5 | 0.686286 | 0.948582 | 0.016242 | 0.967087 | 0.595519 |
| minimal_minmax | 1.0 | 0.675606 | 0.916733 | 0.046703 | 0.948790 | 0.587135 |
| minimal_minmax | 1.5 | 0.661932 | 0.877467 | 0.083190 | 0.916343 | 0.575328 |

Composite-corruption severity 1, directly compared with the original VAE:

| view | model | BA | agreement | latent drift | MSE |
| --- | --- | --- | --- | --- | --- |
| arpls_minmax | VAE-100 β=1 | 0.609595 | 0.699518 | 0.154594 | 0.018035 |
| arpls_minmax | VAE-500 β=0.25 | 0.639166 | 0.726920 | 0.165625 | 0.014015 |
| minimal_minmax | VAE-100 β=1 | 0.645281 | 0.804269 | 0.153555 | 0.021953 |
| minimal_minmax | VAE-500 β=0.25 | 0.645615 | 0.802535 | 0.152330 | 0.018281 |

The converged VAE improved composite BA, agreement, and reconstruction MSE on
arPLS, but latent drift increased slightly. On minimal, the BA/agreement
change was negligible while MSE improved. Smooth baselines and scale/offset
were more damaging than Gaussian broadening, Gaussian noise, or isolated
spikes. Synthetic robustness is useful diagnostic evidence, not proof of
universal field denoising.

## Same-master cross-instrument geometry

Lower latent distance and negative delta are desirable. A positive separation
margin means different analytes remain farther apart than same-master
replicates.

| view | model | raw same-master | latent same-master | delta | separation margin |
| --- | --- | --- | --- | --- | --- |
| arpls_minmax | AE | 0.675259 | 0.432724 | -0.242535 | 0.138671 |
| arpls_minmax | DAE | 0.675259 | 0.450574 | -0.224686 | 0.174031 |
| arpls_minmax | VAE-100 β=1 | 0.675259 | 0.757807 | 0.082548 | 0.289343 |
| arpls_minmax | VAE-500 β=0.25 | 0.675259 | 0.777328 | 0.102069 | 0.251313 |
| minimal_minmax | AE | 0.760397 | 0.443450 | -0.316947 | 0.091623 |
| minimal_minmax | DAE | 0.760397 | 0.520044 | -0.240353 | 0.121969 |
| minimal_minmax | VAE-100 β=1 | 0.760397 | 0.857641 | 0.097244 | 0.203767 |
| minimal_minmax | VAE-500 β=0.25 | 0.760397 | 0.850862 | 0.090464 | 0.184231 |
| derivative_1 | Siamese | 0.683603 | 0.146114 | -0.537489 | 0.290040 |

On strict arPLS, VAE-500 increased same-master distance by +0.102069; the
original VAE increased it by +0.082548. AE and DAE reduced it, while Siamese
reduced it most. The longer/lower-beta VAE therefore preserved more spectral
detail but made replicate geometry less invariant. Field-stress distances
shrank for all autoencoders, but their separation margins were negative and
field classification was poor, so that shrinkage is not useful invariance.

## Held-out instrument and sensor transfer

`domain_only` holds out a domain; `domain_and_sample` additionally prevents
master-sample overlap and is the stronger generalization test.

| subset | protocol | domain | new BA | old BA | change |
| --- | --- | --- | --- | --- | --- |
| strict_core | domain_and_sample | instrument | 0.596374 | 0.557490 | +0.038884 |
| strict_core | domain_and_sample | sensor_family | 0.584317 | 0.623912 | -0.039595 |
| strict_core | domain_only | instrument | 0.633634 | 0.605729 | +0.027905 |
| strict_core | domain_only | sensor_family | 0.423433 | 0.385036 | +0.038397 |
| quality_pass | domain_and_sample | instrument | 0.593091 | 0.549113 | +0.043978 |
| quality_pass | domain_and_sample | sensor_family | 0.377629 | 0.381807 | -0.004179 |
| quality_pass | domain_only | instrument | 0.647281 | 0.609360 | +0.037922 |
| quality_pass | domain_only | sensor_family | 0.372078 | 0.349585 | +0.022493 |

Instrument transfer improved in all four new-versus-old comparisons. Sensor
family transfer was mixed: strict domain-and-sample fell from 0.623912 to
0.584317, while strict domain-only rose from 0.385036 to 0.423433. These
aggregates also conceal severe domain-specific failures:

| subset | domain type | held-out domain | BA |
| --- | --- | --- | --- |
| quality_pass | instrument | Agilent-1 | 0.712121 |
| quality_pass | instrument | Agilent-3 | 0.487776 |
| quality_pass | instrument | Mira-1 | 0.741703 |
| quality_pass | instrument | Mira-2 | 0.000000 |
| quality_pass | instrument | Mira-3 | 0.459583 |
| quality_pass | instrument | Pendar-1 | 0.812169 |
| quality_pass | instrument | Pendar-2 | 1.000000 |
| quality_pass | instrument | Pendar-3 | 0.512108 |
| quality_pass | instrument | RMX-1 | 0.542088 |
| quality_pass | instrument | RMX-2 | 0.663360 |
| quality_pass | sensor_family | GaN_polymer | 0.538462 |
| quality_pass | sensor_family | H_SERS_H_Kit | NA |
| quality_pass | sensor_family | NRC_Canadian_SERS | 0.216796 |
| quality_pass | sensor_family | pSERS_Metrohm_silver | NA |
| strict_core | instrument | Agilent-1 | 0.789773 |
| strict_core | instrument | Agilent-3 | 0.550739 |
| strict_core | instrument | Mira-1 | 0.742985 |
| strict_core | instrument | Mira-2 | 0.166667 |
| strict_core | instrument | Mira-3 | 0.323750 |
| strict_core | instrument | Pendar-1 | 0.780093 |
| strict_core | instrument | Pendar-2 | 1.000000 |
| strict_core | instrument | Pendar-3 | 0.511035 |
| strict_core | instrument | RMX-1 | 0.454545 |
| strict_core | instrument | RMX-2 | 0.644157 |
| strict_core | sensor_family | GaN_polymer | 0.514815 |
| strict_core | sensor_family | H_SERS_H_Kit | 1.000000 |
| strict_core | sensor_family | NRC_Canadian_SERS | 0.238137 |
| strict_core | sensor_family | pSERS_Metrohm_silver | NA |

The NRC Canadian SERS family was especially weak (~0.22–0.24 BA). Quality
domain-and-sample Mira-2 was 0.0. Some 1.0 values come from small or unusually
supported held-out partitions; unsupported analyte classes remain excluded
from supported-class BA and are retained in failure tables. `NA` means no
supported-class BA could be computed, not perfect or zero performance.

## Descriptive poster substrate transfer

The poster data lack independent preparation IDs, so these are descriptive
leave-one-substrate-family-out results, not an independent validation set.

| view | held-out substrate | BA | correlation | peak recall |
| --- | --- | --- | --- | --- |
| arpls_minmax | Ag | 0.648889 | 0.828140 | 0.457021 |
| arpls_minmax | Au | 0.933333 | 0.945372 | 0.603583 |
| arpls_minmax | PICO | 0.568889 | 0.774848 | 0.293407 |
| arpls_minmax | pSERS | 1.000000 | 0.798859 | 0.592593 |
| minimal_minmax | Ag | 0.640000 | 0.951111 | 0.468320 |
| minimal_minmax | Au | 0.933333 | 0.956991 | 0.734151 |
| minimal_minmax | PICO | 0.666667 | 0.779217 | 0.240803 |
| minimal_minmax | pSERS | 0.991111 | 0.803256 | 0.629747 |

Mean arPLS poster BA fell from 0.927222 for VAE-100 to 0.787778 for VAE-500,
concentrated in held-out Ag and PICO. Mean minimal BA rose from 0.682778 to
0.807778. This direction reversal is another preprocessing-dependent
preservation/invariance warning, not evidence that either view universally
wins.

## Class-level behavior

Pooled spectrum counts below include repeated outer-seed predictions, so they
describe error concentration and are not independent sample counts:

| cohort | class | pooled support | recall | precision |
| --- | --- | --- | --- | --- |
| field_quality_stress | 4_ANPP | 12 | 0.166667 | 0.400000 |
| field_quality_stress | 4_nitrophenol | 54 | 0.444444 | 0.800000 |
| field_quality_stress | acetaminophen | 45 | 0.933333 | 0.210000 |
| field_quality_stress | benzyl_fentanyl | 15 | 0.400000 | 0.461538 |
| field_quality_stress | blank | 48 | 0.250000 | 0.631579 |
| field_quality_stress | ethanol | 69 | 0.173913 | 1.000000 |
| field_quality_stress | ethyl_paraoxon | 51 | 0.156863 | 0.533333 |
| quality_pass | 4_ANPP | 294 | 0.778912 | 0.817857 |
| quality_pass | 4_nitrophenol | 222 | 0.738739 | 0.766355 |
| quality_pass | acetaminophen | 198 | 0.722222 | 0.677725 |
| quality_pass | benzyl_fentanyl | 345 | 0.846377 | 0.906832 |
| quality_pass | blank | 213 | 0.779343 | 0.683128 |
| quality_pass | ethanol | 117 | 0.658120 | 0.687500 |
| quality_pass | ethyl_paraoxon | 111 | 0.531532 | 0.500000 |
| strict_core | 4_ANPP | 306 | 0.771242 | 0.802721 |
| strict_core | 4_nitrophenol | 276 | 0.706522 | 0.750000 |
| strict_core | acetaminophen | 243 | 0.666667 | 0.627907 |
| strict_core | benzyl_fentanyl | 360 | 0.805556 | 0.857988 |
| strict_core | blank | 261 | 0.758621 | 0.675768 |
| strict_core | ethanol | 186 | 0.666667 | 0.685083 |
| strict_core | ethyl_paraoxon | 162 | 0.506173 | 0.482353 |

In strict-core arPLS, ethyl paraoxon was the weakest class (recall 0.506173,
precision 0.482353); benzyl fentanyl was strongest (recall 0.805556). In field
stress the model overpredicted acetaminophen: recall was 0.933333 but precision
only 0.210000. Ethanol, ethyl paraoxon, and 4-ANPP recalls were all below 0.18.

## Instrument-level behavior

These pooled accuracies also repeat spectra across seeds and are diagnostic,
not independent confidence units:

| cohort | instrument | pooled supported rows | accuracy |
| --- | --- | --- | --- |
| field_quality_stress | Agilent-3 | 3 | 1.000000 |
| field_quality_stress | Mira-1 | 6 | 1.000000 |
| field_quality_stress | Mira-2 | 3 | 0.333333 |
| field_quality_stress | Mira-3 | 3 | 0.000000 |
| field_quality_stress | Pendar-1 | 18 | 0.000000 |
| field_quality_stress | Pendar-2 | 168 | 0.303571 |
| field_quality_stress | Pendar-3 | 84 | 0.440476 |
| field_quality_stress | RMX-1 | 6 | 1.000000 |
| field_quality_stress | RMX-2 | 3 | 0.666667 |
| quality_pass | Agilent-1 | 87 | 0.850575 |
| quality_pass | Agilent-3 | 216 | 0.731481 |
| quality_pass | Mira-1 | 63 | 0.793651 |
| quality_pass | Mira-2 | 222 | 0.662162 |
| quality_pass | Mira-3 | 213 | 0.671362 |
| quality_pass | Pendar-1 | 120 | 0.775000 |
| quality_pass | Pendar-2 | 126 | 0.952381 |
| quality_pass | Pendar-3 | 126 | 0.722222 |
| quality_pass | RMX-1 | 135 | 0.659259 |
| quality_pass | RMX-2 | 192 | 0.859375 |
| strict_core | Agilent-1 | 87 | 0.862069 |
| strict_core | Agilent-3 | 219 | 0.721461 |
| strict_core | Mira-1 | 69 | 0.811594 |
| strict_core | Mira-2 | 225 | 0.666667 |
| strict_core | Mira-3 | 216 | 0.638889 |
| strict_core | Pendar-1 | 138 | 0.739130 |
| strict_core | Pendar-2 | 294 | 0.727891 |
| strict_core | Pendar-3 | 210 | 0.628571 |
| strict_core | RMX-1 | 141 | 0.687943 |
| strict_core | RMX-2 | 195 | 0.846154 |

Strict-core performance ranged from 0.628571 on Pendar-3 to 0.862069 on
Agilent-1. The field cohort was dominated by failures on Pendar-1, Pendar-2,
and Pendar-3; instruments with only three or six pooled supported predictions
must not be overinterpreted.

## Failure attribution

1. **Convergence — confirmed material contributor to spectral fidelity.**
   Five hundred epochs repaired correlation and peak recall, but did not
   materially change strict classification.
2. **Architecture/downsampling — not supported as the primary failure.**
   Neither the residual/multiscale nor one-pool candidate gave a converged,
   consistent gain.
3. **Reconstruction objective — peak-aware loss not a solution.** It improved
   correlation but not repeatable-peak recall.
4. **Latent capacity/KL pressure — strong trade-off confirmed.** Width changes
   did not solve it; beta moved preservation and nuisance retention in opposite
   directions.
5. **Data coverage/domain shift — dominant unresolved confirmatory failure.**
   Field stress, NRC Canadian SERS, and several held-out instruments remain
   weak.
6. **Unresolved interaction — ordinary mixed latent inadequate for the desired
   invariance.** A standard VAE has no label telling it which reconstructable
   variance is chemical and which is nuisance.

## Final decision

The converged standard-VAE backbone is scientifically adequate as a reconstruction-capable mixed-latent comparator and initialization for the next study, but it is not adequate as an instrument- and substrate-invariant representation. It remains below the frozen PCA/logistic clean benchmark, fails the registered instrument and same-master gates, and does not solve field or sensor-family shift.

The frozen backbone and training policy may initialize the next structured
study. The next goal should partition or condition chemical and nuisance
information and test that design with the same grouped and locked boundaries.
It must not reopen preprocessing, the 8→16 two-pool backbone, z64 width,
spectral-composite loss, beta 0.25, or 500-epoch policy based on the already
observed locked outcomes.

## Claim limits

- This is still a standard mixed-latent VAE, not evidence of disentanglement.
- Previously observed outer cohorts are confirmatory, not human-blind.
- The poster split is descriptive and lacks independent preparation IDs.
- Synthetic corruption robustness does not establish universal real-field denoising.
- Unsupported unseen analyte classes remain excluded from supported-class metrics and are retained in failure tables.
- Only the 400–1800 cm-1 common axis and frozen preprocessing-v2 population are supported.

## Reproducibility status

The authoritative bundle and a clean rebuild from an empty output directory
are required to agree exactly for canonical tables, JSON decisions, model
tensors, optimizer tensors, embeddings, and reconstructions; floating training
histories use the preregistered `1e-12` tolerance. See
`clean_rebuild_comparison.json`, `validation_report.json`, and
`artifact_hashes.json`.
