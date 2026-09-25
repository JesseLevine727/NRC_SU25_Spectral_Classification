# P05 numerical smoke — accepted bounded recovery

**2026-09-25. Training-only implementation check, not an unseen-instrument benchmark.**

The compact CNN, all-master sampler and three auxiliary-loss variants completed the bounded smoke. All 32 primary fits finished eight epochs with finite gradients and actual weight updates. The approved recovery reproduced the original failed-save fit exactly; both planned deterministic replays also matched. All 34 new checkpoints saved, reloaded and passed an independent state-hash audit.

The initial saving defect is preserved in the [failure/recovery record](../../plan/P05_SMOKE_STATUS.md). It was an infrastructure failure, not evidence that a learning method failed. No original failed artifact or lease was replaced.

## What ran

| Item | Observed result |
|---|---|
| Recipes | D0-M: ordinary classification; D1: + supervised contrastive loss; D2: + same-sample cross-instrument consistency; D3: both |
| Source-fitting groups | CWA: 13 masters / 123 spectra; pills: 9 / 94; surfaces: 13 / 88; sparse surfaces: 4 / 4 |
| Primary fits | 4 recipes × 4 source roles × 2 seeds = 32 |
| Per fit | 8 epochs × 4 sampled batches = 32 optimizer updates; terminal checkpoint, no early stopping or checkpoint selection |
| Model sizes | 208,691 parameters for D0-M/D2; 212,851 for D1/D3 |
| Accepted recovery records | 32 primary fits + 2 planned replays = 34 |
| Total scientific execution accounting | 1 original failed-save fit + 34 recovery executions = **35**, with **1,120 updates** overall |
| Resource use | 21.907 seconds recovery wall time; 21.224 seconds summed fitting time; longest new fit 1.262 seconds; peak allocated CUDA 151.46 MiB |
| Software checks | 523 local tests passed; real serialization tests included; Ruff and public-package checks passed |

The first accepted primary is an explicitly linked replay of the original failed-save fit, not another independent result. The other 33 planned records retain their original specifications. The 309 source-role spectrum memberships are not 309 independent samples; fitting roles can overlap. Only allowlisted source-fitting rows entered training and diagnostics. The representation remains minimal min–max scaling on 400–1800 cm−1; no preprocessing comparison occurred.

## Numerical acceptance

- **32/32 primary fits:** complete, finite and updating the backbone; projection heads update where present.
- **Common random streams:** matching initial backbones, sampling, augmentation and pair digests across all four recipes within every role/seed group.
- **Three exact replay comparisons:** the infrastructure recovery against the retained original history/model digests, plus the two planned replays.
- **Four sparse-control comparisons:** D2 equals D0-M and D3 equals D1 for each of the two seeds, in terminal states and numerical histories.
- **All 34 checkpoints:** reloaded on CPU, finite and matching their saved final-state hashes.
- **Protected state:** original failed evidence, numerical source files, input pins and within-run code/environment identity unchanged.

Where enabled, supervised contrastive loss had support in all 32 batches of every fit. In the sparse role, each batch had two eligible anchors and two without another same-chemical positive. Paired consistency had support in all 32 batches of each relevant dense fit, and **zero support** in the single-instrument sparse role. Its exact zero contribution is the required behavior there—not evidence that instrument invariance was learned.

## What the figures show

Start with [chemical training loss — interactive HTML](figures/P05S01_training_ce.html), or the [PDF](figures/P05S01_training_ce.pdf). Each dot is one epoch's mean weighted chemical cross-entropy over four training batches. Lower means the model assigned more probability to the recorded chemical label on those sampled training views. Colors identify recipes; solid/dashed lines are the two seeds. These are individual trajectories, not confidence intervals.

All **32/32 primary fits** had lower chemical training loss at epoch 8 than epoch 1. Initial two-seed mean losses across the role/recipe groups were approximately 1.03–1.10. The following are epoch-8 losses averaged over the two seeds:

| Source-fitting group | D0-M | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| CWA | 0.581 | 0.841 | 0.621 | 0.840 |
| Pills | 0.479 | 0.828 | 0.521 | 0.839 |
| Surfaces | 0.550 | 0.942 | 0.643 | 0.944 |
| Sparse surfaces | 0.0032 | 0.0383 | 0.0032 | 0.0383 |

D1/D3 reduced the classification component more slowly in these short dense-group runs. Their objective also includes chemical-similarity constraints; this observation cannot establish that they are better or worse on unseen data, or why the training trajectories differ. No recipe or setting was selected from these curves.

The sparse role contains only four training spectra. All recipes reached 100% training balanced accuracy there. That demonstrates how easily a tiny training group can be fitted; it does **not** demonstrate generalization. The overlapping D0-M/D2 and D1/D3 curves are especially useful checks that absent paired supervision was handled correctly.

The [gradient trajectories — interactive HTML](figures/P05S02_gradient_norm.html), or [PDF](figures/P05S02_gradient_norm.pdf), show the mean pre-clipping gradient norm across each epoch's four batches. The gradient describes the proposed direction and size of a parameter update, not classification accuracy. The largest individual pre-clipping batch norm was 20.754; the locked clipping threshold of 5 was activated on **29/1,024 primary-fit updates (2.83%)**. All relevant gradients remained finite, including the projection-head gradients. The plot's epoch means need not equal that largest individual batch value.

Both figures have native TikZ/pgfplots sources: [loss](figures/P05S01_training_ce.tex) and [gradients](figures/P05S02_gradient_norm.tex), plus PNG previews. The offline HTML and TikZ use the same [256-row semantic table](figures/semantic_data.csv). Two planned replay records and the original failed-save attempt are excluded from the plotted primary-fit evidence. HTML can be downloaded and opened locally without a server. The PDF/PNG previews were visually inspected.

## What is established—and what remains open

**Established:** the locked sampler and objectives run on the intended source-fitting data; supported auxiliary branches produce gradients; sparse branches are explicitly unavailable; matched controls and deterministic replay behave as specified; and checkpoint persistence works.

**Not established:** superiority over ordinary CNNs or classical ML, acquisition invariance, chemical/nuisance disentanglement, a best preprocessing method, or an adequate final epoch budget. There were no validation/test predictions, model-selection scores, calibrated probabilities or held-instrument accuracy estimates.

The next gate is a reviewed **source-only development runner and concrete time/storage authorization** for the finite core benchmark. Later training follows the already locked 30–200-epoch, patience-20 source-validation protocol, not this eight-epoch smoke. Model selection must be nested independently within each outer context's source data, with source-master guard checks and D0-M fallback. The later 14,940 inner-fit and at-most-2,880 refit slots are ceilings, **not current permission to execute**. P06 outer evaluation, P08 preprocessing comparisons and P14 RBF/SOM work remain separate.

## Reproducibility record

- Execution code: `bf8735cd43e5a9e22f616b8fc6891fe3e36b276d`.
- Original contract: `60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae`.
- Immutable plan: `a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37`.
- Recovery permit: `01e0835d8a6ece2ee98cee654e9f707e894643788dc9daabc438527c1c19c058`.
- Retained original failed manifest: `d452ec375d7445a0b728f895ceec47f1ae266ee3c2b80b7f731ef65ff2aa7dbe`.
- Accepted private recovery manifest: `4db0a4951d235013e5f34e28ff983dff20f530e87dfde21b62f9e0da73a0fe3b` (141 hashed entries).
- Public figure data: `48e1a384f932994448b539ec2f8e8de886893170c1f8f952359914470f815f3f`; individual export hashes are in the [figure manifest](figures/P05_figure_manifest.json).

Spectral arrays, physical identities, row-level predictions and checkpoints are excluded from this diagnostic export. Existing source-archive publication permissions are unchanged. Hosted CI omits torch, so its optional-test skips are not evidence of numerical validation; the 523-test result above comes from the local numerical environment. See the [supervisor review](../../plan/delegation/P05_CORE_REVIEW.md), [locked core protocol](../../plan/P05_CORE_PROTOCOL.md) and [master plan](../../plan/MASTER_PLAN.md).
