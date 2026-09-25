# NATO SERS: source-validation pilot

**2026-09-25. The approved 36-fit pilot is complete and independently audited. This is development evidence, not held-out test performance or a recipe-selection decision.**

The four fixed CNN recipes trained successfully under the longer source-validation protocol. Their effects differed by station: matched-sample consistency was promising for CWA; both auxiliary losses together were promising for pills; surfaces retained a large training–validation gap. No loss, preprocessing setting, architecture or seed was changed in response. The pilot does not establish improvement over classical ML or the earlier CNN benchmark.

## What was trained

Each station used one split selected from metadata before training, with three initialization seeds per recipe:

| Recipe | Training objective | Parameters |
|---|---|---:|
| D0-M | Matched ordinary classification control | 208,691 |
| D1 | Classification + chemical-similarity loss | 212,851 |
| D2 | Classification + consistency between measurements of the same sample on different instruments | 208,691 |
| D3 | Classification + both auxiliary losses | 212,851 |

The recipes shared the same initial CNN backbone and sampled/augmented observations for corresponding epochs. D0-M is the new matched control, not a rerun of historical P04 D0. Its training sampler differs from that older benchmark.

| Station | Fitting masters / spectra | Validation masters / spectra | Validation question |
|---|---:|---:|---|
| CWA | 12 / 42 | 6 / 12 | Transfer to a withheld source instrument and disjoint samples |
| Pills | 11 / 96 | 5 / 41 | Transfer to disjoint source samples; **not** an unseen-instrument test |
| Surfaces | 7 / 15 | 11 / 15 | Transfer to a withheld source instrument and disjoint samples |

All these roles lie inside their respective outer training partitions. Outer-test predictions were not generated. Repeated spectra remain measurements of the same physical masters, not additional independent samples. In particular, the 41 pills validation spectra represent only five physical samples.

Preprocessing remained the locked **400–1800 cm⁻¹ grid, 1 cm⁻¹ spacing, per-spectrum min–max scaling**. No new baseline subtraction or smoothing was applied. Source-fitting-only augmentation and the original optimizer/loss settings were unchanged. Fits used four sampled batches per epoch, 30–200 epochs, and patience 20. The saved checkpoint maximized source-validation balanced accuracy, then minimized negative log likelihood, with the earliest epoch breaking an exact tie. Training lasted at least 30 epochs, but the selected checkpoint could be earlier.

## What the scores show

**Balanced accuracy (BA)** is the mean fraction correctly classified within each of the three chemical classes. Each class receives equal importance; 33.3% is the uniform-random expectation and the score of an always-one-class predictor when all three classes occur. Scores below are **per-spectrum** BA at the selected checkpoint, averaged across the three seeds. They do not average spectra or combine repeated predictions by sample.

| Station | D0-M | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| CWA | 50.0% | 40.0% | 62.2% | 40.0% |
| Pills | 84.4% | 88.6% | 87.3% | 93.4% |
| Surfaces | 43.2% | 46.3% | 42.0% | 45.4% |

- **CWA:** D2 scored 60.0–66.7% across seeds, versus 40.0–60.0% for D0-M. This is encouraging on this one split, not proof of instrument independence. One D1 and two D3 selected checkpoints predicted a single class: three collapsed predictions among the 36 fits. These fits were numerically valid, but not useful classifiers at those checkpoints.
- **Pills:** D3 scored 92.0–96.1%, versus 79.7–87.7% for D0-M. All three D3 seeds exceeded their matched D0-M seed. This concerns new source samples, not a new instrument.
- **Surfaces:** recipe means remained 42.0–46.3%. D0-M's mean training BA was 89.8%, versus 43.2% on validation; D2's was 92.0% versus 42.0%. Fitting the observed training spectra did not ensure transfer. Small sample support and acquisition differences remain possible contributors; this pilot does not isolate their causes.

Three seeds describe initialization sensitivity on the **same split**. They are not three independent datasets. Validation also selected the checkpoint, so these scores are development estimates, not unbiased final performance. No confidence interval, significance test, global winner or G3 decision is inferred from this table. Station scores should not be interpreted as a controlled ranking of chemical difficulty because their supports and validation modes differ.

## Which figures to open

Start with the **[training-versus-validation scatter plot](figures/P05P02_best_checkpoints.html)** ([PDF](figures/P05P02_best_checkpoints.pdf), [native TikZ](figures/P05P02_best_checkpoints.tex)). Each marker is one trained model at its selected checkpoint: color identifies the recipe, shape the seed. The horizontal coordinate is training BA; the vertical coordinate is validation BA. Points far below the diagonal performed much better on training data. There are 12 points per station, including overlaps; the HTML legend can hide individual traces.

Then open the **[learning curves](figures/P05P01_learning_curves.html)** ([PDF](figures/P05P01_learning_curves.pdf), [native TikZ](figures/P05P01_learning_curves.tex)). Left panels show chemical classification loss on augmented training batches; right panels show unaugmented source-validation BA. Lower training loss did not consistently produce higher validation BA, particularly for surfaces. Each trajectory stops where that fit stopped; none is extended or imputed. CE is the class/master/view-weighted cross-entropy classification component, not the total objective of D1–D3.

Both figures also have PNG previews and use the same [1,720-row semantic table](figures/semantic_data.csv). HTML files are self-contained: download and open locally. Native PDF/PNG and both HTML renderings were inspected. [Per-fit aggregate diagnostics](fit_summary.csv) retain all 36 results, epochs, seed identities and resource counts. NLL measures how much probability was assigned to the true label (lower is better); macro-F1 averages classwise precision/recall balance (higher is better). Neither metric changes the balanced-accuracy comparison above.

## Acceptance and limits

| Check | Observed result |
|---|---|
| Executions | 36 started, 36 completed, 0 failed, 0 retries |
| Epochs | 30–122 completed; selected checkpoints at epochs 1–102 |
| Early checkpoints | 21/36 selected before epoch 30; none reached the 200-epoch ceiling |
| Updates | 6,880; finite gradient diagnostics throughout |
| Classification loss | Lower at the last epoch than at the first in all 36 fits |
| Enabled auxiliary terms | Supported in every sampled batch of the relevant recipes |
| Checkpoints | All 72 best/terminal states independently reloaded and hash-checked |
| Predictions / metrics | All 36 saved validation-logit arrays reproduced exactly; BA, macro-F1 and NLL independently recomputed |
| Shared random streams | All nine station/seed groups passed prefix checks across recipes |
| Resources | 141.664 s total wall time; longest fit 12.566 s; peak allocated CUDA 141.09 MiB; private namespace 63.05 MiB at audit |
| Protected evidence | Within-run code/environment identity unchanged; all 187 earlier core-artifact files unchanged |
| Software | 718 local tests passed, including real serialization and failure cases; public-package and Ruff checks passed |

No fit hit the epoch ceiling, but this does not prove that the stopping rule or architecture is optimal. All three pilot fitting roles had cross-instrument pairs; sparse-role behavior was checked in synthetic tests and the earlier numerical smoke, not newly demonstrated by these three validation units.

## Where this leaves the project

The pilot establishes that the longer training, source-only checkpoint selection, artifact retention and exactly-once accounting work on the selected roles. It provides preliminary, station-dependent evidence for the auxiliary losses, including a clear CWA failure mode. It does **not** establish nuisance removal, disentanglement, instrument independence, a best preprocessing pipeline, or superiority to classical ML.

**Next decision:** review a concrete time/storage allowance for broader, nested source-only development. Any continuation must retain the fixed recipes and choose within each outer context's own source data, including the planned source-master guard folds and D0-M fallback. The accepted 36 slots must be reused, not fitted again. They leave **14,904 of the 14,940 inner slots unexecuted**. The earlier smoke/recovery and this pilot together consumed 71 scientific executions and 8,000 updates. The pilot permit is exhausted; no additional fit, G3 selection, calibration or outer evaluation is authorized by this report. P06 evaluation, P08 preprocessing and P14 RBF/SOM studies remain separate gates.

## Reproducibility

- Execution commit: `b82daec3c58671b45245243f60311f6063fc2729`.
- Original core contract: `60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae`.
- Pilot permit: `652f5c07a1076a907778a9dd80394203ded9084298a95ce791cb5ee2814e576d`.
- Pilot plan: `7c57ba72a98ee018b30973dc35003ef9232babee7d8965fcbfb77b10ada6ca9c`.
- Private run manifest: `847245673400310422cffbdf8f470eb56c5796d9b17a6b43ba24b68149592de1` (185 entries).
- Figure data: `068603f13027122fc53a26b4dc0ce4a146f397d5fb6276fb10ab60c4a6fcc0ff`.

The [aggregate audit](audit.json), [figure manifest](figures/P05_figure_manifest.json), [implementation review](../../plan/delegation/P05_DEVELOPMENT_REVIEW.md), [permit](../../plan/contracts/p05_development_pilot.json) and [master plan](../../plan/MASTER_PLAN.md) record the evidence and boundaries. Spectral arrays, physical identities, row-level predictions and model checkpoints remain private. Hosted CI omits optional torch tests; numerical acceptance rests on the local 718-test run and independent dataset audit.
