# P04 compact D0 execution contract

## Purpose and boundary

P04 implements the ordinary compact deep-learning control for `RQ-P01`. It asks whether a location-preserving one-dimensional residual classifier can learn the station-local chemical task under the same `PP-U-MIN` representation and P02 physical-master partitions used by P03. It does not claim denoising, causal nuisance removal, or chemical/acquisition disentanglement.

The P04 no-fit expansion was completed before successful P04 model outcomes. P03 and P13 results were already available to the investigators, so P04 is not described as analyst-blinded. Instead, the separation is procedural and code-enforced: those earlier outcome tables are prohibited inputs to architecture, optimizer-grid, augmentation, epoch, regularization, seed, calibration, or fallback decisions. The P04 fitting and selection path does not load them; the separate post-freeze comparison path does so only after D0 development is frozen and all D0 held predictions exist.

## Frozen input

- Population: 598 SERS spectra representing 69 physical-master samples.
- Representation: `R_MIN_400_1800`, exactly 1,401 values on the integer 400–1,800 cm⁻¹ axis.
- Preprocessing: `PP-U-MIN`; no background subtraction or smoothing is introduced in the primary learning-strategy comparison.
- Outer design: five P02 repeat seeds, four station-stratified physical-master folds, 13 primary held-instrument domains.
- Station-local vocabulary: three chemical classes per fitted head.
- Unit rule: all rows from one `master_sample_id` remain on the same side of every train/validation/test boundary.

The no-fit validator reconstructs every UID role, verifies the 598-by-1,401 tensor, checks exact row order and row min–max invariants, and confirms that every T3 fitting or validation role excludes both the held instrument and outer-test masters.

## Architecture

`D0-ERM` uses the exact locked channel, kernel, dilation, and ordered-pooling sequence:

1. `Conv1d 1→24`, kernel 11, padding 5; GroupNorm; GELU.
2. Two 24-channel residual blocks, kernel 7, dilations 1 and 2.
3. `Conv1d 24→48`, kernel 5, stride 2; GroupNorm; GELU.
4. Two 48-channel residual blocks, kernel 7, dilations 1 and 2.
5. `Conv1d 48→64`, kernel 5, stride 2; GroupNorm; GELU.
6. Two 64-channel residual blocks, kernel 5, dilations 1 and 2.
7. Adaptive mean pooling to 16 ordered bins.
8. Projection `1024→96→64` with GELU and dropout 0.2.
9. A station-local `64→3` linear classification head.

The earlier plan did not specify the internal number of convolutions in a residual block. Before fitting, P04 resolved each block as one pre-activation `GroupNorm→GELU→dilated Conv1d` branch plus its identity skip. A conventional two-convolution branch would exceed the already-locked 250,000-parameter ceiling. The implemented model contains exactly **208,691 trainable parameters**, has no BatchNorm module, maps `2×1×1401 → 2×64 → 2×3` in its audit, and leaves 41,309 parameters of headroom.

PyTorch 2.9 does not expose a deterministic CUDA backward implementation for `AdaptiveAvgPool1d`. After a retained terminal smoke-test error and before any successful model outcome, the same adaptive-average operation was expressed explicitly using the standard `floor(iL/16):ceil((i+1)L/16)` bin boundaries and tensor means. Forward parity with PyTorch adaptive pooling is unit-tested. This changes no layer, tensor shape, parameter, or statistic.

## Optimization and selection

Every authorized inner selection unit evaluates the full Cartesian grid:

- learning rate `{0.0003, 0.001}`;
- weight decay `{0.00001, 0.0001, 0.001}`;
- neural seeds `{20260805, 20260817, 20260829}`;
- AdamW, batch size 48, inverse-fitting-spectrum-frequency class-balanced cross-entropy;
- at most 200 epochs, at least 30 epochs, patience 20, validation every epoch;
- gradient norm clipped at 5.

The best checkpoint within a fit is the highest validation balanced accuracy, then lowest validation negative log likelihood, then earliest epoch. Candidate selection first averages the three seeds within each source validation unit and then ranks candidates by mean unit balanced accuracy, worst unit balanced accuracy, mean macro-F1, fixed complexity rank, and declared order.

For a final outer refit, each seed uses the rounded median of that seed's best epochs across the selected candidate's authorized inner units, clipped to `[30,200]`. The model trains on all permitted outer fitting rows for exactly this fixed count. Test data are not passed to stopping, candidate selection, temperature fitting, or any retraining decision.

`EXP-N00-DEV` uses the three non-test physical-master folds as inner validation units. `EXP-N00-T3` uses P02-supported leave-one-source-instrument-out pseudo-domains; where fewer than two are supported, it uses the exact P02 three-fold source-master fallback. No substitute pseudo-domain is invented.

## Training-only augmentation

Each fitting example receives a deterministically replayable draw of:

- wavenumber translation uniformly in ±2 cm⁻¹;
- multiplicative intensity 0.9–1.1;
- signed linear baseline span up to 5% of the normalized row range;
- Gaussian noise at zero or the fitting-role 25th, 50th, or 75th quantile of `first_difference_noise_mad / intensity_range`.

The first-difference MAD level is converted to a white-noise standard deviation by division by `0.9538725524`. The augmented row is min–max rescaled only after intensity, baseline, noise, and translation are applied. Validation and test spectra are never augmented. Each fit stores the fitting-only noise levels, deterministic draw count, and draw digest; fit ID, seed, epoch, batch ordinal, and the versioned algorithm reconstruct every per-example draw. Target-derived noise, arbitrary warping, peak deletion, and different-chemical mixup are prohibited.

## Calibration and seed ensemble

One scalar temperature is fitted separately for each final context and neural seed using only selected-candidate inner validation logits. Logits are averaged within a physical master before temperature optimization so each calibration master has equal influence. The temperature is then applied to that seed's untouched outer-test logits. Calibrated probabilities are averaged across all three registered seeds before M01 or M06 scoring.

## Exact no-fit expansion

The deterministic plan contains:

- 320 outer contexts: 60 development and 260 T3;
- 15,498 inner-selection fits;
- 960 conditional final refits;
- 16,458 total planned fits;
- 3,420 development fits and 13,038 T3 fits;
- 132,392 private UID-role rows;
- 320 independently resumable context shards.

The earlier compute file contained only a nonauthorizing rough estimate. The exact expansion is larger because leakage-safe tuning repeats all six optimizer settings, three seeds, and every source-only selection unit inside every outer context. Replacing this with one global setting would reduce compute but allow a held instrument or an outer-test physical sample to influence some later endpoint.

## Failure handling and G2

Each fit terminates as complete, numerical failure, resource failure, data failure, fit failure, or protocol exclusion. Failed and collapsed fits are never silently restarted. Histories classify collapse, underfit, overfit, and optimization instability using the definitions in `p04_execution_contract.json`.

G2 is evaluated only from development inner-selection execution records. At least 95% must produce finite checkpoints, and every development context must select exactly one candidate. Development outer-test accuracy, P03 outcomes, and P13 outcomes are not G2 inputs. A failed G2 blocks T3.

Three immutable smoke-test runs are retained outside the repository:

- two runs terminated because deterministic cuBLAS requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- one run terminated because CUDA adaptive-average backward is nondeterministic.

Neither failure produced a validation or test prediction. The locked environment setting and mathematically equivalent pooling implementation were versioned before the first successful outcome.

## Private evidence and public release

Private context shards retain fit status, full histories, source-validation logits, final seed predictions, temperatures, final checkpoints, timings, memory, and hashes. The public release contains only disclosure-safe aggregate training, endpoint, domain, comparison, and figure tables. It excludes observation UIDs, master IDs, context/fold identities, row probabilities, and checkpoints.

Figures F19, F20, and F48 are generated from one frozen semantic table apiece as native TikZ, vector PDF, 300-DPI PNG, and standalone HTML. Hash parity prevents the static and interactive versions from showing different statistics.

## Comparison estimands and limits

The P04 baseline report retains two explicitly labelled summaries. The endpoint
table averages balanced accuracy over complete paired outer cells. The paired
bootstrap first pools out-of-fold class recalls within each held domain and
averages repeated predictions; its point estimate can therefore differ from
the mean of fold-level balanced accuracies when fold sizes differ. It draws
5,000 physical-master samples, stratified by station and class, preserving a
master's repeated and cross-instrument predictions together. Domain-specific
figures report the actual number of masters in each domain. The overall
interval conditions on the observed set of domains.
Draws that lose a required class in any included domain are rejected, so the
interval also conditions on retaining three-class support. Small fold-level
test cells can lack a class: their inherited BA averages the classes present,
whereas the pooled domain statistic requires all three station-local classes.

This is the P04 baseline diagnostic, not the P11 final primary inference.
P11 additionally resamples domains, uses 10,000 draws, and applies the declared
leave-domain and leave-instrument sensitivities. A P04 interval cannot by
itself pass G4 or demonstrate equivalence. The full research question about
acquisition-aware learning remains open until P05/P06/P11.

The P13 extension partitions frozen P04 predictions into the exact PP-U-MIN
substrate views for 15 of the 16 eligible P13 domains, including all 13
confirmatory domains. The exploratory CWA/Agilent-3 domain is outside P04's
frozen instrument-domain set. The extension uses P13's 10,000-draw held-recovery
interval and 0.60 lower-bound rule. It does not supply P13's matched-source-loss
endpoint or its SG/arPLS deep sensitivities, so it does not complete the full
P13 deep portability experiments or issue a dual-margin portability verdict.
Moreover, P04 fits all source substrates within a station, while the P13
classical protocol restricts fitting to the substrate family being evaluated.
Shared held-test UIDs therefore establish test-view parity only. The reused
D0-versus-P13-classical differences are descriptive and combine model and
training-support differences. Completing the P13 deep learner comparison
requires substrate-restricted deep refits on the exact P13 source roles.

## Commands

```bash
python scripts/run_p04.py plan
python scripts/run_p04.py validate-plan
python scripts/run_p04.py execute-batch --phase development
python scripts/run_p04.py freeze-development
python scripts/run_p04.py execute-batch --phase held_evaluation
python scripts/run_p04.py aggregate
python scripts/run_p04.py compare
python scripts/publish_p04_results.py
python scripts/validate_p04_public_release.py
```

All commands require the private roots documented in the repository README. The plan, development freeze, context shards, aggregate, comparison, and public release each carry separate hashes.
