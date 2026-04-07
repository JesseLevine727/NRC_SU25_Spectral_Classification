# Experiment Log

## 2026-04-07

### Experiment 001: Exhaustive Pair NNLS

Script:
- `Scripts/run_exhaustive_pair_nnls.py`

Results:
- `Results/exhaustive_pair_nnls/`

Setup:
- Expanded reference library using original `reference_v2.csv` plus pt2 pure compounds
- Dictionary built from class-mean spectra
- Exhaustive search over all binary pairs
- NNLS solve per candidate pair
- Evaluated in two preprocessing modes:
  - `raw`
  - `baseline_corrected`

Headline results:
- `raw`
  - existing real mixtures: exact `0.560`, micro-F1 `0.675`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- `baseline_corrected`
  - existing real mixtures: exact `0.864`, micro-F1 `0.932`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`

Interpretation:
- Sparse binary unmixing is much better aligned with the task than the Siamese + MLP pipeline
- Baseline correction helps substantially on the original real-mixture set
- Perfect pt2 performance suggests the expanded-library framing is correct for this task, but still needed validation against same-family reference leakage

### Experiment 002: Pair NNLS Reject Rule And Split Validation

Script:
- `Scripts/analyze_pair_nnls_reject.py`

Results:
- `Results/pair_nnls_reject_and_split_validation/`

Setup:
- Used the `baseline_corrected` pair-NNLS configuration
- Built a simple reject rule from:
  - relative reconstruction residual
  - minor coefficient share in the best binary pair
- Validated pt2 stability by rebuilding the pt2-added portion of the reference from only half of the pt2 pure replicates across 5 random seeds

Headline results:
- existing real mixtures: exact `0.864`, micro-F1 `0.932`
- pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- learned reject rule:
  - residual_rel `<= 0.4863`
  - minor_share `>= 0.0824`
- reject calibration:
  - mixture accept TPR `0.998`
  - pure reject TNR `1.000`
  - pure accept rate `0.000`
- pt2 split validation:
  - exact `1.000` for all 5 seeds

Interpretation:
- The binary NNLS solution is stable under a basic held-out split of pt2 pure references
- The reject rule cleanly distinguishes pure spectra from binary mixtures in this setup
- This is still not true external cross-batch validation, but it is stronger than the naive all-replicates-in-library case

### Experiment 003: Non-Negative Elastic Net

Script:
- `Scripts/run_nonnegative_elastic_net.py`

Results:
- `Results/nonnegative_elastic_net/`

Setup:
- Expanded reference library using original `reference_v2.csv` plus pt2 pure compounds
- Dictionary built from class-mean spectra
- Full-library non-negative elastic net
- Hyperparameters tuned on original real mixtures
- Grid:
  - `alpha`: `1e-4` to `1e-1`
  - `l1_ratio`: `0.2, 0.5, 0.8, 0.95, 1.0`
  - support threshold by coefficient share
- Evaluated in two preprocessing modes:
  - `raw`
  - `baseline_corrected`

Headline results:
- `raw`
  - selected: `alpha=0.1`, `l1_ratio=0.8`, `share_threshold=0.2`
  - existing real mixtures: exact `0.567`, micro-F1 `0.766`
  - pt2 real mixtures: exact `0.981`, micro-F1 `0.995`
- `baseline_corrected`
  - selected: `alpha=0.03`, `l1_ratio=0.8`, `share_threshold=0.15`
  - existing real mixtures: exact `0.669`, micro-F1 `0.868`
  - pt2 real mixtures: exact `0.889`, micro-F1 `0.971`

Interpretation:
- Non-negative elastic net is strong, but it did not beat exhaustive pair NNLS
- `raw` elastic net generalized better to pt2 than `baseline_corrected` elastic net
- `baseline_corrected` elastic net was better on the original real-mixture set than raw elastic net
- As of now, exhaustive pair NNLS remains the best baseline in this new unmixing track

### Per-Compound Diagnostics

These diagnostics were extracted from the saved prediction files after the main aggregate experiments.

#### Best current model: exhaustive pair NNLS, baseline-corrected

Original real mixtures:

- `benzene`: precision `1.000`, recall `1.000`
- `etoh`: precision `1.000`, recall `1.000`
- `pyridine`: precision `1.000`, recall `1.000`
- `n,n-dimethylformamide`: precision `1.000`, recall `1.000`
- `meoh`: precision `1.000`, recall `0.938`
- `1-dodecanethiol`: precision `1.000`, recall `0.885`
- `6-mercapto-1-hexanol`: precision `1.000`, recall `0.667`
- `benzenethiol`: precision `0.667`, recall `1.000`

Interpretation:

- This model is extremely clean overall
- Main remaining issues are:
  - missed `6-mercapto-1-hexanol`
  - missed `1-dodecanethiol`
  - some false-positive `benzenethiol`

PT2 real mixtures:

- `pyridine`: precision `1.000`, recall `1.000`
- `benzene`: precision `1.000`, recall `1.000`
- `dichloromethane`: precision `1.000`, recall `1.000`
- `n-hexane`: precision `1.000`, recall `1.000`
- `acetonitrile`: precision `1.000`, recall `1.000`
- `6-mercapto-1-hexanol`: precision `1.000`, recall `1.000`
- `diethylamine`: precision `1.000`, recall `1.000`
- `toluene`: precision `1.000`, recall `1.000`

Interpretation:

- Pair NNLS is perfect per compound on the current pt2 evaluation

#### Non-negative elastic net

Original real mixtures, raw:

- `benzene`, `pyridine`, `n,n-dimethylformamide`: precision `1.000`, recall `1.000`
- `etoh`: precision `0.890`, recall `1.000`
- `benzenethiol`: precision `0.886`, recall `0.972`
- `6-mercapto-1-hexanol`: precision `0.686`, recall `0.667`
- `1-dodecanethiol`: precision `1.000`, recall `0.490`
- `meoh`: precision `1.000`, recall `0.486`

Original real mixtures, baseline-corrected:

- `benzene`, `etoh`, `6-mercapto-1-hexanol`, `pyridine`, `n,n-dimethylformamide`: precision `1.000`, recall `1.000`
- `1-dodecanethiol`: precision `1.000`, recall `0.951`
- `meoh`: precision `1.000`, recall `0.556`
- `benzenethiol`: precision `0.667`, recall `1.000`

Interpretation:

- Elastic net is more conservative than the deep model
- On the original real-mixture set, its main weakness is missed `meoh`
- Baseline correction helps `1-dodecanethiol` strongly

PT2 real mixtures, raw:

- all present compounds were precision `1.000`, recall `1.000`

PT2 real mixtures, baseline-corrected:

- all present compounds were precision `1.000`, recall `1.000` except:
  - `n-hexane`: precision `1.000`, recall `0.667`

Interpretation:

- The pt2 errors in baseline-corrected elastic net are localized rather than broad
- `n-hexane` is the only clear weak point in that setting

#### Expanded-reference deep retrain

PT2 real mixtures:

- `pyridine`, `benzene`, `dichloromethane`, `acetonitrile`, `toluene`: precision `1.000`, recall `1.000`
- `n-hexane`: precision `1.000`, recall `0.667`
- `diethylamine`: precision `0.971`, recall `0.944`
- `6-mercapto-1-hexanol`: precision `0.000`, recall `0.000`

Interpretation:

- The deep retrained model still has a real compound-level failure mode
- It completely misses `6-mercapto-1-hexanol` on pt2
- This reinforces the decision to prioritize sparse unmixing over the old deep pipeline

## Current Ranking

### Experiment 004: Pair NNLS With Nuisance Baseline Atoms

Script:
- `Scripts/run_pair_nnls_with_baseline_atoms.py`

Results:
- `Results/pair_nnls_with_baseline_atoms/`

Setup:
- Same expanded reference library as the earlier pair-NNLS baseline
- Same exhaustive binary pair search
- Added smooth nonnegative nuisance baseline atoms to each candidate solve
- Used Bernstein basis functions as baseline atoms
- Swept:
  - preprocessing mode: `raw`, `baseline_corrected`
  - Bernstein degree: `0, 1, 2, 3`

Headline results:
- `raw`, degree `0`
  - existing real mixtures: exact `0.667`, micro-F1 `0.789`
  - pt2 real mixtures: exact `0.926`, micro-F1 `0.963`
- `raw`, degree `1`
  - existing real mixtures: exact `0.812`, micro-F1 `0.892`
  - pt2 real mixtures: exact `0.923`, micro-F1 `0.961`
- `raw`, degree `2`
  - existing real mixtures: exact `0.819`, micro-F1 `0.908`
  - pt2 real mixtures: exact `0.991`, micro-F1 `0.995`
- `raw`, degree `3`
  - existing real mixtures: exact `0.816`, micro-F1 `0.908`
  - pt2 real mixtures: exact `0.972`, micro-F1 `0.986`
- `baseline_corrected`, degree `0`
  - existing real mixtures: exact `0.888`, micro-F1 `0.944`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- `baseline_corrected`, degree `1`
  - existing real mixtures: exact `0.879`, micro-F1 `0.939`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- `baseline_corrected`, degree `2`
  - existing real mixtures: exact `0.879`, micro-F1 `0.939`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- `baseline_corrected`, degree `3`
  - existing real mixtures: exact `0.879`, micro-F1 `0.939`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`

Best configuration:

- `baseline_corrected`, degree `0`
- this corresponds to the original pair-NNLS solver plus one nonnegative constant nuisance atom

Interpretation:

- A very small amount of nuisance background flexibility helps on the original real-mixture set
- More flexible baseline bases did not help further once spectra were already baseline-corrected
- This improves the current best result from:
  - exact `0.864` to `0.888`
  - micro-F1 `0.932` to `0.944`
- The pt2 result stayed perfect

### Experiment 005: Pair NNLS With Replicate-Aware Compound Dictionaries

Script:
- `Scripts/run_pair_nnls_replicate_dictionary.py`

Results:
- `Results/pair_nnls_replicate_dictionary/`

Setup:
- Built on the current best model:
  - baseline-corrected spectra
  - binary pair NNLS
  - one nonnegative constant nuisance atom
- Replaced each single compound-mean atom with:
  - the compound mean atom
  - plus a small set of representative pure spectra from that compound
- Representative spectra were chosen by farthest-point sampling in preprocessed spectral space
- Swept extra representative counts:
  - `0, 2, 4, 9`

Headline results:
- extra reps `0`
  - existing real mixtures: exact `0.888`, micro-F1 `0.944`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- extra reps `2`
  - existing real mixtures: exact `0.821`, micro-F1 `0.910`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- extra reps `4`
  - existing real mixtures: exact `0.902`, micro-F1 `0.951`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`
- extra reps `9`
  - existing real mixtures: exact `0.907`, micro-F1 `0.953`
  - pt2 real mixtures: exact `1.000`, micro-F1 `1.000`

Best configuration:

- extra reps `9`
- up to `10` atoms per compound including the mean atom

Interpretation:

- Replicate-aware dictionaries do help
- The improvement is not monotonic at very small representative counts:
  - `2` extra representatives made things worse
  - `4` and `9` improved over the current best baseline
- The best result now improves the running benchmark from:
  - exact `0.888` to `0.907`
  - micro-F1 `0.944` to `0.953`
- The pt2 result stayed perfect
- This is evidence that within-compound variability matters for the original real-mixture set
- It also strengthens the case that the current ceiling is being set by dictionary quality rather than by needing deep learning yet

### Experiment 006: Harder Validation Of The Binary Anchor

Script:
- `Scripts/run_binary_anchor_harder_validation.py`

Results:
- `Results/binary_anchor_harder_validation/`

Setup:
- Took the current best binary anchor:
  - baseline-corrected spectra
  - binary pair NNLS
  - one constant nuisance baseline atom
  - replicate-aware compound dictionaries with up to `9` extra representatives
- Rebuilt the pure reference library repeatedly with restricted numbers of spectra per compound
- Used shared caps for both:
  - original reference compounds
  - pt2-added pure compounds
- Evaluated across 5 random seeds for each cap:
  - `1, 2, 4, 8, 16, all`

Headline results:
- cap `1`
  - existing real mixtures: exact `0.817 ± 0.123`, micro-F1 `0.907 ± 0.063`
  - pt2 real mixtures: exact `1.000 ± 0.000`, micro-F1 `1.000 ± 0.000`
- cap `2`
  - existing real mixtures: exact `0.803 ± 0.112`, micro-F1 `0.899 ± 0.058`
  - pt2 real mixtures: exact `1.000 ± 0.000`, micro-F1 `1.000 ± 0.000`
- cap `4`
  - existing real mixtures: exact `0.872 ± 0.029`, micro-F1 `0.935 ± 0.015`
  - pt2 real mixtures: exact `1.000 ± 0.000`, micro-F1 `1.000 ± 0.000`
- cap `8`
  - existing real mixtures: exact `0.889 ± 0.013`, micro-F1 `0.944 ± 0.007`
  - pt2 real mixtures: exact `1.000 ± 0.000`, micro-F1 `1.000 ± 0.000`
- cap `16`
  - existing real mixtures: exact `0.896 ± 0.008`, micro-F1 `0.948 ± 0.004`
  - pt2 real mixtures: exact `1.000 ± 0.000`, micro-F1 `1.000 ± 0.000`
- cap `all`
  - existing real mixtures: exact `0.907 ± 0.000`, micro-F1 `0.953 ± 0.000`
  - pt2 real mixtures: exact `1.000 ± 0.000`, micro-F1 `1.000 ± 0.000`

Interpretation:

- The current binary anchor is robust to thinner reference libraries
- On the original real-mixture set, performance degrades gradually rather than collapsing
- The model still performs strongly even with very limited library support per compound
- The pt2 set remains perfect under this validation protocol, which means it is not sensitive to moderate reductions in pt2 pure-library density
- This is a stronger result than the earlier naive full-library evaluation, but it is still not a full open-cardinality validation
- The main conclusion remains: classical sparse unmixing is still improving and still well justified

Implication for next steps:

- We now have a strong and reasonably stress-tested binary anchor
- This is the right point to consider relaxing the binary assumption in the classical setting before moving to deep methods

### Experiment 007: Cardinality-Adaptive NNLS

Script:
- `Scripts/run_cardinality_adaptive_nnls.py`

Results:
- `Results/cardinality_adaptive_nnls/`

Setup:
- Built on the current best dictionary and preprocessing stack:
  - baseline-corrected spectra
  - replicate-aware compound dictionaries with up to `9` extra representatives
  - one constant nuisance baseline atom
- Relaxed the fixed-binary assumption by allowing support sizes `1`, `2`, or `3`
- For each spectrum:
  - screened all single-compound supports
  - screened all binary supports
  - built a shortlist of candidate compounds from the best single and pair fits
  - evaluated candidate supports up to size `3`
- Selected the final support with a calibrated support-size penalty and minimum-share rule

Selected hyperparameters:

- size penalty: `0.01`
- minimum share threshold: `0.03`

Headline results:
- original real mixtures:
  - exact `0.900`
  - micro-F1 `0.969`
- original pure spectra:
  - exact `1.000`
  - micro-F1 `1.000`
- pt2 pure spectra:
  - exact `1.000`
  - micro-F1 `1.000`
- pt2 real mixtures:
  - exact `0.889`
  - micro-F1 `0.973`

Interpretation:

- The cardinality-adaptive solver successfully handles both pure and mixture spectra in one framework
- It achieves very strong performance on the original real-mixture set while staying perfect on both pure sets
- However, it does not beat the current binary anchor overall because it overpredicts a third component on one pt2 mixture family
- The main observed failure mode is extra support rather than missed support
- This means the direction is scientifically valid, but the current support-selection rule is still too permissive for promotion to the top benchmark

Current conclusion:

- keep the binary anchor as the main benchmark
- treat cardinality-adaptive NNLS as the active open-cardinality prototype to improve next
- focus next on better support-size selection rather than changing the dictionary again immediately

### Experiment 008: Operational Calibration Of The Binary Anchor

Script:
- `Scripts/run_binary_anchor_operational_calibration.py`

Results:
- `Results/binary_anchor_operational_calibration/`

Setup:
- Started from the current best binary anchor:
  - baseline-corrected spectra
  - replicate-aware compound dictionaries with up to `9` extra representatives
  - one constant nuisance baseline atom
- Kept binary pair inference fixed
- Added a calibrated accept / reject gate for the question:
  - "does this spectrum look like a trustworthy binary-mixture prediction?"
- Calibrated the gate using:
  - positives: original real binary mixtures
  - negatives: original pure spectra and pt2 pure spectra
- Swept thresholds over:
  - relative reconstruction residual
  - minor coefficient share
  - pair-gap ratio

Selected thresholds:

- residual relative threshold: `0.4237`
- minor share threshold: `0.0979`
- gap ratio threshold: `0.0000`

Headline results:
- existing real mixtures:
  - coverage `0.969`
  - accepted exact `0.909`
  - accepted micro-F1 `0.955`
- pt2 real mixtures:
  - coverage `1.000`
  - accepted exact `1.000`
  - accepted micro-F1 `1.000`
- original pure spectra:
  - binary reject rate `1.000`
- pt2 pure spectra:
  - binary reject rate `1.000`

Interpretation:

- This does not replace the binary anchor as the primary benchmark because it introduces abstention
- It does finalize the current operational story:
  - use the binary anchor when the spectrum passes the binary-compatibility gate
  - reject / escalate spectra that do not look like trustworthy binary mixtures
- In practice, the calibrated gate rejects all pure spectra while preserving almost all real binary mixtures
- The pair-gap feature was not needed in the selected solution; the useful calibration came from residual and minor-share structure
- This supports the current framing:
  - binary-constrained inference is appropriate for the present dataset
  - open-cardinality handling remains exploratory rather than the main deployment path

### Experiment 009: Targeted Audit Of Remaining Original-Mixture Errors

Script:
- `Scripts/run_existing_real_failure_audit.py`

Results:
- `Results/existing_real_failure_audit/`

Setup:
- Audited the current best binary anchor on the original real-mixture set only
- Merged:
  - the best replicate-aware pair-NNLS predictions
  - the operational binary gate outputs
- Split the remaining non-perfect behavior into:
  - accepted misclassifications
  - rejected but actually correct binary predictions
  - rejected and incorrect predictions
- Grouped results by true mixture pair and by confusion pattern

Headline results:
- original real samples audited: `580`
- anchor exact match: `0.9069`
- total anchor errors: `54`
- operational accepted count: `562`
- operational rejected count: `18`
- accepted errors: `51`
- rejected but anchor-correct: `15`
- rejected and anchor-incorrect: `3`

Dominant failure modes:
- `6-mercapto-1-hexanol + pyridine` -> `benzenethiol + pyridine`
  - `36 / 36` samples
- `1-dodecanethiol + meoh` family
  - `9` samples predicted as `1-dodecanethiol + diethylamine`
  - `8` samples predicted as `1-undecanethiol + meoh`
  - `1` sample predicted as `1-dodecanethiol + tris(2-ethylhexyl) phosphate`

Interpretation:

- The classical ceiling is not being set by broad model weakness anymore
- Almost all remaining error is concentrated in two chemistry-specific regions:
  - a complete confusion between `6-mercapto-1-hexanol` and `benzenethiol` when paired with `pyridine`
  - a smaller ambiguity cluster around `1-dodecanethiol + meoh`
- The operational gate mostly rejects borderline `1-dodecanethiol + meoh` cases
- It does not catch the `6-mercapto-1-hexanol + pyridine` failure because those predictions are made confidently under the current features

Implication for next steps:

- The next classical round should be targeted, not broad
- Priority order:
  - inspect the `6-mercapto-1-hexanol` vs `benzenethiol` dictionary geometry in the presence of `pyridine`
  - inspect the `1-dodecanethiol + meoh` neighborhood for within-class reference overlap and near-tie pair fits
- This is still not a reason to pivot to deep learning yet

### Experiment 010: Selective Low-Baseline Fallback For Pair NNLS

Scripts:
- `Scripts/run_targeted_pair_diagnostics.py`
- `Scripts/run_pair_nnls_baseline_penalty.py`
- `Scripts/run_pair_nnls_baseline_fallback.py`

Results:
- `Results/targeted_pair_diagnostics/`
- `Results/pair_nnls_baseline_penalty/`
- `Results/pair_nnls_baseline_fallback/`

Setup:
- Started from the current best binary anchor:
  - baseline-corrected spectra
  - replicate-aware compound dictionaries with up to `9` extra representatives
  - one constant nuisance baseline atom
- First diagnosed the two remaining dominant error regions:
  - `6-mercapto-1-hexanol + pyridine`
  - `1-dodecanethiol + meoh`
- Diagnostics showed:
  - `1-dodecanethiol + meoh` is mainly a near-tie problem
  - `6-mercapto-1-hexanol + pyridine` is a systematic ranking problem
  - the wrong `benzenethiol + pyridine` solution relies strongly on the nuisance baseline atom, while the true pair does not
- Tested two fixes:
  - a global baseline-use penalty
  - a selective fallback rule that only reranks when:
    - the best pair uses unusually high baseline mass
    - and a low-baseline alternative is close in residual

Diagnostic headline:
- `1-dodecanethiol + meoh`
  - top-1 exact `0.886`
  - mean true-pair rank `1.44`
  - mean residual gap to best pair `0.0008`
- `6-mercapto-1-hexanol + pyridine`
  - top-1 exact `0.000`
  - mean true-pair rank `2.89`
  - mean residual gap to best pair `0.0389`

Global-penalty result:
- a large baseline penalty could fix `6-mercapto-1-hexanol + pyridine`
- but it damaged `1-dodecanethiol + meoh` too much
- not promoted

Best selective fallback configuration:
- baseline relative threshold: `0.025`
- low-baseline alternative threshold: `0.005`
- residual margin: `0.05`

Headline results:
- existing real mixtures:
  - exact `0.969`
  - micro-F1 `0.984`
- pt2 real mixtures:
  - exact `1.000`
  - micro-F1 `1.000`
- target pair accuracy:
  - `6-mercapto-1-hexanol + pyridine`: `1.000`
  - `1-dodecanethiol + meoh`: `0.886`
- fallback trigger rate:
  - existing real: `0.062`
  - pt2 real: `0.000`

Interpretation:

- This is the first classical change that materially improves the benchmark after the replicate-aware dictionary
- The fix is targeted and interpretable:
  - only override the raw best pair when that best pair appears to be leaning too heavily on the nuisance baseline atom
  - and only when a low-baseline alternative is still competitive in reconstruction error
- It completely resolves the `6-mercapto-1-hexanol + pyridine` failure mode without harming pt2
- The remaining main weakness is now concentrated almost entirely in `1-dodecanethiol + meoh`

Implication for next steps:

- keep this selective-fallback binary solver as the new benchmark
- the final major classical target is now the `1-dodecanethiol + meoh` family
- deep learning is still not justified yet because the error surface has become even more localized and class-specific

### Experiment 011: Family-Specific Near-Tie Fallback For `1-dodecanethiol + meoh`

Script:
- `Scripts/run_pair_nnls_family_fallback.py`

Results:
- `Results/pair_nnls_family_fallback/`

Setup:
- Built directly on the current best selective low-baseline fallback solver
- Added one additional narrow reranking rule for the remaining `1-dodecanethiol + meoh` cluster
- Applied only when:
  - the current selected pair was one of:
    - `1-undecanethiol + meoh`
    - `1-dodecanethiol + tris(2-ethylhexyl) phosphate`
  - and `1-dodecanethiol + meoh` was present as a close alternative within a very small residual margin

Best configuration:
- family residual margin: `0.002`

Headline results:
- existing real mixtures:
  - exact `0.984`
  - micro-F1 `0.992`
- pt2 real mixtures:
  - exact `1.000`
  - micro-F1 `1.000`
- target pair accuracy:
  - `6-mercapto-1-hexanol + pyridine`: `1.000`
  - `1-dodecanethiol + meoh`: `0.943`
- fallback trigger rates:
  - existing low-baseline fallback: `0.062`
  - existing family fallback: `0.016`
  - pt2 low-baseline fallback: `0.000`
  - pt2 family fallback: `0.000`

Interpretation:

- This is the strongest classical result in the repo so far
- The new family fallback improves the remaining `1-dodecanethiol + meoh` neighborhood substantially:
  - from `0.886` to `0.943`
- The improvement comes from correcting the very tight local near-ties:
  - `1-undecanethiol + meoh`
  - occasional `1-dodecanethiol + tris(2-ethylhexyl) phosphate`
- The harder `1-dodecanethiol + diethylamine` subset still remains and now dominates the residual error budget
- pt2 remains untouched, which is important because the fallback never triggered there

Implication for next steps:

- This is likely the practical classical ceiling for the current binary-first framing
- The remaining error surface is now extremely small and highly localized
- The next decision is no longer "broad classical versus deep"
- It is now:
  - either stop and freeze this classical benchmark
  - or do one last targeted analysis of the `1-dodecanethiol + diethylamine` collapse to decide whether it is recoverable classically at all

## Current Benchmark View

### Clean benchmark ranking

These are the methods that should count as the main scientific comparison set because they do not rely on pair- or family-specific override rules.

1. Pair NNLS + nuisance baseline atom + replicate-aware dictionary, `baseline_corrected`, extra reps `9`
2. Pair NNLS + nuisance baseline atom, `baseline_corrected`, degree `0`
3. Exhaustive pair NNLS, `baseline_corrected`
4. Non-negative elastic net, `baseline_corrected`
5. Non-negative elastic net, `raw`
6. Exhaustive pair NNLS, `raw`

### Diagnostic engineering variants

These variants improved performance, but they should be treated separately from the clean benchmark because they use localized decision logic discovered from the observed confusion structure.

1. Pair NNLS + replicate-aware dictionary + selective low-baseline fallback + family near-tie fallback, `baseline_corrected`, extra reps `9`
2. Pair NNLS + replicate-aware dictionary + selective low-baseline fallback, `baseline_corrected`, extra reps `9`

Not promoted into either main ranking:

- Cardinality-adaptive NNLS prototype
  - strong on the original real-mixture set
  - not yet benchmark-beating overall because pt2 binary-mixture exactness dropped from `1.000` to `0.889`
- Operationally calibrated binary anchor
  - useful as a deployment-facing reject / abstain layer
  - not directly comparable because it can abstain
- Global baseline-penalty pair NNLS
  - fixes the `6-mercapto-1-hexanol + pyridine` failure only at the cost of broader regression

Benchmark criterion:
- priority to real-mixture exact support recovery
- then micro-F1
- then simplicity and interpretability

Current interpretation:

- if the goal is a clean, defensible general method, the replicate-aware binary pair NNLS anchor remains the main benchmark
- if the goal is maximum performance on the current dataset, the localized fallback variants currently perform best
- this split is the main reason a deep pivot is now reasonable to discuss

## Next Candidate Experiments

1. Add nuisance baseline atoms / polynomial background terms to the pair-NNLS solver
2. Move from class-mean dictionary atoms to replicate-level dictionary atoms
3. Test 3-sparse exhaustive search if ternary mixtures become relevant
4. Add true external cross-batch validation when more independent pure references are available
