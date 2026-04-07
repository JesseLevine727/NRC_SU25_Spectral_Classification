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

## Current Ranking

1. Pair NNLS + nuisance baseline atom + replicate-aware dictionary, `baseline_corrected`, extra reps `9`
2. Pair NNLS + nuisance baseline atom, `baseline_corrected`, degree `0`
3. Exhaustive pair NNLS, `baseline_corrected`
4. Non-negative elastic net, `baseline_corrected`
5. Non-negative elastic net, `raw`
6. Exhaustive pair NNLS, `raw`

Not promoted into the ranking:

- Cardinality-adaptive NNLS prototype
  - strong on the original real-mixture set
  - not yet benchmark-beating overall because pt2 binary-mixture exactness dropped from `1.000` to `0.889`

Ranking criterion:
- priority to real-mixture exact support recovery
- then micro-F1
- then simplicity and interpretability

## Next Candidate Experiments

1. Add nuisance baseline atoms / polynomial background terms to the pair-NNLS solver
2. Move from class-mean dictionary atoms to replicate-level dictionary atoms
3. Test 3-sparse exhaustive search if ternary mixtures become relevant
4. Add true external cross-batch validation when more independent pure references are available
