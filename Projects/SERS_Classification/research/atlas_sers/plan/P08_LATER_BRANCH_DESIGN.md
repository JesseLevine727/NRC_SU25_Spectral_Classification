# P08 later branches: numerical constraints and remaining decisions

**Status:** no-fit design audit, not a numerical execution lock or permit.

This note separates the approved universal preprocessing comparison from the later controls in [Master Plan §16](MASTER_PLAN.md#16-sub-plan-p08--preprocessing-policy-factorial-and-robustness). It records arithmetic and metadata findings, not new classification results. The universal MIN/SG/arPLS comparison, the 54-context adaptive subset and the 206 complete MIN fallbacks are unchanged.

## 1. Normalization controls must remain distinct

The three primary pipelines end in per-spectrum min–max scaling. The exploratory normalization controls do not. Their frozen definitions in the [P01 contract](contracts/p01_governance_contract.json) are:

| Representation | Final operation and scale | Interpretation |
|---|---|---|
| `R_MIN_400_1800`, `R_SG_400_1800`, `R_ARPLS_400_1800` | Min–max; minimum 0 and maximum 1 for valid rows | Compare the complete primary pipelines on the same output range |
| `R_SNV_400_1800` | Subtract row mean and divide by row standard deviation | Zero mean and unit standard deviation; negative values are expected |
| `R_VECTOR_400_1800` | Divide the original row by its L2 norm | Unit vector norm; no additional minimum subtraction |
| `R_AREA_400_1800` | Subtract the row minimum, then divide by trapezoidal area over the wavenumber axis | Unit integrated area, not unit sum or unit maximum |
| `R_D1_400_1800` | Frozen impulse replacement, Savitzky–Golay first derivative, then SNV | Destructive control with signed, standardized derivative values |

For a nonconstant row that passes the normalization and final min–max nondegeneracy checks, each of SNV, vector normalization and area normalization is a positive affine transformation, `T(x) = a x + b`, with `a > 0`. Consequently, `minmax(T(x)) = minmax(x)`, apart from numerical rounding. Adding final min–max scaling would erase the intended normalization contrast. This identity does not apply to baseline removal, smoothing or differentiation, which can change spectral shape.

Each control therefore retains its own common scaling rule within that experimental arm. This does not change the user's requested [0,1] scale for the three primary pipelines. Nonfinite or degenerate rows retain the frozen validity rules; epsilon handling is not evidence that an invalid row became scientifically usable. The derivative representation's `D1` name is unrelated to the neural recipe named D1.

The [synthetic design tests](../tests/test_p08_later_design.py) check these identities, native invariants, zero-row rejection and the frozen operation lists. They contain invented arrays only. No scientific spectrum was transformed for this audit.

## 2. A full classical selector is a substantial experiment

The pending choice of a “classical champion” must be defined by source data, not by the best held-test score. Repeating the full historical selector means considering all nine families, not just the three classical families in the universal panel. The [metadata-only accounting audit](../results/p08_readiness/later_classical_selector_bounds.json) authenticates the original candidate registry and fit manifest.

| Family | Hyperparameter candidates | Candidate–seed pairs |
|---|---:|---:|
| Class-prior reference | 2 | 2 |
| Spectral matching | 3 | 3 |
| Nearest centroid | 8 | 8 |
| PCA–LDA | 10 | 10 |
| PLS–DA | 5 | 5 |
| Elastic-net logistic regression | 30 | 30 |
| RBF-SVM | 36 | 36 |
| Random Forest | 16 | 48 |
| Extra Trees | 16 | 48 |
| Total | **126** | **190** |

Across the unchanged 681 source-selection units, one new representation would require 190 × 681 = **129,390 source-fit slots**. The selected family then requires at most 1,152 fresh calibration fits and 780 final fits. These ceilings use three seeds if a stochastic tree family wins; a deterministic winner uses one seed. They do not sum the mutually exclusive one-seed and three-seed outcomes.

Exact within-representation role matching can reuse 396 calibration roles from source selection, corresponding to at most 1,188 seed-level prediction aliases. After that authenticated reuse, the ceiling is **131,322 model fits per representation**, plus at most **260 scalar temperature fits**. Without that reuse, the literal model-fit ceiling is 132,510. Historical MIN estimators or predictions cannot process a different representation merely because the roles match.

If the same full-selector procedure is approved for all three normalization controls and the derivative control, the combined ceiling is **525,288 model fits and 1,040 scalar temperatures**. This conditional bound is not an approved budget, exact executable job graph or prediction of runtime. It excludes the range and population branches. Full wall-time, memory, artifact and prediction accounting remains necessary before a launch request.

The subsequent [N1 resource proposal](P08_RESOURCE_PROPOSAL.md#conditional-n1-proposal-full-selector-normalization-controls) adds finite, conditional ceilings without approving the selector choice. Its [timing basis](../results/p08_readiness/later_classical_timing_basis.json) preserves historical rank and convergence failures. The recorded source-fit durations sum to 123,953.952 seconds per historical selector; four copies give 137.73 sequential CPU-hours before calibration, final fitting, inference and persistence. These recorded costs do not guarantee new-representation runtimes or establish a launchable operation ledger.

The alternative is to freeze the family chosen from MIN source data within each context and retune only its hyperparameters for each control. That answers a narrower question: how normalization affects a fixed source-selected family. The following audit supplies its cost from the actual source-selection map. The eight missing historical selected-classical final references must not be imputed, but their absence does not by itself prove that source family selections are missing.

**Alternative audited, 2026-10-02:** the [saved-family audit](../results/p08_readiness/fixed_family_alternative_audit.json) confirms complete source-only family selections for all **260 contexts**, including the **eight** whose historical final predictions are unavailable. The family map is authenticated, not recomputed. It spans eight of the nine registered families; the class-prior reference was never selected. Selection frequencies describe the recorded source procedure, not a ranking by held-test performance.

| MIN-selected family | Contexts | Candidate–seed source-fit slots per new control |
|---|---:|---:|
| PLS–DA | 80 | 1,150 |
| Elastic-net logistic regression | 64 | 4,800 |
| Spectral matching | 58 | 456 |
| PCA–LDA | 18 | 410 |
| Nearest centroid | 13 | 264 |
| RBF-SVM | 12 | 972 |
| Random Forest | 11 | 1,392 |
| Extra Trees | 4 | 432 |
| Total | **260** | **9,876** |

After exact fit/validation-role matching, this alternative adds **426 fresh calibration fits** and **290 final fits** per representation. Its **444 calibration prediction aliases** are within-representation role reuse, not reuse of old MIN predictions on new inputs. The ceiling is therefore **10,592 model fits and 260 scalar temperatures per control**, or **42,368 fits and 1,040 scalar temperatures** across the four controls. The corresponding full-selector ceilings remain **525,288** and **1,040**. The [conditional resource proposals](P08_RESOURCE_PROPOSAL.md#13-conditional-n2-proposal-frozen-min-selected-families) are alternatives, not cumulative permission to execute both.

Neither choice repairs the historical **252/260** final-result coverage. Even complete new-control results would pair with at most 252 historical C-SELECTED contexts under the current evidence lock. A complete family choice is not a complete fitted/calibrated test result. If every candidate within a frozen family fails for a new control, the context remains unavailable; the implementation may not silently switch families or substitute MIN.

The audited alternative's historical source-fitting records contain **9,488 complete attempts**, **320 convergence failures** and **68 rank failures**. All recorded costs are retained. Four copies of their **7,988.616-second** sum give **8.88 sequential CPU-hours** for source fitting alone, not a measured control runtime or completion guarantee. The owner has been asked to choose between the broader full-selector procedure and this narrower fixed-family sensitivity using these audited costs. No answer or training permission is inferred.

## 3. Rigid shifts need an explicit edge rule

Use the stated convention `y_shift(v) = y(v − delta)`. The model's primary input spans 400–1,800 cm⁻¹. The frozen interpolated common-support input spans 400–1,849 cm⁻¹. Coordinate arithmetic gives:

| Available input support | Shift | Unsupported coordinates needed for the primary model input |
|---|---:|---:|
| 400–1,849 cm⁻¹ | +5 cm⁻¹ | 5, at 395–399 cm⁻¹ |
| 400–1,849 cm⁻¹ | −5 cm⁻¹ | 0; required support is 405–1,805 cm⁻¹ |
| 400–1,800 cm⁻¹ | +5 cm⁻¹ | 5, at the lower edge |
| 400–1,800 cm⁻¹ | −5 cm⁻¹ | 5, at the upper edge |

These counts concern the registered arrays, not a claim that every original instrument lacks all lower-wavenumber measurements. The synthetic tests check only coordinate membership. They implement no shift, padding or extrapolation.

The later perturbation lock must state how unsupported edges are handled. Endpoint extension would be a declared synthetic boundary assumption, not recovered measured signal. Returning to native measurements would require a separately governed input definition. Dropping channels would change the model input. None of these choices follows automatically from the approved shift range, and none has been implemented or approved here.

The owner has now been asked whether to approve constant endpoint extension for this labelled synthetic stress test only. The recommendation retains the registered model width and leaves measured-data preprocessing unchanged. It remains pending, and cannot be described as measured spectral support or a physically validated instrument error model.

Perturbation placement also remains pending. Corruption before the numerical preprocessing stages asks whether preprocessing protects identification; corruption after those stages asks about classifier-input sensitivity. The frozen common-support input is already interpolated, so “before preprocessing” must identify its starting array rather than imply a native irregular-axis experiment. The two questions cannot be silently interchanged.

## 4. Range and population controls need separate role accounting

`R_MIN_400_1849` contains 1,450 channels, compared with 1,401 in the primary input. The range sensitivity must verify the frozen architecture's dimensional interface and declare any required P08-specific input specification. A historical input-width guard must not be weakened globally. Extra Trees was approved for the universal three-policy panel; that approval does not silently expand every later panel.

The [frozen range-input audit](../results/p08_readiness/range_input_audit.json) authenticates all 598 rows, their original ordering, the 400–1,849 cm⁻¹ axis, stored validity flags and final [0,1] normalization. No row is invalid in this representation. The input therefore preserves the primary population and can retain its registered source/test roles; the audit does not authorize fitting or establish that the wider range improves identification.

This branch compares complete input definitions, not the isolated benefit of 49 extra channels. Min–max normalization over the wider range can rescale channels in the shared interval. The existing neural encoder uses adaptive pooling, whose bin boundaries also depend on input width. Architecture compatibility must therefore be distinguished from identical internal responses or prediction parity. The comparison retains RBF-SVM, Random Forest, D0-M and the context-local source-selected strategy; Extra Trees remains outside this range branch.

The range branch remains separate from the universal policy-effect families in the [statistical protocol](P08_STATISTICAL_PROTOCOL.md#3-contrast-registry-and-multiplicity). Its four methods and two endpoints imply eight paired range contrasts. Two deep strategies against two classical methods imply eight range–model interactions. The [numerical range addendum](P08_RANGE_INFERENCE.md) now fixes those two separate multiplicity families, missing-cell rules and the inherited conditional uncertainty procedure before outcomes. It adds no tests to the primary universal families and grants no permission to resample.

The [metadata-only range ledger](../results/p08_readiness/range_ledger_audit.json) now enumerates 131,199 operations, including 62,981 future fits and 1,417 scalar calibrations. Every source/test role, candidate, seed and dependency was compared with the authenticated primary graph. The [separate R1 proposal](P08_RESOURCE_PROPOSAL.md#12-r1-proposal-frozen-wider-range-sensitivity) supplies finite but unapproved ceilings. The numerical inference specification is now recorded separately; metadata accounting does not establish inference implementation, runtime acceptance or sufficient storage.

Population controls regenerate master-separated splits for the notes-clear and Mira-1-excluded tiers. The 260 primary contexts and their source-role counts cannot simply be copied to the regenerated populations. Support must be recomputed from membership metadata, with newly unsupported domains reported. Reusing a context label is not proof that its fitting, selection or test rows match.

The owner has been asked whether these regenerated tiers should repeat source-only model selection and then freeze neural recipe identity, or use a separately specified fixed-model comparison. No answer is inferred here. Neither a previously inspected held score nor the primary context's recipe may select a model for a changed source population by convenience.

## 5. Next decision boundary

The three submitted decisions remain open: full versus fixed-family normalization selection, source selection for regenerated populations, and perturbation placement. A later perturbation specification must additionally resolve edge handling, discrete severity grids, impulse magnitude, random replicates, source-noise estimation, zero-dose parity and statistical contrast families. These details must be locked before new outcomes, not filled in after viewing degradation curves.

The universal comparison remains the first proposed numerical stage. Later-branch uncertainty does not authorize narrowing their scientific scope or launching them. The [resource proposal](P08_RESOURCE_PROPOSAL.md) and [readiness record](P08_READINESS.md) retain the separate runtime-review and execution-approval gates. Native TikZ and offline HTML figures remain planned outputs; no new P08 outcome plot exists.
