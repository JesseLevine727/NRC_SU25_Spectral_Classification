# P08 synthetic spectral stress tests

**Specified:** 2026-10-07. **State:** pre-outcome numerical design with owner-approved model panel and QC interpretation; not an execution permit. The [registry](contracts/p08_perturbation_design.json) records the numerical choices below, and the [case inventory](../results/p08_readiness/perturbation_case_inventory.json) enumerates their descriptors. Exact prediction/reconstruction accounting, finite resources and numerical acceptance remain incomplete.

**Inference checkpoint:** the subsequent [comparison specification](P08_PERTURBATION_INFERENCE.md) and its registry now define all robustness effects, interactions, multiplicity families, support, uncertainty and missingness. They supersede the pending-inference wording retained below. Numerical inference acceptance and execution remain unauthorized; the disturbance geometry and 96-case definition are unchanged.

## 1. Question and scope

These tests ask whether the registered preprocessing-and-classification pipelines retain identification performance under a defined change to the test spectrum. They do not estimate the frequency of that disturbance in the field trial, reconstruct clean chemistry or establish a particular instrument's physical failure mechanism. All source selection, model weights, training durations, calibration parameters and preprocessing parameters remain fixed. No model is trained on the disturbed held data.

Use the primary 598-spectrum population and its existing 260 held contexts, representing 557 distinct test spectra and 69 physical masters. Do not combine this experiment with range, normalization or filtered-population sensitivities. The three universal pipelines are MIN, SG and arPLS. Owner decision P08-A10 includes all five universal methods: RBF-SVM, Random Forest, Extra Trees, ordinary CNN (D0-M) and the context-local source-selected CNN (P05-SELECTED). The earlier four-method alternative is not selected for this universal robustness branch. This decision does not select a model from stress-test outcomes or resolve the distinct filtered-population panel question.

The main comparison uses the same registered test rows, source-only model decisions and matched disturbances for every method/policy. Contexts, technical seeds and synthetic repetitions do not create additional independent samples. A failed required cell remains unavailable; it cannot silently disappear from a full-support curve or area comparison.

## 2. Starting values and operation order

Start from the authenticated, stored **raw common-support interpolation**, spanning 400–1,849 cm⁻¹ at 1 cm⁻¹ spacing. “Raw” here means no subsequent numerical preprocessing; the values are already interpolated and are not the instrument's native grid. The P01 loader uses this stored array as its source. Do not substitute the min–max representation, re-interpolate native files by convenience or overwrite a frozen P01 array.

Let `y(v)` be that row, `y0` its unmodified 400–1,800 cm⁻¹ portion, and `R = max(y0) − min(y0)`. Require finite values and `R > float64 epsilon`. Every amplitude and clipping threshold below uses this unmodified primary portion, not an already disturbed or renormalized row. Let `t = (v − 400)/1400` on the primary grid.

For a shift, first read the required coordinates from the wider stored input, including supported values above 1,800 cm⁻¹. For the other families, modify the primary portion directly. Each family acts independently: do not compose noise, slope, clipping or shifts. Then apply the unchanged P01 transform on the **1,401-point primary grid**, using float64 arithmetic and float32 final serialization. P01 crops before despiking, smoothing or baseline correction; operating those transforms on the wider grid and cropping afterward would be a different pipeline.

The inherited operations remain MIN alone, isolated-impulse replacement followed by SG 11/3 and MIN, or isolated-impulse replacement followed by arPLS and MIN. All parameters and early exits are unchanged. Future disturbed inputs are private transient stress-test artifacts, not replacement P01 representations or additional preprocessing candidates. Producing them on scientific data requires a separate permit.

## 3. Finite disturbance design

The ranges and named levels originate in Master Plan §16.5. The intermediate slope/background levels, impulse height and ten stochastic repetitions are explicit planning choices made before new outcomes, not settings inferred from an observed best result.

| Family | Registered levels | Operation on the unmodified row | Nonzero realizations |
|---|---|---|---:|
| Wavenumber shift | −5 through +5 cm⁻¹, step 1 | `y_shift(v) = y(v − delta)` | 10 |
| Linear slope | −0.10 through +0.10 of R, step 0.025 | Add `s R (t − 0.5)` | 8 |
| Broad quadratic background | 0, 0.025, 0.05, 0.075, 0.10 of R | Add `b R 4t(1 − t)` | 4 |
| Gaussian noise | Source-proxy quantiles 0.50, 0.75, 0.90, 0.95; plus no noise | Add `R sigma_q z`; ten repetitions per nonzero quantile label | 40 |
| Isolated positive impulses | 0, 1, 3, 5 locations | Add R at each selected location; ten repetitions per nonzero count | 30 |
| Upper clipping | Quantiles 1.00, 0.999, 0.995, 0.99 | `min(y0, quantile(y0,q))`, linear quantile interpolation | 3 |

The slope parameter is its end-to-end change divided by R; each end is displaced by half that signed amount. The quadratic is a nonnegative central hump, zero at both primary endpoints and equal to bR at 1,100 cm⁻¹. Its specific shape is a declared stress geometry, not a fitted fluorescence model. The impulse amplitude is one original row range above the value at that point; it is not a percentile or a random multiple selected afterward. Clipping changes intensity values but does not delete rows or channels.

All families share one clean case. Thus the design has **95 nonzero-labelled realizations plus one shared clean case**, or **96 unique cases**. The six family views contain **101 case memberships** because the clean case appears in each view. These are disturbance descriptors, not model fits, prediction jobs or independent experimental samples. If a source-noise quantile is zero, retain its labelled cases and record the zero amplitude; do not redefine the design after inspecting values.

### Shift boundaries

Use the owner-approved constant nearest-endpoint extension only outside 400–1,849 cm⁻¹. Positive shifts of one through five channels require one through five synthetic lower-edge values, respectively. Negative shifts through −5 cm⁻¹ use available wider-support values and need no extension for the primary model input. Record each affected coordinate. Do not introduce upper-edge extension by cropping the available input too early. Extension is a synthetic assumption, not recovered measured signal.

### Source-only noise reference

For each outer context, use only the final source-fitting observations shared by the compared universal procedures. The held instrument and test masters remain excluded. Use their frozen native-QC `first_difference_noise_mad` and positive `intensity_range`, with one contribution per stored fitting spectrum. Do not use target rows, target-batch statistics, post-SG/arPLS residuals, or labels to estimate the dose.

The native-QC field already divides the first-difference median absolute deviation by **0.67448975**. Define the normalized proxy as

`sigma_proxy = first_difference_noise_mad / (sqrt(2) * intensity_range)`.

The additional square-root-of-two factor accounts for the difference of independent equal-variance noise samples under an idealized white-Gaussian model. Real peaks, nonuniform native sampling and correlated noise can also affect this diagnostic. It is therefore a source-referenced disturbance scale, not a validated estimate of pure detector noise. Take its four registered quantiles with float64 linear interpolation; use the same context-level values across models and policies. An invalid required fitting ingredient makes that context's noise experiment unavailable; do not silently discard the row or replace the reference with held data.

This new stress-test definition does **not** alter the completed neural benchmark or its inherited augmentation, including the existing `0.9538725524` divisor. Training augmentation and this explicit test-time proxy have different declared definitions. No scientific noise reference is calculated during readiness.

## 4. Matched stochastic realizations

Use NumPy `Generator(PCG64)` with a seed derived from the first eight SHA-256 bytes, interpreted as an unsigned big-endian integer, of canonical JSON containing the namespace, disturbance family, observation-ID string and repetition index. The namespace is `nato-sers-p08-stress-random-v1`; repetition indices are 0 through 9. Model, policy, technical seed, outer context and severity do not enter this key.

For Gaussian noise, draw a 1,401-element standard-normal vector. Reuse that vector across the four quantile levels, models, policies and appearances of the same observation; its amplitude can differ across contexts because their source-only quantiles differ. Independent observations and repetitions have distinct keys. This pairing reduces unrelated Monte Carlo variation without treating repeated contexts as independent data.

For impulses, draw a permutation of indices 0 through 1400. Greedily accept the first five indices separated from all previously accepted indices by at least three channels. The one- and three-impulse cases use prefixes of the same five-index sequence. This makes contamination nested across counts and keeps injected points isolated from one another. It does not guarantee isolation from naturally occurring spikes or analyte peaks. Edge positions remain eligible.

Retain private realization hashes and source-noise-reference bindings. Public reports contain aggregate descriptions, not observation keys, per-row positions or random vectors. Readiness may enumerate case descriptors and test invented inputs, but generates no scientific random perturbation.

## 5. Zero-dose and model-reuse gates

Before accepting any disturbed comparison, the clean numerical path must reproduce the stored action's axis, row order and validity exactly. Require float32 output agreement with maximum absolute error at most **10⁻⁶**, with zero relative tolerance; also record exact-equality fraction and maximum error. This uses the existing representation-level numerical scale, not an outcome-selected tolerance. A no-op implementation must not conceal a different transform order by simply returning the saved array without testing the numerical path.

This is the newly declared clean-path gate, not a revision of P01's separate historical candidate-reproduction tolerance of **5 × 10⁻⁶**. Both the original P01 parameter contract and transform source are byte-pinned in the design registry. Numerical feasibility on scientific rows remains untested under the no-fit authority.

Replaying the clean model input must reproduce the authenticated reference probabilities to absolute tolerance **10⁻⁷**, zero relative tolerance, with exactly the same final class decisions. The same check applies to combined predictions. Check per-seed values where retained and the final technical-seed aggregate in every case. Record discrepancies before computing stress-test scores. Failure stops the affected comparison and preserves evidence; it does not authorize relaxed tolerances, repeated fits or a replacement clean reference.

The historical classical MIN run did not retain fitted estimators. Reconstruct only the exact saved source-selected specifications on their original fitting rows and seeds, under a separately enumerated permit. Do not repeat hyperparameter selection, refit temperatures or repair the eight missing historical C-SELECTED results. Those missing results concern a different comparator; they are not excuses to replace fixed-family reference endpoints. Retained P05 neural and future universal/QC estimators require authenticated input/model/calibration bindings and clean prediction parity before reuse.

A fallback under a disturbance must use the MIN-trained estimator on the **disturbed MIN input**. It cannot alias the original clean prediction. Family-policy aliases and the 206 structurally unsupported QC contexts may reuse the matched **disturbed** MIN endpoint after exact binding checks. New universal final estimators should be retained for this later stage. Exact reconstruction, zero-dose verification, disturbance preprocessing, prediction, calibration application and reporting jobs remain separate accounting items.

## 6. Approved fixed-route adaptive-QC sensitivity

The existing QC gate consumes diagnostics calculated on each original native spectrum, including its native range and sampling. The approved disturbances instead start from the stored interpolation. Recomputing those diagnostics on the primary grid would change the measurement procedure and may change the clean route. Reusing the old thresholds with those new diagnostic definitions is not an authenticated test of the same gate.

Owner decision P08-A11 selects the conditional sensitivity that freezes each row's clean gate decision and applies its selected numerical action to the disturbed primary row. It uses the same already trained mixed-route QC estimator, not a row-wise mixture of universal-model predictions. Registered invalid-action MIN fallback remains explicit; it does not become evidence that the gate detected contamination. Such a result measures robustness **conditional on the original routing decision**, not the QC gate's reaction to new contamination. Report the 54-context eligible subset and the complete 260-context fallback-inclusive result separately. The adaptive panel remains the existing four methods; universal Extra Trees inclusion does not add an adaptive Extra Trees model.

The alternative requires an additional native-grid disturbance design and a consistent path for recalculating native QC before interpolation. That changes the approved starting-input experiment and needs a separate specification and budget; it is not selected here. Family-aware fallback remains a structural MIN identity, not an independent robustness result. The standard, unperturbed adaptive experiment is unchanged by the fixed-route stress-test decision.

## 7. Curves and normalized loss areas

Use the existing M01 individual-spectrum and M06 combined-probability definitions. For each stochastic repetition, first perform the complete model prediction, calibration, technical-seed aggregation and within-master M06 combination. Then calculate that repetition's context balanced accuracy. Average the ten **scores**, not the ten probability vectors; averaging probabilities across synthetic repetitions would create a new prediction ensemble.

Define the loss at a dose as `BA(clean) − BA(disturbed)`. Negative losses remain negative: an accidental improvement is not clipped to zero. Retain both signed shift and slope curves. For their area summaries, average positive- and negative-direction losses at each absolute dose, then integrate by the trapezoidal rule. Also report the poorer direction rather than hiding it in the average.

Normalize each family's severity axis to [0,1]: absolute shift divided by 5; absolute slope or quadratic fraction divided by 0.10; impulse count divided by 5; and clipping removed-tail fraction `(1−q)` divided by 0.01. For noise, the axis contains a separately labelled no-noise point at zero followed by source quantile ranks divided by 0.95. That last axis is a **quantile-index axis**, not equal increments in physical noise amplitude; publish the source-proxy amplitude summaries beside it. Its zero point is not the empirical zeroth quantile. The trapezoidal area describes the declared grid, not an observed continuous dose response.

Calculate context loss areas before averaging contexts equally within domains and domains equally, using unchanged support. Do not combine the six families into one cross-family area or a new leaderboard. A policy's robustness benefit is `loss_area(MIN) − loss_area(policy)`: positive values mean less degradation relative to that policy's own clean performance. Report clean and disturbed absolute accuracy alongside this contrast, since a poor classifier can have little remaining accuracy to lose.

The later inference registry must enumerate separate robustness families under the approved model-panel and QC decisions. It must retain paired support, the inherited shared master/instrument weights, missing-cell rules, conditional-uncertainty limits and original-hierarchy feasibility checks. Pointwise curves are descriptive; repetition spread is Monte Carlo variability, not an interval over new samples or instruments. Neither a favorable area nor a conditional interval automatically passes G4. No score or resampling draw is computed by this design.

## 8. Figures, remaining gates and authority

P08-F08 will show directional degradation curves with dose markers and paired domain scatter of MIN versus nonminimal loss area. Display both endpoints, poor domains and unavailable cells. Private review may include matched disturbed spectral examples; public spectral aggregates retain the two-master minimum and privacy rules in the [figure protocol](P08_FIGURE_PROTOCOL.md). Native TikZ, offline HTML, vector PDF and PNG must share reviewed semantic data and black standard-font text.

The two scope decisions are approved in P08-A10–A11. Remaining gates are the exact reconstruction/prediction ledger, finite resource proposal, complete inference registry, numerical implementation and synthetic tests, scientific-input authentication and a separate execution permit. The current case enumeration is not that job ledger. Universal preprocessing remains first, adaptive work second and robustness later. No training, new prediction, QC calculation, preprocessing-array rebuild, perturbation run or automatic retry is authorized.

## 9. Universal model-reference accounting

The [procedure audit](../results/p08_readiness/universal_stress_procedure_audit.json) binds each universal pipeline to its existing final-fit, calibration, clean-prediction and ensemble metadata. It verifies **3,237 distinct procedures** across MIN, SG and arPLS, with **8,151 seed-specific estimator slots**. Each preprocessing action has **1,079 procedures** and **2,717 estimator slots**. A procedure is one context–preprocessing–model combination; technical seeds are its component estimators, not additional physical samples.

| Model source | Seed-specific slots | Required later handling |
|---|---:|---|
| Historical classical MIN models | 1,820 | Reconstruct the saved selected specifications once on their original fitting rows and seeds. Require clean prediction parity; do not retune or recalibrate. |
| Historical neural MIN models | 897 | Authenticate and reuse the saved checkpoints and calibration states, subject to clean prediction parity. |
| Future SG/arPLS final models | 5,434 | Retain the estimators produced by the separately approved universal experiment. These are upstream fits, not additional robustness fits. |

The **1,560 neural reporting aliases** preserve the ordinary and source-selected views without counting a shared D0-M procedure twice. Classical predictions retain seed averaging before their single temperature calibration; neural predictions retain per-seed calibration before averaging. These operations must not be interchanged when the case-level ledger is assembled.

Independent verification reconciled **24,570 upstream operation references**, including **8,190 historical MIN bindings**, and rehashed **6,826 existing files** containing **810,255,576 bytes**. Prediction values and checkpoint tensors were not parsed. The full reference catalog remains private; the public aggregate contains counts and hashes only. Its authenticated metadata does not establish successful reconstruction, numerical equality or preprocessing benefit.

This closes the universal procedure-reference slice only. The fixed-route QC procedures, disturbed-MIN fallback aliases, case expansion, numerical transform/noise-reference dependencies, calibration applications, scoring/inference jobs and finite resources still require complete accounting. The metadata adapter always denies scientific execution; no model was reconstructed and no disturbed spectrum or prediction was generated.
