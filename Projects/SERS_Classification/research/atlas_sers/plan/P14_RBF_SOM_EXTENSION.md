# P14 — RBF networks and self-organizing maps for NATO SERS

**Version:** `nato-sers-p14-plan-v1`, 2026-09-24.

**Status:** planning specification; implementation, support audit, and scientific fits have not started.

**Evidence timing:** added after P03, P13-classical, and P04 outcomes were available. This is a secondary/exploratory extension, not an amendment to their frozen results or a prospectively blinded study.

**Companions:** [publication strategy](P14_PUBLICATION_STRATEGY.md), [machine-readable registry](registries/p14_extension_registry.json), [master plan](MASTER_PLAN.md), [question map](RESEARCH_QUESTION_MAP.md).

## 1. Purpose and place in the project

The extension asks whether a small collection of representative spectral patterns can support chemical classification, reveal acquisition dependence, and help identify unreliable predictions. A radial basis function neural network (RBF network) tests classification using local similarity to prototypes. A self-organizing map (SOM, or Kohonen map) tests how such prototypes organize the observed spectra. Neither method is assumed to remove background or recover a clean chemical spectrum.

The project already has 598 attributable SERS spectra from 69 physical masters, ten acquisition instruments, four substrate families, and three station-specific three-class tasks. P00–P02 established the data and evaluation design. P03 classical and P04 ordinary compact neural baselines are complete. P13 classical portability is complete, but its controlled deep comparison remains outstanding. P05 acquisition-aware development, P06 final comparison, and P07–P12 secondary/final analyses remain planned.

P04's 208,691-parameter D0 reached 0.711 mean unseen-instrument spectrum balanced accuracy, with a 0.379 worst domain. It did not establish superiority over Random Forest or Extra Trees. That motivates testing compact alternatives without assuming that additional neural complexity helps. These are recorded P04 results, not P14 results.

The original next step remains P05's source-only development audit. P14 can proceed alongside it after its own no-fit audit because its core uses existing P01/P02 data and does not require D1–D5. P14 may not choose P05 losses, augmentations, epochs, or advancement rules from its observed test outcomes. Conversely, P05 results do not choose P14 settings. Frozen v1 and P13 registries are not edited to insert the extension.

| Existing question | P14 contribution | Boundary |
|---|---|---|
| RQ-P01: learning under acquisition shift | Additional shallow prototype classifiers and matched controls | Secondary comparison; the original primary contrast and G4 remain unchanged |
| RQ-E01: representation and failure structure | SOM geometry, prototypes, repeated-view movement, and instrument associations | Association and prediction, not causal disentanglement |
| RQ-S01–S03: preprocessing | First universal MIN/SG/arPLS contrasts; later consume source-frozen family/QC policies if P08 supplies them | P14 does not reselect the original P08 policy panel |
| RQ-S05/S06: reliable operation and narrow unknown rejection | Prototype distance for error detection; optional held-chemical experiment later | A distant spectrum is not automatically an unknown analyte |
| RQ-S07: substrate portability | Optional exact substrate-restricted refits and same-master comparisons | Existing P13 support and two-margin rule apply |

## 2. Research questions and intended evidence

The identifiers below belong only to P14. RQ-P14-01 is the lead extension question, not a new project-wide primary endpoint. RQ-P14-01–05 form the core package; 06–07 are conditional additions.

| ID | Precise question | Main comparison and endpoint | Meaningful outcome |
|---|---|---|---|
| RQ-P14-01 | Can a small RBF classifier identify station-supported chemicals on an excluded instrument and unseen physical masters competitively with established baselines? | Source-selected RBF versus Random Forest, Extra Trees, RBF-SVM, C-SELECTED, and D0 on identical P02 test views; pooled three-class domain BA | A bounded predictive/cost tradeoff, superiority over a named comparator, or a documented loss |
| RQ-P14-02 | Do SOM neighbourhoods organize by chemical category, acquisition instrument, or substrate, and how stable is that organization? | Label-free SOM fitting followed by station-stratified label overlays; node association, occupancy, quantization/topographic error, seed stability; PCA and equal-size K-means controls | Reproducible structure with explicit confounding limits, not visual proof of chemical separation |
| RQ-P14-03 | Are repeated views of one sample more consistent across instruments than chemically different samples under comparable acquisition conditions? | Same-master cross-instrument distances and probability agreement versus same-analyte/different-master and different-analyte reference pairs | Lower cross-view distance is useful only with retained chemical contrast and correct identification |
| RQ-P14-04 | Do universal smoothing or baseline correction improve prototype-based transfer while preserving chemical distinctions? | SG−MIN and arPLS−MIN within a frozen fitting recipe on paired views; domain BA, chemical contrast, association, and peak/shape audit | Whether preprocessing benefits are consistent, acquisition-dependent, or model-dependent |
| RQ-P14-05 | How does performance change with independent training-master count and prototype budget? | Matched master-level learning curves and fixed-budget RBF/SOM controls | Evidence about sample efficiency and capacity within the observed range; no invented learning-curve extrapolation |
| RQ-P14-06 | Does distance from familiar prototypes identify errors on an unfamiliar instrument better than classifier confidence alone? | Risk–coverage and error-detection curves versus maximum probability and existing source-only confidence scores | Useful abstention at documented coverage; does not establish novel-chemical detection |
| RQ-P14-07 | Does SOM organization add value to RBF centres, or does a prototype head add value to a frozen neural representation? | SOM versus K-means centres at identical K/features/output rule; separately linear versus RBF head on identical fold-specific frozen embeddings | Isolates prototype construction or head choice; does not imply benefit from an unrestricted hybrid search |

Substrate portability is a conditional application of these questions, not an eighth RQ that silently changes P13's claim. A null or negative result for any row remains an answer.

## 3. Data, access, and evaluation contract

### 3.1 Shared data and tasks

- Use the unchanged 598-row P01 primary population. Notes-clear and Mira-1-excluded tiers are later sensitivities, not replacements selected by results.
- Fit separate CWA, Pills, and Surfaces tasks: 24, 20, and 25 masters respectively. Do not use pooled seven-class accuracy as the main chemical endpoint.
- Reuse exact P02 repeated physical-master partitions, the 13 primary station–instrument domains, source-only inner roles, and held-instrument exclusions.
- Keep all spectra from a master out of training when that master is tested. Keep the held instrument out of every fitted component, even when labels are not used.
- Use 400–1,800 cm⁻¹, 1,401 channels, and the immutable MIN/SG/arPLS arrays. SG includes the existing impulse treatment and SG(11,3); arPLS uses the existing impulse treatment and lambda 100000. No new baseline parameter or transformation is introduced here.
- Record the physical-master count in every train/validation/test role. The 598 spectra do not become 598 independent preparations.

### 3.2 Descriptive maps versus predictive evidence

The first core maps are fitted on source-training spectra and then held spectra are projected without updating the map. Their predictions use only source labels. A later full-dataset map is permitted solely as a labelled descriptive figure after recipes are frozen; it cannot select preprocessing, model size, hyperparameters, a classifier, or rejection thresholds. A full-dataset SOM followed by cross-validation of its classifier would leak test information and is prohibited.

The no-fit audit reads the existing manifest, split/support records, and input hashes. It does not calculate new clustering, distances, or model scores. Its PCA feasibility check uses metadata upper bounds only; numeric-rank verification belongs to the later training-only fit stage. After that gate, source-only development may fit PCA, centres, widths, maps, classifiers, and calibration. Final held-test output is revealed only after the extension recipe and selection algorithm are frozen. Earlier study outcomes are disclosed prior knowledge throughout.

### 3.3 Weighting and feature transforms

For unsupervised PCA/codebooks/SOMs, give each source master equal total weight, each instrument view within a master equal weight, and each stored spectrum within a view equal weight. With M masters, J_m instruments for master m, and R_mj spectra in its view, row weight is `1/(M J_m R_mj)`. This preserves repeated views without making frequently measured masters dominate. Report an additional equal-substrate-view sensitivity only in the supported P13 branch.

For a supervised output layer, balance classes first, then masters within class, instruments within master, and spectra within instrument: `1/(C M_y J_m R_mj)`. These choices are part of the P14 pipeline. Comparisons with previously frozen models are comparisons of complete procedures; they do not isolate network architecture if weighting, augmentation, or calibration differs. Mechanistic centre/head comparisons use identical weights and features.

Feature candidates are full MIN spectra, weighted source-only PCA with 5 components, and weighted source-only PCA with 10 components. Do not whiten or standardize channels in the core. Weighted PCA centres using the training weights and derives components from the weighted covariance/SVD. Transform held rows with the fitted training mean and components. PCA is not claimed to remove nuisance.

To avoid misleading capacity claims, report all fitted PCA coefficients/means, stored prototype coordinates and widths, supervised output coefficients, and any encoder parameters separately. Lower supervised parameter count does not erase unsupervised estimation or guarantee better generalization.

## 4. Models and finite search space

These are specified defaults for implementation. Feasibility is resolved from metadata and training-only numeric rank before any held outcome; a changed grid requires a dated P14 amendment. The registry records the same defaults. No successful scientific fit has yet validated them.

### 4.1 Core RBF classifier

Use Gaussian units `phi_k(z) = exp(-||z-c_k||²/(2 sigma²))`, followed by L2-regularized multinomial logistic regression. This is a shallow RBF network with a two-stage fit, not the previously tested RBF-kernel SVM. Centres are fitted by weighted K-means; the core does not backpropagate class loss into centres or widths.

| Component | Specification |
|---|---|
| Features | full spectrum, PCA5, PCA10 |
| Centre count K | 3, 6, 9 |
| Initialization | weighted K-means++ with one initialization per recorded seed |
| Codebook iteration cap | 300; deterministic ties and convergence tolerance 1e-6 |
| Base width | weighted median positive training distance to the nearest fitted centre |
| Width multiplier | 0.5, 1, 2 |
| Width floor | 1e-8 times the source RMS radius, with an absolute floor 1e-12; zero-radius inputs are terminal degenerate inputs |
| Output regularization C | 0.1, 1, 10; weighted mean cross-entropy plus `||W||²/(2C)`, intercept unpenalized |
| Output optimizer | deterministic L-BFGS, maximum 10000 iterations, convergence tolerance 1e-6 |
| Seeds | 20260805, 20260817, 20260829; all retained, no best-seed reporting |

This gives at most 81 candidate recipes per selection context before feasibility filters. Initialization seeds are repeats of a recipe, not additional choices. The implementation must explicitly match the stated normalized loss rather than assume a library's C has identical weight-scaling semantics.

If all nearest-centre distances are zero, use the positive weighted median pairwise training distance; if none exists, record a degenerate fit. Empty K-means clusters are deterministically reseeded to the highest weighted-error training row, tie-broken by stable row ID, for at most K reseeds; failure thereafter remains terminal. Do not insert a centre per stored spectrum.

Fit the same-feature regularized linear classifier as a control: it receives full/PCA features directly without RBF units. A same-feature RBF-SVM is an additional diagnostic control. Frozen P03 forests, C-SELECTED, RBF-SVM, and P04 D0 remain external baseline procedures; fresh fits are required when a comparison changes training-master count or substrate support.

The matched linear control uses exactly the RBF output layer's C grid, normalized loss, solver, and supervised weights. The matched SVM uses full/PCA5/PCA10, C={0.1,1,10}, and gamma multipliers {0.1,1,10} times `1/(2 r²)`, where r² is the weighted mean squared source distance from its weighted mean in that feature space. A zero radius is a terminal degenerate fit. Use one-versus-one C-SVC with supervised row weights summing to one, no additional class weighting, tolerance 1e-6, and a 100000-iteration cap; record convergence warnings as failures. Calibrate its three-class decision-function scores with the same source roles rather than library-internal ungrouped probability fitting. This is at most nine linear and 27 SVM recipes per context. The matched SVM is a new control, not a rewrite of the frozen P03 RBF-SVM. Forest learning-curve controls reuse the family-specific search grids in the [original hyperparameter registry](contracts/hyperparameter_registry.json), but all fitting and selection are repeated on each allowed subset.

### 4.2 Core SOM and SOM classifier

Use a rectangular, non-toroidal SOM with four-neighbour lattice adjacency and Euclidean feature distance. Fit batch SOMs with the same unsupervised row weights as above. A Gaussian lattice neighbourhood shrinks geometrically from `max(grid width, grid height)/2` to 0.5 over 100 epochs. Batch updates use weighted neighbourhood means. Initialize node prototypes from distinct source-master representatives, then one recorded spectrum per representative using the declared seed; sampling follows the training-view weights. These representatives initialize the map but do not replace the training population.

| Component | Specification |
|---|---|
| Features | full spectrum, PCA5, PCA10 |
| Grid sizes | 2×2 and 3×3 (4 and 9 nodes) |
| Epochs | fixed 100; no test-based stopping |
| Seeds | the three RBF seeds above |
| Map-only control | weighted K-means with exactly the same number of prototypes and feature transform |
| Classification | source-label distribution at the best-matching node, with one master-equivalent pseudocount per class |
| Empty-node rule | use the nearest occupied node in feature space, stable-node-index ties; log every fallback |

A node distribution uses per-master/per-instrument weights before global M normalization, so pseudocounts have an interpretable master-equivalent scale. Classification remains source-labelled even though map fitting is unsupervised. Selecting grid/features using source classification scores makes the complete predictive pipeline label-informed; it must not be advertised as wholly unsupervised classification.

Core descriptive plots retain both sizes and all features, with the first declared seed as the display anchor. Other seeds appear in stability summaries. Do not choose a prettier map after seeing held labels. Display occupancy and training-master counts; a large empty map is not evidence of more chemical states. A 100-epoch run is a declared fit budget, not a guarantee of convergence; source-only error trajectories document stability.

### 4.3 Feasibility and source selection

Require `K <= number of distinct source-fitting masters` for codebooks and SOM nodes. Require PCA dimension no greater than both numerical training rank and `source-master count - 1`. The extra master-count restriction is a conservative design choice, not a mathematical claim that repeated spectra cannot increase matrix rank. A candidate must be feasible across all its required inner roles; do not evaluate candidates on whichever folds happened to succeed.

Candidates are ranked by mean source pseudo-instrument pooled three-class BA, then worst source-domain BA, then fewer prototypes, fewer feature dimensions, stronger output regularization, and registry order. SOM selection omits the regularization tie-break. Seeds are averaged before ranking. Any recipe missing a required inner endpoint is ineligible; if no recipe survives, retain the outer context as unavailable with reasons. Degenerate single-class predictions are valid poor results, not grounds for silent exclusion.

Use the same source-only calibration roles as the existing design. Temperature calibration uses cross-fitted selected-recipe logits and equal master contribution; the no-fit/implementation gate must verify the exact role mapping before fitting. For SOMs, use log node probabilities as logits. Record raw and calibrated probabilities, temperature, optimizer status, and all failures. If calibration fails, report discrimination where defined and calibration as unavailable; do not silently substitute a different calibration method.

### 4.4 Optional models, explicitly outside the core grid

- **SOM-centred RBF:** compare against K-means-centred RBF at exactly K=4 or K=9, identical features, widths, output regularization, weighting, and validation. The K=4 RBF is an explicitly separate control, not an unnoticed core-grid change.
- **Frozen-encoder RBF head:** use the exact source-fitted encoder for each outer role. Compare RBF and regularized linear heads using the same embeddings, labels, and calibration roles. Inner selection must use encoders fitted without the inner validation masters; an outer encoder trained on those masters cannot supply supposedly cross-fitted inner features. Account for any necessary encoder refits before authorizing this branch.
- **Learned centres or deep RBF layers:** deferred; require a new bounded optimization specification. They are not a remedy selected from P14 held-test failures.
- **SKiNET/LVQ-style refinement:** literature-motivated future comparator, not included in the first SOM experiment.

## 5. Ordered sub-plans and stopping points

| Experiment | Work and output | Prerequisite and finish condition |
|---|---|---|
| EXP-P14-00 | No-fit manifest: inherited hashes, exact roles, metadata feature/K feasibility bounds, same-master pair and crossover support, fit ledger and compute estimate | Planning complete; finish with a reproducible audit containing zero scientific fits |
| EXP-P14-01 | SOM/K-means/PCA source maps, held projection, occupancy, prototypes, seed and master-resampling stability | 00 passes; all declared map configurations have outputs or reasons |
| EXP-P14-02 | RBF and SOM classification; same-feature linear/SVM controls; frozen baseline comparisons | 00 passes; source selections frozen before held scoring, all endpoints reconciled |
| EXP-P14-03 | Same-master cross-instrument and supported substrate pair diagnostics | 01/02 frozen; compare correctness and chemical contrast as well as agreement |
| EXP-P14-04 | Universal MIN/SG/arPLS factorial using the same P14 model recipes and source roles | 01–03 core MIN recipes frozen; paired support, preservation, and interactions reported |
| EXP-P14-05 | Physical-master learning curves and prototype-budget curves | 02 recipe library fixed; subset-only fits and identical subsets across compared methods |
| EXP-P14-06 | Distance-based error detection and risk–coverage | 02 complete; source-only score/threshold policy specified before held error analysis |
| EXP-P14-07 | SOM-centred versus K-means-centred RBF ablation | Optional after core reconciliation and separate exact budget; both equal-size codebooks retained regardless of winner |
| EXP-P14-08 | Frozen-encoder RBF versus linear head | Optional after source/inner encoder provenance and extra refits are resolved |
| EXP-P14-09 | Exact P13 substrate-restricted prototype refits | Optional; use unchanged P13 support, source roles, margins, matched-source endpoint, and all unavailable cells |
| EXP-P14-10 | Question-wise statistics, figure package, bounded claim table, and manuscript integration | Core 00–05 complete; 06–09 explicitly completed, deferred, or unsupported |

Core analyses are not conditional on achieving a positive score. Optional branches are admitted on scientific distinctness, source support, and resources; held results cannot decide to try more settings until a win appears. Final paper preparation does not wait for all optional branches or completion of every future idea in the master plan.

## 6. Preprocessing, pair analysis, and sample-efficiency details

### 6.1 Universal preprocessing

The first paired question changes only the input action: MIN, SG, or arPLS. Freeze the selected MIN feature dimension, K/grid, width multiplier, regularization, epochs, and seed rule, then refit numeric PCA components, centres, widths, maps, output weights, and calibration on each action's source data. Never apply a MIN-trained model to corrected inputs and call that a fair retrained preprocessing comparison.

An unavailable rank/support/fit under a sensitivity action is retained. If later policy-specific source reselection is added, give it a different experiment/analysis label: it estimates the effect of a complete selection procedure, not the fixed-recipe preprocessing effect. Family/QC policies can be consumed only after P08 freezes them independently; P14 does not select a policy by the held instrument's performance.

Assess spectra with the existing P01 preservation measures and display nearby actual training spectra alongside prototypes. SOM/RBF prototypes can average acquisition effects, weak signals, or mixtures. Neither a prototype nor inverse-PCA reconstruction is labelled a pure chemical reference. Reduced acquisition association without retained target information and transfer is treated as possible collapse.

### 6.2 Repeated measurements and incomplete factors

Use three reference-pair types within station: same-master/different-instrument, same-analyte/different-master, and different-analyte/different-master. Where support permits, match substrate family and acquisition-instrument pair. Compare instrument changes at fixed substrate first. Evaluate substrate changes at fixed instrument separately. Cases in which both change are labelled joint changes.

Do not fill the sample×substrate×instrument matrix with generated observations. The 67 cross-instrument masters and 32 crossover-capable masters describe overall support; each outer role has less support that the audit must enumerate. A master producing many pairs gets the same total contribution as another master. Pair counts are reported but are not independent sample counts.

Report feature-space distance, prototype-vector distance, map-node distance, prediction divergence, both-correct fraction, and agreement. Compare within-master distances with both reference-pair types. Short map paths or identical wrong predictions alone do not demonstrate invariance. Use a target-adjusted instrument probe only on source-fitted features and authorized grouped roles; unsupported class/instrument contrasts stay descriptive.

### 6.3 Learning curves

Use training subsets with `{2,3,4,6}` masters per station class when feasible, five deterministic subset draws, and fixed evaluation masters. Thus attempted totals are 6, 9, 12, and 18 physical masters, with actual support shown. Make draws nested within each seed and share exactly the same subsets across RBF, SOM, and fresh forest/SVM controls. Refit PCA, codebooks, tuning, and calibration using only the subset. If nested selection/calibration lacks support, mark that size unsupported instead of borrowing full-source data.

Plot BA and log loss against actual master count, and BA against prototype count at fixed sample size. Fit counts, runtime, model storage, and distinct masters per prototype accompany the curves. Do not fit an extrapolated curve and claim how many future samples will guarantee success. D0 learning curves require matched subset-specific training, so they are deferred until separately budgeted; the full-data D0 endpoint is not a low-data control.

## 7. Endpoints, uncertainty, and decisions

### 7.1 Prediction endpoints

Use pooled out-of-fold predictions within each station–instrument domain and repeat, compute recall for all three station classes, average class recalls, then average repeats and domains with equal domain weight. A repeat/domain missing required class support is unavailable. This avoids confusing P04's fold-mean summary with P03's pooled-repeat summary. Recompute comparator endpoints from frozen row predictions using this same rule; do not subtract published headline numbers with different aggregations.

Report M01 spectrum BA, class recall, macro-F1, worst domain, NLL, Brier score, calibration curves, and endpoint coverage. For M06, average probabilities across spectra within an instrument view, then weight available instrument views equally for each master. Classify the largest averaged probability. Raw spectra are not averaged for M06. Report unavailable runs and missing calibration separately from valid wrong predictions.

### 7.2 Map and distance endpoints

- Quantization error: weighted mean distance to the best-matching prototype, plus normalization by source RMS radius in that same feature space. Cross-representation comparisons are qualified because PCA changes the metric and information retained.
- Topographic error: fraction of rows whose first two best-matching nodes are not four-neighbours. A small value assesses lattice behaviour, not chemical accuracy.
- Occupancy: spectra, distinct masters, instruments, and targets per node; empty nodes remain visible.
- Association: target/instrument/substrate adjusted mutual information by station and support-qualified conditional views. Node labels are categorical. No pooled station-confounded chemical claim.
- Stability: co-assignment/neighbour agreement across seeds and master resamples; map rotation/reflection is aligned using source prototypes only for illustration. Never align using held chemical labels.
- Pair consistency: aggregate per master before uncertainty; retain chemical-distance contrasts and both-correct outcomes.

Use 100 source-master resampling refits per map configuration for the stability analysis, with all views carried together and PCA/codebooks refitted within each draw. Score co-assignment on the common original source views; those are a stability reference, not an independent accuracy test. Record draws that lose feasibility. This refitting budget is separate from the 10000 fixed-prediction bootstrap draws below and must appear in EXP-P14-00.

### 7.3 Inference and failure handling

Average seed predictions before scoring and average technical split repeats before uncertainty. Follow P11's 10000-draw paired domain/master bootstrap logic. Implement master draws globally within station/class and carry all that master's method, instrument, and repeat views together; separately resample the eligible domains, preserving paired methods. Do not independently resample the same master anew for every instrument domain. Use the same recorded rejection/undefined-class rule for every method, disclose retained support, and give leave-one-instrument and leave-one-domain sensitivities. Prototype stability that requires refitting uses a separately budgeted resampling loop; uncertainty on fixed predictions does not measure training instability.

P14 is post-outcome secondary/exploratory research. Report effect sizes and intervals with this timing. If descriptive p-values are included, apply Holm separately to (a) RBF-versus-five-baseline contrasts and (b) two universal-action contrasts across the frozen model panel; register the exact families at EXP-P14-00. State that multiplicity adjustment does not undo historical analyst exposure. Pair geometry, learning curves, and optional hybrids do not acquire confirmatory status from nominal significance.

Use `complete`, `unsupported_by_design`, `unavailable_terminal_failure`, `deferred`, and `not_started` distinctly. Complete-case effects are accompanied by coverage and a sensitivity assigning chance BA to terminal unavailable three-class endpoints, never to unmeasured design cells. A conclusion reversed by missing-endpoint sensitivity is inconclusive. Calibration failures cannot be converted into successful NLL values by dropping rows.

### 7.4 Extension decision rules

No new practical superiority margin is declared here without scientific justification. A favourable mean alone is insufficient. A claim of better classification must name the comparator, show its paired interval and domain pattern, disclose worst-domain behaviour and failure sensitivity, and remain bounded to this reused dataset. Claiming that methods are equivalent requires a separately justified equivalence margin; failure to detect a difference is not equivalence.

Proceed to core synthesis once 00–05 reconcile, even if all new methods lose. Add 06 if the source roles support independent score calibration. Add 07–09 only through their already specified contrasts and resource audit. The original project's Route A still requires P05/P06/P11 and G4; a favourable P14 result cannot substitute for that test.

## 8. Planned figure package

These are figure specifications, not generated results. All require native TikZ/PGFPlots, vector PDF, 300-DPI preview PNG, self-contained HTML, and a shared hashed semantic table. HTML must work offline. Full figure/caption/access rules are in [the figure contract](FIGURE_STYLE_AND_REGENERATION.md).

| Figure | Scientific view | Marks, panels, and retained meaning |
|---|---|---|
| F-P14-01 | SOM atlas and PCA reference | Same SOM coordinates recoloured by target, instrument, and substrate; source/held markers; PC1–PC2 with variance labels; occupancy shown |
| F-P14-02 | Prototype spectral gallery | Prototype curves plus nearest actual source spectra; MIN/SG/arPLS labels; source-master count and spectral range; no pure-spectrum label |
| F-P14-03 | Same-sample acquisition movement | Fixed-substrate cross-instrument connected points, separate fixed-instrument substrate panels, paired feature-distance distributions, and both-correct outcomes |
| F-P14-04 | Classification across instruments | Paired-dot domain BA comparisons and named-comparator effect intervals; all 13 primary domains and missing endpoints retained |
| F-P14-05 | Preprocessing benefit and chemical retention | SG/arPLS minus MIN paired scatter; chemical contrast versus instrument association; selected representative spectra determined before held-outcome inspection |
| F-P14-06 | Sample-efficiency curves | BA/log loss versus distinct training masters with paired intervals and unsupported sizes; curves use common evaluation views |
| F-P14-07 | Prototype budget and computational cost | BA versus K, storage, and measured runtime; point sizes/hover give master counts; PCA/codebook/head state counted separately |
| F-P14-08 | Map stability and confounding audit | Co-assignment stability, occupancy, normalized quantization/topographic errors, station-specific target/instrument associations; no prettier-seed selection |
| F-P14-09 | Reliability and abstention | Distance-versus-error scatter, risk–coverage and reliability curves; realized coverage and source-chosen operating thresholds |
| F-P14-10 | Matched hybrid ablation | Paired effects for SOM/K-means centres or linear/RBF heads, equal feature/K/source-role panels; conditional figure |
| F-P14-11 | Supported substrate portability | Existing P13 support matrix and recovery-versus-loss view for new exact refits; unmeasured and failed cells visibly distinct; conditional figure |
| F-P14-12 | Study workflow and publication evidence map | Native TikZ diagram linking data/roles, independent model/preprocessing axes, endpoints, and manuscript decisions; planning schematic, never a result plot |

Public plots use approved aggregate metadata. Individual source-spectrum or master-ID displays must be explicitly included in the publication review, following the existing policy. Internal diagnostics can retain reversible IDs outside routine release. Missing nodes/cells are hatched in TikZ and labelled in HTML. Grayscale-readable markers accompany colours. Quantitative points cannot be omitted merely to improve the appearance of a map.

## 9. Compute, artifacts, and implementation handoff

EXP-P14-00 must enumerate every outer role × inner role × candidate × seed × preprocessing action × subset draw before execution. The 81 RBF recipes and six SOM feature/grid configurations do not by themselves describe the total fitting cost. Factor in calibration, refits, controls, learning subsets, and optional encoders. No runtime promise or unbounded search is authorized by this plan.

Cache only within identical source-role/feature/seed hashes: PCA and codebooks can be reused across output regularization values without sharing information across folds. Record cache reuse and count statistical fits separately from wall time. Benchmark implementation cost on synthetic arrays and source-authorized smoke roles only after the code gate; do not use held performance to justify a larger budget. If compute is excessive, revise/defer optional work before revealing outcomes and record the amendment.

Implementation should extend `src/atlas_sers/models`, `exploration`, `evaluation`, and `visualization`, with a thin future `scripts/run_p14.py` CLI. Those interfaces are planned; the CLI is not present yet. Intended commands are `plan`, `validate-plan`, `run`, `aggregate`, `publish`, and `validate`, with immutable manifests and restart-safe outputs matching existing phase conventions.

Use a separate governed artifact namespace `p14/<protocol>/<run_id>` under the configured artifact root. Store fit roles, selected recipes, codebooks/PCA, per-epoch SOM diagnostics, row probabilities, pair membership, coverage, intervals, and decisions. Public aggregate results will live under `results/p14_prototypes/` only after reconciliation. New figure files use `F-P14-*` names and do not overwrite F00–F48.

The P14 registry has planning status only. Its gates require: inherited-data integrity; no fitted test-data access; exact support and budget enumeration; synthetic implementation checks for grouping/weighting/determinism/failures; source-only recipe freeze; endpoint reconciliation; native-TikZ/HTML semantic parity; and a bounded claim audit. These are implementation gates to satisfy, not completed checks.

## 10. Completion and immediate next task

Planning is complete when this document, the publication strategy, registry, master plan, question map, and dashboard agree on scope/status/IDs and their links validate. Scientific completion requires all core experiment outcomes or explicit supported reasons, paired comparator accounting, source and test provenance, figures, and question-wise conclusions. Writing a plan does not complete any scientific experiment.

The immediate next task is **EXP-P14-00: construct the no-fit support/role/compute manifest**, followed by implementation and source-only smoke validation. Then run the fixed core SOM maps and RBF/SOM classifiers, pair diagnostics, universal preprocessing contrasts, and sample-efficiency curves. P05 retains its separate already planned development path. Publication synthesis begins at core completion rather than waiting for every optional branch.

The scientific motivation and literature-to-design distinctions are documented in [P14_PUBLICATION_STRATEGY.md](P14_PUBLICATION_STRATEGY.md). P14 seeks informative positive, negative, and inconclusive results; it does not assume that a neural method must win.
