# NATO SERS parallel research-question map

**Amendment date:** 2026-08-07

**Field-trial-purpose amendment:** proposed 2026-09-01 after P03 and locked 2026-09-04 before any P13 outcome calculation

**Evidence timing:** after disclosed pilots and P01 descriptive preprocessing evidence; before P02 and every registered definitive predictive outcome

**Independent scientific unit:** physical `master_sample_id`; the primary transfer contrast is summarized over 13 station–instrument domains

This document is the concise human-readable map from scientific questions to the master plan. `registries/research_question_registry.csv` remains the machine-readable authority for the original v1 questions. `RQ-S07` is intentionally not inserted into that frozen file and is instead represented by the separately versioned and locked `registries/p13_*` files. The questions run in parallel, but only `RQ-P01` determines whether acquisition-aware deep learning is promoted.

## Shared design

The study separates three axes:

| Axis | Levels |
|---|---|
| preprocessing policy | universal minimal, universal SG, universal arPLS, source-selected platform-family, source-selected identity-blind row-QC |
| learning strategy | fixed/selected classical, ordinary compact deep, frozen acquisition-aware deep |
| target information | none, platform-family metadata, current-row QC, unlabeled target masters, paired target masters, labelled target masters |

All candidate preprocessing actions use immutable P01 arrays on 400–1,800 cm⁻¹ with 1,401 coordinates and final per-row `[0,1]` scaling. Family/QC policies select an existing action; they do not estimate a new transform. Within a comparison, policies and models use identical outer splits, test UIDs, metrics, and aggregation. Policy and model selection have separate source-only roles and hashes.

The primary comparison is deliberately one cell: `PP-U-MIN × {selected classical, D0, selected acquisition-aware deep}`. Universal/family/QC preprocessing and target access are secondary. Selecting a transform because it scored best on a named held instrument is prohibited.

## RQ-P01 — learning strategy under acquisition shift (primary)

- **Motivation:** The central publication question is whether structured acquisition-aware supervision adds value beyond rigorous classical methods on this small, heterogeneous field dataset.
- **Formal question:** Does a source-only acquisition-aware representation improve station-supported chemical identification when both the instrument and physical masters are absent from fitting?
- **Comparison:** frozen acquisition-aware deep pipeline minus the development-selected classical pipeline. D0 is the ordinary-deep control.
- **Population/unit:** 598 primary spectra; 13 support-qualified station–instrument domains; physical masters resampled within domain.
- **Information/split:** T3-ZS, five repeat seeds, four master-grouped folds, full held-instrument exclusion; source pseudo-instrument development only.
- **Preprocessing/models:** `PP-U-MIN`; `C-SELECTED`, `D0-ERM`, `D-SELECTED`.
- **Metrics/figures:** M01, M02, M05, M06; F14, F15, F34.
- **Plan coverage:** P02 freezes partitions; P03 establishes classical; P04/P05 establish deep candidates; P06 executes the definitive paired comparison; P11 applies G4; P12 reports it.
- **Success:** G4 supports a bounded acquisition-aware methods claim: mean difference ≥0.03, interval lower bound >0, at least 8/13 domains improve, worst-domain and T1 safety gates pass.
- **Failure:** Route B reports the stronger classical benchmark or Route C reports instability/identifiability limits. Failure is publishable and not repaired by secondary results.
- **Prohibited claim:** arbitrary unseen-instrument, concentration, substrate, or chemical generalization.

## RQ-S01 — universal preprocessing sensitivity

- **Motivation:** Some systems show strong background or noise, but P01 showed that arPLS can alter systems unequally and reduce chemistry association along with instrument association.
- **Formal question:** Does applying one universal SG or arPLS rule to every spectrum improve unseen-instrument transfer relative to minimal min–max, and is the answer shared across model families?
- **Comparison:** `PP-U-SG − PP-U-MIN` and `PP-U-ARPLS − PP-U-MIN` within each fixed model; deep-minus-classical difference-in-differences.
- **Population/unit:** all 13 primary domains; paired test rows and masters; equal domain weight.
- **Information/split:** identical T3-ZS splits; no held identity, family, QC distribution, label, or outcome chooses the policy.
- **Preprocessing/models:** `PP-U-MIN`, `PP-U-SG`, `PP-U-ARPLS`; RBF SVM, Random Forest, D0, frozen acquisition-aware deep.
- **Metrics/figures:** M01, M26, M27, M30; F25 and F37.
- **Plan coverage:** P01 supplies immutable actions/preservation; P02 supplies identical roles; P08 retrains the fixed panel and computes effects; P10 audits transformation harm; P11 performs paired inference.
- **Success:** a consistent paired gain with acceptable worst-domain and preservation behavior supports that universal policy for this population.
- **Failure:** loss, heterogeneity, or model-dependent effects support retaining minimal preprocessing and reporting why a single background rule is unsafe.
- **Prohibited claim:** clean-spectrum recovery, denoising, or universal background-correction superiority.

## RQ-S02 — platform-family-aware preprocessing

- **Motivation:** P01 preservation differs by acquisition system, so a represented platform family may warrant a source-learned action without tuning on the new unit itself.
- **Formal question:** For a held acquisition unit whose platform family has other source units, does a source-selected family action improve transfer over universal minimal preprocessing?
- **Comparison:** `PP-FAMILY-SRC − PP-U-MIN` for (1) all domains with fallbacks retained and (2) the metadata-defined supported-family subset.
- **Population/unit:** all 13 domains for the operational estimand; supported-family domains for the additional estimand; physical-master resampling.
- **Information/split:** held unit family ID only. Its spectra, QC distribution, labels, and outcomes are absent from policy selection. Source leave-one-unit-out pseudo-domains select the mapping.
- **Support/fallback:** at least two distinct supported source units. P02 chooses the largest viable masters-per-class threshold from `{2,3,4}` using metadata only. Unknown/unsupported families and invalid actions fall back to `PP-U-MIN`.
- **Preprocessing/models:** `PP-FAMILY-SRC` versus `PP-U-MIN`; the same RBF SVM, Random Forest, D0, and frozen acquisition-aware panel.
- **Metrics/figures:** M01, M26, M28, M29, M30; F35.
- **Plan coverage:** P02 freezes family identity/support/fallback; P03/P04 supply the source-only panel; P08 selects/evaluates; P10 audits actions; P11 reports both estimands.
- **Success:** gain, adequate coverage, stable selection, and acceptable preservation support family-conditioned preprocessing for represented platform families.
- **Failure:** no gain, unstable mappings, or heavy fallback favors universal or identity-blind QC processing.
- **Prohibited claim:** arbitrary per-instrument tuning or transfer to an unrepresented platform family.

## RQ-S03 — identity-blind QC-adaptive preprocessing

- **Motivation:** A genuinely unfamiliar instrument may have no recognized platform-family mapping, but each spectrum exposes row-local evidence of baseline, noise, spikes, and negative intensity.
- **Formal question:** Can a source-frozen row-local QC gate route each held-instrument spectrum to minimal, SG, or arPLS preprocessing without using identity, family, target-batch statistics, or outcomes?
- **Comparison:** `PP-QC-SRC − PP-U-MIN` on identical rows/domains plus preprocessing × model interactions.
- **Population/unit:** all 13 domains; masters resampled within domain; no supported-family restriction.
- **Information/split:** permitted current-row QC only. Source-training quantiles `{0.50,0.75,0.90}` instantiate a finite one-/two-trigger gate library; source pseudo-instrument domains choose the gate.
- **Fallback:** missing/nonfinite QC, invalid action, or insufficient source pseudo-domains uses minimal preprocessing and remains in the denominator.
- **Preprocessing/models:** `PP-QC-SRC` versus `PP-U-MIN`; the same four-model panel.
- **Metrics/figures:** M01, M26, M28, M29, M30; F36 and F37.
- **Plan coverage:** P01 supplies QC/action evidence; P02 freezes source roles and candidates; P08 selects/evaluates; P10 audits routing; P11 performs paired inference.
- **Success:** robust gain with stable gates, limited fallback, and acceptable preservation supports an identity-blind policy within the observed shift envelope.
- **Failure:** no gain, unstable actions, or heavy fallback means the current QC features cannot safely outperform universal minimal processing.
- **Prohibited claim:** guaranteed performance on every new instrument, denoising, or causal nuisance removal.

## RQ-S04 — value of target-instrument access

- **Motivation:** Operational users may be able to collect a few target-unit spectra. The value of those data must not be confused with zero-shot generalization.
- **Formal question:** How much do disjoint unlabeled, paired, or labelled target calibration masters improve over frozen zero-shot pipelines?
- **Comparison:** UDA, paired calibration, and `k={1,2,3,5}` few-shot learning curves versus their zero-shot baselines.
- **Population/unit:** support-permitting T3 domains; physical masters, not spectra, on the x axis and in uncertainty.
- **Information/split:** adaptation/calibration masters are disjoint from evaluation masters. UDA hides labels/pairs; paired calibration exposes pair IDs; few-shot exposes target labels.
- **Preprocessing/models:** `PP-U-MIN` primary and source-frozen `PP-QC-SRC` sensitivity; `A-UDA`, `A-PAIRED-CAL`, `A-FEWSHOT`. Target data do not retune preprocessing in v1.
- **Metrics/figures:** M01, M05, M07–M09; F28.
- **Plan coverage:** P02 freezes role separation; P07 executes curves; P11 reports regimes separately.
- **Success/failure:** curves quantify whether and which target information is useful; no gain identifies insufficient/unstable adaptation rather than invalidating zero-shot results.
- **Prohibited claim:** calling any target-informed result zero-shot.

## RQ-S05 — calibrated robustness and selective operation

- **Motivation:** Accuracy alone is insufficient if confidence is unreliable or performance collapses under modest spectral stress/quality changes.
- **Formal question:** Are frozen pipelines calibrated, selectively useful, and stable to declared perturbations and population tiers?
- **Comparison:** risk–coverage and degradation relative to each unperturbed frozen pipeline; notes-clear and acquisition-system sensitivity populations remain separate.
- **Population/unit:** primary and declared sensitivity tiers; domain and physical master.
- **Information/split:** zero-shot source-only; calibration uses development-known data only; perturbation outcomes cannot select models.
- **Preprocessing/models:** `PP-U-MIN`; frozen `E-PERTURBATION`/`E-FIXED-PIPELINES` evaluations.
- **Metrics/figures:** M07–M10, M19; F17, F18, F24, F26.
- **Plan coverage:** P06 freezes unperturbed predictions; P08 evaluates perturbations/tiers; P11 reports uncertainty.
- **Success/failure:** stable degradation and useful coverage support bounded operation; heterogeneous collapse defines explicit unsafe conditions.
- **Prohibited claim:** that simulated perturbations reproduce a specific physical instrument.

## RQ-S06 — narrow unknown-chemical rejection

- **Motivation:** The dataset supports a limited held-chemical stress test, not broad open-world discovery.
- **Formal question:** Can a station-conditioned frozen model reject one nonblank chemical absent from fitting while retaining known-class performance?
- **Comparison:** classical and deep known-only scores over eight station/held-chemical tasks.
- **Population/unit:** station subsets; held station–chemical task and physical master.
- **Information/split:** the held chemical is absent from fitting, score selection, calibration, and thresholding.
- **Preprocessing/models:** `PP-U-MIN`; `OS-CLASSICAL`, `OS-DEEP`.
- **Metrics/figures:** M11–M14; F29–F31.
- **Plan coverage:** P02 freezes held tasks; P09 evaluates; P11 reports task-level uncertainty.
- **Success/failure:** consistent results support only the declared narrow stress-test claim; failure documents field open-set fragility.
- **Prohibited claim:** general open-world or arbitrary unseen-chemical identification.

## RQ-E01 — representation mechanism and failure structure

- **Motivation:** Metric learning can improve predictions without removing nuisance, and lower domain predictability can also reflect chemistry collapse.
- **Formal question:** Do acquisition-aware objectives improve cross-view consistency and reduce target-adjusted domain information while retaining chemistry and transfer?
- **Comparison:** D0 versus the frozen acquisition-aware model using probes, same-master diagnostics, attribution controls, and structured errors.
- **Population/unit:** frozen primary embeddings/predictions; domain, physical master, and same-master pair.
- **Information/split:** training-only probes and frozen models; no diagnostic selects a model or policy.
- **Preprocessing/models:** `PP-U-MIN` primary; adaptive-policy action audits remain descriptive; `PR-LINEAR`, `PR-PAIR`, `PR-ATTR`.
- **Metrics/figures:** M15–M18; F22, F23, F32, F33.
- **Plan coverage:** P10 executes; G5 and P11 enforce the joint chemistry/transfer/invariance interpretation.
- **Success:** lower target-adjusted domain increment matters only with retained chemistry and improved transfer.
- **Failure:** reduced domain information with lost chemistry is collapse, not disentanglement.
- **Prohibited claim:** causal chemical/nuisance disentanglement, clean-spectrum recovery, or unsupported bond assignment.

## RQ-S07 — field-trial substrate portability amendment

- **Motivation:** Li-Lin clarified that the field trial was intended to ask whether the SERS substrates recover the required analyte signal independently of the acquisition instrument.
- **Formal question:** Within each support-qualified station and substrate family, is three-class analyte-discriminative signal recoverable on an instrument excluded from fitting without a practically important instrument-specific loss?
- **Comparison:** held-instrument recoverability and matched source-to-held loss within station × substrate-family × held-instrument domains, plus same-master substrate-by-instrument crossover effects. Portability requires `LCB95(held BA) >= 0.60` and `UCB95(source BA - held BA) <= 0.10`.
- **Population/unit:** 598 spectra but only 69 independent physical masters; 67 masters have at least two instruments, 39 have at least two substrate families, and 32 support at least one complete two-substrate-by-two-instrument crossover.
- **Information/split:** held instrument excluded from fitting, transformation fitting, selection, calibration, and stopping; every split and interval is master-grouped.
- **Preprocessing/models:** `PP-U-MIN` primary, universal SG/arPLS paired sensitivities; source-only `C-SELECTED` primary, fixed RBF SVM main sensitivity, other frozen classical models secondary, and compact deep models later on identical eligible cells.
- **Metrics/figures:** held-instrument balanced accuracy and per-analyte recall, worst-instrument loss, calibration, crossover interactions; F44–F47.
- **Plan coverage:** the P13 freeze found 13 confirmatory, three exploratory, and 18 unsupported observed domains plus eight confirmatory, seven exploratory, and 19 descriptive crossover blocks. Classical recoverability executes first; compact DL follows where support permits.
- **Success:** a bounded statement that a named substrate family preserves practically useful analyte recoverability over the tested instruments and supported station/analyte cells.
- **Failure:** evidence of a meaningful instrument loss, an inconclusive interval, or insufficient crossed support; these outcomes are not conflated.
- **Prohibited claim:** universal instrument independence, global substrate ranking, physical adsorption/enhancement proof, or counting 598 correlated spectra as 598 independent chemical samples.

Figure F44 is the design audit: each row is one recorded physical-master ID, panels are substrate families, columns are instruments, and cell shade is the stored-spectrum count. Gray means unmeasured, not failed detection. The full 69-by-4-by-10 grid has 374 observed and 2,386 missing combinations.

## Immediate next phase

P03 and P13 classical execution are complete. P13 retained all 34 observed
domains but found no substrate family that passed the locked portability rule
across every confirmatory domain. The primary result contained zero supporting,
five inconclusive, two inferior, and six terminally unavailable confirmatory
domains. The fixed RBF-SVM completed all 13 domains but still produced only two
supporting, three inferior, and eight inconclusive results. Same-master
crossover evidence and the arPLS sensitivity both showed meaningful
condition-dependence.

P04 is now complete. Its 208,691-parameter ordinary D0 control achieved mean
unseen-instrument spectrum BA 0.711 (worst domain 0.379), with 260/260 complete
held endpoint cells. The pooled paired gain over C-SELECTED was +0.050
(conditional 95% interval +0.022 to +0.078), but there was no clear gain over
fixed Random Forest or Extra Trees. Spectrum log loss was 1.635, worse than
the uniform-probability reference 1.099. Thus RQ-P01 has a useful ordinary-deep
baseline, not a completed acquisition-aware claim or solved calibration.

The immediate next phase is P05's no-fit pair/loss/role expansion, followed by
source-only supervised-contrastive and paired-consistency development under
G3. D0 is frozen; D1–D5 have not been trained. All earlier held outcomes remain
excluded from loss, architecture, preprocessing, epoch, and advancement
selection. This is procedural separation, not analyst blinding.

For RQ-S07, P04 predictions were reused on 15 exact PP-U-MIN P13 test views,
including all 13 confirmatory domains, with seven passing held recovery alone.
This is not a controlled learner comparison: D0 trained across source
substrates, unlike P13's substrate-restricted classical fits. Exact P13 deep
source refits, matched-source loss, and preprocessing sensitivities remain
outstanding. P13 support and margins are unchanged. See
[P04 results](../results/p04_deep/P04_RESULTS.md) and F19/F20/F48.
