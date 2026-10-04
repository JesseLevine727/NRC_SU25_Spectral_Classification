# P08 filtered-population support: membership and regenerated roles

**Date:** 2026-10-04. **Authority:** no-fit planning under P08-A07. No new split, QC cutpoint, model, prediction or score has been produced by this prescreen.

Sections 1–3 retain the membership-prescreen checkpoint. Section 4 records the subsequent regenerated-role audit; it supersedes the earlier statement that fold support had not yet been checked. Neither checkpoint authorizes scientific execution.

## 1. What the filters change

The [prescreen](../results/p08_readiness/population_membership_prescreen.json) authenticated the three stored P01 manifests against the frozen artifact manifest. It verified exact row membership and unchanged metadata; it did not create replacement datasets or load spectra. All three populations retain the same 69 recorded physical masters.

| Population | Stored spectra | Physical masters | Instruments | Original primary domains passing the pooled rule |
|---|---:|---:|---:|---:|
| Primary | 598 | 69 | 10 | 13 |
| Notes-clear | 500 | 69 | 10 | 11 |
| Mira-1 excluded | 575 | 69 | 9 | 12 |

The inherited pooled-domain rule requires three observed task classes and at least 15 physical masters. It is a domain-level eligibility rule, not a guarantee that every individual fold or source-validation role is supported.

Notes-clear removes enough recorded observations to make two original primary domains ineligible: CWA/Pendar-2 retains ten spectra from seven masters and only one class; pills/Pendar-3 retains eleven spectra from eleven masters and three classes. The latter fails the master-count threshold, not the class-count threshold. Mira-1 exclusion removes all observations from surfaces/Mira-1. These are consequences of the declared filters, not new evidence of sensor failure.

The filters remove repeated views rather than independent samples from the overall dataset. Retaining all 69 masters does not preserve every instrument/analyte combination or source-training role. A change in aggregate classification after filtering could reflect changed domain coverage, changed source information or both; it cannot automatically be attributed to cleaner spectra.

## 2. Regeneration requirements

P08-A07 still requires regenerated, population-bound physical-master roles and fresh registered source-only model selection. Reuse the original split algorithm, seeds, stratification labels and eligibility logic; do not choose new seeds to force a different partition. Because each filter retains the same master/station/target records, the deterministic outer master assignment should be unchanged under identical splitter inputs and software. That is an inference from metadata, not yet an authenticated regenerated-role result. Rebuild and compare it explicitly.

Even if outer master assignments agree, the available spectral rows differ. Source instrument exclusion, pseudo-domain validation, fallback master-CV, calibration roles and test-role hashes must be reconstructed from each filtered population. A primary context label or equal master count cannot justify reusing its fitted model, recipe decision, numerical preprocessing-policy decision or prediction.

The inherited P02/P04 builders contain primary-population counts and predeclared domain scopes. Do not call their full primary-population validators on a filtered dataset and suppress resulting failures. The next bounded implementation must preserve the original code and build population-specific metadata accounting around the existing split definitions. Distinguish pooled ineligibility, sparse individual test folds and unavailable source-selection/calibration roles. Preserve every reason rather than silently deleting a difficult fold.

For each supported regenerated context, repeat the registered classical tuning and source-only neural-recipe selection, then freeze that context's neural recipe across preprocessing actions. The primary recipe map is not copied. No extra candidate grid, architecture, target-label selection or scientific execution is authorized by this planning step.

## 3. What remains before a population execution proposal

The pooled checks give 11 and 12 eligible original primary domains; they do not yet establish executable context or fit counts. The next audit must authenticate regenerated outer/inner/calibration roles, class support, master separation, instrument exclusion and exact dependencies. It must enumerate selected-recipe development, per-policy fitting, calibration and prediction operations before costing finite resources.

Population-specific statistical rules must distinguish within-tier preprocessing effects from comparisons with the primary dataset. The eligible domain sets differ, so a direct difference between two all-domain means is not a matched-domain filtering effect. Preserve unsupported domains in reporting and define any common-domain sensitivity before outcomes. Neither a smaller evaluation set nor removal of a difficult instrument constitutes evidence of improved instrument generalization.

This note does not amend the primary 260 contexts, the 54/206 QC adaptive/fallback rule, the approved N2 normalization graph or the universal-first sequence. Figures remain future native TikZ/offline HTML/vector PDF/PNG outputs under the existing figure protocol. No new predictive result is reported here.

## 4. Regenerated role support

The [independent metadata audit](../results/p08_readiness/population_role_support_audit.json) regenerated the outer assignments with the inherited splitter and original seeds. Each population reproduces all 345 historical master/repeat assignments exactly. Master sample IDs must be read as text before sorting, as in the original P02 normalization. An initial diagnostic used inferred numeric IDs and did not reproduce the frozen folds; correcting the import type restored exact agreement without changing any seed or sample assignment. The agreement was checked with both the historical scikit-learn 1.9.0 environment and the local 1.8.0 environment.

A context is one eligible held-instrument domain, one repeat and one outer test fold. The calibration counts below concern the classical three-fold master-CV procedure. Neural temperature calibration instead consumes selected source-validation logits; fresh neural guard-role and calibration readiness are not established by this audit. The shared selection-role audit gives:

| Population | Eligible domains | Eligible contexts | Model-selection support | Classical calibration support | Both audited stages supported |
|---|---:|---:|---:|---:|---:|
| Primary reference | 13 | 260 | 260 | 260 | 260 |
| Notes-clear | 11 | 220 | 216 | 215 | 215 |
| Mira-1 excluded | 12 | 240 | 240 | 240 | 240 |

Notes-clear has five CWA/Mira-2 contexts with insufficient source-master support for the registered three-fold classical calibration. Four also lack supported model selection; the fifth retains supported pseudo-instrument selection but not classical calibration. Preserve all five as unavailable for the complete registered classical pipeline. This does not automatically remove a neural method whose distinct source-calibration requirements might be supported; its additional roles must be audited first. Do not lower the fold count, borrow held observations, insert another population's calibration or report only successful contexts as if support were complete.

Model-selection modes are 128 pseudo-instrument and 132 master-CV contexts in the primary reference; 94 pseudo-instrument, 122 master-CV and four unsupported contexts in notes-clear; and 83 pseudo-instrument and 157 master-CV contexts without Mira-1. These correspond to 681, 583 and 637 selection units, respectively. The supported calibration roles contain 780, 645 and 720 units. A unit is a fitting/validation partition, not a model fit: candidate models, technical seeds and policies still have to be enumerated separately.

The eligible context sets contain 41, 39 and 32 class-sparse held folds, respectively, but no empty held folds. They remain in the planned evaluation. Sparse held-fold class coverage differs from insufficient source-selection or calibration support; it does not justify rebuilding a split. Every generated calibration fitting and validation unit retains all station classes in these audited populations.

All 13 original domains remain in the metadata audit. Pooled-ineligible domains account for 40 notes-clear and 20 Mira-1-excluded domain/repeat/fold records. Those records are not newly supported experiments or failed sensor measurements. The primary reference remains unchanged; the filtered tiers do not acquire the primary models, recipe choices or predictions through matching outer assignments.

The eligible held roles contain 557, 449 and 534 distinct spectra in the primary, notes-clear and Mira-1-excluded populations, respectively. Each retains all 69 masters across those roles. These are distinct-spectrum counts before the source-support restriction, not calibrated prediction counts or independent sample counts. Five repeats produce 2,785, 2,245 and 2,670 spectrum/context appearances; they do not multiply the independent physical samples.

The [population planner](../src/atlas_sers/evaluation/p08_population_plan.py) binds each context to its population, input metadata and split contracts. Its identity-bearing tables remain private. The supervisor authenticated 14 input files, independently reconstructed source roles, checked primary parity against the frozen P02 tables, and verified the canonical hashes after reloading all 24 generated metadata tables. No intensity array or saved outcome was loaded.

Fresh neural guard-role and source-logit calibration audits, exact model-operation ledgers, population-specific inferential comparisons and finite resource proposals remain required. In particular, a complete 220-context calibrated classical notes-clear result cannot be promised under the registered support rules. Any conditional support summary must retain the five unavailable contexts in its coverage accounting and must not be called a full-population result. The planner's `metadata_ready` field denotes its audited outer/selection/classical-calibration conditions, not complete neural or execution readiness.

## 5. Neural guard and calibration-role support

The subsequent [neural metadata audit](../results/p08_readiness/population_neural_support_audit.json) checks the distinct CNN requirements. It supersedes Section 4's pending neural-role check, not its classical support counts. The supervisor authenticated 33 input files and independently reconstructed the guard assignments from source-master metadata. No spectrum intensity, saved logit, prediction or outcome was loaded.

| Population | Eligible contexts | Ordinary CNN and source-calibration roles supported | Acquisition-aware recipe comparison structurally supported | Ordinary-CNN fallback required by support | Neural roles unavailable |
|---|---:|---:|---:|---:|---:|
| Primary reference | 260 | 260 | 128 | 132 | 0 |
| Notes-clear | 220 | 216 | 93 | 123 | 4 |
| Mira-1 excluded | 240 | 240 | 83 | 157 | 0 |

Neural temperature calibration uses the registered source-selection validation outputs, averaged by physical master with equal master weight. It does not require the additional three-fold calibration procedure used by the classical pipeline. The notes-clear tier therefore retains one ordinary-CNN case beyond its 215 classically supported contexts. This case has supported source validation but one unsupported neural guard fold. The inherited selection rule blocks acquisition-aware advancement and retains the ordinary-CNN fallback if its later numerical fits succeed. It does not justify inventing another fold or borrowing held data.

The notes-clear fallback count comprises 122 master-CV contexts and this one guard-limited pseudo-instrument context. Four further CWA/Mira-2 contexts lack supported source selection and remain unavailable. Without Mira-1, all 240 eligible contexts support ordinary neural fitting/calibration metadata; 157 lack pseudo-instrument selection and therefore require the ordinary-CNN fallback. A structural fallback is not a measured recipe-selection result or evidence that the ordinary CNN is more accurate.

The three reference/tier audits contain 384, 282 and 249 guard units, respectively. All are supported except one notes-clear unit, which remains explicitly excluded. Maximum audited batch capacities, including outer source refits, are 38, 37 and 38 spectra; each is below the locked ceiling of 48. These checks establish metadata feasibility, not successful optimization, finite logits, calibration quality or GPU-memory compliance.

Guard construction retains the inherited chemical-stratified SHA-ordered assignment but uses the newly population-bound context IDs. Regenerated guard identities are consequently not asserted to equal historical P05 identities. The primary audit is a metadata reference, not permission to replace the completed primary recipe map. Guard-validation, classical-calibration and held-test rows are excluded from neural temperature-calibration inputs.

Later classical-versus-neural contrasts must use matched contexts: metadata supports at most 215 common notes-clear contexts and 240 common Mira-1-excluded contexts. Retain each method's full coverage separately, and record further numerical failures rather than treating these ceilings as completed predictions. The differing 215/216 denominators must not enter an unpaired apparent model advantage.

The [neural adapter](../src/atlas_sers/evaluation/p08_population_neural.py) checks all inherited units, physical-master isolation, held-instrument exclusion, population provenance and role hashes. Its caller-supplied hashes establish internal consistency; the supervisor separately authenticates the upstream artifacts. Exact population operations, selection-dependent reuse, inference definitions and finite resources remain incomplete. No new recipe has been selected and no model, temperature or outcome has been computed.

## 6. Conditional population operation accounting

The [operation audit](../results/p08_readiness/population_slot_ledger_audit.json) now enumerates the filtered-population comparisons for MIN, Savitzky–Golay smoothing and arPLS baseline correction. It closes the exact-accounting gap above, not the inference, resource or numerical-execution gates. The four-method alternative contains RBF-SVM, Random Forest, the ordinary CNN and the source-selected CNN strategy. The five-method alternative additionally contains Extra Trees. Whether the population sensitivity retains Extra Trees is an explicit scope clarification awaiting the owner's response; neither alternative is authorized to run.

Every classical family repeats its registered source-only hyperparameter search within each population and preprocessing action. The existing 598-row representation arrays remain unchanged; authenticated population and role identities select their rows. The catalog does not create filtered intensity arrays, inherit a primary fitted estimator or transplant the primary neural recipe map.

Neural development evaluates all four registered recipes on each supported context's MIN source-selection roles and supported guard roles. This includes contexts whose structural support already requires the ordinary-CNN fallback: the inherited readiness rule still requires the registered, nonexcluded development records. The single unsupported notes-clear guard creates 12 excluded recipe/seed slots, not 12 attempted fits. A later numerical failure does not become permission to omit evidence or use an unverified fallback.

One future source-only decision selects a recipe per context. The selected identity is then fixed across the three preprocessing actions. For a structurally comparable context, the catalog lists D1, D2 and D3 alternatives, but at most one can activate. If the decision selects D0-M, the selected strategy references the ordinary-CNN pipeline without another refit. Both strategies still wait for successful source readiness. An unknown or failed decision is not a D0-M selection.

| Population | Panel alternative | Prospective model fits, lower–upper | Scalar calibrations, lower–upper | Logical evaluated pipelines |
|---|---|---:|---:|---:|
| Notes-clear | Four methods | 168,150–170,277 | 3,234–4,071 | 2,586 |
| Notes-clear | Five methods | 256,260–258,387 | 3,879–4,716 | 3,231 |
| Mira-1 excluded | Four methods | 183,006–184,749 | 3,600–4,347 | 2,880 |
| Mira-1 excluded | Five methods | 279,135–280,878 | 4,320–5,067 | 3,600 |

These are successful-completion bounds without retries, not measured executions or independent sample counts. The lower bound assumes every selected strategy resolves to D0-M. The upper bound allows one acquisition-aware winner in every structurally comparable context. Classical grid searches, source-validation roles, technical seeds and preprocessing actions account for most fits. The number of physical samples remains 69 per tier.

Across both populations, the four-method ceiling is 355,026 model fits; the five-method ceiling is 539,265. Extra Trees adds 184,239 fits and 1,365 scalar calibrations. These population counts are separate from the primary universal comparison and cannot borrow its proposed resources. Time, RAM, GPU and retained-artifact ceilings still require their own cost basis and finite proposal.

The catalog contains 549,384 and 595,608 operation records for the five-method notes-clear and Mira-1-excluded alternatives, respectively. Those catalogs deliberately include mutually exclusive recipe branches. Their executable upper bounds are 536,970 and 585,150 operations; summing all listed branches would overstate the planned work. Per-recipe upper bounds are likewise nonadditive. Logical pipeline counts retain two neural strategy endpoints even where both reference one fitted pipeline; they are not independent models or statistical replicates.

Classical calibration retains exact master-CV prediction aliases and fresh pseudo-domain calibration fits. Forests average seed probabilities before one temperature; neural models fit a temperature per seed before averaging calibrated probabilities. Neural temperature and epoch selection use inherited source-selection predictions, never guard, classical-calibration or held-test evidence. The graph binds these dependencies and orders every operation after its prerequisites.

The supervisor's input bridge authenticated 29 files, checked the regenerated role and neural-support digests, and matched 366 notes-clear and 471 Mira-1-excluded master-CV calibration roles to their selection roles. An independent arithmetic and dependency audit verified both panel alternatives for both populations, including compressed-JSON readback. Identity-bearing graphs remain private; the public audit contains aggregate counts and hashes. Caller-provided hashes alone still do not prove provenance or physical-master separation.

No new fit, temperature, prediction, score, uncertainty draw, QC cutpoint or preprocessing array was calculated. Population-specific inference, the panel clarification, finite resource proposals, later perturbation specifications and final readiness review remain open. Universal preprocessing remains first; these sensitivity branches do not acquire automatic execution permission.
