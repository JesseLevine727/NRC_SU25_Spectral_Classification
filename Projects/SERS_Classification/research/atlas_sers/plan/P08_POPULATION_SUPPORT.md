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
