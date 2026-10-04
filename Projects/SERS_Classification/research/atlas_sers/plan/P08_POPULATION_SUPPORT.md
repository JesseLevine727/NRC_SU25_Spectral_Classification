# P08 filtered-population support: metadata prescreen

**Date:** 2026-10-04. **Authority:** no-fit planning under P08-A07. No new split, QC cutpoint, model, prediction or score has been produced by this prescreen.

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
