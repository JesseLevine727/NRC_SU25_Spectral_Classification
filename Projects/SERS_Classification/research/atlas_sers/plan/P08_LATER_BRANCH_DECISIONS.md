# P08 later-branch decisions: approved direction

**Approved:** 2026-10-04. **Authority:** scientific planning only. The owner accepted the four recommendations and resumed the existing no-fit readiness goal. No fit, scalar calibration, prediction, resampling draw, representation rebuild or perturbation run is authorized.

## 1. Decision record

| Decision | Approved choice | Question and evidence boundary |
|---|---|---|
| P08-A06: normalization | N2: retain each primary context's saved MIN-source-selected classical family; retune that family's registered hyperparameters and train new estimators for each frozen control. | How does normalization affect a family already selected under MIN? This is a conditional exploratory comparison, not selection of the best family for each normalization. |
| P08-A07: filtered populations | Regenerate physical-master-separated roles for the notes-clear and Mira-1-excluded tiers; repeat registered source-only tuning and neural-recipe selection within those roles. | How do the registered comparisons behave in these changed populations? Earlier selections are not transplanted into a different training/test partition. |
| P08-A08: perturbation placement | Apply test-time disturbances to the stored common-support interpolated spectrum before the numerical preprocessing pipeline. | Does the complete preprocessing pipeline protect identification against the imposed disturbance? The starting input is already interpolated, not native raw instrument data. |
| P08-A09: synthetic shift edges | Use constant endpoint extension for unavailable coordinates in the prespecified shifts. | Preserve the registered model-input width while labelling the extension as a synthetic boundary assumption, not recovered Raman measurements. |

The same approval retains universal preprocessing first, adaptive preprocessing second, and robustness/exploratory branches as separately costed stages. Approval does not remove a later branch, expand a model panel, change a frozen input, or transfer an earlier training permit. The universal model panel, primary P05 recipe map and P08-A01–A05 support/statistical decisions are unchanged.

## 2. Normalization scope and accounting

N2 covers `R_SNV_400_1800`, `R_VECTOR_400_1800`, `R_AREA_400_1800` and the destructive derivative control `R_D1_400_1800`. Keep their frozen native scaling; appending min–max scaling would erase the intended SNV/vector/area contrasts. The derivative control remains exploratory and cannot become a primary policy because it produces a favorable result.

The [audited saved-family mapping](../results/p08_readiness/fixed_family_alternative_audit.json) covers all 260 primary contexts. Within each control, the existing count is 9,876 source fits, 426 fresh calibration fits, 290 final fits and 444 exact-role calibration prediction aliases. This gives 10,592 model fits and 260 scalar temperatures per control, or 42,368 model fits and 1,040 scalar temperatures for all four controls. The [exact metadata operation graph](../results/p08_readiness/normalization_slot_ledger_audit.json) now verifies these counts and their dependencies; it is not execution authority.

N1's 525,288-fit full-family selector is not selected for this branch. It remains documented as the broader alternative, not an additional experiment. N2's family choices favor the MIN selection procedure by construction and can miss families better suited to another normalization. State that conditioning beside the results.

New-control tuning must consider every registered candidate within the frozen family using source data only. The original MIN winning hyperparameters and fitted estimator are not reused on the new representation. If no candidate in that family remains valid, report the context unavailable; do not switch family or substitute MIN. Technical-seed aggregation and scalar calibration retain the inherited classical order.

The eight missing historical selected-classical final endpoints remain missing. New-control jobs may cover all 260 contexts, but comparison with the existing selected-classical reference has at most 252 paired contexts. Do not repair old outcomes or silently call that subset a complete all-context comparison. The [normalization inference specification](P08_NORMALIZATION_INFERENCE.md) now fixes that reference-supported subset, its contrast registry and uncertainty rules before new outcomes.

## 3. Regenerated population scope

Filter using the existing tier definitions, then regenerate master-separated roles and apply the same eligibility logic. Every fitting, selection, stopping and calibration step must exclude the new held instrument and new test masters. Repeat the registered classical tuning and source-only neural-recipe selection within the resulting source population, then freeze each new context's neural recipe across preprocessing actions.

This approval permits planning those regenerated roles and selection jobs; it does not approve new architectures, new candidate grids, numerical selection outcomes or reuse of the original 260-context recipe map. Preserve unsupported domains and compute exact model/role/job counts before proposing finite execution ceilings. A changed population can change task support and evaluation denominators; that is not automatically an improvement in instrument generalization.

## 4. Perturbation placement and edges

The starting representation is the existing interpolated 400–1,849 cm⁻¹ common-support spectrum. The later numerical specification must make the crop/perturbation/pipeline order explicit and retain the primary 400–1,800 cm⁻¹ model input. Apply the same disturbance realization to matched methods before their registered numerical preprocessing; do not retrain on corrupted held spectra. Severity and randomization choices must be fixed without inspecting new held outcomes.

For `y_shift(v) = y(v − delta)`, use the existing interpolated value where the requested coordinate is supported. Outside available support, repeat the nearest endpoint value. At a +5 cm⁻¹ shift, primary inputs require five unsupported lower-edge coordinates, 395–399 cm⁻¹; these are labelled synthetic. Under the shared wider input, the −5 cm⁻¹ primary-input coordinates are supported. Do not add padding merely because a narrower intermediate crop would have removed available values.

Record affected edge coordinates and distinguish them from measured/interpolated support. Endpoint extension can alter edge peaks; the rule is not a physically validated instrument model. Unperturbed inputs are unchanged, and the zero-dose calculation must reproduce the registered pipeline within its locked numerical tolerance before perturbation results are accepted.

Discrete severity grids where still unspecified, impulse magnitude, random replicates, source-noise estimation, paired random streams, zero-dose tolerances, degradation summaries and statistical contrast families remain technical specification work. This approval resolves placement and edge treatment; it does not invent those remaining values or authorize their calculation on scientific data.

## 5. Implementation and release gates

DeepSeek V4.1 Flash implements the metadata-only N2 operation planner from public source and invented fixtures; the supervisor reviews and tests it. The full identity-bearing graph and input bindings remain private. Public evidence may include aggregate stage counts, code/specification hashes and the private-archive digest after review. A passing synthetic graph test is not authentication of the actual dataset or a validated training runtime.

The reviewed planner enumerates 89,632 operation slots across four controls, including 42,368 model fits, 1,040 scalar calibrations and 1,776 within-control calibration prediction aliases. All 681 source units were reconciled with the saved classical fit manifest, and 396 master-CV calibration roles matched source roles exactly. The audit authenticated 22 input files and verified their unchanged bytes. It builds metadata only; it does not estimate new model parameters, choose new hyperparameters or generate predictions. N2 inference rules are now specified; their numerical implementation and runtime acceptance remain incomplete.

The [resource proposal](P08_RESOURCE_PROPOSAL.md) remains a set of unapproved finite limits. N2 scope approval does not approve its proposed time, storage, memory or fitting allowances. The first possible scientific request remains the separately reviewed U0 source-only smoke; successful U0 work would count toward U1 only after exact input/specification verification. There are no automatic retries or automatic stage advancement.

Figures retain the [P08 format and disclosure rules](P08_FIGURE_PROTOCOL.md): native TikZ, offline HTML, vector PDF and PNG from shared reviewed semantic data. These decisions create no new classification, preservation or degradation result and no new outcome figure.
