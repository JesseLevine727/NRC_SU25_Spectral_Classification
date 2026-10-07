# P08 later-branch decisions: approved direction

**Approved:** 2026-10-04. **Authority:** scientific planning only. The owner accepted the four recommendations and resumed the existing no-fit readiness goal. No fit, scalar calibration, prediction, resampling draw, representation rebuild or perturbation run is authorized.

**Current decision status, 2026-10-07:** P08-A12 in Section 7 resolves the filtered-population panel in favor of including Extra Trees in both sensitivities. This supersedes earlier unanswered-panel wording. P08-A10–A11 in Section 6 retain the five-method universal robustness panel and fixed-clean-route QC interpretation. The owner has also raised concern about excessive planning; the panel answer does not approve or reject the separate proposed simplification or pause. This update records the answer only, without expanding implementation or authorizing execution. Existing machine-readable pending-state fields have not yet been synchronized with this decision; their numerical definitions and execution denials remain unchanged.

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

## 6. Robustness scope decisions, approved 2026-10-07

The owner separately approved the two recommendations below. They supplement P08-A08–A09 without approving execution, finite resource ceilings or additional population experiments.

| Decision | Recorded answer | Approved scope and boundary |
|---|---|---|
| P08-A10: universal robustness panel | Use all five universal-panel methods (Recommended) | Test MIN, SG and arPLS with RBF-SVM, Random Forest, Extra Trees, D0-M and the context-local source-selected CNN. This resolves the robustness panel only; the filtered-population panel remains a separate unanswered question. |
| P08-A11: QC stress interpretation | Plan a clearly labelled fixed-route QC sensitivity (Recommended) | Hold each row's clean native-QC preprocessing decision fixed, disturb the common-grid input and apply that selected action using the existing mixed-route estimator. This tests robustness after routing, not whether the gate detects new contamination. No native-grid gate-reaction experiment is added. |

The [stress-test protocol](P08_PERTURBATION_PROTOCOL.md) defines the measurement-grid distinction and the fixed-route result's interpretation. The adaptive panel remains the existing four methods; adding Extra Trees to universal robustness does not expand it. Keep the 54-context supported adaptive subset separate from the 260-context fallback-inclusive result. Complete MIN fallbacks must use the matched disturbed MIN prediction, not the saved clean prediction.

The 96-case descriptor inventory is metadata only. Exact reconstruction/prediction accounting, finite resource proposals, numerical inference and a separate execution permit remain required. No new model, calibration, preprocessing result, prediction, QC route or disturbance was computed by these approvals.

## 7. Filtered-population panel, approved 2026-10-07

**P08-A12. Recorded answer:** “Include Extra Trees in both sensitivities.”

The notes-clear and Mira-1-excluded sensitivity studies will use RBF-SVM, Random Forest, Extra Trees, D0-M and the source-selected CNN procedure. This selects the already documented five-method alternative; it does not add a new model family or candidate grid. The combined prospective fit ceiling is **539,265**, rather than **355,026** for four methods, a difference of **184,239**. These are planning counts, not approved executions or completed fits.

This decision does not change the main five-method universal preprocessing or robustness panels. It does not approve resource limits, launch either sensitivity, authorize the U0 pilot, or answer whether to pause or narrow the current planning goal. The four-method catalogs remain historical alternatives, not additional work to execute. No new implementation, scientific calculation or training accompanies this decision record.
