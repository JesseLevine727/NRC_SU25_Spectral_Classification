# P08 historical evidence and complete-pipeline reuse

**Recorded:** 2026-09-30. **Authority:** read-only evidence audit; no fitting, prediction, calibration or scoring is authorized.

The [operation audit](../results/p08_readiness/minimal_operation_reuse_audit.json) binds all **202,407 MIN operation slots** to the previously reviewed scientific evidence. It rehashed **20,934 files**, including source/specification records, role registries, stored predictions, neural checkpoints and the immutable MIN array. It did not rerun an estimator, optimize a temperature, generate a prediction or calculate a metric.

The mapping supports reuse of complete saved predictions on identical observations under the same fitting procedure. It does not imply that every historical intermediate exists as a separate reusable object. The distinction matters because the future SG/arPLS operation graph is more detailed than parts of the original classical serialization.

## 1. What the evidence proves

| Evidence class | MIN operation slots | Permitted interpretation |
|---|---:|---|
| Existing record or array with an exact file/row binding | 105,108 | Inspect or reuse that recorded object only within its authenticated role/specification scope |
| Completed classical fit record without a persisted estimator | 94,400 | The fit occurred and its saved outputs can support the comparison; the record cannot infer on a new input |
| Complete saved model–context endpoint | 1,079 | Reuse the original final predictions for the identical MIN procedure and test observations |
| Single-seed SVM raw scores contained in its final endpoint | 260 | The single-seed scores are present within the authenticated final file |
| Tree seed-level held slot represented only by its final ensemble | 1,560 | The ensemble is available; separate constituent seed predictions are not reconstructible from that average |
| Total | **202,407** | Evidence coverage, not a blanket runtime cache permit |

The 1,079 distinct endpoints comprise 780 classical model–context procedures and 299 unique neural recipe–context procedures. Reporting D0-M and P05-SELECTED separately gives 1,300 method–context cells, but those neural strategies coincide in 221 contexts. The audit checked exact saved probability equality in those coincident cases. Neither operation counts nor repeated method cells increase the number of physical samples.

Classical source-selection predictions retain their candidate, seed and role bindings. Fresh pseudo-domain calibration bundles retain individual seeds. Cached master-CV calibration bundles retain their seed average, with constituent source predictions still available in the original selection shards. Classical final prediction files retain the technical-seed aggregate. Neural evidence includes the original per-seed checkpoints, calibration states and final predictions.

The audit resolves source-selected classical candidates and neural refit durations from saved records. It does not select them again. Every reference binds the relevant model specification, input representation, source/test identities and historical artifact bytes. The original source-only selection, master isolation and held-instrument exclusion remain unchanged.

## 2. Complete MIN fallback aliases

Owner-approved P08-A04 makes an unsupported held-family deployment use the **MIN-trained estimator and MIN test input**. P08-A05 applies the same complete-pipeline fallback to unsupported QC contexts. The authenticated alias ledger contains:

- **1,040 family-policy endpoints:** four methods × all 260 contexts;
- **824 QC fallback endpoints:** four methods × 206 unsupported contexts.

Each alias records its reason, context-local model or neural recipe, MIN input hash, fitting/test hashes, model-specification hash and target endpoint evidence. It references an existing complete prediction procedure; it does not route individual rows between separately trained models. Extra Trees is not part of the adaptive panel.

These aliases add **zero fits and zero predictions**. Family-specific transfer remains unavailable because no held-family context is supported. QC adaptation remains prospective in the 54 supported CWA contexts. Equal outputs forced by a fallback definition are structural identities, not empirical evidence that adaptive preprocessing succeeds or fails.

## 3. Reuse boundaries for the future executor

The MIN reporting path should consume the authenticated complete endpoints directly. It must not demand artificial completion of missing seed-level intermediates, silently retrain MIN to match a newer file layout, or label a fit-status row as a serialized estimator.

Existing MIN weights or predictions cannot substitute for SG/arPLS fits: the input-representation hashes differ. Within a new policy, selected master-CV calibration predictions can be reused only after the same role, candidate, seed and input hashes agree. QC reuse additionally requires identical source-fitted threshold/routing bindings and complete selection/stopping/calibration dependencies. Similar scores or equal action counts are insufficient.

Historical classical models cannot process perturbed inputs because their fitted objects were not saved. A later approved reconstruction must reproduce the original selected specification on the same source observations, preserve all original evidence, and pass the declared prediction-parity check. That is a separately budgeted operation, not reuse of the saved predictions. No reconstruction is authorized by this audit.

The private operation/alias ledger contains identity-bearing pointers. Only its aggregate findings and archive digest are public. The later executor must verify those bindings at admission and must still obtain a separate scientific permit for every newly executed operation.
