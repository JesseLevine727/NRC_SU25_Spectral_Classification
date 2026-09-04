# P13 classical portability completion audit

**Protocol:** `nato-sers-p13-v1-locked`

**No-fit plan:** `P13PLAN-c0d28caeacb8088107fc09bc`

**Execution:** `P13-3d21aa17c7d6cd750ca9d286`

**Completed:** 2026-09-04

This is a post-outcome completion record. It does not modify the frozen P13
protocol, thresholds, support tiers, estimator roles, split rules,
preprocessing roles, multiplicity rule, or failure handling.

## Outcome-blind authorization

The no-fit expansion was completed before scientific fitting. It contained 960
contexts, 3,200 role records, 6,720 procedure/fold records, 1,680 endpoints,
42,360 planned fit invocations, and 240 resumable shards. Every context mapped
to a declared support cell and every fit mapped to one context. Master and
instrument exclusions passed for outer fitting and cross-fit calibration. The
validation report recorded zero model fits and authorized execution.

## Terminal reconciliation

| Requirement | Evidence | Result |
| --- | --- | --- |
| All prediction shards valid | 240 expected / 240 valid / 0 quarantined | pass |
| Fit IDs complete and unique | 42,360 planned / 42,360 terminal | pass |
| Fold endpoints retained | 6,720 / 6,720 | pass |
| Complete endpoint repeats | minimum five out-of-fold predictions per view | pass |
| Held instrument excluded | role and fit-manifest audit | pass |
| Physical-master isolation | train/validation/test membership audit | pass |
| Preprocessing comparison parity | identical common-success view sets | pass |
| Domain accounting | 336 eligible domain/procedure/policy rows | pass |
| Primary claim accounting | 34 domains and 102 class cells | pass |
| Crossover accounting | 34 blocks for each of seven procedures | pass |
| Bootstrap contract | 10,000 resamples for every generated interval | pass |
| Public boundary | no observation or master identifier in aggregate tables | pass |

The private execution validation report SHA-256 is
`d846ef57524ac45efc0df1b46f9e1a7a74373d39204d3ede1a164154c686d053`.
The execution protected-state SHA-256 is
`3d21aa17c7d6cd750ca9d28604fa601768ab35eb22f73263ce13d121963254ed`;
the aggregation protected-state SHA-256 is
`2733c5f6bbb656a871c49154e0722fa66ff122b1839b03bd874c4c162861884c`.

## Evidence completion

- `EXP-P13-C01`: complete; 34 primary domain states and 102 class-cell states.
- `EXP-P13-C02`: complete under the locked missing-support rules; all 238 rows
  retained, with 13 predictive four-cell rows and all representation contrasts.
- `EXP-P13-C03`: complete secondary field-log corroboration; 35 rows.
- `EXP-P13-C04`: complete paired universal preprocessing sensitivity; 181
  common-view comparisons.
- F45–F47: complete as native TikZ/PGFPlots, vector PDF, PNG, and standalone
  HTML from byte-matched semantic CSVs.
- Public result narrative and aggregate tables: complete under
  `results/p13_portability/`.
- Private row predictions, master-view predictions, calibration status, fit
  status, and shard state: retained outside the public repository.

## Reproducibility

Re-running aggregation against unchanged protected state returns a verified
skip. Re-publishing twice produced identical hashes for all 30 generated P13 result and
figure artifacts in the reproducibility comparison. The public release is
validated independently by `scripts/validate_p13_public_release.py` and by the
repository-wide scaffold validator.

The authoritative scientific interpretation is
`results/p13_portability/P13_RESULTS.md`. P04 compact deep development remains
separate: P13 outcomes may not select its architecture, epoch policy,
optimizer, regularization, or preprocessing.
