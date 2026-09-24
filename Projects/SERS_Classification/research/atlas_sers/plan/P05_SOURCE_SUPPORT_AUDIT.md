# P05 source-support audit

**Date:** 2026-09-24. **Scope:** outcome-blind metadata checks; no new neural fits.

The repeated-measurement structure is useful for acquisition-aware learning, but it is not available equally in every leakage-safe training group. The next model must handle this explicitly. This document records inherited P04 split support, not an approved P05 training manifest or an improvement in classification.

## What was inspected

The inputs are the frozen 598-spectrum/69-master manifest and the completed P04 planning context and role registries. The prerequisite pointers, completion status, validation-report hashes, file hashes, and P04-to-P01 manifest link were checked before the audit. No intensity array, new prediction, or held-test performance table was opened for this audit.

There are 320 inherited outer contexts: 60 within-station contexts and 260 unseen-instrument contexts. They contain 861 inner model-selection units: 180 and 681, respectively. The 861 inner fitting roles plus 320 outer fitting roles give **1,181 fitting roles**. Repeated roles do not add specimens to the dataset.

The full role table has 2,362 roles and 132,392 row assignments, including validation and test roles. These large counts reflect repeated evaluation assignments of the original observations, not 132,392 independent spectra. Held metadata is used only to verify exclusions; support statistics use fitting roles only.

## The important support finding

| Evaluation setting | Station | Inner fitting roles | Distinct masters per fitting role | Roles without cross-instrument same-master pairs | Roles lacking two chemicals with at least two masters each |
|---|---|---:|---:|---:|---:|
| Within-station | CWA | 60 | 11–13 | 0 | 0 |
| Within-station | Pills | 60 | 9–11 | 0 | 0 |
| Within-station | Surfaces | 60 | 12–13 | 0 | 0 |
| Unseen-instrument source selection | CWA | 126 | 10–14 | 0 | 0 |
| Unseen-instrument source selection | Pills | 300 | 9–11 | 0 | 0 |
| Unseen-instrument source selection | Surfaces | 255 | 4–13 | 87 | 36 |

The **87/255 surface inner fitting roles (34.1%)** without cross-instrument master pairs contain only one instrument. They also have no same-chemical/different-master cross-instrument positives. These are pseudo-domain fitting roles: excluding the pseudo-validation instrument and its physical masters can leave only one instrument in fitting. This is a consequence of the sparse crossed design plus the intended leakage exclusions, not a failed measurement or a demonstrated model failure.

All 320 outer fitting roles retain cross-instrument same-master pairs. It would therefore be misleading to check only the outer/full training data and declare the inner tuning procedure adequately supported everywhere.

All 861 inner fitting roles retain the three station-specific chemical classes. Nevertheless, **36/255 surface inner roles (14.1%)** cannot supply two distinct masters in each of two chemical classes. **177/255 (69.4%)** have at least one chemical represented by fewer than two fitting masters. These conditions are different from lacking a chemical class or lacking two spectra. The sampler must distinguish them.

There are also **87/255 surface inner roles** containing at least one spectrum with no other same-chemical spectrum in that fitting role. This is an observation-level positive-pair limitation, distinct from the independent-master criterion above. Zero-positive anchors need an explicit loss rule rather than an assumed positive partner.

## What these counts mean for the models

- Supervised contrastive learning can still use same-chemical pairs within an instrument, where those pairs exist. Availability of pairs alone does not establish chemical generalization or guarantee feasible minibatches.
- Cross-instrument paired consistency has no eligible pairs in the 87 single-instrument roles. Cross-instrument alignment and instrument adversarial discrimination are likewise unavailable there. It would be misleading to report a zero auxiliary loss as evidence that nuisance was removed.
- More recorded repeats can produce many pairs without adding independent physical samples. Master-aware sampling and separate observation/master counts remain essential.
- No split should be relaxed to obtain nicer pair support. Explicit loss availability, fallback behavior, and advancement denominators must be specified before training.

These are design constraints, not evidence against trying the acquisition-aware CNN.

## Exact arithmetic exposes an unresolved compute decision

The public loss inventory contains 147 potential configurations across D1–D5. A full crossing with six optimizer settings and three seeds yields 2,646 fits per inner selection unit.

| Illustrative Cartesian scenario | Inner selection units | Inner fits |
|---|---:|---:|
| Within-station contexts only | 180 | 476,280 |
| Unseen-instrument source-selection contexts only | 681 | 1,801,926 |
| Both sets | 861 | 2,278,206 |

**This is not the execution plan or an authorized budget.** It ignores the sequential/conditional development ladder and excludes final refits, calibration, matched D0 controls, failures/retries, and any supplementary analyses. It is not a GPU-hour estimate. The older rough 300–700-fit estimate does not define a reproducible schedule for these inherited roles. A finite staged design and resource ledger must be approved before training.

## Provenance and boundaries

| Input | SHA-256 |
|---|---|
| Primary manifest | `db1f298a76aeb9962db004776a9f41d6c9afe5b76c39aa9277a24848108d5f90` |
| P04 context registry | `12be22701e6c9847301bff7978bd6e4a4cd2f4abade84c4a025f1ed2c24810fb` |
| P04 role registry | `224891579d5df84c42a1c3c827590b6515ec83a13bfe4f07d48f249432f38d91` |

The program uses caller-pinned file hashes. The supervisor's prerequisite check supplies those pins; a caller who fabricates both data and hashes has not independently authenticated the original split plan. Audit success does not inherit P04's training authorization. Pair-identity proposals and role-support summaries are not executable P05 sampler/fit registries.

The [design handoff](P05_DESIGN_HANDOFF.md) records the next decisions. The [worker assignment](delegation/P05_TASK_003_SOURCE_SUPPORT.md) defines the implementation boundary. Scientific training remains blocked until the remaining design, source-only advancement, and finite-budget gates are resolved.

The [machine-readable aggregate summary](registries/p05_source_support_summary.json) contains denominators, grouped ranges, input hashes, the full local report digest, and the audit-module digest. It excludes per-role records and source identifiers. The [supervisor review](delegation/P05_TASK_003_REVIEW.md) records independent checks and limitations.
