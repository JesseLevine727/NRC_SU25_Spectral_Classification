# P08-U1 universal preprocessing execution boundary

Owner approval, 2026-10-08: **“Approve these limits and proceed after review.”** This supersedes the earlier proposal-only status for U1 resources. It does not certify an implementation or indicate that training has started. Storage has been freed at the owner's request; admission must still use a fresh measurement.

## Scientific scope

Compare authenticated historical minimal preprocessing with the frozen Savitzky–Golay and arPLS pipelines on the 400–1800 cm⁻¹ grid, retaining their complete action order and final [0,1] scaling. The panel is RBF-SVM, Random Forest, Extra Trees, matched ordinary CNN (D0-M), and the context-local CNN recipe already selected from source data. All 260 registered outer contexts and 13 held station–instrument domains remain in scope.

Classical hyperparameters are selected afresh within each policy using source data only. CNN recipe identities remain fixed across preprocessing; source-selected refit duration and calibration remain policy-specific. Neural source fits retain 30–200 epochs, patience 20, and the inherited best-checkpoint rule. Data membership, physical-master separation, held-instrument exclusion, candidates, seeds, sampling and losses are unchanged.

The frozen universal graph has plan SHA-256 `179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03`. Its archived graph and input/specification identities are recorded in the [slot audit](../results/p08_readiness/universal_slot_ledger_audit.json). Historical MIN endpoints are references under the [reuse protocol](P08_REUSE_PROTOCOL.md), not new fits.

## Original approved ceilings and carry-over

| Item | Approved boundary |
|---|---|
| Unique SG/arPLS model-fit slots | 195,202, including 78 accepted U0 slots |
| Unstarted slots after exact U0 reuse | 195,124 |
| Historical recovery overhead | Five retained attempts, separate from unique slots |
| Cumulative fit-attempt ceiling | 195,207 |
| Scalar calibrations | 3,354 |
| Cumulative active execution | 48 hours, including accepted U0 charge |
| Private artifacts | 80 GiB, including U0 evidence and new monitoring/reporting outputs |
| Process-tree RAM | 24 GiB |
| Allocated GPU memory | 8 GiB; reserved/device-used memory logged separately |
| Parallel fitting | At most four single-thread CPU workers and one GPU worker |
| Disk admission | 30 GiB reserve beyond the remaining artifact allowance |
| Automatic retries | Zero |

The accepted U0 carry-over is **1,357.694676884 seconds** and **64,846,625 logical artifact bytes** across original and recovery evidence. The five overhead attempts comprise four duplicated successful classical fits and one failed neural attempt. No counter resets at controller restart. The outer GPU allowance does not change the inherited per-neural-fit **120-second/4-GiB** guards.

Reuse requires fresh verification of exact job, input, role, specification and retained artifact identities. A missing or mismatched pilot artifact is not permission to replace its fit. Failed or interrupted new attempts remain consumed and stop further admissions for review. Finite collapsed models remain results, not retry requests.

## Approved R1 recovery amendment

The first U1 run stopped because the fresh GPU worker had not explicitly initialized CUDA before the unchanged training kernel reset its memory statistics. This was a launcher setup error, before the first neural epoch, not a model-performance result. The original run remains closed and preserved.

The owner subsequently approved the exact five-fit recovery. This changes execution accounting only:

| Item | R1 boundary |
|---|---|
| Verified reusable fit/prediction pairs | 84: 78 U0 pairs plus six completed U1 pairs |
| Exact replay fits | Five: one setup failure, three interrupted fits, one completed fit lacking its independent prediction verification |
| Remaining fit executions | 195,118: five replays plus 195,113 previously unstarted slots |
| Unique fit slots / scalar calibrations | Unchanged: 195,202 / 3,354 |
| Historical overhead / total attempt ceiling | Ten / 195,212 |
| Cumulative carry into R1 | 1,516 active seconds and 1,427,760,770 artifact bytes |

All hardware, disk-reserve, wall-time, scientific and inner-fit limits above remain unchanged. Carry-over preserves the previous artifact high-water mark even though database closure reduced its current file size. No further automatic retry is authorized. R1 must authenticate the closed parent inventory and exact reuse/replay identities, initialize CUDA before any scientific admission, and pass fresh-process GPU and end-to-end synthetic regression tests. The immutable R1 accounting profile, accepted runtime, launcher and owner approval are bound into the private launch permit. This amendment does not introduce a new preprocessing or model experiment.

## Approved R2 recovery amendment

R1 completed 8,466 new source fits and their paired predictions, in addition to the 84 imported pairs, and one source-only epoch-selection operation. A concurrent atomic update of the live HTML monitor caused the disk scanner to observe an already-unlinked inode with zero links. The scanner incorrectly treated this as a hard link and stopped admissions. Four source fits were interrupted by the controlled shutdown; there were no failed model outcomes or resource-limit breaches. The closed R1 evidence is preserved.

The owner approved the exact four-fit recovery on 2026-10-09 and requested completion without restarting the benchmark. R2 imports all **8,550 verified fit/prediction pairs and the one completed epoch-selection result**, replays only the four interrupted fits, and executes **186,648 previously unstarted fits**. Thus 186,652 fits remain; the 195,202 unique slots and 3,354 calibrations are unchanged. Historical overhead becomes 14 and the cumulative fit-attempt ceiling becomes **195,216**. No further automatic retries are authorized.

The corrected mutable-output scanner counts a zero-link regular inode using its observed size while continuing to reject actual hard links and symlinks. Immutable artifact authentication still requires exactly one link. The new recovery ledger records the epoch selector separately from fits, predictions and new attempts, and requires its imported dependencies. R2 authenticates every retained R1 file, exact registered job and complete receipt before reuse. The fresh destination and bound permit carry all prior time and artifact usage, including audit overhead; all scientific and resource limits remain unchanged.

## Implementation and execution gates

1. DeepSeek V4.1 Flash implements bounded integrations through the existing supervised workflow. The supervisor independently reviews code, role isolation, numerical parity, persistence, resource stops and tests. Existing U0 limits and frozen scientific kernels are not widened.
2. A reviewed local monitor persists completed epochs and displays job, preprocessing, recipe, seed, losses, source-validation metrics, best epoch and stopping state. Final refits show training only. Classical work displays counts/status/timing, not invented epochs. Monitoring cannot trigger retuning.
3. The supervisor binds approval to the reviewed executable/input identities and private destination, verifies pilot reuse, and checks fresh resources before launching. The full workflow covers source fitting, selection, calibration, refits and held evaluation; implementation tests alone do not pass this gate.
4. Independent completeness, scoring, uncertainty, disclosure and visual reviews precede publication. Use the existing [statistical protocol](P08_STATISTICAL_PROTOCOL.md), [reporting accounting](P08_REPORTING_ACCOUNTING.md) and [figure protocol](P08_FIGURE_PROTOCOL.md). U1 delivers P08-F01–F04/F07 and training diagnostics, with native TikZ and offline HTML from identical semantic data. Publish reviewed work to `main` and verify CI.

Family-specific/QC-adaptive preprocessing, perturbations, range/normalization/population sensitivities and new models are excluded. They do not extend U1's completion boundary. At most two worker correction rounds per bounded slice precede supervisor reassessment; the token ceiling is not a spending target. Required completion is the reviewed universal benchmark and release, not another planning-only milestone.

Current acceptance evidence is maintained in the [U1 review log](delegation/P08_U1_REVIEW.md).

## Owner-approved standing recovery and eight-CPU amendment, 2026-10-09

The owner authorized bounded infrastructure recovery once, without repeated approval requests: at most 32 model-fit/scalar replay attempts, 40 replay attempts across all stages, two per operation and eight recovery generations. Every completed operation must be authenticated and retained. Only interrupted or failed operations covered by that authority may be replayed; completed fits must not be repeated. Prior attempts, time and artifact high-water usage carry forward. This supersedes the earlier prohibition on further automatic recovery only within these explicit limits; it does not authorize scientific retuning.

After requesting higher throughput and permitting memory above 30 GiB if safe, the owner explicitly approved **eight single-thread CPU workers**. The reviewed recommendation retains **one GPU worker** and raises the whole-process-tree RAM ceiling from 24 to **36 GiB**. The 48-hour cumulative active-time ceiling, 80-GiB artifact ceiling, 8-GiB allocated-GPU ceiling, disk reserve and inner neural-fit guards remain unchanged. The unique 195,202 model-fit slots, 3,354 scalar calibrations and all scientific choices are unchanged.

The change is operational, not a new experiment. Candidate code is tested separately while the four-CPU run continues. A controlled handover must preserve paired fit/prediction boundaries, authenticate the closed parent, carry all recovery history and bind the exact two-file resource-cap change into a new permit. No live runtime or permit is edited in place. CPU throughput must be measured after launch; a doubling is not assumed. If the amended memory guard is exceeded, admissions stop and retained evidence is preserved.
