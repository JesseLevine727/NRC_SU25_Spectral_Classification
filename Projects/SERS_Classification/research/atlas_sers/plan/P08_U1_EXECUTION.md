# P08-U1 universal preprocessing execution boundary

Owner approval, 2026-10-08: **“Approve these limits and proceed after review.”** This supersedes the earlier proposal-only status for U1 resources. It does not certify an implementation or indicate that training has started. Storage has been freed at the owner's request; admission must still use a fresh measurement.

## Scientific scope

Compare authenticated historical minimal preprocessing with the frozen Savitzky–Golay and arPLS pipelines on the 400–1800 cm⁻¹ grid, retaining their complete action order and final [0,1] scaling. The panel is RBF-SVM, Random Forest, Extra Trees, matched ordinary CNN (D0-M), and the context-local CNN recipe already selected from source data. All 260 registered outer contexts and 13 held station–instrument domains remain in scope.

Classical hyperparameters are selected afresh within each policy using source data only. CNN recipe identities remain fixed across preprocessing; source-selected refit duration and calibration remain policy-specific. Neural source fits retain 30–200 epochs, patience 20, and the inherited best-checkpoint rule. Data membership, physical-master separation, held-instrument exclusion, candidates, seeds, sampling and losses are unchanged.

The frozen universal graph has plan SHA-256 `179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03`. Its archived graph and input/specification identities are recorded in the [slot audit](../results/p08_readiness/universal_slot_ledger_audit.json). Historical MIN endpoints are references under the [reuse protocol](P08_REUSE_PROTOCOL.md), not new fits.

## Approved ceilings and carry-over

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

## Implementation and execution gates

1. DeepSeek V4.1 Flash implements bounded integrations through the existing supervised workflow. The supervisor independently reviews code, role isolation, numerical parity, persistence, resource stops and tests. Existing U0 limits and frozen scientific kernels are not widened.
2. A reviewed local monitor persists completed epochs and displays job, preprocessing, recipe, seed, losses, source-validation metrics, best epoch and stopping state. Final refits show training only. Classical work displays counts/status/timing, not invented epochs. Monitoring cannot trigger retuning.
3. The supervisor binds approval to the reviewed executable/input identities and private destination, verifies pilot reuse, and checks fresh resources before launching. The full workflow covers source fitting, selection, calibration, refits and held evaluation; implementation tests alone do not pass this gate.
4. Independent completeness, scoring, uncertainty, disclosure and visual reviews precede publication. Use the existing [statistical protocol](P08_STATISTICAL_PROTOCOL.md), [reporting accounting](P08_REPORTING_ACCOUNTING.md) and [figure protocol](P08_FIGURE_PROTOCOL.md). U1 delivers P08-F01–F04/F07 and training diagnostics, with native TikZ and offline HTML from identical semantic data. Publish reviewed work to `main` and verify CI.

Family-specific/QC-adaptive preprocessing, perturbations, range/normalization/population sensitivities and new models are excluded. They do not extend U1's completion boundary. At most two worker correction rounds per bounded slice precede supervisor reassessment; the token ceiling is not a spending target. Required completion is the reviewed universal benchmark and release, not another planning-only milestone.

Current acceptance evidence is maintained in the [U1 review log](delegation/P08_U1_REVIEW.md).
