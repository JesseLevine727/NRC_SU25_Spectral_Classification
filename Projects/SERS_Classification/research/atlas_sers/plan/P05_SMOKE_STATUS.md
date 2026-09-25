# P05 smoke status — failed attempt preserved, recovery accepted

**2026-09-25. Recovery complete and numerically accepted; not a predictive benchmark.** See the [actual results and figures](../results/p05_smoke/P05_SMOKE_RESULTS.md). The original implementation milestone was pushed as `5b44c193`; its first scientific attempt stopped at checkpoint persistence. The corrected recovery ran at `bf8735cd` and passed. The original failed attempt remains recorded below.

## What ran in the original failed attempt

| Item | Retained evidence |
|---|---|
| Scheduled executions | 32 primary fits plus two exact replays |
| Executions started | 1: CWA source-fitting D0-M, seed 20260805 |
| Completed numerical work | 8 epochs, 32 optimizer updates, 32 finite-gradient batches |
| Fit duration / peak allocated CUDA memory | 2.599 seconds / 158,724,096 bytes |
| Saved training history | All eight epochs, including the epoch-callback journal |
| Saved checkpoint | None: writing failed after the fit returned |
| Remaining unstarted executions | 33 |
| Held-test or validation evaluation | None |

The numerical result contains initial/final model digests and support/gradient records. The failed lease, reservation, history, result, failure summary and artifact manifest remain private and unchanged. The original private manifest SHA-256 is `d452ec375d7445a0b728f895ceec47f1ae266ee3c2b80b7f731ef65ff2aa7dbe`. No failed attempt has been erased, replaced or counted as an accepted complete run.

## Cause and correction

The installed PyTorch writer rejected an extensionless hidden temporary pathname (`.ckpt-*`). The defect was reproduced twice through the real saver with a two-element synthetic tensor. Controlled filename probes showed that ordinary basenames and hidden names with a `.pt` extension work; using an open binary file handle also works. This was a persistence defect, not a nonfinite loss, an exhausted GPU, or evidence that the learning method failed.

The diagnose workflow added real serializer and complete execution-persistence regressions before the fix. Two tests reproduced the error; both pass after changing the saver to a binary file handle with flush/fsync before atomic replacement. A tiny save/load preflight now occurs before a scientific lease can be acquired. Persistence exceptions also retain an explicit failed ledger event and diagnostic digest. The tensors round-trip exactly in an independent check. These changes do not alter the model, data, sampler, losses or training seeds.

The pre-run suite's 499 passing tests did not cover the actual serializer seam. That gap is recorded rather than presenting the earlier green suite as sufficient end-to-end evidence. The checkpoint tests require torch locally; the existing hosted CI omits torch, so its optional-dependency skips must not be described as neural/persistence validation.

## Approved bounded recovery

On 2026-09-25 the owner explicitly approved one recorded recovery replay of the first numerical fit plus the 33 unstarted executions, increasing the overall ceiling from 34 to **35 executions / 1,120 updates**. The [separate recovery permit](contracts/p05_checkpoint_recovery.json), canonical SHA-256 `01e0835d8a6ece2ee98cee654e9f707e894643788dc9daabc438527c1c19c058`, preserves the original contract and metadata plan because their identities enter random streams. Recovery must preserve the original failed attempt and keep all numerical fit identities/settings fixed. The replay must reproduce the retained original terminal-state and full semantic history before continuation.

This is an approved infrastructure-recovery amendment, not a new scientific experiment or permission for automatic retries. [T012](delegation/P05_TASK_012_BOUNDED_RECOVERY.md) delegated its implementation and tests to DeepSeek. The new exclusive recovery lease now records all 34 completed new executions and cannot be reused. The first fit matched the original state/stream digests and full history before any of the other 33 started.

The accepted recovery produced 32 primary records plus two planned exact replays. All 34 checkpoints were independently reloaded and verified. Both planned replays, four sparse-control comparisons and protected-state checks passed. Recovery wall time was 21.907 seconds; peak allocated CUDA memory was 151.46 MiB. The retained original and accepted recovery total exactly **35 scientific executions / 1,120 updates**, exhausting this authority.

The [public figures](../results/p05_smoke/P05_SMOKE_RESULTS.md#what-the-figures-show) now use actual primary-fit diagnostics, not the earlier synthetic rendering tests. Full P05 development, G3 advancement and P06 testing remain incomplete and unauthorized. Passing this smoke establishes numerical implementation behavior, not predictive improvement.
