# Comprehensive recovery: implementation review

## Authority and interruption evidence

The owner approved one bounded replay on 2026-09-27 and raised the overall allocated CUDA ceiling to 8 GiB. The [recovery protocol](../P05_COMPREHENSIVE_RECOVERY.md) and separate [permit](../contracts/p05_comprehensive_recovery.json) preserve the original scientific settings, contracts and seven numerical-file pins. The recovery permit's canonical SHA-256 is `dfd19af6546de2e891f35da27f496c3e4ebc8e4fd000c4dc35fac8a9d75e913e`.

The original source process was killed by Linux during global host-memory exhaustion. Post-crash read-only checks verified the sealed-unit manifests, bounded evidence for the eight completions in the unsealed unit, original pilot metadata and the frozen numerical pins. Neither a completed source receipt nor final held-out results exist. The original files and consumed leases remain unchanged.

## First implementation gate: pure planner and resource authority

OpenCode Go `opencode-go/deepseek-v4.1-flash` authored the recovery design, pure planner, independent synthetic planner tests, and authority/resource-guard draft. Each invocation had tools denied and only explicitly supplied public source snapshots. No private evidence, checkpoints, spectra or identities were supplied. The supervisor applied the returned patches, inspected the interfaces and executed the tests independently.

The pure planner validates the exact registered interruption prefix, rather than accepting an arbitrary list of completed fits. It checks canonical unit/recipe/seed order, journal transitions and counters, complete pilot-unit coverage, selector identities, original per-slot leases and the interrupted epoch/update lower bound. Its output explicitly has `execution_authorized = false` and starts no fits. Actual file and model authentication remains a separate required boundary.

The first draft passed 61 synthetic tests. Supervisor-added tests exposed nine failures involving floating-point pilot seeds, incomplete pilot-record equality, non-string JSON keys, malformed Unicode, recursive input, unrepresentable metrics and inconsistent optional slot-context fields. After correction, all 78 planner tests passed. The tests construct the full 14,940-slot shape using artificial identifiers and outcomes, not private chemical data.

A subsequent read-only application to the authenticated real ledger, saved event and selector journals, original lease set and interrupted history reproduced the recorded counts: 8,720 original completions, one interrupted attempt, 6,183 unstarted slots and 6,184 recovery fits. The private plan ID is `325b676ab1f58ab6e46476628bc9d5fc920065cb9811ae3eb736ac0b5aff8782`. The check took 13.719735 seconds and made no writes or fits. It validates the recovery schedule, not every saved model or source-validation prediction.

The authority loader accepts only the separately pinned recovery permit. It rejects malformed JSON, duplicate keys, changed payloads, symbolic links and nonregular or oversized files. Operational guards require at least 16 GiB host `MemAvailable` before launch and 8 GiB before fits and after epochs, at least 9 GiB free CUDA before launch, and at most 8 GiB allocated CUDA. The frozen numerical source kernel still enforces its stricter 4-GiB per-fit cap. The guards detect low headroom at check time; they cannot guarantee protection against another process abruptly exhausting the machine.

The authority draft passed 62 synthetic tests. Supervisor-added cases exposed malformed-path, recursive-input, meminfo parsing and inconsistent CUDA-metric handling. After correction, all 82 authority tests passed. These tests use temporary fake meminfo files and an injected fake CUDA interface; they launch no numerical experiment. No new recovery fit has started.

The combined first gate contains 160 passing focused tests. Independent regression of the immutable review snapshot passed: **2,026 tests passed, four CUDA-dependent tests skipped, 832 warnings, 404.82 seconds**. CUDA was intentionally hidden for this synthetic regression; it is not a scientific GPU acceptance run. The public-file validator passed for 591 files, and the CI lint scope (`src`, `scripts`, `tests`) passed. All seven numerical-file hashes and the separate recovery-permit pin remained unchanged.

The initial regression attempt exposed a missing validation-only Parquet dependency: its previous installation had been under a temporary directory removed by the reboot. That attempt was stopped and is not counted as successful. The declared `pyarrow` dependency was restored in a separate persistent validation directory, without changing the frozen training environment, and the full suite was rerun successfully. The accepted first gate adds planning and resource checks only; it does not authorize scientific execution before the remaining gates below.

## Remaining recovery gates

1. Authenticate the exact original-evidence anchor, complete inventories, lease files and saved model evidence; seal a private recovery input inventory and plan without fitting.
2. Implement an additive recovered source stage below the original governed run root. Keep the original `develop` files unchanged; copy all needed completed evidence independently, with hash and storage accounting. Do not drop checkpoints or use links to evade accounting. A separately named recovery receipt and deterministic source resolver must distinguish the recovered source view from the original incomplete stage.
3. Implement the one-shot replay lease, original unstarted-slot leases, host/CUDA/time/storage guards, immutable launch provenance and explicit successful-versus-interrupted attempt/update accounting. Reconstruct all 12 results before closing the partial unit's shared-prefix and sparse-equivalence checks.
4. Update downstream selection/refit/evaluation/reporting/publication checks to authenticate the recovery receipt and carry its charged prior time and interrupted-update upper bound. Preserve legacy clean-run behavior. Do not fake zero failures, exact interrupted updates or a new time allowance.
5. Independently test the integrated runner and downstream boundaries, publish reviewed code to `main`, launch from an immutable checkout with persistent non-temporary logs, and complete the full remaining benchmark under the approved cumulative limits.

This review accepts no scientific restart merely because the planner and resource guards pass. Full recovery integration, scientific acceptance, comparisons and figure publication remain unfinished.
