# P05 bounded-core execution handoff

The [core contract](contracts/p05_core_contract.json) and [protocol](P05_CORE_PROTOCOL.md) fix the science. This handoff describes the separate private runner. The older `run_p05.py readiness/check` commands inventory the historical broad grid; they do not authorize this core experiment.

**Execution complete:** the original run remains failed after one numerical fit and a checkpoint-write error; its separately approved recovery is complete and accepted. See [actual results](../results/p05_smoke/P05_SMOKE_RESULTS.md) and [smoke status](P05_SMOKE_STATUS.md). Do not rerun either command or remove either lease. The [recovery permit](contracts/p05_checkpoint_recovery.json) is exhausted: 34 new executions, 35 overall including the original failed-save attempt. Neither the original contract nor its numerical identities was rewritten.

## Authority and frozen identities

- Canonical contract SHA-256: `60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae`.
- Metadata plan SHA-256 / ID: `a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37`.
- Permitted execution: 32 primary fits plus two exact replays, each eight epochs and four draws per epoch; maximum 34 executions and 1,088 optimizer updates. No retries, held metrics, checkpoint selection or calibration.
- The later 17,820 neural-fit ceiling is not execution permission. D4/D5 and P06 evaluation are not reachable from this CLI.

## Before execution

Use the maintained package as the working directory and its validated environment. Set `NATO_SERS_ARTIFACT_ROOT` to the existing private artifact root outside the entire Git repository. The runner authenticates the successful pinned P01/P04PLAN states, reports and input files, rejects symlinked paths, reconstructs the metadata plan, and verifies the representation and source-fitting QC alignment.

The metadata-only command creates or verifies the immutable plan; it hashes the representation file but does not load intensities:

```bash
python scripts/run_p05_core.py --project-root . \
  --artifact-root "$NATO_SERS_ARTIFACT_ROOT" \
  --contract plan/contracts/p05_core_contract.json \
  --contract-sha256 60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae plan
```

Only after the supervisor accepts the implementation and tests may the same command prefix use `smoke --plan-id a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37`. This is not an instruction to launch a second run. One contract-level lease prevents re-execution even when code changes. Never delete or bypass that lease to retry a failure.

Freeze package contents and Git state throughout the governed smoke. CUDA is selected only when at least 5 GiB is free; CPU is selected before fitting otherwise. No mid-fit device fallback is allowed. The kernel enforces 120 seconds per fit and 4 GiB peak allocated CUDA memory; the runner enforces 900 seconds overall. An external process timeout provides a final watchdog without authorizing a retry.

## Separately approved persistence recovery

The owner approved a single infrastructure recovery on 2026-09-25. Its permit SHA-256 is `01e0835d8a6ece2ee98cee654e9f707e894643788dc9daabc438527c1c19c058`. The original plan, contract and numerical source hashes remain pinned. The replacement execution has the same numerical fit specification and seed, but a distinct execution identifier and a recovery-of link.

After implementation review, use this no-fit preflight:

```bash
python scripts/recover_p05_smoke.py --project-root . \
  --artifact-root "$NATO_SERS_ARTIFACT_ROOT" \
  --contract plan/contracts/p05_core_contract.json \
  --contract-sha256 60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae \
  --plan-id a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37 preflight
```

The same command ending in `recover` consumes the one approved recovery after the supervisor accepts all tests. It is not a resumption flag for the original runner. Artifacts live under `p05core/recovery/<permit-sha>/{lease,run}`; either an existing lease or an existing run prohibits another attempt. Never delete them to retry. The first replay must match the retained original state/stream digests and complete history before the 33 unstarted executions proceed. Every new checkpoint is reloaded and its state hash checked. The accepted set still contains 32 primary fits plus two planned replays; total optimization accounting includes the original failed-save execution: 35 executions and 1,120 updates. No full development is authorized.

## Evidence and interpretation

Private artifacts live under `p05core/plans/<plan-id>`, `p05core/leases/<contract-sha>` and `p05core/runs/<contract-sha>`. Preserve the plan, reservation ledger, partial epoch histories, numerical result records, terminal checkpoints, before/after provenance, summary and output-hash manifest. A crash or failure consumes its recorded attempts and remains visible.

Acceptance checks numerical completion, actual weight updates, supported auxiliary branches, common random streams, exact sparse-control equivalence, both replay comparisons and unchanged protected state. Training accuracy is not an acceptance or promotion threshold. The smoke cannot establish chemical/nuisance disentanglement, unseen-instrument generalization, a preferred recipe or an optimal epoch budget.

After execution has ended, export only the reviewed, whitelisted primary-fit training curves to native TikZ, PDF/PNG and offline HTML. Replays are excluded from plotted evidence. Scientific-data rows, identities, prediction rows and checkpoints remain outside this export. The [supervisor review](delegation/P05_CORE_REVIEW.md) records the actual gates and outcome.
