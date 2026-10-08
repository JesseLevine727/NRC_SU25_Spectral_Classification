# Preprocessing pilot: first-attempt review

**2026-10-08. Decision: NO-GO for the full benchmark.**

Four classical fits and their source-validation checks completed. The first CNN attempt refused training because the supervisor's launch command omitted `CUBLAS_WORKSPACE_CONFIG=:4096:8`, an existing requirement of the inherited CUDA kernel. This is a launch-configuration failure, not evidence against preprocessing, the CNN architecture or the dataset. No automatic retry was made.

| Operation | Planned | Succeeded | Failed | Unstarted |
|---|---:|---:|---:|---:|
| Source fits | 78 | 4 | 1 | 73 |
| Source-validation operations | 78 | 4 | 0 | 74 |

The failed ordinary-CNN (D0-M) attempt recorded **zero epochs and zero optimizer steps**, but consumes one attempted fit slot. Its saved status is `resource_failure`, with reason `cublas_workspace_config_invalid`; this does not indicate exhausted GPU memory. The outer launcher reported the less informative `internal_error`.

The terminal record measures **5.526092778 seconds before terminal-file persistence**, not the complete cleanup interval. Run-wide RAM/GPU peak acceptance is unavailable after early termination. Zero neural allocation before model construction does not establish zero launcher GPU use.

## Review evidence

Independent review replayed **86 journal events**, checked the canonical hashes and artifact bytes of **nine terminal receipts**, and verified **four saved source-prediction parity records**. The private evidence remained unchanged. The [aggregate review](../results/p08_u0/pilot_review.json) includes retained size and an inventory digest without sample identifiers or private paths. An initial supervisor diagnostic passed the full journal to a receipt helper expecting a different event boundary and was rejected; the subsequent independent hash-and-artifact checks passed without changing evidence.

Preflight authenticated **174 project sources** and loaded **41 project modules**. All **78 pairs** matched the frozen source roles, arrays and settings: **42 classical** and **36 neural**. Deployment checks passed **80 tests in 11.54 seconds**. A syntax-tree comparison confirmed that only the approved private-permit digest changed executable launcher behavior; all scientific kernels remained unchanged. DeepSeek V4.1 Flash authored the deployment patch and refusal test; the supervisor reviewed them and supplied the launch command.

The tests used CPU fixtures and simulated GPU observations, so they did not catch the missing real CUDA environment setting. This is a preflight gap and a supervisor launch error. It is not a justification to change models, data or preprocessing.

## Interpretation and recovery boundary

Four completed pairs demonstrate partial execution and saved-output consistency, not a balanced preprocessing comparison. No held-test prediction, calibration, winner selection or full-benchmark evaluation occurred. Performance plots and preprocessing-superiority claims are not justified by this incomplete pilot.

Recovery requires separate approval. First require the existing `CUBLAS_WORKSPACE_CONFIG=:4096:8` setting in a non-training preflight before fits or CUDA initialization. Then specify an exact continuation or replay schedule accounting for the **five consumed fit attempts**. The current launcher refuses prior destinations; do not bypass it or silently use a fresh destination under the same permit. Preserve all completed evidence.

No broader planning or framework redesign is needed. Full U1 execution remains unapproved. This is a reviewed no-go outcome, not a successfully completed 78-fit pilot. The [execution approval](../plan/P08_U0_EXECUTION.md) retains the original limits. Lenarizer guided the separation of measured, tested and unavailable evidence. Publication and final CI remain separate checks.
