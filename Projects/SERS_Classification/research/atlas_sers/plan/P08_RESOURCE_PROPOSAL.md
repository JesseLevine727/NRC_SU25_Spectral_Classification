# P08 staged resource proposal

**Date:** 2026-09-30. **Status:** proposed limits, not an execution permit. Current authorized scientific operations remain zero. Previous P03–P05 permissions do not transfer to P08.

Universal preprocessing is the first executable research stage to prepare. It directly tests whether the frozen SG or arPLS pipeline changes unseen-instrument identification relative to MIN. QC-adaptive selection is a separate, substantially larger stage; its feasibility does not need to delay this universal comparison once the remaining universal readiness checks pass.

## 1. Evidence used for the estimate

The authenticated [MIN classical timing records](../results/p08_readiness/historical_timing_basis.json) give 91,334.361 seconds of classical fitting per universal action after exact master-CV calibration reuse. Doubling this sum gives **50.74 sequential CPU fitting hours** for SG and arPLS. The relevant [neural records](../results/p08_readiness/minimal_neural_reuse_audit.json) give **4.83 summed GPU fitting hours** for both actions. These sums exclude scheduling, serialization, inference, calibration, reporting and new-policy changes to early stopping. Dividing CPU time by a worker count is not a guaranteed wall-time estimate.

The [universal slot ledger](../results/p08_readiness/universal_slot_ledger_audit.json) contains 195,202 prospective new model fits and 3,354 scalar calibrations across SG and arPLS. The ledger also enumerates source, calibration and held predictions, source-only choices and aliases. Historical MIN evidence remains a separately authenticated reference, not a third retraining arm.

The machine audit found approximately 187 GiB free on the private-artifact filesystem, 50 GiB of available host RAM and 15 GiB of free GPU memory. These are point-in-time observations, not reserved resources. Every execution permit must repeat the resource check before starting. Existing data, checkpoints and user files must not be deleted to satisfy a budget.

## 2. Proposed staged ceilings

| Stage | Model-fit ceiling | Scalar calibrations | Active wall-time ceiling | New private artifacts | Process-tree RAM | Allocated GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| U0: reviewed source-only universal smoke | 78 | 0 | 90 minutes | 8 GiB | 16 GiB | 8 GiB |
| U1: complete SG/arPLS universal comparison, including accepted U0 work | 195,202 | 3,354 | 48 hours | 80 GiB | 24 GiB | 8 GiB |
| Q1: literal complete QC-adaptive sweep, before proven reuse | 1,630,980 | 53,880 | 240 hours | 400 GiB | 24 GiB | 8 GiB |

These are finite **proposals**, not measured runtimes, promises of completion or cumulative permission. U0 requires its own approval after implementation review. U1 requires a new approval after the U0 gate. Q1 requires a separate approval and storage-feasibility gate. A successful earlier stage does not launch the next stage automatically. Family fallback produces aliases to accepted MIN results and adds no scientific fits under the approved complete-pipeline interpretation.

U0 permits at most two single-threaded CPU fitting workers and one GPU fitting worker. U1 and Q1 permit at most four single-threaded CPU fitting workers and one GPU fitting worker. Model-internal parallelism, BLAS threads and PyTorch CPU threads are fixed at one per worker. The GPU limit concerns allocated CUDA memory; reserved and device-reported memory must also be logged, rather than described as if they were the same measurement.

Wall time accumulates over every active execution interval, including failed attempts, orchestration and artifact writes. Pauses between execution sessions do not erase consumed time. A failure consumes its attempted slot; there are zero automatic retries. Crossing a time, memory or storage threshold stops new work and preserves the completed evidence. Restarting a controller must reconstruct cumulative counters before any job admission.

The proposed storage admission rule reserves 30 GiB of free filesystem space beyond the stage's still-unconsumed artifact allowance. The U1 budget therefore requires at least 110 GiB free when starting from zero new artifacts. Q1 requires at least 430 GiB free before an uncompressed full-ceiling launch. A later exact reuse ledger may reduce a proposed allowance, but a count-only or approximate match cannot justify that reduction.

## 3. Exact U0 smoke scope

The [metadata-only smoke audit](../results/p08_readiness/universal_smoke_proposal_audit.json) binds **78 existing source-fit slots and their 78 source-validation prediction slots** from the universal graph. No new experiment or split was invented for the smoke.

Selection is fixed before any new outcome:

1. At each of the three stations, take the lexicographically first held context and its first registered source selection unit. This is a deterministic implementation-coverage rule, not representative sampling.
2. For that unit, include the first candidate in the frozen declared order for each of RBF-SVM, Random Forest and Extra Trees, plus D0-M.
3. Add the first context and first source unit carrying each frozen non-D0 recipe, D1, D2 and D3. Do not change the source-selected recipe map.
4. Cross these choices with SG and arPLS. Retain the deterministic SVM seed and all three registered stochastic seeds: 20260805, 20260817 and 20260829.

| Model | U0 fits across both actions |
|---|---:|
| RBF-SVM | 6 |
| Random Forest | 18 |
| Extra Trees | 18 |
| D0-M | 18 |
| D1 | 6 |
| D2 | 6 |
| D3 | 6 |
| Total | **78** |

Neural fits retain 30–200 epochs, patience 20 and the inherited checkpoint rule. The best epoch may precede the minimum stopping epoch, as in the accepted original procedure. Source validation is available for those inherited stopping rules; outer held rows are not. U0 contains no final refits, scalar temperature fitting, hyperparameter winner selection, QC cutpoints, held predictions, resampling or perturbations.

Acceptance requires exact input and role hashes; complete candidate, recipe and seed accounting; no held access; finite outputs or a documented valid collapse; loadable checkpoints; terminal-state and source-prediction integrity; resource accounting; and successful synthetic restart/stop tests for the controller. A finite low-scoring classifier is not itself an implementation failure. Failed or incomplete slots are not silently replaced. Successful U0 artifacts can count toward U1 only after full input/specification matching; they are not 78 additional fits beyond U1.

## 4. Why the adaptive stage has a separate gate

The [nested QC protocol](P08_QC_NESTING.md) uses 124 gates, 108 policy-validation units and three deeper estimator folds per unit. Its literal ceiling contains 161,316 neural fits. Applying the observed D0-M source/refit mean fitting times to the gate-ranking block alone gives approximately **123.31 neural fitting hours**; this is an extrapolation, not a QC measurement. Gate routing, smaller fitting sets, stopping behavior and I/O can change the realized cost substantially.

The historical retention pattern stores both best and terminal checkpoints for neural source fits and a terminal checkpoint for each refit. Relevant old checkpoint files average approximately 0.85 MB each. Applying that pattern to 120,936 QC source fits and 40,380 QC refits gives approximately **223.33 GiB of checkpoints**, before histories, routing records, predictions or reports. This exceeds the observed free disk space even before U1 artifacts are added.

The 400-GiB proposal is therefore not currently storage-feasible. Before Q1, either provide sufficient private storage or approve a separately documented retention/resource amendment. Do not discard historical or newly required evidence automatically. Source-only threshold/routing preflight could establish exact duplicate routed inputs and reduce actual work, but that preflight itself needs a separate scientific permit; no cutpoints have been calculated during readiness.

## 5. Retention and later branches

For new universal runs, preserve original inputs, role/specification hashes, source prediction records, candidate statuses, selections, calibration states, terminal ledgers and final predictions. Retain neural source best/terminal checkpoints and final refit checkpoints. Retain new classical final estimators for later perturbation testing; classical grid-fit estimators need not be persisted when their original procedure requires only source predictions and fit records. Packaging records into immutable shards is permitted only with per-record identities and hashes retained.

The original P03 MIN run did not save fitted classical estimator objects. A future perturbation branch therefore needs separately budgeted exact-selected-estimator reconstruction, not a claim that stored predictions can process new inputs. Across all three fixed classical families and 260 contexts, that reconstruction would contain at most 1,820 fits, before any explicit panel restriction. Reconstruction must preserve old evidence and verify prediction parity before it is used.

Range, normalization, derivative-control, population-tier and test-time perturbation branches remain required planning items under master-plan Sections 16.1 and 16.5–16.7. They are not included in U0, U1 or Q1. Their exact role/model mappings, numerical perturbation definitions, reconstruction requirement and finite job budgets must be recorded separately. This proposal does not silently remove them or grant them U1's resources.

## 6. Current decision boundary

The first possible execution request is **U0 only**, after the no-fit package and runtime review gates close. It is not being launched by publishing this proposal. Full scientific readiness still requires the complete operation-to-artifact reuse ledger, adaptive job catalog, later-branch budgets and reviewed admission/restart implementation. No planning approval is treated as a training permit.

## 7. Resource snapshots and cumulative attempt accounting

The pure resource checker compares caller-supplied usage and measurements with the proposed U0, U1 and Q1 ceilings. It does not query hardware or launch work. Fit counters include failed and unfinished attempts. Exhausted fit capacity prevents another fit, but does not itself forbid a pending prediction from an already completed fit. Exhausted active time or artifact headroom stops new work. Memory and worker counts may equal their ceilings but may not exceed them; model, BLAS and PyTorch thread settings must each equal one.

The storage check requires the remaining artifact allowance plus the 30-GiB reserve. U0 therefore requires 38 GiB free before its first artifact. Retained-artifact usage is a cumulative high-water mark, not a sum that charges every overwrite as a new complete file. Usage cannot be reset by pausing, restarting or deleting evidence. Allocated CUDA memory is checked against its own limit; reserved and device-reported memory remain separate recorded measurements.

The pure attempt-journal kernel reconstructs usage from hash-linked session events. An attempt is charged when it starts, before its outcome is known. A prediction requires its fitting dependency to have succeeded. A failed or interrupted job cannot be started again under the same no-retry authority. A session may close only when all its attempts are terminal; a later clean session starts its local clock at zero while retaining prior durations, attempts and artifact usage.

An open journal session is not proof of a crash or proof of a live owner. The kernel reports that a clean restart is unavailable, even if no job is currently marked in flight. Its elapsed-time total covers recorded intervals only; it cannot infer time after the last durable observation. The future runtime must authenticate the exact smoke-job mapping, journal records and head, preserve exclusive ownership, and obtain reviewed recovery evidence for an incomplete session. It may not assume the unobserved interval was a free pause.

These numerical and state-machine checks use synthetic inputs. Their acceptance is recorded in the [review log](delegation/P08_REVIEW.md). They do not establish durable filesystem behavior, authorize a retry, prove a receipt file exists, or grant a scientific execution permit. The generic journal's bounded synthetic manifests do not replace the exact 78-fit and 78-prediction U0 manifest. Full admission/restart acceptance remains open until those integration and durability checks pass.
