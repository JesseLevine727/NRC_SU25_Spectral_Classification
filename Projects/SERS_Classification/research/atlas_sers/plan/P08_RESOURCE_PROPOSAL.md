# P08 staged resource proposal

**Date:** 2026-09-30. **Status:** proposed limits, not an execution permit. Current authorized scientific operations remain zero. Previous P03–P05 permissions do not transfer to P08.

**Scope update, 2026-10-04:** the owner selected the N2 fixed-family normalization design under [P08-A06](P08_LATER_BRANCH_DECISIONS.md), with the other three later-branch decisions approved concurrently. N1 is retained below as the unselected historical alternative. Every resource ceiling in this document, including N2's, remains proposed and unapproved. The universal-first sequence and separate adaptive-resource gate are unchanged.

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

The [source-stage mapping](P08_SOURCE_STAGE_MAPPING.md) distinguishes these logical jobs from individual model calls. Both inherited fitting kernels already calculate source-validation outputs; the dependent prediction stage authenticates and verifies them without another optimization. Internal validation and verification passes remain inside the proposed wall-time, memory and artifact ceilings. Lost in-memory classical estimators cannot be replaced through unrecorded fits. The mapping does not change job identities, dependencies or execution authority.

Acceptance requires exact input and role hashes; complete candidate, recipe and seed accounting; no held access; finite outputs or a documented valid collapse; loadable checkpoints; terminal-state and source-prediction integrity; resource accounting; and successful synthetic restart/stop tests for the controller. A finite low-scoring classifier is not itself an implementation failure. Failed or incomplete slots are not silently replaced. Successful U0 artifacts can count toward U1 only after full input/specification matching; they are not 78 additional fits beyond U1.

## 4. Why the adaptive stage has a separate gate

The [nested QC protocol](P08_QC_NESTING.md) uses 124 gates, 108 policy-validation units and three deeper estimator folds per unit. Its literal ceiling contains 161,316 neural fits. Applying the observed D0-M source/refit mean fitting times to the gate-ranking block alone gives approximately **123.31 neural fitting hours**; this is an extrapolation, not a QC measurement. Gate routing, smaller fitting sets, stopping behavior and I/O can change the realized cost substantially.

The historical retention pattern stores both best and terminal checkpoints for neural source fits and a terminal checkpoint for each refit. Relevant old checkpoint files average approximately 0.85 MB each. Applying that pattern to 120,936 QC source fits and 40,380 QC refits gives approximately **223.33 GiB of checkpoints**, before histories, routing records, predictions or reports. This exceeds the observed free disk space even before U1 artifacts are added.

The 400-GiB proposal is therefore not currently storage-feasible. Before Q1, either provide sufficient private storage or approve a separately documented retention/resource amendment. Do not discard historical or newly required evidence automatically. Source-only threshold/routing preflight could establish exact duplicate routed inputs and reduce actual work, but that preflight itself needs a separate scientific permit; no cutpoints have been calculated during readiness.

## 5. Retention and later branches

For new universal runs, preserve original inputs, role/specification hashes, source prediction records, candidate statuses, selections, calibration states, terminal ledgers and final predictions. Retain neural source best/terminal checkpoints and final refit checkpoints. Retain new classical final estimators for later perturbation testing; classical grid-fit estimators need not be persisted when their original procedure requires only source predictions and fit records. Packaging records into immutable shards is permitted only with per-record identities and hashes retained.

The original P03 MIN run did not save fitted classical estimator objects. A future perturbation branch therefore needs separately budgeted exact-selected-estimator reconstruction, not a claim that stored predictions can process new inputs. Across all three fixed classical families and 260 contexts, that reconstruction would contain at most 1,820 fits, before any explicit panel restriction. Reconstruction must preserve old evidence and verify prediction parity before it is used.

Range, normalization, derivative-control, population-tier and test-time perturbation branches remain required planning items under master-plan Sections 16.1 and 16.5–16.7. They are not included in U0, U1 or Q1. Their exact role/model mappings, numerical perturbation definitions, reconstruction requirement and finite job budgets must be recorded separately. This proposal does not silently remove them or grant them U1's resources.

### Conditional N1 proposal: full-selector normalization controls

This historical proposal would apply to repeating the full source-only classical selector for SNV, vector and area normalization and the derivative control. The owner instead selected N2 under P08-A06; N1 is not part of the approved branch design. The [conditional slot audit](../results/p08_readiness/later_classical_selector_bounds.json) gives a ceiling of **525,288 model fits and 1,040 scalar calibrations** across those four controls, after authenticated within-representation calibration-role reuse.

| Proposed stage | Active wall time | New private artifacts | Process-tree RAM | Allocated GPU memory | CPU workers |
|---|---:|---:|---:|---:|---:|
| N1: four controls with the full classical selector | 96 hours | 80 GiB | 24 GiB | 0 | 4, single-threaded |

The [authenticated timing basis](../results/p08_readiness/later_classical_timing_basis.json) covers all 129,390 historical source-selection slots across nine classical families. It retains 127,182 completed candidates, 1,278 rank failures and 930 convergence failures. Every slot has a recorded duration; failed candidates are not silently removed from the timing sum. The sum is **123,953.952 seconds**. Multiplying it by four gives **137.73 sequential CPU-hours** for source fitting alone, under unchanged historical durations. This is not a P08 measurement or an upper runtime bound.

The 96-hour wall ceiling allows additional time beyond the idealized four-worker source-fitting duration for calibration, final fits, predictions, persistence and scheduling. It does not guarantee completion: new representations can change numerical cost and valid-candidate support. The artifact and RAM ceilings are explicit operational proposals, not measurements of future use. No GPU work or automatic retry is proposed. Existing failed-candidate rules and source-only selection remain unchanged.

N1 requires its own approval, an exact operation ledger and a fresh storage check. Starting from zero N1 artifacts requires at least 110 GiB free under the 30-GiB reserve rule; earlier stages' retained artifacts still occupy disk. This allowance is separate from U1 and Q1 and cannot be borrowed automatically. The incomplete historical selected-classical reference is preserved; N1 does not authorize filling its missing cells. Range, regenerated-population and perturbation branches still need their own accounting.

## 6. Current decision boundary

The first possible execution request is **U0 only**, after the no-fit package and its release checks close. It is not being launched by publishing this proposal. The [current requirement audit](P08_COMPLETION_AUDIT.md) reconciles the operation catalogs, conditional later-branch specifications, finite proposals and reviewed invented-data launcher. The separate filtered-population Extra Trees choice remains unresolved. Real resource admission and each later stage's numerical acceptance remain separate from the planning lock and synthetic tests. No planning approval is treated as a training permit.

## 7. Resource snapshots and cumulative attempt accounting

The pure resource checker compares caller-supplied usage and measurements with the proposed U0, U1 and Q1 ceilings. It does not query hardware or launch work. Fit counters include failed and unfinished attempts. Exhausted fit capacity prevents another fit, but does not itself forbid a pending prediction from an already completed fit. Exhausted active time or artifact headroom stops new work. Memory and worker counts may equal their ceilings but may not exceed them; model, BLAS and PyTorch thread settings must each equal one.

The storage check requires the remaining artifact allowance plus the 30-GiB reserve. U0 therefore requires 38 GiB free before its first artifact. Retained-artifact usage is a cumulative high-water mark, not a sum that charges every overwrite as a new complete file. Usage cannot be reset by pausing, restarting or deleting evidence. Allocated CUDA memory is checked against its own limit; reserved and device-reported memory remain separate recorded measurements.

The pure attempt-journal kernel reconstructs usage from hash-linked session events. An attempt is charged when it starts, before its outcome is known. A prediction requires its fitting dependency to have succeeded. A failed or interrupted job cannot be started again under the same no-retry authority. A session may close only when all its attempts are terminal; a later clean session starts its local clock at zero while retaining prior durations, attempts and artifact usage.

An open journal session is not proof of a crash or proof of a live owner. The kernel reports that a clean restart is unavailable, even if no job is currently marked in flight. Its elapsed-time total covers recorded intervals only; it cannot infer time after the last durable observation. The future runtime must authenticate the exact smoke-job mapping, journal records and head, preserve exclusive ownership, and obtain reviewed recovery evidence for an incomplete session. It may not assume the unobserved interval was a free pause.

These numerical and state-machine checks use synthetic inputs. Their acceptance is recorded in the [review log](delegation/P08_REVIEW.md). They do not establish durable filesystem behavior, authorize a retry, prove a receipt file exists, or grant a scientific execution permit. The generic journal's bounded synthetic manifests do not replace the exact 78-fit and 78-prediction U0 manifest. Full admission/restart acceptance remains open until those integration and durability checks pass.

## 8. Exact smoke-candidate checks

The candidate adapter binds the published proposal and attempt-manifest digests, replays the supplied journal, and checks the proposed U0 resource limits. It accepts no caller-supplied usage summary or replacement manifest digest. Its worker counts must agree with the replayed in-flight attempts. Admission arithmetic includes the next candidate: a worker pool already at capacity cannot accept another worker of that class. A pending prediction does not consume another fitting slot.

The proposed U0 controller stops new admissions after any failed or interrupted attempt and requires review. This rule does not classify a finite low score or a documented valid collapse as an implementation failure. Previously attempted jobs remain consumed and cannot be retried by opening another session. Successful dependencies, an open session and remaining time, memory, storage and worker capacity are necessary conditions, not execution authority.

The pure adapter neither reserves a slot nor writes an attempt event. It cannot establish that a resource reading is fresh, that a receipt file exists, or that another controller is excluded. Its structural eligibility result remains separate from a permit; scientific execution is still unconditionally denied. Durable ownership, event/head persistence and interruption handling require subsequent integration tests before a real smoke request can be accepted.

## 9. Local journal persistence and incomplete sessions

The local store joins the exact candidate guard to a hash-linked journal under a single POSIX controller lease. Before writing an attempt-start event, it checks the existing history, candidate dependency and prospective resource limits. The event must use the time and artifact counters already present in the latest durable record. If either counter advances, a progress event must be written first. This ordering checks consistency; it does not establish that an observation is fresh or truthful.

Each event is created exclusively and synchronized before the head is replaced. The head update uses a separate pending file and directory synchronization. Metadata reads reject symlinks, hardlinks, nonregular files, duplicate JSON keys, nonfinite values, oversized records and inconsistent inventories. Parser limits are 16,384 events, 1,048,576 bytes per manifest, 65,536 bytes per event and 4,096 bytes per head. These are defensive metadata bounds, not additional scientific allowances.

A write failure disables further use of that store object and preserves partial files. Reopening requires an independently supplied expected head and a coherent, not-started or cleanly closed session. An open session cannot become a free pause merely because its process has exited. Inspection can report its recorded state but cannot authorize recovery, infer unobserved elapsed time or turn running attempts into completed ones. Closing the object releases descriptors and the lease; it does not invent a session-close event.

The [review log](delegation/P08_REVIEW.md) records functional, fault-injection and cross-process acceptance separately. These tests use invented metadata and temporary files. They do not simulate physical power loss or authenticate model checkpoints, live hardware readings or scientific receipts. Runtime measurement, permit binding, receipt validation and any incomplete-session recovery remain separate gates. The implementation continues to deny scientific execution unconditionally.

## 10. Terminal receipt and artifact-byte verification

A proposed completion event must bind to the exact manifest, current journal head, running job and original attempt-start event. Its receipt must agree on the job, source-fit or source-validation stage, worker, session and terminal status. The existing replay kernel checks the proposed transition before any artifact is read. A digest-shaped receipt reference alone is insufficient.

The receipt reader checks an independently specified list of required artifact names, each file's byte length and SHA-256 digest. It uses the accepted descriptor-relative readers and rejects duplicate JSON keys, nonfinite values, unknown fields, unsafe names, symlinks, hardlinks and nonregular files. Reader bounds are 65,536 bytes per receipt, eight artifacts and 64 MiB per artifact. These are defensive parsing limits, not new scientific resource allowances.

This check is read-only: it neither appends the proposed event nor opens an execution session. Its report must distinguish verified byte integrity from unverified scientific semantics. Matching bytes do not establish that a checkpoint loads, that a model has the required architecture, or that predictions contain the expected source observations and valid values. Those semantic checks must use the authenticated job specification before a future controller accepts a successful attempt. Failure or interruption records do not grant a retry.

The caller must derive required artifact names from the reviewed stage specification, not from the receipt it is checking. Verification describes the bytes read at that time; later consumers must preserve or reauthenticate them. Live ownership, resource freshness, the scientific permit and incomplete-session recovery remain separate requirements. Acceptance of the synthetic reader tests is recorded in the [review log](delegation/P08_REVIEW.md); this specification alone establishes none of those runtime properties.

### Neural checkpoint-content requirement

The prospective source-fit acceptance path must verify the contents of each required best and terminal checkpoint, not just its file digest. The content check consumes the already read bytes and independently supplied expected file/state digests, recipe and class count. It accepts only the inherited `state_dict` wrapper and loads the same bytes with `weights_only=True` on CPU. A byte mismatch must fail before deserialization; no permissive deserialization fallback is allowed.

The loaded state must match the existing acquisition classifier's exact tensor names, shapes and float32 types. Every tensor must be an ordinary, finite, dense CPU tensor. D0-M/D2 omit the projection head; D1/D3 include it. With two classes, the corresponding parameter counts are **208,626** and **212,786**; with three classes, they are **208,691** and **212,851**. Checking both architectural interfaces does not authorize changing any registered job's class vocabulary. The inherited tensor-state digest must also match. Constructing the reference architecture must preserve the caller's random-number state and default data type, and must not initialize CUDA or perform inference.

The input-byte ceiling is **64 MiB**, consistent with the receipt reader's per-artifact bound. This is not a hard bound on deserialization memory or a sandbox for arbitrary hostile checkpoints. The future caller must first authenticate the artifact and enforce its process resource limits. Only synthetic, locally constructed checkpoints may be used during the current no-fit review.

Content validity alone cannot identify the fitting role, preprocessing input, seed, ordered chemical labels or selected epoch. Recipes sharing an architecture cannot be distinguished from their tensors alone. The check also cannot prove that training finished or that saved predictions came from that checkpoint. The checkpoint report must leave those acceptance flags false. A successful scientific source fit still needs the authenticated operation/receipt binding, complete training history and stopping checks; a successful source-prediction job additionally needs the exact validation-row/class order and checkpoint–prediction agreement. Failed or interrupted runs are not required to manufacture a successful checkpoint. This content requirement does not alter their recorded status, consumed budget or review requirement.

### Source-prediction content and job binding

The prospective U0 acceptance path must bind each source-validation prediction to its exact source-fit job. Both identifiers must match the authenticated operation registry, not merely hashes supplied by the prediction artifact. The two jobs must agree on policy, input, context, model, candidate, specification, seed and fitting/validation/test-role hashes. Their only differences are stage, dependency and derived job identifier. This check retains SG/arPLS source operations; it cannot accept a held-prediction job, a strategy alias or an unregistered branch.

The inherited classical score tables and neural logits archives remain unchanged. After authenticating their bytes, the future adapter must extract the score matrix, ordered observation identifiers and ordered class vocabulary without relabelling, sorting or coercion. The shared semantic check requires the exact registered validation order, the exact class-column order and the registered validation-set hash. Set equality alone is insufficient: a row permutation can attach a prediction to the wrong observation even when the set hash is unchanged.

The matrix must contain finite float64 scores, with one row per validation observation and one column per registered class. These are uncalibrated decision scores or logits, not probabilities; negative values, values above one and rows that do not sum to one are legitimate. A finite collapsed model also remains a valid scientific outcome. Input checks must not select models by prediction quality or silently normalize their scores.

A scalar report records the declared job pair, ordered-content hashes and dimensions without returning identifiers, chemical names or raw scores. Its content hash is not a file hash or evidence that the stated model produced those values. Registry membership, byte authentication, physical-master/instrument isolation, training completion and checkpoint–prediction agreement remain independently required. The shared check performs no model call, metric calculation, calibration or scientific operation, and grants no execution authority. Synthetic acceptance is recorded in the [review](delegation/P08_REVIEW.md).

### Neural completion-record requirement

The prospective source-fit adapter must compare the recorded training history with the inherited stopping and checkpoint-selection rules. It extracts the required fields from the existing development result or private summary; this is not a replacement artifact format. Exact types, finite metric ranges, epoch order, optimizer counts and recipe flags must be checked before using the inherited metadata helpers. String or Boolean values cannot substitute for numerical metrics or counters.

The inherited development kernel requires **three classes** in each fitting and validation role. The checkpoint checker's two-class architectural interface does not extend that training contract. The completion-record check retains **30–200 epochs**, **four updates per epoch**, **20 non-improving epochs** of patience and the earliest eligible stop. Checkpoint selection maximizes validation balanced accuracy, then minimizes negative log likelihood; exact ties retain the earlier epoch. Recorded improvement flags, running best epochs, patience counts and selected metrics must agree with that sequence.

The record's best and terminal state digests must match independently checked checkpoint contents. Its final sampling, augmentation and pair digests must match the last history entry. If the selected epoch is the last epoch, best and terminal state digests must agree. Finite collapsed predictions, zero-gradient batches and unchanged states are not rejected merely for being uninformative. Failure or interruption records cannot be relabelled as successful completion.

Gradient counters must also agree arithmetically. Subtracting zero-gradient batches from completed updates gives the number of batches with nonzero gradients. Their cumulative nonzero-element count must lie between that batch count and its product with the model's parameter count. This checks the inherited counter definitions; it does not infer whether weights changed or whether the learned model is useful.

The inherited **120-second** and **4-GiB** inner limits are checked against the declared record. This does not measure live usage, change the proposed outer ceiling or transfer a historical permit. A consistent record is not proof that training occurred: authenticated artifact loading, exact registry/role binding, checkpoint–prediction agreement, durable accounting and live enforcement remain required. Reports must retain false training-authenticity, prediction-parity and execution-authority flags until those independent obligations are met. Current tests may use invented records and checkpoints only.

### Combined neural source-artifact requirement

The neural source adapter must connect the declared job pair, training record, best/terminal checkpoints and saved validation-logits archive before proposing a successful result. Checking those artifacts independently is insufficient when their recipe, seed, class order or state digests disagree. The adapter must retain the existing `validation_logits.npz` format and the accepted training-record projection. Its supplied record pin is a canonical content digest, not the byte digest of the full private summary.

The adapter checks the exact source-fit and dependent source-validation jobs before parsing artifacts. Recipe and seed come from that verified pair; the fitting-role identifier comes from separate role metadata, not the operation's unit identifier. It checks the projected training record against its supplied content pin, then binds both checkpoint contents to that record's state digests. Every file digest is checked against an independently supplied expected value. The future controller remains responsible for authenticating those expected values and their operation-registry membership.

The prediction reader accepts only the inherited three named arrays: logits, ordered observation identifiers and ordered classes. It checks the archive's byte digest before parsing and validates each array header, type, shape and exact payload length before array allocation. Pickled objects, duplicate or unexpected members, incompatible shapes and nonfinite logits are rejected. The defensive archive and total expanded-payload bounds are **2 MiB** each; these are parsing bounds, not additional scientific storage allowances. The arrays retain their stored values and order without normalization, relabelling or sorting.

The combined report must remain scalar/hash-only and distinguish consistency against supplied pins from external provenance. A finite score matrix can agree with every structural requirement without being produced by the saved checkpoint. Therefore prediction parity, training authenticity, physical-role isolation, live-resource enforcement and execution authority remain unverified. This reader performs no inference, metric calculation or fitting. CPU reference-model construction inside the accepted checkpoint checker is permitted; model forward evaluation and CUDA initialization are not. Synthetic composition must pass review before this component is accepted, and acceptance is not a launch permit.

## 11. Serial smoke measurements

The proposed first smoke schedule runs one operation at a time in the controller's Python process. It retains the same 78 fits and 78 source-validation prediction jobs and stays below the existing worker ceilings. Serial scheduling simplifies resource attribution but does not validate later parallel execution or guarantee completion within 90 minutes. The scientific scope and proposed ceilings are unchanged.

Before a candidate check, the serial adapter must replay the exact journal and require an open session with no active, failed or interrupted jobs. It measures the current process's resident memory, checks each process thread for child processes, and reads free space through the output directory's descriptor. Process-tree memory equals the measured process memory only under this child-free execution condition. Two child checks cannot exclude an unobserved short-lived process; the reviewed runner must itself prohibit subprocess work.

Loaded numerical thread pools and PyTorch's intra-operation and inter-operation thread counts are checked without changing their settings. The model's thread count is an explicit runner-supplied configuration, not an independent inspection of the fitted estimator. The runner remains responsible for enforcing that configuration and for avoiding later imports or subprocesses that invalidate the measurement.

For an initialized CUDA context, the adapter records allocated, reserved, device-used and lifetime peak allocated bytes separately. The proposed allocation ceiling applies to the recorded peak as well as the current allocation. A GPU candidate requires an initialized context; the adapter must not initialize CUDA or reset its counters. For a CPU candidate without an initialized CUDA context, zero device-used bytes is explicitly an unobserved placeholder, not a claim that the device is unused.

The measured resources feed the existing prospective candidate check. A monotonic clock bounds the entire sampling and checking interval to one second. This defensive freshness threshold is not additional scientific execution time or a reservation of memory. The controller must capture readings under its live lease immediately before admission; a saved report cannot be reused as permanently fresh evidence.

This read-only adapter neither records durable progress nor starts a job. It cannot reconstruct unrecorded elapsed time, measure cumulative artifact growth, validate a checkpoint, grant a permit or enforce memory peaks between observations. Those obligations remain with the reviewed runtime. Synthetic test acceptance is recorded separately in the [review log](delegation/P08_REVIEW.md); no scientific operation is authorized by this scheduling proposal.

### Combined no-fit validation boundary

The integration tests compose the existing components in the proposed serial order: record progress under the store lease, inspect the candidate and resource observations, record an invented start, verify an invented terminal receipt, and append its terminal event. A dependent prediction remains unavailable until its fitting dependency succeeds. Clean session closure preserves cumulative counters; releasing a lease without session closure requires review before reopening. A failed or interrupted attempt remains consumed after a clean reopen and prevents further admission under the unchanged no-retry authority.

These tests replace hardware and clock observations with explicit synthetic values; their files contain invented non-model bytes. They verify record consistency and byte-check composition, not measured artifact growth, checkpoint loadability, valid scientific predictions or enforcement of a live permit. The store itself does not automatically call the receipt verifier. A future runtime must enforce that ordering, bind the stage-specific required artifacts and perform semantic acceptance before recording scientific success. The [review](delegation/P08_REVIEW.md) records which combined checks passed; no test callback is a production runner.

## 12. R1 proposal: frozen wider-range sensitivity

The [range ledger audit](../results/p08_readiness/range_ledger_audit.json) binds **131,199 prospective operations** across the unchanged **260 contexts**. Its four-method panel is RBF-SVM, Random Forest, D0-M and the frozen context-local P05-selected recipe. Extra Trees is excluded. Source/test roles, seeds and classical candidate grids are inherited; range input and model-specification hashes are new. No historical MIN estimator or prediction is reused as wider-range evidence.

| Fitting component | Prospective fits |
|---|---:|
| RBF-SVM | 25,160 |
| Random Forest | 34,620 |
| Distinct neural recipes, after identical-strategy sharing | 3,201 |
| Total | **62,981** |

The graph separately contains **1,417 scalar calibrations**, **59,508 source-validation prediction jobs**, **1,536 fresh calibration prediction jobs**, **1,937 held-prediction jobs** and **819 seed-ensemble jobs**. It also retains **1,584 within-range calibration aliases**, **520 classical selection jobs**, **897 epoch-selection jobs** and **520 neural strategy aliases**. These counts describe repeated roles and candidates, not additional independent samples. Structural aliases do not authorize cross-representation cache reuse.

| Proposed stage | Active wall time | New private artifacts | Process-tree RAM | Allocated GPU memory | Fitting workers |
|---|---:|---:|---:|---:|---|
| R1, separate from U0/U1/Q1/N1 | 24 hours | 40 GiB | 24 GiB | 8 GiB | At most four single-thread CPU workers and one GPU worker |

R1 proposes at most **62,981 model fits** and **1,417 scalar calibrations**, with no automatic retries. The same cumulative wall-time, failed-attempt and evidence-retention rules apply. Retain source prediction/status records, neural best/terminal source checkpoints and final refit checkpoints, and classical final estimators. Classical grid estimators need not be persisted when the inherited procedure requires only their predictions and fit records. No existing evidence may be deleted to meet the allowance.

Authenticated MIN fitting durations for this panel sum to **59,087.649 seconds** for the classical models and **8,695.609 seconds** for the neural models: approximately **16.41 sequential CPU-hours** and **2.42 summed neural fitting hours**. These historical durations exclude new range-dependent changes, prediction, scalar calibration, serialization and orchestration. They are neither measured range timings nor an upper bound. The 24-hour ceiling is a finite proposal, not a completion guarantee.

R1 requires at least **70 GiB free** initially under the existing **30-GiB reserve**. Earlier stages' retained artifacts still occupy disk; stage allowances cannot be borrowed automatically. Synthetic input-width compatibility does not validate a range executor. The historical primary-width guards remain fixed, and range-specific runtime acceptance and the numerical inference addendum are still required before a separate execution request.

### Current storage feasibility, 2026-10-02

A fresh filesystem check reported **8,824,156,160 bytes available**, approximately **8.2 GiB**. This superseded the earlier capacity observation at that checkpoint; it was not reserved space. It was below the initial requirements for U0 (**38 GiB**), U1/N1 (**110 GiB**), R1 (**70 GiB**) and Q1 (**430 GiB**). None was storage-feasible under its proposed full allowance at that observation. Additional private capacity or an explicitly approved storage/retention amendment was required; no cleanup, evidence deletion or reduced reserve was inferred. Repeat the check before any future launch request.

## 13. Conditional N2 proposal: frozen MIN-selected families

The owner selected N2's scientific design on 2026-10-04: keep each context's recorded MIN-selected classical family and retune that family's hyperparameters for each normalization control. It replaces the full-selector N1 interpretation for this exploratory branch; its resource proposal below is not approved. Both designs retain SNV, vector, area and the derivative destructive control; neither permits held-test model selection or repairs the eight missing historical final results.

The [authenticated accounting](../results/p08_readiness/fixed_family_alternative_audit.json) gives **42,368 model-fit slots** and **1,040 scalar temperatures** across these four controls. Per control, **9,876 source fits**, **426 fresh calibration fits** and **290 final fits** account for every fitting slot after **444 exact-role calibration aliases**. The subsequent [N2 slot ledger](../results/p08_readiness/normalization_slot_ledger_audit.json) enumerates **89,632 operations** with exact selection, calibration and prediction dependencies and binds the frozen input hashes. It is metadata planning evidence, not an accepted numerical runtime or an execution permit.

| Proposed alternative | Active wall time | New private artifacts | Process-tree RAM | Allocated GPU memory | CPU fitting workers |
|---|---:|---:|---:|---:|---:|
| N2, selected design; resources unapproved | 24 hours | 20 GiB | 24 GiB | 0 GiB | At most four single-thread workers |

All failed and interrupted attempts consume their slots and time; there are no automatic retries. The same evidence-retention and **30-GiB reserve** rules apply. N2 requires at least **50 GiB free** initially, including the space occupied by earlier retained stages. Retain source records/predictions, selection and calibration states, and final estimators; do not save or delete evidence by convenience to meet a ceiling. N2 does not change the universal-first sequence or borrow an earlier permit.

The relevant historical source durations sum to **7,988.616 seconds**; four copies give **8.88 sequential CPU-hours**, including the recorded costs of failed candidates. This excludes new-control changes, calibration, final fitting, predictions, persistence and scheduling. The 24-hour and 20-GiB limits are proposed operational bounds, not measured needs or promised completion. Scope selection, the metadata ledger and [numerical inference specification](P08_NORMALIZATION_INFERENCE.md) are resolved; numerical implementation review, runtime acceptance, fresh capacity checks and a separate scientific permit remain required.

**Subsequent capacity observation:** at **2026-10-03T00:05:39Z** (the evening of **2026-10-02** locally), the filesystem reported **57,421,008,896 bytes free**, approximately **53.5 GiB**. This clears the proposed initial U0 and conditional N2 storage thresholds at that instant, but not U1/N1, R1 or Q1. No cleanup was performed as part of this audit. The increased capacity is not reserved, does not approve N2, and does not authorize a launch or a change to the universal-first sequence.

**Latest capacity observation:** at **2026-10-04T18:08:50Z**, the filesystem reported **210,259,017,728 bytes free**, approximately **195.8 GiB**. This meets the proposed initial storage thresholds for U0, U1, R1 and the selected N2 design individually. It remains below Q1's **430-GiB** initial requirement. The observation does not reserve space for concurrent or cumulative stages, approve any allowance, or resolve adaptive-stage storage. No cleanup was performed during this audit; capacity must be checked again at any authorized launch.

## 14. Filtered-population resource proposals

The [population catalog](../results/p08_readiness/population_slot_ledger_audit.json) enumerates fresh source-only model selection in the notes-clear and Mira-1-excluded tiers. These are later sensitivity stages, not additions to U0 or U1. Both four- and five-method alternatives remain explicit until the owner resolves Extra Trees inclusion. Their [cost audit](../results/p08_readiness/population_resource_proposal_audit.json) specifies finite ceilings without selecting a panel, transferring prior permissions or launching work.

### Historical cost basis

The new [historical neural audit](../results/p08_readiness/population_historical_neural_cost.json) authenticates **14,428 files** and extracts **12,780 source-fit** and **1,635 final-refit** duration records from the **260** held-evaluation contexts. It excludes **60** development contexts. The source count contains **8,172 inherited-selection fits** and **4,608 guard fits** across all four recipes. The **36** reused pilot fits appear once, not as additional executions. Interrupted-run accounting charges are not substituted for measured kernel durations.

The source kernel's elapsed time includes its internal validation. External scalar calibration, persistence, orchestration, reporting and interruptions remain outside these durations. Source summaries contain historical validation metrics, but the audit did not extract or use them. It loaded no prediction archive or checkpoint tensor. Checkpoint sizes come from authenticated inventory entries checked against current file sizes, not a fresh tensor-content verification. The D1/D2 final-refit samples contain only **42** and **33** selected historical cases, respectively; their means are not representative timing guarantees for a new population.

Classical rates use the existing [MIN timing basis](../results/p08_readiness/historical_timing_basis.json), separately for source search, fresh calibration fitting and final refitting. Divide durations by the number of recorded fits, not by counts that include reused calibration aliases. Historical serialized estimator sizes were measured in memory; they do not establish that primary classical checkpoints exist.

For each catalog, multiply unconditional model/stage counts by their corresponding historical mean kernel duration. For the remaining conditional source/refit counts, use the largest D1/D2/D3 mean rate separately at each stage. This is a historical-rate envelope, not a bound on future time. It must not sum mutually exclusive candidate branches. The checkpoint estimate retains neural best and terminal source checkpoints and terminal final checkpoints. Classical size estimates retain final estimators; grid/calibration prediction and status records remain required, but their estimators need not be stored.

| Population | Panel | Sequential classical kernel hours | Neural kernel-hour envelope | Serial kernel-hour envelope | Checkpoint/estimator estimate, GiB |
|---|---|---:|---:|---:|---:|
| Notes-clear | Four methods | 41.81 | 17.97 | 59.78 | 31.40 |
| Notes-clear | Five methods | 64.64 | 17.97 | 82.61 | 40.55 |
| Mira-1 excluded | Four methods | 45.66 | 18.15 | 63.80 | 32.55 |
| Mira-1 excluded | Five methods | 70.59 | 18.15 | 88.73 | 42.76 |

These are arithmetic projections from historical MIN runs, not new-population measurements, GPU profiler times or parallel wall-time forecasts. Filtering and preprocessing may change convergence and selected hyperparameters. Do not divide these sums by a worker count and present the result as a completion promise.

### Finite, unapproved ceilings

The planning rule doubles the serial kernel-hour envelope and rounds upward to the next **24-hour** block. It doubles the checkpoint/estimator estimate and rounds upward to the next **16-GiB** block. These margins cover unspecified overhead and variability provisionally; they are not measured total-cost bounds or statistical confidence limits. The resulting ceilings limit execution even if a stage remains incomplete.

| Proposed stage | Model fits | Scalar calibrations | Active wall time | New artifacts | Initial free space including reserve |
|---|---:|---:|---:|---:|---:|
| POP-NOTES-4 | 170,277 | 4,071 | 120 hours | 64 GiB | 94 GiB |
| POP-NOTES-5 | 258,387 | 4,716 | 168 hours | 96 GiB | 126 GiB |
| POP-MIRA-4 | 184,749 | 4,347 | 144 hours | 80 GiB | 110 GiB |
| POP-MIRA-5 | 280,878 | 5,067 | 192 hours | 96 GiB | 126 GiB |

The four- and five-method stages are alternatives within a population, not cumulative permissions. Every stage proposes **24 GiB process-tree RAM**, **8 GiB allocated GPU memory**, at most **four single-thread CPU fitting workers** and **one GPU fitting worker**. Model-internal, BLAS and PyTorch CPU thread counts remain one per worker. The inherited inner neural limits are unchanged. Record reserved and device-reported GPU memory separately from allocation.

Active wall time includes setup, failed attempts, fitting, predictions, calibration, persistence and reporting. Every attempted operation consumes its slot; automatic retries remain zero. Preserve completed and failed evidence. Retained catalogs, status records, predictions, selection/calibration records and logs count toward actual stage storage, even though they are absent from the checkpoint-only estimate. A ceiling breach stops new admission; it does not authorize deletion, a new allowance or a reduced scientific experiment. Numerical runtime review and independent live enforcement remain necessary before any permit.

### Capacity and cumulative retention

At **2026-10-04T21:04:53Z**, the private-artifact filesystem reported **211,101,495,296 bytes free**, approximately **196.6 GiB**. Host available RAM was **52,779,343,872 bytes**; device-reported free GPU memory was **15,187 MiB**. These observations are not reservations or runtime acceptance. Each population stage individually meets its proposed initial storage threshold at that instant; Q1 still does not meet its **430-GiB** threshold.

The **2026-10-07T13:30:53Z** recheck reported **192,519,241,728 bytes free**, approximately **179.3 GiB**, with **53,036,097,536 bytes** of available host RAM and **15,145 MiB** of device-reported free GPU memory. Each population stage still clears its initial threshold individually. The reduced free space illustrates why a dated observation cannot substitute for admission-time checks. No cleanup or reservation was performed.

The stages cannot all assume the same free space. Retaining the full **80-GiB** U1 allowance plus both four-method population allowances and the **30-GiB** reserve requires **254 GiB**. The five-method counterparts require **302 GiB**. Both exceed the observed capacity even before range, normalization or adaptive artifacts. Actual retained sizes may be lower, but their size cannot be assumed in advance. Recheck capacity after each accepted stage; additional storage or an explicitly approved retention amendment is required if the next stage fails admission. No cleanup is authorized or performed by this audit.

Universal preprocessing remains the first scientific stage to request. These larger sensitivities do not delay implementation work on the remaining readiness specifications and do not become authorized because their costs are now explicit. Panel resolution, numerical runtime/inference review and separate stage-specific permits remain outstanding.

## 15. S1 proposal: registered test-time robustness

The [S1 resource proposal](contracts/p08_stress_resources.json) bounds the approved 96-case robustness design after the required universal and QC estimators are available. It crosses neither the wider range nor the normalization or filtered-population branches. The fixed-clean-route QC interpretation remains unchanged. This proposal does not authorize native-grid gate-reaction testing, new hyperparameter selection, neural optimization or a scientific launch.

Only the **1,820 historical classical MIN estimators** require new fitting: **260 RBF-SVM**, **780 Random Forest** and **780 Extra Trees** fits. Reconstruct their exact saved source-selected specifications, original fitting rows and seeds. No scalar temperature is refitted. The other **6,751 estimator references** and **5,343 calibration references** come from retained upstream artifacts; a missing or incompatible artifact cannot trigger a replacement fit under S1.

| Proposed stage | Active wall time | New private artifacts | Process-tree RAM | Allocated GPU memory | Workers |
|---|---:|---:|---:|---:|---|
| S1, complete registered robustness design | 72 hours | 64 GiB | 24 GiB | 8 GiB | At most four single-thread CPU workers and one GPU worker |

The CPU workers may reconstruct classical estimators or execute bounded processing jobs. The GPU worker performs retained neural inference, not optimization. The wall ceiling includes authentication, setup, reconstruction, transforms, predictions, inference, diagnostics, persistence and reporting. Failed attempts consume their slots and time. Automatic retries remain zero; a ceiling breach stops new admission and preserves evidence. These limits are proposals, not transferred U0/U1/Q1 permissions.

### Cost evidence and unmeasured work

The authenticated historical final-refit records contain **1,786.700 seconds** of summed fitting time and **6,142,752,908 bytes** of in-memory serialized estimators (approximately **0.50 hours** and **5.72 GiB**). These are historical kernel measurements, not saved estimator files or a forecast of the full stress experiment. They omit disturbed preprocessing, prediction, uncertainty, serialization and scheduling. No stress runtime has been measured, so the **72-hour** ceiling is an operational limit rather than a promised completion time.

Payload arithmetic provides scale without claiming a measured storage requirement. The **142,592** primary-grid raw cases would occupy **1,598,171,136 bytes** at float64. The **427,776** transformed rows would occupy **2,397,256,704 bytes** at their declared float32 serialization, or **4,794,513,408 bytes** if all float64 working values were retained simultaneously. These figures exclude predictions, metadata, receipts, logs, compression and filesystem overhead; runtime work need not hold every row simultaneously.

The inherited master/instrument weight arrays contain **6,320,000 float64 bytes** in total and must be referenced once by identity and hash. A single reporting view of all **456 contrasts**, **three weighting modes** and **10,000 draws** has **109,440,000 bytes** of scalar terminal draws. Allowing four views for fixed and conditionally available paired-support reporting gives **437,760,000 bytes**. This is a capacity allowance, not permission to invent another inferential family or reduced-support method. The original hierarchical scalar results and defined-draw mask require **36,480,000** and **4,560,000 bytes**, respectively, before metadata. Exact conditional activation and analysis-job accounting remain separately reviewed requirements.

Process uncertainty in batches of at most **128 draws**, or **79 batches** for each registered stream. Retain terminal contrast draws, defined/undefined status, support and provenance. Do not persist a draw-by-context-by-case tensor: it is not required evidence, and its size would dominate these terminal records. Shared weights, frozen inputs, calibrated predictions and the exact algorithm must permit reconstruction of the calculation without repeating model fits. No random values or statistical outputs were generated to obtain these arithmetic counts.

### Retention and capacity

Retain reconstructed MIN estimators, realization/source-noise bindings, transformed rows, seed predictions, calibrated prediction units, case scores, clean-parity receipts, terminal contrast draws, missingness reasons, diagnostic summaries and figure provenance. Reference existing upstream models instead of copying or refitting them. Immutable shards may package records only when every record keeps its identity and hash. Failed attempts, control files and logs count toward the **64-GiB** allowance. The proposal neither deletes earlier evidence nor relaxes its retention obligations.

S1 requires **94 GiB free** initially under the existing **30-GiB reserve**. At **2026-10-07 18:09:10 UTC**, available filesystem space was **135,042,387,968 bytes** (approximately **125.8 GiB**), with **49,015,124 kB** of available host RAM and **14,699 MiB** of device-reported free GPU memory. This clears S1's individual initial thresholds at that instant; it does not reserve capacity or establish live enforcement.

The required stages cannot each spend that same free space. Retaining the full U1 and Q1 allowances, S1 and the reserve requires **574 GiB**, before range, normalization or population artifacts. The present machine does not meet that cumulative requirement. Additional private storage or an explicitly approved retention/resource amendment is required before admitting a stage that lacks capacity; smaller actual artifact sizes must be observed, not assumed. No cleanup or storage expansion was performed by this audit.

S1 still requires complete analysis accounting, numerical implementation review, scientific clean-path parity, actual admission-time capacity and its own execution permit. It does not delay the separately bounded U0 request once readiness closes, and it cannot start automatically after an earlier stage succeeds.
