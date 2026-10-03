# P08 source-stage mapping

**Status:** no-fit runtime specification. This document grants no scientific execution authority and changes no registered job, dependency, model, seed or fitting limit.

## 1. Logical jobs are not individual model calls

The U0 proposal contains **78 source-fit jobs** and **78 dependent source-validation prediction jobs**. The inherited fitting routines already evaluate source validation before returning. Therefore these two logical stages cannot be interpreted as fitting without validation followed by the first prediction call.

For classical models, [`run_candidate_fit`](../src/atlas_sers/evaluation/p03_runtime.py) fits the estimator, calculates validation scores and diagnostic metrics, and returns both the estimator and the prediction frame. For neural models, [`train_development_fit`](../src/atlas_sers/evaluation/p05_development.py) uses source validation for checkpoint selection and stopping, restores the best state and returns its validation logits. These behaviors are part of the inherited algorithms and must not be removed merely to simplify accounting.

| Logical stage | Required work | What it must not imply |
|---|---|---|
| `source_fit` | Execute one registered fitting procedure, including its internal source-validation evaluations; preserve the result and required checkpoints/history | That validation was unavailable during fitting, or that every forward pass is a separate model fit |
| `source_validation_prediction` | Authenticate the successful fit's output, check the exact validation/class order and numerical agreement with the fitted model, and record the accepted source-prediction artifact | A second optimization, an automatic replacement fit, or permission to predict held-test rows |

The controller must authenticate the complete fit–prediction pair and its source roles before entering the fitting routine. A permit for a metadata record alone cannot authorize its embedded numerical work. Any future U0 permit must explicitly cover inherited within-fit validation and the verification passes below. The separate prediction job remains dependent on successful fit completion; no journal dependency is weakened to start it early.

## 2. Fit-stage acceptance

Before invoking a kernel, the future controller must hold its execution lease, verify the permit and fixed input/specification identities, record fresh cumulative resource usage and persist the fit-attempt start. The supplied training and validation rows must match their authenticated source roles. Held rows remain inaccessible.

The inherited kernel is called once for the registered model, candidate, recipe and seed. Its internal validation calls, elapsed time, memory and generated artifacts belong to that attempt's accounting. A returned prediction is provisional evidence until the dependent prediction stage accepts it. A finite collapsed classifier remains a legitimate result; it is not a reason to launch a replacement fit.

The classical result must preserve the original terminal status, estimator fit audit and prediction records. The neural result must pass the reviewed history, checkpoint-selection, gradient-count and checkpoint-content checks. The required files must be written and authenticated before a successful terminal receipt is accepted. A fit failure or interruption consumes the attempt, retains partial evidence and stops further admissions for review; it does not create a successful dependent prediction.

## 3. Prediction-stage acceptance without refitting

After authenticating the fit's successful receipt and source-role bindings, the controller persists the prediction-attempt start. It must consume the same authenticated bytes or reauthenticate any bytes it rereads. A previously passing receipt does not establish the identity of a subsequently replaced file.

For neural models, reuse the restored-best verification semantics in [`check_completed_result`](../src/atlas_sers/evaluation/p05_pilot.py): load the saved best checkpoint, use the identical ordered source-validation inputs and class vocabulary, and require exact equality of the recomputed and stored logits. The stored balanced accuracy, negative log likelihood, macro-F1 and predicted-class count must also agree. Verification uses the original execution device and numerical environment; passing a CPU fixture does not establish cross-device bitwise equivalence.

For classical models, retain the returned fitted estimator until its dependent source-prediction stage completes. Use the inherited aligned-score function on the same ordered validation inputs and require exact equality with the stored uncalibrated scores. No model optimization is required for this check. Historical P03 grid estimators were not persisted; this requirement concerns new P08 execution, not retrospective reconstruction of missing objects.

Verification passes consume wall time, memory and artifact accounting but do not add model-fit jobs. The **78 prediction jobs** count accepted source-prediction operations, not every internal model forward call. The original **78-fit ceiling** remains unchanged. No scalar calibration, source winner selection or held-test evaluation is added to U0.

A classical estimator lost before its dependent prediction is accepted cannot be recreated through an unrecorded fit. The controller must retain the unfinished status and require recovery review. It must not close a session as resumable while an unfinished prediction depends on an in-memory-only estimator. Any future persistence or recovery alternative requires its own reviewed implementation and accounting; this specification does not choose or authorize one.

## 4. Synthetic evidence and remaining gate

The [synthetic integration tests](../tests/test_p08_source_kernel_integration.py) connect the actual inherited CPU kernels, serializers, restored-best verifier and accepted P08 artifact checks using invented observations only. All **19 tests** pass across four neural recipes and three classical methods. Their two policy labels test job/format compatibility; those fixtures are not scientific SG/arPLS transformations or a preprocessing comparison. Test results and corrections are recorded in the [review log](delegation/P08_REVIEW.md#inherited-source-kernel-integration-review).

The combined artifact checker deliberately leaves prediction-parity and external-authenticity flags false: valid hashes and finite values alone do not establish that the fitted model produced those scores. The subsequent numerical check is a separate requirement. Passing a synthetic composition does not establish live controller ordering, authenticated permit enforcement, durable cumulative measurements or safe incomplete-session recovery.

The [resource proposal](P08_RESOURCE_PROPOSAL.md), original job graph and all zero-execution allowances remain in force. The full controller must be reviewed against this mapping before a separate scientific smoke request can be accepted.

## 5. Fixed graph and source-role binding

The [metadata binding audit](../results/p08_readiness/u0_source_binding_audit.json) connects the exact proposed jobs to their permitted source rows. Its reader authenticates the proposal, compact attempt manifest, primary manifest, context registry and role registry against fixed byte hashes before parsing. It then checks the full fit–prediction pairs against the journal projection and resolves the ordered source roles from recorded membership, not hash differences alone.

The **78 fit–prediction pairs** share **five source units in five contexts**: four pseudo-instrument units and one physical-master cross-validation unit. Each training and validation role contains the same **three chemicals**. Explicit checks exclude shared physical masters between training and validation, exclude held-test masters from the outer source set, and exclude the held instrument from source rows. The returned private objects retain only ordered source observations and job records; the public report contains counts, hashes and verification flags.

An independent read-only reconstruction matched every returned observation, metadata field, class order and full job record. All five input files retained their hashes. This authenticates recorded metadata against reviewed pins, not independent historical provenance or numerical execution. Arrays, model parameters, live permit enforcement, cumulative accounting and recovery remain separate checks. The binding object is not an execution capability; its scientific launch function always refuses execution.

## 6. Fixed source arrays and inherited noise reference

The [source-array audit](../results/p08_readiness/u0_source_array_audit.json) connects those ordered roles to the frozen MIN, SG and arPLS representations. Each archive contains **598 spectra × 1,401 channels** on the **400–1,800 cm⁻¹** grid. The reader authenticates all three archive hashes before parsing, checks array contents and row order, and returns only the proposed SG/arPLS source-training and source-validation rows. The **78 fit–prediction pairs** share **ten prepared inputs**: five source units under each of the two new policies. Authenticating MIN does not introduce additional MIN fits.

Every selected float32 row matches the original array bytes in the required order. Returned array views are read-only and backed by immutable bytes. The fitting-row quality-control metadata matches the original manifest parser, and each call returns a fresh frame. The public aggregate contains hashes and counts, not spectra, sample identifiers, source paths or noise values. Archive/header bounds are defensive reader limits, not a runtime memory guarantee.

### Noise augmentation is fixed across preprocessing policies

The inherited [neural protocol](P05_CORE_PROTOCOL.md#4-frozen-optimization-and-numerical-smoke) retains the P04 augmentation implementation and its source-only noise rule. The original P01 manifest supplies `first_difference_noise_mad` and `intensity_range`; these describe the native measured spectra, not residual noise recomputed after smoothing or baseline correction. For a given fitting role, the same recorded values are supplied under every preprocessing policy. Validation and held-test QC values do not determine that role's augmentation noise.

During later authorized fitting, the unchanged kernel divides the recorded noise estimate by the recorded intensity range and obtains its source-only noise levels. This preparation step neither computes those levels nor augments any spectrum. It does not retune noise magnitude for SG or arPLS.

The planned comparison therefore tests preprocessing under a fixed training recipe, not the best separately optimized augmentation for each pipeline. A fixed absolute augmentation can have a different strength relative to the remaining spectral structure after preprocessing. That is a limitation of the comparison, not an observed performance effect. Changing the noise reference would change the training recipe and requires a separately specified experiment.

The reader does not load models, calculate predictions, prove model-output agreement or enforce a live permit. Specification loading, controller ordering, durable resource accounting and reviewed recovery remain required before scientific execution can be requested.

## 7. Frozen model settings and recorded substrate metadata

The [runtime-input audit](../results/p08_readiness/u0_runtime_input_audit.json) connects the prepared source arrays to the frozen model settings. The adapter authenticates the specification audit, candidate registry and **18 inherited source files** before assembling kernel arguments. Every fit and prediction job must match its recorded model-specification digest. Classical jobs use the first declared candidate for each of RBF-SVM, Random Forest and Extra Trees; this is the fixed smoke proposal, not selection by validation performance. Neural jobs use their registered D0-M, D1, D2 or D3 recipe.

The inherited neural stopping rule remains **30–200 epochs**, with **20-epoch patience**. The existing development kernel retains its **120-second** per-fit and **4-GiB** allocated-CUDA guards. These inner limits are distinct from the proposed outer resource allowance and do not inherit historical execution permission. Device choice, cumulative deadline and progress callbacks belong to the future controller rather than this static adapter.

Substrate metadata is part of the inherited contrastive objective. The adapter maps each source row's recorded `sensor_family` directly to the sampler's `substrate` field, preserving blanks and unknown tokens. It does not substitute the instrument, sensor variant or a fabricated common family. This preserves the existing pair-weighting rule; it does not infer which substrates are chemically equivalent. Only authenticated source observations reach the kernel arguments.

The independent read-only check matched all **78 pairs**, including **42 classical** and **36 neural** pairs, to the original parameters, ordered arrays, role metadata and native fitting-row QC. Classical arguments retain the inherited float64 conversion; neural arrays remain float32. The public report contains aggregate counts and hashes, not the private candidate table or sample records.

Authenticating supplied source bytes does not prove that already imported Python code came from those bytes. The future controller must verify its loaded implementation before invoking any kernel. This adapter does not construct an estimator, sample a batch, estimate noise, augment spectra, fit a model or calculate predictions. Live code identity, permit enforcement, cumulative resource measurement, stage dispatch and reviewed recovery remain open; its execution entry point always refuses.

## 8. Numerical backend and saved-output verification

The [source-stage backend](../src/atlas_sers/evaluation/p08_u0_stage_backend.py) connects a prepared source pair to the inherited fitting kernel. It calls that kernel once, returns the original result and prepares immutable artifact bytes separately. Separating invocation from serialization lets the caller retain a completed or failed result if serialization raises an error. Neural arrays are copied into writable float32 buffers before the inherited kernel receives them; this changes buffer ownership, not preprocessing, values or row order.

A complete result must pass structural checks before the backend labels its fit artifacts successful. Neural checks cover the actual recipe, seed, role, training history, checkpoints and prediction layout. Classical checks cover the candidate, seed, source hashes, estimator fit audit and ordered prediction records. The repeated-master hash in the fit outcome remains distinct from the unique-master hash in the estimator audit. Failed results preserve their diagnostics without fabricating a successful identity or requiring success-only artifacts.

The dependent verifier authenticates every supplied saved byte string against the prepared snapshot before numerical inference. It repeats structural checks, then compares restored-best neural logits or retained-estimator classical scores exactly with the saved outputs. Diagnostic metrics must also agree. Changed file bytes are rejected before inference; finite, consistently repinned values still fail if they differ from the fitted model. Classical verification retains the original estimator and never replaces a missing estimator with another fit.

Verification uses in-memory snapshots, not unchecked pathname rereads. CPU neural verification restores the random state, including on failure. The report contains only counts, hashes and bounded verification flags. It does not establish that a controller wrote those bytes, authenticated its loaded code, held an execution permit or measured cumulative resources. The backend is not the complete controller and never authorizes scientific execution.

An independent invented-data CPU check completed one fit for each of the three classical methods and four neural recipes. All seven saved outputs matched their models, original prepared arrays remained unchanged, and the check passed with warnings treated as errors. This is numerical compatibility evidence, not a field-trial preprocessing result. The proposed U0 job counts, model settings and resource ceilings remain unchanged.

## 9. Integrated session requirements

This section specifies the next controller implementation. It is not evidence that the controller exists or has passed review. The reviewed backend remains separate from loaded-code authentication, an independently approved execution permit and durable session orchestration.

### Ownership and operation order

The session core must hold the existing store lease and compare all **78 prepared pairs** with the exact **156-job** compact manifest before opening a session. Full model jobs do not contain the compact manifest's worker field; the projection must derive that field from the registered model, without changing its CPU/GPU assignment. An object supplied by a caller is not, by itself, proof of authenticated inputs or execution authority.

Artifacts require a new private directory beside the journal directory, under the same verified parent. They must not be written inside the journal store, whose permitted inventory is limited to its manifest, head, events and lock. Existing output paths, including partial prior outputs, must be preserved rather than overwritten or repaired automatically. Descriptor-relative exclusive writes, synchronization and identity checks must cover both the artifact directory and its files.

For each pair, the controller must record fresh usage and resources, persist the fit start, call the inherited kernel once, retain the result, check its structure, and write and read back its artifacts. Only then may a verified receipt support fit success. The dependent prediction follows a new admission check and start event. It rereads the saved fit artifacts, authenticates those exact bytes against the retained result, checks numerical agreement without fitting again, and persists its verification evidence before prediction success. Its receipt references the original fit artifacts rather than creating duplicate copies. Another pair cannot begin while this pair remains unfinished.

### Time and retained-file accounting

The monotonic session clock must begin before session setup. Later progress records must include setup, fitting, internal validation, verification, serialization and artifact writes. Neural epoch callbacks provide intermediate observations; the existing classical kernel has no equivalent callback. Before/after checks therefore cannot be described as hard preemption of a native classical call or proof that an unobserved memory peak stayed below the ceiling. A detected breach stops further admissions and preserves the attempted work.

Artifact usage must cover both dedicated directories: model outputs, receipts, failed or partial files, journal events, manifest and head. The accounting basis is retained regular-file logical size, not allocated filesystem blocks or the sum of every historical overwrite. Unrelated files in the parent directory are outside this run's usage. Free-space checks remain necessary because logical size does not reserve physical capacity.

Appending progress itself writes an event and a pending replacement head. The following attempt-start event must nevertheless use exactly the progress record's time and artifact counters, as required by the existing store. The controller must therefore account conservatively for both impending appends, including transient old/new head coexistence, before recording progress. A bounded self-consistency calculation must resolve the encoded counter and record sizes. Observed file size and this conservative charge must remain distinguishable; neither may silently reset the cumulative high-water mark.

Charging bytes before writing them does not make them physically allocated. The controller must check actual available space for the impending writes as well as the existing reserve/remaining-allowance rule. It must not treat a prospective charge as consumed disk space when that would understate the free space still required. Failure diagnostics also consume time and storage; logging them does not grant another scientific attempt.

Resource observations must be taken under the lease and remain within the existing **one-second** freshness window immediately before admission. Measurements during an active operation must use worker counts reconstructed from the journal, not the idle counts produced by a between-job sampler. CUDA current, reserved, device-used and lifetime-peak values remain distinct. Observation must not initialize CUDA, reset allocator counters or change numerical thread settings.

### Finalization, failure and reviewed recovery

A final durable timestamp precedes the filesystem work needed to persist that timestamp. The controller must report its measured post-close finalization interval separately rather than claim that the journal includes time it could not yet observe. This interval is not a free pause. Any later recovery or budget transfer must reconcile it with independently retained controller observations. An incomplete process cannot supply a trustworthy final interval merely because its journal has a digest or a lock file exists.

The initial session core must refuse automatic entry into any previous session, including a cleanly closed one. It must inspect and retain the recorded cumulative usage before refusing; it may not reset counters or create replacement output paths to obtain fresh capacity. A future reviewed continuation must authenticate its prior head, consumed work, finalization accounting and execution authority. No recovery permit is created by this specification.

Failure after a start consumes that attempt. The controller must retain the original result and partial artifacts, attempt a bound failed/interrupted receipt, and stop further work. If writing or verifying the terminal evidence fails, the attempt remains unfinished for review. Genuine interrupts must propagate, not become successful results. A successful classical fit with an unfinished prediction still depends on its in-memory estimator; the controller must not report that state as safely closed or recreate the estimator through an unrecorded fit.

Acceptance requires an invented-data fit through actual storage and receipt verification, not only a mocked sequence of successful callbacks. Fault cases must cover stale usage, changed bytes, wrong job projections, storage failures, resource exhaustion, interrupted attempts and previous-session refusal. Mock GPU observations must remain labelled synthetic. Passing these tests would establish the tested session behavior only; loaded-runtime identity, independent permit binding and the final goal-wide readiness review remain separate gates.
