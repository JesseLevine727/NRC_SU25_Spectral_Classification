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
