# P08 U0: outer launch boundary

**Status:** implementation specification, not an execution permit. This document connects the existing source-input, session and import components. It does not replace the original [readiness requirements](P08_COMPLETION_AUDIT.md), resolve the outstanding scientific choices or authorize training.

## 1. Scope and authority

The proposed smoke remains **78 source fits and 78 dependent source-validation prediction jobs**. It contains **42 classical pairs and 36 neural pairs**, using the frozen SG/arPLS inputs and existing candidate, recipe and seed identities. Within-fit validation and saved-output verification are included in those operations; neither permits a replacement fit. Final refits, scalar calibration, held predictions, new model/policy selection, numerical QC routing, resampling, perturbations and representation rebuilds remain excluded.

The proposed ceilings remain **90 minutes**, **8 GiB** of new private artifacts, **16 GiB** process-tree RAM and **8 GiB** allocated GPU memory, with a **30-GiB** free-space reserve and zero automatic retries. Initial execution remains serial, below the existing two-CPU-worker and one-GPU-worker ceilings. The neural kernel retains its separate **120-second** and **4-GiB** inner guards. None of these proposals is an approved allowance.

An executable entry point must start with no accepted scientific permit. A readiness contract, previous benchmark permit, caller-supplied `approved` flag or earlier audit report cannot enable it. After separate owner approval, a reviewed release must bind an independently recorded permit digest to the exact smoke graph, source catalog, input/specification identities, ceilings and private output destination. The command line must not accept a replacement authority digest or an alternate output directory that silently creates another run under the same permit.

The permit is a private deployment record; public records retain only its digest and non-identifying scope. Merely generating a proposed permit does not make it approved. Interpretation of the four pending later-branch choices remains separate from this implementation specification.

## 2. Required order

| Boundary | Required action | Refusal condition |
|---|---|---|
| Authority | Validate the independently approved permit before private scientific input reads, CUDA initialization or output creation | Missing, unapproved, changed, wrong-stage or expanded permit |
| Launch ownership | Claim the exact new private destination exclusively and retain the attempt's identity | Existing destination, competing owner, changed parent or attempted alternate destination |
| Project imports | Authenticate the catalog and captured source bytes in a fresh isolated interpreter; retain the owned importer throughout the run | Preloaded, unlisted, replaced or no-longer-owned project code |
| Input assembly | Use the existing fixed metadata, array and model-setting adapters on bounded authenticated snapshots | Changed bytes, wrong roles, altered settings or incorrect prepared-pair mapping |
| Runtime preparation | Establish the declared numerical thread/device settings; observe actual resources and include all setup costs | Unavailable device, invalid thread state, insufficient capacity or exhausted time |
| Source execution | Consume each registered pair through the existing session: admit, fit once, persist, authenticate, verify and complete | Failure, interrupt, stale observations, changed artifacts or any breached ceiling |
| Closure | Preserve terminal evidence, measured finalization costs and unresolved attempts; release owned resources | A pending in-memory estimator dependency cannot become a cleanly resumable state |

This order does not require weakening a failed check to reach a later boundary. No numerical outcome is available to choose the smoke jobs or alter this sequence.

## 3. Same-process code identity

The standalone import audit proves identity only for its own process. The scientific entry point must retain its authenticated finder and captured source buffers while preparing inputs, running kernels, verifying saved outputs and closing the session. Lazy project imports must use those captured buffers as well; restoring the normal finder before fitting would reopen the gap.

The owned scope must reject use after closure, use from another process or thread, replacement of its finder, and replacement of imported project module objects. Its validity check must cover the currently loaded project namespace, not only the modules present immediately after startup. Caller-visible source snapshots must be immutable bytes, not a mutable catalog or loader registry.

The import guard itself grants no execution authority. Its general active-scope report must not assert that no training occurred or that CUDA remains uninitialized after a separately authorized consumer has used the scope. Those assertions belong to the standalone no-fit audit only. The interpreter, reviewed bootstrap and external dependencies remain trusted rather than byte-authenticated by the project-source catalog.

## 4. Setup time, retained files and failed launches

The outer monotonic clock must start before permit validation, project imports, input assembly and output preparation. The session currently starts its clock in its own constructor; it cannot retrospectively infer earlier setup time. Integration must carry the outer start into accounting or otherwise charge that interval explicitly, without changing the proposed ceiling or counting it twice. A successful fit must not receive a new full allowance merely because setup happened outside the internal session.

An approved launch must retain exclusive evidence of its attempt before project imports or input preparation can fail. A failed setup does not consume a model-fit slot unless a fit-start event was recorded. It does consume observed active time and retained storage, and cannot become an automatic fresh launch. Unknown time after a crash remains unknown; a later process cannot declare that interval a free pause.

The existing session counts its journal and model-artifact directories. Any outer launch record, ownership metadata or retained failure record is additional run storage and must also enter the cumulative high-water accounting. Such records must not be inserted into the journal directory, whose inventory is already fixed. A future control directory must have an explicit inventory, bounded writes and authenticated ownership, with its bytes included in admission and free-space checks. A new filename or sibling directory cannot be used to reset usage.

The first outer entry point must refuse all previous destinations, including ones with no completed fit or an apparently clean journal. Inspection and recovery are separate operations. Recovery requires review of the retained attempt, elapsed-time uncertainty, consumed slots, artifacts and permit; it is not a default command-line mode.

Measured finalization includes outer cleanup and any retained final report, not only the internal journal-close timestamp. Durable counters and later measured intervals must remain distinguishable. Reports must not describe a pre-write timestamp as covering the write that followed it.

## 5. Inputs, devices and failure propagation

The complete frozen archives may be reauthenticated and structurally checked, as in the accepted adapters. Only the registered source-training and source-validation slices may reach numerical kernels. This is a source-only scientific computation boundary, not a claim that an archive containing all stored observations was never opened. No held prediction or held-outcome selection is permitted.

Input paths and private output bindings must be explicit and bounded. Reject symbolic links, nonregular input files, changed identities and path traversal; do not discover alternate files automatically. Reauthenticate any bytes reread after an earlier check. Source catalog verification must not be confused with authentication of separate JSON model contracts or private array archives.

CUDA initialization belongs after authority, ownership, code and input checks. Resource observation must not initialize it implicitly. Keep allocated, reserved, device-used and lifetime-peak memory distinct. Numerical thread settings must be established before fitting and verified by the existing observer; no global machine configuration or dependency installation is part of a launch.

The outer layer must preserve the original interrupt or execution failure while attempting bounded diagnostics and cleanup. A failure while recording a failure cannot turn the run into a success, erase the initial exception or authorize another attempt. The journal and partial outputs must survive for inspection. Classical in-memory estimator loss before dependent verification still forbids an unrecorded refit.

## 6. Acceptance evidence

Acceptance requires a cohesive entry-point test, not only independent helper tests. Invented fixtures must demonstrate permit refusal before input reads and mutations; exclusive launch ownership; retained lazy-import protection; complete job/input binding; inclusion of pre-session time and control-file storage; a numerical CPU pair through actual session persistence; interruption preservation; and refusal of a second launch using the same destination.

Additional fault cases must change the permit, input bytes, catalog, loaded module, finder, output parent and resource observations independently. Tests must demonstrate that earlier-stage failures cannot reach fitting. GPU-labelled fixtures must remain clearly distinguished from actual GPU execution. Existing tests of the internal session and standalone import audit are reused as component evidence, not represented as this end-to-end acceptance.

The future scientific smoke must still obtain separate approval and repeat real capacity checks. Passing an invented-data entry test would establish tested orchestration behavior; it would not establish preprocessing benefit, cross-device numerical equivalence or completion of the full P08 benchmark.
