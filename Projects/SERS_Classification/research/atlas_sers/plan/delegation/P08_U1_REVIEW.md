# P08-U1 supervised implementation review

## Authority and current status

The owner approved the [U1 execution limits](../P08_U1_EXECUTION.md) on 2026-10-08, conditional on implementation review. Two infrastructure stops and their explicitly approved recoveries are recorded below. The latest R2 section supersedes earlier execution checkpoints; accepted U0 evidence and all frozen scientific choices remain unchanged. No scalar calibration, final refit or held prediction was reached in either stopped run.

DeepSeek V4.1 Flash is the implementation worker through the existing tool-disabled OpenCode patch-author wrapper. Only selected public source snapshots and synthetic fixture instructions are supplied. The supervisor applies patches, runs checks and controls scientific execution and publication. Worker statements are not acceptance evidence.

## T271–T273: per-fit live epoch monitor

Accepted as an observational component, not as a full-run controller or scientific launch gate.

- Added `visualization/p08_live_monitor.py` and focused synthetic tests. The module uses the standard library only and does not import training libraries or perform fitting/inference.
- Completed epochs are appended to private JSONL with flush/fsync; offline HTML refreshes after accepted epochs. Final semantic JSON and native pgfplots/TikZ use identical coordinates and axis definitions. Console/HTML identify an opaque job token, preprocessing, recipe, seed and stage.
- Training objective/components are distinguished from clean-evaluation training NLL and source-validation NLL/accuracy. Source validation is not held-test performance. Fixed-duration refits have no invented validation curve or best-checkpoint claim.
- Invalid records do not alter accepted evidence. I/O/render/stream failures close the monitor to further epochs and propagate. Clock observations must be finite and nondecreasing; final elapsed time is frozen. Persistence is not claimed to be an atomic multi-file transaction, and partial files remain evidence.
- Supervisor review rejected the initial HTML/TikZ axis mismatch, hidden write errors, missing visible metadata and incomplete timing checks. Two bounded worker correction rounds addressed these; the supervisor then corrected the synthetic default recipe label to D3 because that fixture includes both auxiliary losses. No scientific kernel changed.
- Focused result: **70 passed**, including real `pdflatex` compilation for source-fitting and refit plots. Ruff passed. The supervisor inspected rendered native PDF/PNG and offline browser HTML for readable legends, labels and agreement. These displays contain invented test values, not benchmark results.

Command from the package directory:

```text
../../.venv/bin/python -m pytest tests/test_p08_live_monitor.py -q --tb=short
../../.venv/bin/ruff check src/atlas_sers/visualization/p08_live_monitor.py tests/test_p08_live_monitor.py
```

Remaining integration: connect the callback to the governed worker, expose a campaign-level index/count view, and include monitoring overhead in cumulative resource observations. Passing these component tests does not establish live GPU resource enforcement or full-benchmark completion.

## T274/T277/T280: all-candidate source inputs

Accepted as an input adapter, not as execution authority. The factory authenticates the full universal graph, metadata, three representation archives, candidate registry and inherited specification sources. It retains source pairs only and uses bounded role-array caching. Kernel arguments preserve the existing physical-master roles, instrument exclusion, candidate parameters, class ordering, QC values, substrate strings and neural RNG role IDs.

The first worker response was truncated and was not applied. Review of its completed replacement found an important integration defect: a source-only validator was being applied to non-source and MIN graph jobs. The final bounded correction validates global graph identities before filtering supported source pairs. Other corrections covered metadata pins, exact pilot-container access, candidate coverage, cache eviction and mutation tests. Supervisor changes were limited to patch-format normalization and formatting.

Focused source-input tests: **17 passed**. A separate read-only check authenticated the actual 607,221-job graph and resolved all **184,392 SG/arPLS source-fit configurations**, across 260 contexts and 681 source units. All **78 pilot input pairs** matched the existing U0 adapter exactly, including arrays, observations, parameters and QC. This check took 27.69 seconds and peaked at 2.02 GiB RSS; it performed **zero fits or predictions**. Its aggregate factory-report SHA-256 is `6c1549fe8692ee5832d8a4ccfe3cd93ea0a2d038d44f95baf459fd5009df94e5`.

## T276/T279: source-only selection

Accepted as a pure selection adapter. Classical settings use the unchanged inherited lexicographic objective, complete candidate/unit/seed coverage and declared tie-breaking order. Neural refit duration uses the inherited per-seed median-best-epoch rule, rounded and clipped to 30–200 epochs; recipe identities are not reselected.

Review corrected job-key handling and neural specification identity. Two final synthetic fixture corrections made the intended tests meaningful: equal complexity for a declared-order tie and a deterministic SVM registry row. Neither changes scientific selection. Focused result: **41 passed**; Ruff passed. No real-data selection has been performed.

## T275/T278/T284: durable execution ledger

The reviewed core records immutable job identities, dependency/attempt accounting, pilot reuse and cumulative resource observations in a single-controller SQLite ledger. It does not measure hardware or authorize numerical kernels. Initial corrections removed per-start full-table budget scans, strengthened journal/table consistency checks, preserved failed-create evidence, enforced CPU/GPU lanes and retained breached-resource observations. The worker's repeated/truncated output tail was not applied; only its complete, nonduplicated patch sections were used.

Accepted as a ledger primitive after the final bounded correction. Stored job payloads are now bound to their content-addressed IDs on reopen, and database-write failures latch the instance against further starts. Focused result: **30 passed**, with Ruff passing. The supervisor normalized two worker patch paths to the explicitly assigned files; no scientific logic was changed. Full controller acceptance has not been granted.

## Pilot reuse and next integration

A fresh read-only inventory of the accepted recovery directory confirmed all **3,515 files**, **64,739,703 bytes**, and inventory SHA-256 `f0ed157a0d928a7114344e2520adbbef5daddaed39ce4f4b0e8fd41a74dabd4f`. No retained evidence was modified. Together with exact source-input parity, this supports reuse; importing the pilot's terminal receipts into the new controller still requires review.

T282 integrates post-source physical roles and fitting/held-only input separation. T283 integrates the frozen scalar-calibration and seed-aggregation order. Both are bounded implementation tasks with synthetic tests, not scientific execution. Full worker/controller integration, live resource measurement, loaded-runtime binding and end-to-end admission remain required before launch. No later preprocessing branch is included.

## Reviewed component release

The four accepted components above pass **158 focused tests** together. A clean tracked-package export with only those reviewed additions passed the public scaffold validator (**1,063 files checked**) and package-wide Ruff. The export excludes unrelated user changes, local LaTeX build files and pending implementation drafts. This is an implementation milestone, not a benchmark-result release; remote CI is checked separately after the main-branch push.

The next integration drafts remain outside this release. T282 produced no code before its worker output limit; T286 is its bounded completion attempt with a smaller source context. T283 initially passed 18 synthetic tests, but metadata/seed validation and audit-count corrections were requested in T287. T285 composes the existing calibration-fit/final-refit/held-prediction kernels without changing them. None of these drafts grants execution permission or changes the frozen experiment.

Calibration wording must distinguish physical-master-separated cross-fit roles from equal-master weighting. The inherited classical helper fits temperature to its cross-fitted observation rows after tree-seed aggregation; the neural helper first averages logits by physical master. U1 preserves both inherited numerical procedures. Neither a new weighting rule nor a calibration experiment is introduced by this adapter work.

## Full integration and first launch

The post-source input, fitting, calibration, prediction, artifact, process, dispatcher and controller integrations were accepted after supervisor corrections and 288 focused tests. Eight private resource-guard tests also passed. The real-data read-only post-input audit resolved 14,618 distinct bindings without fitting. Full synthetic SVM and ordinary-CNN dependency chains reached held prediction and ensemble output using invented spectra. The reviewed integration was pushed on `main` at `6475a0a9536af70dcd1c8384802b95968c3509ce`; its remote CI passed.

The first launch reused all 78 U0 pairs and admitted 11 new source fits: seven completed, one failed at CUDA setup and three were interrupted by the controlled stop. Six new independent prediction verifications completed. The seventh completed fit had not reached its paired verification, and its estimator was not persisted; it is therefore included in the approved exact replay rather than treated as verified reuse. Zero scalar calibrations, final refits or held predictions ran. All five workers exited and the ledger closed cleanly.

The private audit verified all 404,814 registered job hashes, 374 hash-linked events, 156 imported operation artifact sets and 14 new terminal artifact sets. The closed directory contains 601 files and 711,585,724 bytes, with inventory SHA-256 `2e9cd51d336f079f1e9240a5cb8ac796495802c3efb27af5926b49cdebd0b9f8`. Its larger recorded artifact high-water mark remains charged.

## T297/T298: approved R1 recovery

The supervisor's launcher omitted explicit CUDA initialization. A fresh-process synthetic reproduction matched the failed job's exception digest exactly, before any epoch or optimizer update. GPU availability and memory telemetry alone do not initialize CUDA. The inherited training kernel is unchanged; the correction initializes the worker device and verifies memory-statistic access before the worker becomes ready.

DeepSeek V4.1 Flash authored the device-preparation helper and immutable recovery-accounting adapter. Supervisor review tightened boolean validation, required sorted distinct job-ID lists, removed an unused binding alias, corrected test fixtures and formatting, and added the recovered-attempt ceiling regression. The optional profile leaves original-run defaults intact, requires all six new reusable fits at sealing, prohibits reuse of the five replay identities, and carries previous time/storage/attempt charges into a distinct recovery directory. Failed original stores remain failure-latched.

The read-only recovery bridge authenticated 84 exact fit/prediction pairs across both retained runs in 29.82 seconds, with zero new fits. Twelve device-helper tests passed, including a real fresh-subprocess CUDA test. Five end-to-end dispatcher tests passed, including ordinary and auxiliary-loss CNN paths on the GPU and the real unchanged kernels on invented spectra. The 62 combined ledger, recovery-accounting and controller tests passed. The final combined suite passed **326 tests in 80.47 seconds**, including all U1 components, the live monitor and eleven private resource-guard tests. Package-wide Ruff passed. This accepts the reviewed recovery implementation for its bound launch; actual benchmark completion remains pending.

The recovery is limited to five exact replays and 195,113 previously unstarted fits. Its total fit-attempt ceiling is 195,212, including ten retained overhead attempts. All scientific choices and resource ceilings remain unchanged. These are infrastructure-recovery findings, not evidence that smoothing, baseline correction or a classifier improves chemical identification.

## T305/T308/T309: approved R2 recovery

R1 completed 8,466 new source-fit/prediction pairs and one source-only epoch selection. Together with imported evidence, 8,550 complete pairs are reusable. A live-monitor atomic replacement raced the supervisor's disk scanner: the kernel reported a regular inode with zero remaining links, which the scanner incorrectly rejected as a hard link. The supervisor reproduced the same exception using the actual monitor writer; no hard link or scientific-kernel failure was involved. Three Random Forest fits and one ordinary-CNN fit were interrupted when admissions stopped. The ledger closed cleanly and all owned workers exited. No resource ceiling was exceeded.

The stopped-run audit checked SQLite integrity, all 404,814 registered job hashes, 68,068 journal events, every one of 17,101 complete operation receipts and artifacts, exact fit/prediction pairing, and the epoch selector's dependencies. It inventoried 55,310 retained files (2,170,190,041 bytes); inventory SHA-256 is `2cf4e30fd57ab28fb5f8a970beecf0c84df88a58b0f533f0e51fd207092ece2f`. Earlier high-water usage remains charged. The owner explicitly approved four exact replays and completion of the 186,648 unstarted fits; the new total attempt ceiling is 195,216, not a reset.

DeepSeek V4.1 Flash implemented the narrow scanner fix, R2 accounting/selector-import extension, and fresh private launcher. The supervisor reviewed all patches, normalized patch paths, added explicit rejection of unknown reuse kinds and a separate public-summary selector count, and corrected test formatting/strict iteration. The mutable-output scanner now counts observed zero-link regular inodes but rejects actual hard links and symlinks. Immutable artifact authentication remains unchanged. R2 requires the exact 8,550-fit identity set, the one declared selector and imported source dependencies; the selector is neither a fit nor a prediction nor a new attempt. R0/R1 accounting and failure latches remain intact.

Validation on the reviewed code: **339 U1/monitor/scanner tests passed in 98.00 seconds**, including real CPU/GPU kernel integration on invented fixtures; **six private launcher tests passed**. Package-wide Ruff passed. A fresh read-only actual-data admission check verified all 8,550 pairs and the completed selector against the frozen jobs and retained receipts in **41.81 seconds**. A separate **4.18-second** registry check reconciled 404,814 operations, 186,652 remaining fits and 3,354 calibrations. No fit was performed by these admission checks. The conservative carry into R2 is 4,000 active seconds and 4,400,000,000 artifact bytes, including prior usage and audit overhead.

This accepts the bounded recovery implementation, not benchmark completion. The reviewed source, launcher, approval, accounting, dependencies and parent audit must be sealed into a fresh permit; the immutable destination must remain absent until launch. Source fitting, calibration, held evaluation, statistical/visual review and final publication remain required. No later preprocessing experiment or automatic further retry is authorized.

## R2 retention audit and T314 resource-monitor correction

R2 resumed real field-data fitting and completed 4,411 additional source-fit/prediction pairs, three source-only epoch selections and three scalar calibrations. At 04:34:36 UTC on 2026-10-09, the supervisor stopped admissions because a worker's resource heartbeat was older than five seconds. Three Random Forest fits and one ordinary-CNN fit were interrupted. The ledger closed cleanly and all owned workers exited. This was a resource-monitor exception, not a recorded numerical-model failure. No final refits or held-test endpoints had completed.

The read-only retention audit passed in 54.65 seconds. It checked all 404,814 registered job identities, 52,681 event-chain entries, 25,929 completed operation receipts and their artifacts, and 81,243 retained files (2,888,860,980 bytes). The reusable evidence contains **12,961 source-fit/prediction pairs, four epoch selections and three scalar calibrations**. Inventory SHA-256 is `8156dfbce717720c5c580eec0d3df78cc74f73988579c9fcd9bc75f2068e9ca7`. Cumulative recorded usage reached 5,540.94 active seconds and 7,940,729,260 charged artifact bytes; subsequent audit overhead must also carry forward.

The supervisor reproduced the exact stale-heartbeat exception using the actual process adapter and an invented, non-training task that holds the child process's Python interpreter lock for seven seconds. The worker remained alive and subsequently returned successfully. This demonstrates the monitor's false-stop mechanism, but does not identify the exact worker or trigger in R2: those diagnostics were not retained. Contemporaneous kernel memory-reclamation activity and a fall in benchmark resident memory from 18.70 to approximately 9.23 GB support memory pressure as another plausible trigger. A particular trigger is not claimed proven.

DeepSeek V4.1 Flash implemented T314. The supervisor reviewed the complete patch and corrected the test's native-call argument type, finite-age validation and conservative integer-MiB accounting. The adapter now distinguishes a valid but stale sample from unhealthy, invalid, unavailable or locked telemetry. Its five-second freshness threshold is unchanged. A separate helper can replace only the former with explicitly labelled resource evidence: the verified CPU-only device contract, or a fresh per-process GPU-memory upper bound from the driver. The latter includes one additional MiB to cover display precision. Stale allocator values are never reported as fresh observations; missing process readings, query failures, dead workers and invalid samples still cause refusal. The driver metric represents context memory, not just tensor allocation; see the [NVIDIA monitoring reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html#processes).

Validation: **82 transport, resource-helper, controller, scanner and device tests passed in 15.07 seconds**. These are software checks on invented fixtures, not additional field-data fits. The real seven-second process regression completed successfully with the new helper. Ruff passed over the CI scope (`src`, `scripts`, `tests`). A broader lint invocation additionally reached an unchanged historical report generator with 54 pre-existing style errors; that file was not altered.

This accepts the monitor **component**, not a new recovery launch. A future launcher must wire the helper into whole-process-tree accounting, record fallback identity/age/source evidence, and preserve all existing time, memory, storage and inner-fit limits. The original R2 run and permit remain immutable. A request to preserve every completed operation and replay only the four interrupted fits is pending owner approval; no further scientific retry is authorized by this review. The corresponding total fit-attempt ceiling would become 195,220, with 195,202 unique slots unchanged. Reporting-code drafts remain private and are not benchmark results.

## T331/T332: approved eight-CPU throughput handover

This section supersedes the preceding pending-approval checkpoint. The owner subsequently granted bounded standing recovery and approved eight CPU workers after a recommendation of eight CPU plus one GPU and a 36-GiB whole-process-tree RAM guard; see the [execution amendment](../P08_U1_EXECUTION.md). The two-CPU and four-CPU continuations performed real field-data training. Before this handover, a five-minute four-CPU window measured 305 completed model fits/minute; this is a local throughput observation, not a fixed campaign rate.

DeepSeek V4.1 Flash authored the narrow resource-cap amendment and private continuation bridge. The supervisor independently reviewed both. Only two public runtime files change: the ledger's CPU/RAM ceilings and the controller's total worker ceiling, derived from CPU plus GPU limits. Model architectures, numerical kernels, input arrays, seeds, losses, selection and calibration remain unchanged. Regression coverage explicitly admits eight CPU workers and one GPU, refuses a ninth CPU or second GPU, and tests the 36-GiB boundary and persistent resource-stop latch. Supervisor corrections were limited to test formatting and private launcher description.

The isolated candidate passed **301 U1 tests in 216.55 seconds** and **20 private handover tests in 1.58 seconds**. Package-wide Ruff passed. The first isolated test collection lacked copied public fixtures; those fixtures were added before the successful run. This did not alter or stop scientific training. An exact two-file runtime-hash check also rejected an altered approved hash, an unrelated scientific edit and an extra runtime file.

At 15:13:35 UTC on 2026-10-09, the approved handover stopped the verified four-CPU supervisor at a complete fit/prediction boundary. All owned processes exited; the ledger closed cleanly without worker-shutdown errors. **Five in-progress source fits** were interrupted by this planned switch, not by numerical failure. The read-only closed-parent audit verified **64,666 completed operations**, including **32,203 source fits and their predictions**, 104 epoch selections and 89 scalar calibrations. It authenticated **201,785 retained files (7,145,265,041 bytes)** in 38.78 seconds; inventory SHA-256 is `a7c9064d999b05d14d65fac8579c87c25b925e5df65ead76eba600d19af9c72f`. Final-refit and other completed operations are also retained in the full stage inventory.

The new generation must retain all completed operations, add only these five interruptions to the existing replay history, carry cumulative resource usage and pass the real-data read-only admission check before launch. The completed benchmark, held-endpoint audit, statistical analysis, figures and report remain outstanding. This operational review makes no claim about a winning preprocessing method or classifier.

## T334/T335: approved twelve-CPU throughput handover

The owner approved twelve single-thread CPU workers plus one GPU worker, with a 44-GiB whole-process-tree RAM guard. DeepSeek V4.1 Flash implemented the narrow resource amendment and private continuation bridge; the supervisor independently reviewed both. Only one public runtime file changes: the ledger's CPU/RAM constants. The controller already derives the combined worker ceiling. Model architectures, numerical kernels, arrays, seeds, losses, selection and calibration are unchanged.

The isolated candidate passed **301 U1 tests in 155.34 seconds** and **20 private handover tests in 1.67 seconds**. The tests admit twelve CPU workers plus one GPU, refuse a thirteenth CPU or second GPU, and check the 44-GiB boundary and persistent resource-stop latch. The exact runtime-delta check requires the approved old/new ledger hashes and unchanged identities for all other 204 runtime files. The eight-worker scientific run remained active throughout candidate testing.

At 15:41:55 UTC on 2026-10-09, the approved controlled handover stopped the verified eight-worker supervisor at a complete fit/prediction boundary. All owned processes exited and the ledger closed cleanly without worker-shutdown errors. Nine in-progress source fits were interrupted by this planned switch, not by numerical failure. The read-only audit authenticated **83,375 completed operations**, including **41,453 source fits and paired predictions**, 169 epoch selections and 169 scalar calibrations. The complete stage inventory also retains final refits and other completed operations. The audit checked **258,153 retained files (8,019,754,223 bytes)** in 59.10 seconds; inventory SHA-256 is `6a3b62e8af88adc5432eb4dee4772c78e171d9c90f4dfe8681f5779465e89d26`.

The new generation must preserve all completed work, exact interrupted identities, previous replay counts, cumulative active time and artifact high-water usage. This review accepts the operational change, not benchmark completion or a scientific performance claim. Admission still requires the real-data retention check and a fresh bound permit; final held-endpoint, statistical, visual and disclosure reviews remain outstanding.
