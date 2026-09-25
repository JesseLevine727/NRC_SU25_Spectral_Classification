# Supervisor review — P05 core implementation and bounded smoke

**Date:** 2026-09-25. **Status:** implementation accepted for the single bounded smoke; no scientific-data fits performed at this checkpoint.

## Authority and scope

The owner requested locking the sparse-split, sampling/loss and finite-budget rules, delegating implementation to DeepSeek, and running a small reviewed smoke. The owner then explicitly chose nested source-only model selection independently within each outer test split. [P05_CORE_PROTOCOL.md](../P05_CORE_PROTOCOL.md) and its contract record the dated core amendment without rewriting frozen P00–P04/P13 artifacts. Authorization is limited to the finite smoke, not the later 17,820-fit development/refit ceiling.

The supervisor confirmed the maintained package was clean at starting commit `7a3b7d7c`; unrelated historical worktree changes remain out of scope. The same completed P01/P04 planning pointers and input pins were found. Outcome-blind metadata confirms 128 T3 pseudo-domain contexts, 132 T3 master-CV fallback contexts, and at most 19 masters per fitting role. Thus every role fits the all-master/two-view batch cap (at most 38 rows). The four deterministically selected smoke roles have 13, 9, 13, 4 masters and 123, 94, 88, 4 spectra. The sparse role has two anchors without another same-chemical spectrum.

The unchanged three-class P04 model has 208,691 parameters; the single 64→64 auxiliary linear head adds 4,160, for 212,851. These counts were independently checked using the installed torch runtime. GPU availability was checked, but availability alone is not the eventual resource/launch gate.

## Worker supervision

OpenCode 1.18.32 is used with the explicitly requested `opencode-go/deepseek-v4.1-flash`, external plugins disabled, session sharing disabled, and per-process configuration only. The first T006 session attempted six reads (more than the assigned four), but all were denied by the client's path-permission matching; it delivered no code and had no data access. This deviation is recorded rather than hidden. The continuation receives the complete four authorized public source snapshots in its prompt, with every tool denied. No global permission change, alternative model, or credential access is used.

T006 and T007 author disjoint patch sets from approved source snapshots. The supervisor applies only allowlisted paths, independently validates them, and returns defects to DeepSeek. Worker statements of correctness or completion do not pass a gate by themselves.

## Initial implementation review

The initial T006 code passed its own 35 synthetic tests but failed independent counterexamples: paired consistency accepted contradictory chemical labels for one physical master, and a NaN contrastive temperature produced a NaN rather than rejection. Its paired tests themselves used contradictory labels. Review also identified delimiter-based rather than canonical sampler hashes, epoch-zero acceptance, incomplete finite/schema/parameter checks, weak single-instrument fixtures and two Ruff findings. The worker is correcting these and adding non-vacuous regressions; the initial green test count was not accepted as sufficient evidence.

T007's first response reached its combined output/reasoning limit mid-module, without any complete patch. It was not applied. Delivery was split into a compact complete module followed by tests. The entire locked contract is authenticated by its independently computed canonical digest rather than duplicating a large contract literal in the implementation. T008 authors the in-memory numerical kernel separately; it has no data loader, filesystem writer or fitting authority.

## Intermediate acceptance gates

The corrected primitives pass 53 synthetic tests, including finite-difference gradient checks, multi-positive weighting, unequal pair-count master averaging, canonical sampler identity and invalid-metadata rejection. The planner adds 42 passing tests. Independent reconstruction from the pinned metadata selects the expected four roles, registers 34 smoke executions and 5,585 eligible unordered positive observation pairs, and reconciles 14,940 later inner-fit slots. All 384 guard units retain three classes in both roles; no guard slots are excluded in this pinned population. Shuffling all three input tables produces the identical full-plan canonical digest `a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37`. The 309 smoke-role observation memberships are not 309 independent physical samples.

Review corrected T007's erroneous restriction of the sparse source role to the development phase; its four-master source fitting role belongs to a held-evaluation context, but no outer-test row is used. T008's first kernel used epoch zero against the locked one-based sampler, incompletely retained terminal states on failure, and lacked several deterministic/resource checks. These were returned to DeepSeek, corrected and tested. The kernel now passes 24 synthetic tests. Independent GPU diagnostics additionally exercise all four recipes on dense and sparse synthetic roles: initial backbones and sampling/augmentation/pair digests agree across recipes, all gradients are finite, and sparse D0-M/D2 and D1/D3 terminal hashes are exactly equal. These are synthetic numerical checks, not scientific-data fits or performance evidence.

Before adding the kernel tests, the fixed-package full regression suite passed 424 tests in 139.45 seconds. Ruff passes after mechanical formatting/import fixes. A final full suite is still required after runner and figure acceptance.

The initial T009 runner patch was rejected before application: it invented import paths, misread inherited state schemas, confused new and inherited role IDs, treated table lists as counts, mismatched replay identifiers, incorrectly expected sparse SupCon to be absent, and could compare a replay to itself. The initial T010 figure patch was also rejected before application because it invented role labels and omitted essential TikZ legends/claim boundaries. Both workers received precise correction requests. Neither draft has accessed actual spectra or run a scientific fit. The contract and authorized scientific budget remain unchanged.

## Implementation acceptance before scientific execution

The corrected boundary passes 22 tests; the figure exporter passes 29. All four actual source-fitting arrays, QC orderings and sampler capacities were independently validated without optimization. The immutable private plan was created with the same previously verified digest. The smoke lease does not yet exist. Both synthetic figure PDFs compile, and the native TikZ layout was visually inspected; the HTML regression preserves all four subplot labels and its sparse-support disclaimer.

The final fixed-package suite passes **499 tests in 134.60 seconds**. Ruff passes, and the clean public-package audit passes. One preceding full run had 498 passes and one fixture-import failure after the supervisor incorrectly suggested changing the worker's package-qualified import. Inspection confirmed `tests/__init__.py` exists; restoring the worker's original import resolved it. This was a test-import defect, not a protocol or numerical change. Worker transcripts remain outside the repository and no worker process remains active.

The accepted implementation is authorized only for the [bounded execution handoff](../P05_CORE_EXECUTION.md). Final scientific smoke acceptance still requires the actual 34 records, numerical/replay/control checks, protected-state verification, checkpoint audit and reviewed diagnostic figures. No claim of acquisition invariance or predictive improvement follows from this implementation gate.
