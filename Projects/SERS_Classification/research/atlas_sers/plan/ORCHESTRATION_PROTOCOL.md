# Supervised implementation workflow

**Established:** 2026-09-24, at the project owner's request.

**Current phase:** P05 core implementation and bounded numerical smoke are accepted after the completed P04 ordinary CNN. The approved checkpoint recovery is exhausted at 35 total executions; [results and figures](../results/p05_smoke/P05_SMOKE_RESULTS.md) record the evidence. Full source-only development/evaluation require a separate reviewed runner and execution/resource authorization.

**Implementation worker:** OpenCode Go, `opencode-go/deepseek-v4.1-flash`.

**Scientific planner and reviewer:** the supervising assistant in the project conversation.

## Responsibility split

The supervisor maintains the scientific plan, defines bounded assignments, checks the worker's code and evidence, independently runs relevant checks, requests corrections, and decides whether a review gate passes. The worker reads the assignment, implements the approved slice, adds tests, reports exact commands and outcomes, and stops at its assigned boundary. Worker self-assessment does not constitute scientific acceptance.

The project owner retains decisions that materially alter research scope, hypotheses, data access, frozen specifications, practical margins, computational budget, or external publication. Approval of this working arrangement does not authorize all future experiments or an unattended, unlimited sweep. The owner additionally authorized committing and pushing substantive milestones after supervisor review concludes the work is valid; this is not authorization to publish unfinished or unreviewed outputs.

The supervisor normally edits planning/review documents, not scientific implementation. Code revisions are returned to the worker for implementation. Diagnostic commands and independent test execution remain supervisor responsibilities. Only one implementation worker writes within a slice; the supervisor does not edit the same implementation files concurrently.

## Operating loop

1. Issue a versioned task with objective, required readings, allowed files/actions, forbidden inputs/actions, deliverables, acceptance tests, and stop conditions.
2. Record starting repository state so pre-existing user changes are not attributed to the worker.
3. Use a dedicated OpenCode session with the requested model explicitly selected. Constrain permissions to the task; do not change global permissions, share the session publicly, or silently substitute a model/provider.
4. Monitor the active session and inspect its actual tool use. Keep transcripts local and outside routine publication. Send concise user updates at meaningful gates or when intervention is needed.
5. Review the diff and run the relevant tests independently. Audit master isolation, instrument exclusion, source-only selection/calibration, failure handling, deterministic seeds, immutable hashes, and endpoint definitions.
6. Record accepted work, rejected claims, unresolved decisions, commands, and the next allowed slice. Request corrections rather than accepting a worker's completion statement without evidence.
7. Proceed only within the approved scope. A stopped task is not permission to execute its suggested next command.

Supervision occurs while the orchestration session is active. Do not promise persistent observation after a conversation ends. Before handing control back, finish or stop the worker and report its state; no scientific run should be left unattended under an implied monitoring promise.

Hold the entire protected package fixed during governed validation or execution. These routines fingerprint planning documents and repository state as well as implementation files. Do not edit documentation, stage files, or commit while a repeatability test or scientific run is comparing protected identities. Finish edits first; validation of the fixed state follows.

## P05 gate sequence

| Gate | Worker deliverable | Supervisor acceptance boundary |
|---|---|---|
| A: readiness | Read-only code/protocol audit and unresolved-decision list | Verify evidence pointers; distinguish requirements from worker proposals |
| B: no-fit implementation | Planning CLI, deterministic candidate/role/pair/support manifests, unit tests, budget ledger | No model optimization, validation scores, held predictions, or scientific fit |
| C: loss/sampler implementation | Tested master-aware batches, contrastive/consistency losses, D0-compatible interfaces | Synthetic gradient, masking, determinism, support, and failure tests; no scientific training |
| D: source-only smoke | Explicitly authorized minimal source-role smoke and diagnostics | Fixed resources, no held-test access, retain failures; reject numerical or leakage defects |
| E: development | Separately authorized finite D0-M/D1/D2/D3 core and nested source-only G3 decisions | Reconcile candidates, costs, source-only advancement, collapse and unavailable runs; D4/D5 remain deferred |
| F: evaluation/reporting | Frozen-candidate P06 implementation and later approved execution | Identical evaluation roles, correct aggregation/uncertainty, reviewed figures and bounded claims |

Only Gate A is issued by the initial readiness task. Gate B and later tasks need a new explicit supervisor assignment within project-owner authority. Unspecified numerical choices must be resolved and versioned before use, not chosen from held-test results. Broader preprocessing remains P08; RBF/SOM remains the separate planned P14 extension.

## Repository and publication boundaries

- Preserve the existing working tree, especially uncommitted P14 planning changes and unrelated historical files.
- Work on the existing `main` branch. The worker may not commit, push, create/delete branches, reset, clean, or rewrite history. After a substantive milestone passes independent review, the supervisor may stage its exact reviewed paths, commit, push `main` without force, and verify the remote revision and available CI status. Exclude unrelated dirty files, local logs, and unfinished work. This implements the owner's explicit milestone-push authorization and overrides the historical draft-PR recommendation in CONTRIBUTING for this workflow. A failed push or CI check must be reported separately from successful local validation.
- Preserve frozen P00–P04/P13 contracts, completed results, and scientific run directories. New P05 artifacts have a separate namespace.
- Do not read credentials, authentication stores, unrelated home files, or unrestricted logs. Use the already configured provider.
- Do not install dependencies, alter global tool configuration, or enable external plugins without a justified scoped assignment.
- Published quantitative figures must use native TikZ and offline HTML from the same semantic data, with the existing PDF/PNG/hash requirements.
- If permission, cost, protocol meaning, or resource support is uncertain, report the limitation; do not silently widen scope.

OpenCode's documented [CLI](https://opencode.ai/docs/cli/) and [per-agent permissions](https://opencode.ai/docs/permissions/) support explicit model/session selection and task-scoped access. Local installed help and effective configuration must be verified before relying on a flag; permission settings are guardrails, not a substitute for review.

## First assignment

[P05-T001: read-only readiness audit](delegation/P05_TASK_001_READINESS.md). The expected output is an evidence-backed handoff for the no-fit implementation, not a training result.

## Current handoff

The [T001 review](delegation/P05_TASK_001_REVIEW.md) records accepted findings and corrected worker assumptions. [T002](delegation/P05_TASK_002_READINESS_IMPLEMENTATION.md) then authorized only a contract-only readiness inventory; its [review record](delegation/P05_TASK_002_REVIEW.md) lists code, tests, corrections, and the remaining boundary. Passing this small implementation slice does not pass the full Gate B role/support/budget audit or authorize Gate D/E training.

[T003](delegation/P05_TASK_003_SOURCE_SUPPORT.md) adds the metadata-only inherited-role support audit. Its [review](delegation/P05_TASK_003_REVIEW.md), [aggregate findings](P05_SOURCE_SUPPORT_AUDIT.md), and [design handoff](P05_DESIGN_HANDOFF.md) are the current continuation points. Read-only metadata access is explicitly supervisor-controlled; the worker implements against synthetic fixtures without access to the actual scientific dataset. Audited P04 role provenance is not authorization to run a new P05 experiment.

On 2026-09-25 the owner authorized the next bounded implementation/smoke and separately approved nested source-only selection per outer test split. The [core protocol](P05_CORE_PROTOCOL.md) and [contract](contracts/p05_core_contract.json) define the exact scope and ceilings. T006 covers synthetic-tested numerical primitives; T007 covers metadata-only registries. Worker sessions return patches from scoped source snapshots with tools denied, while the supervisor applies/reviews/tests them. Source snapshots are used because the client's intended exact-path read allowance was rejected; permissions were not broadened. Independent patch-authoring slices may run concurrently on disjoint allowed paths, but no worker directly edits files or executes scientific data. Full-package validation and actual smoke execution still require a fixed protected workspace and no concurrent edits.
