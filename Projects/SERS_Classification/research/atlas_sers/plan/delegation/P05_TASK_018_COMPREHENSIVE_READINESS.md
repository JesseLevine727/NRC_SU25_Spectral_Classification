# P05-T018 — comprehensive benchmark execution design

**2026-09-26.** The owner asked to proceed beyond the pilot and complete the comprehensive benchmark. This task implements that request within the locked D0-M/D1/D2/D3 scope. A concrete proposal of 48 hours cumulative execution, 100 GiB private artifacts and 4 GiB allocated CUDA has been submitted for owner confirmation; it is not yet a numerical launch permit. Preprocessing experiments, D4/D5, adaptation, P14 and definitive P11 inference remain separate.

Worker: OpenCode Go / `opencode-go/deepseek-v4.1-flash`. All tools denied; public source snapshots only. Do not access scientific arrays, private metadata, credentials or previous prediction rows. Do not run scientific fits, commit, push or edit frozen contracts. Supervisor controls application, testing, execution and publication. Return a concise design review as an add-file patch for `plan/delegation/P05_TASK_018_WORKER_DESIGN.md`; no other file edits in this slice.

## Required design

**Owner approval, 2026-09-26:** the owner subsequently selected “Approve these limits and complete the benchmark” for comprehensive D0-M/D1/D2/D3 execution, source-only selection and final held-out evaluation, with 48 hours cumulative scientific execution, 100 GiB private artifacts and 4 GiB allocated CUDA. This supersedes the pending resource decision above, not the frozen scientific settings. The separate comprehensive permit records the bounds; implementation still requires supervisor acceptance before launch.

Use the authenticated 14,940-slot development ledger across 320 outer contexts, comprising 861 inherited and 384 guard units. Reuse the 36 accepted pilot slots, leaving 14,904 unstarted inner fits. Reuse means authenticated checkpoint, history, validation-logit, source-role, code and immutable-manifest evidence, not copying scores without their provenance. Shared original slot leases prohibit duplicate fitting. No automatic retries. No new loss weights, architecture, optimizer, augmentation, preprocessing or seed choices.

Describe bounded implementation slices and exact reusable APIs for:

1. Full metadata preparation, pilot-evidence import and complete slot reconciliation.
2. Serial, finite source-validation execution with durable per-slot journals, 120 seconds per fit, one GPU process, private storage/time caps, actual update accounting, saved-checkpoint acceptance, common random-stream prefix and sparse-equivalence audits. Do not keep 14,940 model states in RAM. A completed slot may be reused; an incomplete consumed slot may not be retried. Infrastructure failure stops execution and preserves the evidence. Predictive collapse is recorded, not disguised as an infrastructure error.
3. Context-local G3 decisions and per-seed refit epochs, exclusively from that context's source roles. All 320 decisions must be persisted and frozen before outer-test prediction. Master-CV-only contexts return D0-M; fixed D3 remains a control. No global winner or test-informed choice.
4. A new fixed-epoch P05 refit kernel using the identical frozen all-master numerical recipe, with no validation or test input in its training API. P04's fixed trainer has a different sampler/objective and is not a substitute. Refit D0-M, selected recipe and fixed D3, aliasing identical specifications rather than executing them twice. At most 2,880 refits and 2,880 source-only scalar calibrations. Inherited validation logits, not guard or test logits, supply equal-master temperature fitting. Freeze/checkpoint models and calibration before outer prediction.
5. Complete three-seed probabilities, instrument-balanced aggregation, exact P03/P04 comparison alignment and coverage/failure reporting. Preserve existing per-spectrum, instrument-view and master estimands. P11's final hierarchical inference is not silently replaced by a simpler interval or declared complete by this benchmark.
6. Native TikZ/offline HTML figures with matched semantic data: source-selection effects, paired domain/classical/deep comparisons, scatter of source versus held performance, calibration/reliability, and support/learning diagnostics. Any uncertainty method must be separately pinned before reading new held outcomes.

## Long-run workspace isolation

The current runner fingerprints package contents and Git state. A multi-hour run should use an immutable, private, sparse local checkout of the reviewed commit on local `main`, without creating a new remote branch or touching the owner's unrelated worktree. Private arrays/artifacts remain outside that checkout. Record the original repository commit and code-tree/dependency hashes. The active execution checkout remains fixed through all protected checks. Worker activity and report edits must not mutate that checkout. Outline how existing provenance/path guards can operate unchanged on it.

## Acceptance and stop

Identify actual interfaces, required source/result fields, incompatibilities, unresolved scientific decisions and failure modes. Separate implementation choices from changes needing owner approval. Prefer small modules and synthetic tests over a new monolithic runner. Do not claim tests or experiments were run. Keep the response bounded: approximately 120 lines, not implementation code. Stop after this design handoff.
