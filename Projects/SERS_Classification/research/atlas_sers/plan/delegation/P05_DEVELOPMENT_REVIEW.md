# Supervisor review — source-validation implementation and bounded pilot

**Date:** 2026-09-25. **Status:** implementation accepted, approved 36-fit pilot completed, independent post-run audit passed. No expansion is authorized. The implementation/preflight observations below precede the scientific execution recorded at the end.

## Execution authority

The owner approved the [36-fit pilot](../contracts/p05_development_pilot.json), followed by review before expansion. It uses one metadata-preselected source fitting/validation unit per station, four fixed recipes and three seeds. The bounds are 30–200 epochs, patience 20, four sampled batches per epoch, 120 seconds per fit, 90 minutes overall, 4 GiB allocated CUDA and 2 GiB private output. No automatic retry, G3 decision, calibration or held-test prediction is authorized. These executions consume original slots within the existing 14,940-inner-fit ceiling.

The original eight-epoch smoke, failed checkpoint attempt and approved recovery remain unchanged. That completed stage consumed 35 executions and 1,120 updates; it is not part of the new 36-fit permit. Starting implementation revision: `0bf2dd8620ecf030384354b6276f7b62b42cebd1`. Unrelated historical worktree changes are excluded from this milestone.

## Implementation and review

OpenCode Go / `opencode-go/deepseek-v4.1-flash` authors the implementation from scoped public source snapshots with all worker tools denied. The supervisor applies allowlisted patches, checks the numerical and scientific boundaries, runs tests, and returns substantive defects for correction. No private spectra or metadata are supplied to the worker. CLI event delivery stalled on several completed continuation requests; exact-session exports confirmed completed Flash responses with no tool calls. Only those finished worker processes were terminated. Later requests use the session's actual project directory. Per-process snapshots and automatic updating are disabled; global configuration is unchanged. OpenCode documents the [snapshot setting](https://opencode.ai/docs/config/#snapshot).

T013 adds a source-validation trainer without changing the accepted smoke kernel. It retains recipe-independent initialization, sampling, augmentation and dropout streams. Checkpoint selection uses validation balanced accuracy, then negative log likelihood, then the earliest epoch. The 30-epoch minimum constrains training duration, not the selected checkpoint. Best and terminal states are retained separately. Failure records preserve actual updates, partial histories and available states; an individual zero-gradient batch is recorded rather than treated as a failure after convergence.

Review corrected an omitted augmentation import, incomplete source-noise diagnostics, loss of the terminal state after restoring the best model, and resource checks that omitted finalization. Synthetic tests cover all four recipes, the 200-epoch ceiling, a 35th-epoch optimum followed by stopping at epoch 55, tie rules, sparse equivalence, source-only noise, callback/resource/numerical failures and actual filesystem checkpoint round trips. Two additional worker fixtures required correction: an initially flat trajectory would stop before its claimed epoch-35 optimum, and mocked per-epoch metrics cannot be equated with genuinely recomputed final metrics.

T014 implements context-local G3 selection without filesystem access or model fitting. Decimal-string fractions preserve the exact 0.02 boundary; this avoids changing the scientific margin through binary floating-point subtraction. The selector retains failed, missing and excluded evidence, verifies the complete recipe/seed ladder, and distinguishes a fallback identity from usable completed evidence. Review rejected excluded records carrying scores, added the kernel's reason-code mapping, and corrected one missing baseline entry in a test fixture. No G3 selection has been applied to dataset outcomes.

T015 reconstructs and validates all source-role memberships, guard units and canonical slot identities. Review corrected epoch lookup against the real contract, fitting-only sampler capacity, missing support pins and duplicate selection-role identities. The actual frozen plan reproduces canonical ID `a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37`; the development ledger is `P05DEV-8bf60eeca36d4b4663441eda`.

## Metadata findings and focused checks

The authenticated ledger contains 320 outer contexts, 861 inherited selection units, 384 guard units and 14,940 eligible fit slots. No guard unit is excluded in this population. The largest inner fitting role contains 14 masters, and the maximum sampled batch is 28 spectra, below the 48-spectrum cap. The minimum/maximum scheduled inner optimization counts are 1,792,800/11,952,000 updates. These are schedule bounds, not consumed computation.

The pilot plan has canonical ID `7c57ba72a98ee018b30973dc35003ef9232babee7d8965fcbfb77b10ada6ca9c`. Its selection is independent of validation scores:

| Station | Validation mode | Fitting masters | Validation masters | Maximum sampled batch |
|---|---|---:|---:|---:|
| CWA | source pseudo-instrument | 12 | 6 | 21 |
| Pills | source-master CV | 11 | 5 | 22 |
| Surfaces | source pseudo-instrument | 7 | 11 | 12 |

All three pilot fitting roles contain cross-instrument measurements of at least one common master. Auxiliary support is not a criterion for dropping a role. Pills source-master validation must not be described as unseen-instrument validation.

Focused validation passed 113 trainer/selector tests and 34 ledger tests. A synthetic-only CUDA fit completed 30 epochs and 120 updates, selected epoch 6, and used 114,105,344 allocated bytes. The actual-data metadata/resource preflight passed with no arrays loaded and zero fits started; a real checkpoint serialization probe also passed. Available CUDA memory exceeded the required 5 GiB.

An initial full-suite invocation omitted the already provisioned external Parquet dependency path: 515 tests passed and eight failed with missing-pyarrow errors. The affected files then passed all 15 tests with the correct path; no dependency was installed. The first full implementation suite passed 688 tests; the final fixed-code suite, including pilot orchestration and failure tests, passed **718 tests in 224.61 seconds**. Ruff passed across all package sources, scripts and tests. The direct working-directory public audit encounters pre-existing ignored report compilation logs; publication validation inspects only the exact proposed public files, leaving those local logs untouched.

## Pilot runner and figures: accepted implementation

T016 passed 30 tests covering permit/plan authentication, source-role boundaries, exclusive permit and shared slot leases, resource limits, real checkpoint reload predictions, partial-failure retention and protected provenance. A 36-fit synthetic orchestration passed through the real serializer and reloaded all 72 best/terminal checkpoints. A separate genuinely trained synthetic model passed saved-checkpoint prediction acceptance; corruption of its saved checkpoint was rejected. Metadata-only tests remain usable without torch; neural/serialization tests explicitly skip without it.

Review rejected acceptance helpers that checked only initialization rather than all shared random-stream prefixes, and helpers that recomputed from memory rather than the saved checkpoint. Runner corrections also removed a duplicated lease-acquisition block, added the missing seed identity, moved update accounting before possible persistence/close failures, retained zero-start load failures, and included finalization in resource/provenance checks. Deliberate failures now stop subsequent executions and preserve exact returned update counts, or an explicitly labelled lower bound when the kernel itself raises unexpectedly. Existing slot leases prohibit reuse even under a changed permit. No scientific launch follows from a worker's completion statement.

T017 passed 18 figure tests, including actual native TikZ compilation. The exporter requires all 36 complete fits and retains individual seed trajectories and stopping points. Its CSV, HTML and native coordinates use the same source-validation metrics; no raw spectra, master IDs or row predictions are published. Review corrected overlapping labels and a decimal-roundtrip test fixture, and clarified that the pilot does select epochs but does not select a recipe. Actual-data visual inspection remains a post-run gate.

The final metadata/resource preflight passed with the same pilot-plan digest, more than 5 GiB free CUDA memory, a successful checkpoint probe, no arrays loaded and zero fits started. During execution, the representation container is authenticated in full, but only allowlisted source fitting/validation rows enter optimization or scoring. No outer-test prediction or metric is computed.

## Execution boundary at the implementation checkpoint

After accepted code is committed, the single approved pilot may run under an external timeout. Its records and native TikZ/offline HTML diagnostics require a separate post-run audit. Three initialization seeds on one split per station do not establish generalization or authorize a recipe change. Expansion requires a subsequent review and explicit execution decision.

## Completed scientific pilot and independent review

The supervisor committed and pushed accepted implementation `b82daec3c58671b45245243f60311f6063fc2729` before launch. The exact 36-slot plan ran once on CUDA under a 5,400-second external watchdog. All 36 fits completed, with 6,880 updates and no retries, in 141.664 seconds. Fits stopped after 30–122 epochs; no fit reached 200. Selected checkpoints occurred at epochs 1–102, including 21 before epoch 30. These are observations under the locked stopping rule, not a reason to shorten or alter it.

An independent read-only audit reloaded and hash-checked all 72 saved states, reproduced all 36 saved source-validation logit arrays exactly from the on-disk models, recomputed BA/macro-F1/NLL, reconstructed earliest legal stopping and best-epoch choices, checked nine shared-stream groups, all 36 slot leases, the journal and all 185 run-manifest entries. Within-run protected provenance matched. A before/after file inventory confirmed all 187 original core-artifact files unchanged. Numerical success is distinct from predictive usefulness: one D1 and two D3 CWA selected checkpoints predict a single class.

The native TikZ/vector PDF/PNG and offline HTML exports use 1,720 actual epoch records. Both native figures and both browser-rendered HTML files were inspected. The strict aggregate export excludes private spectra, sample/observation identities, row predictions and checkpoints. Source-validation results, resource use, caveats, all per-fit aggregates and immutable identities are in the [pilot report](../../results/p05_pilot/P05_PILOT_RESULTS.md).

Post-run visual review found partially clipped HTML scatter markers at training BA = 1. The supervisor disabled marker clipping without changing coordinates or axis limits, added a regression test and reran all 19 figure tests successfully. Final exports retain the same semantic-data digest. This display-only correction occurred after the protected training run; model code, selected checkpoints and scientific results were unchanged. Hosted CI passed for the execution commit.

**Acceptance:** the authorized pilot and its post-run review are complete. Source-validation effects differ by station; they do not establish global superiority, instrument independence or preprocessing adequacy. The pilot permit is exhausted, and 36 registered inner slots must be reused by any later runner. The next action is a separately reviewed scope/resource decision for wider nested source-only development, not an automatic full-ladder launch or G3 decision.
