# P05 completion stages: implementation and execution review

## Scope

This record continues the [comprehensive review](P05_COMPREHENSIVE_REVIEW.md) under the approved [execution protocol](../P05_COMPREHENSIVE_EXECUTION.md). DeepSeek V4.1 Flash authors bounded patches from public source snapshots with tools denied. The supervisor applies, reviews, corrects and tests each patch. Worker completion messages do not constitute acceptance.

These stages preserve the registered D0-M/D1/D2/D3 experiment. They do not change preprocessing, model architecture, loss weights, seeds, source-selection rules or the execution ceiling. The corrected permit pin in the execution protocol remains authoritative. All scientific arrays, prediction rows, identifiers and checkpoints remain private.

## Source-stage launch

The source-only runner was released on `main` at commit `2f91a1b40e2bb19c54d56a1174bdf70facfff27e`. The release passed 827 local tests and [GitHub CI](https://github.com/JesseLevine727/NRC_SU25_Spectral_Classification/actions/runs/36280639252). A separate, fixed private checkout of that commit runs the 14,904 new source fits. Subsequent implementation edits occur in the original working tree, not the active execution checkout. The 36 accepted pilot fits occupy their original slots and are reused.

The first completed unit contained 12 fits and 2,140 optimizer updates. Its fits completed 34–62 epochs; the longest took 5.392174910 seconds and peak allocated CUDA memory was 144.84375 MiB. The source runner passed its per-fit checkpoint/logit checks and per-unit shared-randomness and sparse-equivalence checks. A separate supervisor audit verified the closed unit's complete file manifest and all 12 completion records. That audit performed no fitting or prediction. These measurements describe one initial unit, not the maximum cost or predictive performance of the full benchmark.

Source training is in progress at this review checkpoint. No dataset-level G3 decision, final refit, scalar calibration or new held-out prediction has been produced. Scientific execution remains incomplete until the full registered coverage and later acceptance gates pass.

## T026: freeze all source-only decisions

The new selection-freeze boundary authenticates the completed source receipt and manifests, compares the stored full and compact ledgers with the authenticated registries, and reconstructs every selector record from its saved execution summary. Reused pilot records resolve to the original pilot directory. All 320 context-local decisions and refit specifications must be frozen before any outer prediction.

The corrected implementation passed 30 synthetic filesystem/integrity tests. The tests exercise real canonical persistence, manifest verification and storage accounting while replacing scientific inputs and the already-tested pure refit-plan builder with fixtures. They reject changed source/pilot files, missing or duplicate selector records, mismatched summaries and registries, occupied output stages, invalid cumulative time, storage failure and late deadlines. A successful fixture freezes a plan without reading arrays, checkpoints or logits and without starting a fit.

Review moved cumulative-time validation ahead of large integrity scans, restored a dropped compact-ledger check, required the exact strategy-alias ceiling and charged close-stage reconciliation before recording elapsed time. Static writes reserve their actual serialized size plus headroom. A failed consumed stage retains its evidence; there is no automatic retry. These tests accept the implementation boundary, not the actual source evidence, which is still being produced.

## T027: source-only refit input and checkpoint boundary

The first refit I/O draft was rejected before application. It could omit the explicit source UID list, conceal a reversed class order through sorting, overwrite result identities in saved metadata and accept completion without checking the saved summary.

The corrected implementation requires a content-addressed refit specification, the registered seed and epoch range, the current permit, exact source UID/hash agreement and three ordered classes. It checks outer-source membership, held-instrument exclusion and physical-master separation before loading the frozen array container. Only source-fitting rows and source-derived noise metadata are exposed to training. Result identity is checked before persistence, and completion requires consistent saved metadata, finite recorded losses, exact update counts and a verified terminal checkpoint. Correctly identified failed or partial results remain persistable.

Synthetic acceptance tests are pending at this checkpoint. No scientific refit is authorized merely by applying this module. The fixed-epoch numerical kernel has separate CPU/CUDA parity evidence in the earlier comprehensive review.

At the subsequent acceptance gate, all 50 CPU tests passed. They cover specification and role identity, reordered array-container rows, checkpoint round trips, partial failures, saved-summary tampering, invalid losses and update counts. A separate read-only audit checked source input preparation for all 320 actual contexts in 32.666182443 seconds; the largest source role contained 189 spectra. That audit used synthetic 30-epoch specifications, not actual selected refit specifications. It performed zero fits, scalar calibrations or predictions. The array-container audit confirmed that source rows must be indexed by their recorded identifiers, not by assuming a sorted container order.

## Remaining bounded implementation assignments

1. **T028: refit and calibration execution.** Authenticate the frozen selection stage, reserve each unique refit once, inherit the same cumulative resource counters, fit for the registered number of epochs and retain terminal checkpoints and source-only calibration states. Calibration uses only that context/recipe/seed's inherited source-validation logits, with the unchanged equal-master temperature rule. Complete and freeze this entire stage before held-out prediction. Reuse identical selected/control specifications; do not retrain aliases. Test partial-step accounting, failures, storage/time limits, checkpoint reload, calibration provenance and complete alias coverage.
2. **T029: held-out evaluation and aggregation.** Load only frozen models and temperatures. Predict the registered outer rows once per unique specification, reconcile all three seeds and aliases, average calibrated seed probabilities before scoring, and retain the original spectrum and instrument-balanced master endpoints. Freeze new predictions before loading existing P03/P04 prediction evidence for UID-aligned comparisons. No optimizer, new selection rule or test-informed model change belongs in this stage.
3. **Reporting and release.** Report the predeclared descriptive comparisons, support/coverage, source-selection outcomes, collapse diagnostics, calibration and actual costs. Generate native TikZ and offline HTML from matched aggregate data, with PDF/PNG previews. Review numerical claims and rendered figures before the final main-branch push. Definitive P11/G4 inference and the deferred experimental branches remain outside this benchmark.

The active training process must remain the only scientific CUDA process. Subsequent synthetic checks run on CPU with CUDA hidden while training is active. Every later scientific stage requires its own reviewed fixed checkout and an authenticated receipt from its completed prerequisite; a running or partially complete source stage is not sufficient.

## Completion-boundary release gate

The isolated public snapshot passed 903 CPU tests, with four CUDA tests deliberately skipped while the scientific CUDA process remained active. Public-boundary validation checked 524 files and linting passed. An earlier run in the editable worktree passed 902 tests but failed the verified-skip integration test because implementation files changed between its two source fingerprints. The differing recorded code hashes identified that test-isolation error; the fixed-snapshot rerun passed without changing the scientific code or relaxing the test. The release contains T026, T027 and this review record; unfinished later-stage drafts are excluded.
