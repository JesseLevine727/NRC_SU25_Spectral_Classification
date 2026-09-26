# P05-T018 comprehensive benchmark — worker design review (corrected)

Status: design handoff only. No fits run, no arrays read, no numerical launch,
no test or experiment claims. The owner has NOW approved the full benchmark,
including held evaluation and the 48 h / 100 GiB / 4 GiB CUDA envelope. All
original scientific settings are unchanged. Supervisor controls application,
testing, execution and publication.

## 1. Scope

- Locked: D0-M/D1/D2/D3, 320 outer contexts, 14,940 inner slots (861 inherited
  + 384 guard), 36 accepted pilot slots reused, 14,904 unstarted.
- Separate: D4/D5, adaptation, P14, preprocessing selection, definitive P11
  hierarchical inference.
- No new loss weights, architecture, optimizer, augmentation, preprocessing or
  seed choices. Seeds {20260805, 20260817, 20260829}. No global winner.

## 2. Reusable APIs (from snapshots)

Metadata/ledger: `p05_development_plan.build_development_ledger`,
`RECONCILIATION_KEYS`; `p05_core_plan.slot_identity`, `canonical_identity`,
`build_guard_fold_roles`; core helpers via `p05_core_run` (`_authenticate`,
`_load_contract`, `_build_plan`, `_minimal_plan_checks`, `_manifest_uids`,
`_load_representation`, `_noise_frame`, `_capture_provenance`,
`_assert_protected_identity`, `_assert_artifact_location`, `_atomic_write`,
`_mkdir_exclusive`, `_read_json/_read_bytes`, `_save_state`, `_write_manifest`,
`_Recorder`, `_select_device`, `_configure_environment`, `_is_hex64`, `_canon`).

Selection/epoch: `p05_selection.select_context`,
`p05_selection.inherit_refit_epochs`.

Execution kernel (reused unchanged): `p05_development.train_development_fit`;
`p05_development.DevelopmentFitResult`; `p05_pilot.check_completed_result`,
`check_shared_prefixes`, `check_sparse_equivalences`, `check_sparse_support`,
`persist_result`, `persist_checkpoints`, `open_history_recorder`, `execution_id`.

Aggregation/comparison/calibration: `p04_results.ensemble_seed_predictions`,
`endpoint_metrics`, `endpoint_coverage`, `summarize_endpoints`,
`learning_curve_summary`; `p04_comparison.compare_endpoint_metrics`,
`master_clustered_paired_bootstrap`, `normalize_classical_predictions`;
`p04_runtime._state_hash`, `_metric_values`, `_master_equal_calibration`;
`classical.fit_temperature`, `apply_temperature`,
`instrument_balanced_master_probabilities`.

## 3. Corrected interface findings

1. Result adapter is identity-only. `DevelopmentFitResult` already exposes every
   `COMPLETE_METRIC_FIELDS` value (`best_epoch`,
   `best_validation_balanced_accuracy`, `best_validation_nll`,
   `best_validation_macro_f1`, `best_validation_predicted_class_count`) plus
   `status`, `recipe`, `seed`, `role_id`, digests, logits, classes, uids and
   support counts. No metric re-derivation is needed. The adapter adds only the
   slot/context/unit identities the selector requires: `slot_id`, `context_id`,
   `selection_unit_id`, `slot_kind`, `fitting_role_id`, `validation_role_id`,
   and renames `recipe` -> `recipe_id`. Never copy scores without their
   authenticated provenance.
2. P04 fixed trainer is not the P05 refit. `p04_runtime.train_fixed_epochs` uses
   `CompactSERSClassifier`, spectrum-frequency class weights, `p04-aug-v1` and
   permutation sampling. P05 requires `AcquisitionClassifier` plus the
   all-master two-view sampler and equal-master objective. A new fixed-epoch
   kernel is required; only checkpoint/hash scaffolding is reusable.
3. Storage accounting must not rglob per epoch. The pilot's
   `_enforce_storage_cap` -> `_directory_size_bytes` rescans the whole
   `p05development` namespace on every `on_epoch` callback, i.e.
   O(total files x epochs), which is inappropriate for 14,940 fits. Require a
   persisted running byte counter incremented by each artifact's size at write
   time, plus a full `rglob` reconciliation every 64 completed units (and at stage/final acceptance), never per epoch.
4. Pilot helpers carry 36-fit constants indirectly. Reuse only
   `train_development_fit` and the 120 s per-fit limit semantics. Do not reuse
   `p05_pilot.run`, `preflight`, `MAXIMUM_FIT_EXECUTIONS`,
   `MAXIMUM_OPTIMIZER_STEPS`, `MAXIMUM_TOTAL_SECONDS` or
   `PRIVATE_STORAGE_CEILING_BYTES`. Derive a new aggregate optimizer-step bound
   from `later_core_plan` min/max epochs x `BATCHES_PER_EPOCH` (4) x slots, and
   a 100 GiB private ceiling.
5. Collapse stays status complete. `DevelopmentFitResult.collapse` is a bool
   diagnostic; `status` remains `complete` and
   `best_validation_predicted_class_count` is in `[1, 3]`, which the selector
   accepts. Record collapse as a diagnostic, never as a new status and never
   relabelled as infrastructure failure.
6. Lease collision on pilot reuse. Slot leases are keyed by
   `(contract_sha256, core_plan_id, slot_id)`. Treat the 36 existing leases and
   pilot run evidence as immutable inputs; reserve only the 14,904 unstarted
   slots once. No automatic retries.
7. Normalization mismatch. `p04_results.normalize_p04_predictions` hardcodes
   `model_id="D0-ERM"` and P04 experiment ids; `endpoint_metrics` /
   `ensemble_seed_predictions` expect `probability_0..2`, `class_vocabulary`
   and identity columns `context_id, experiment_id, domain, held_instrument,
   outer_repeat, outer_fold, station, candidate_id`. P05 needs its own
   normalizer.
8. Calibration input. `_master_equal_calibration` needs per-master `logit_0..2`,
   `master_sample_id`, `true_label`. Inherited source-validation logits only;
   reject guard and outer-test logits. `fit_temperature` is the only primitive.
9. Isolation. `capture_provenance` shells to git in `repository_root` and hashes
   the public tree; the active project must live inside the immutable sparse
   checkout, with the original commit and code-tree/dependency hashes recorded
   and `_assert_protected_identity` unchanged.

## 4. Bounded slices (small modules, synthetic tests)

- S1 ledger/slot reconciliation: build ledger, assert reconciliation, classify
  the 36 reused slots by `slot_id`, emit 14,904 unstarted.
- S2 pilot-evidence import: authenticate pilot run and 36 leases, re-verify
  checkpoint digests via `_state_hash`, build selector records with identity
  enrichment. No re-fit.
- S3 serial runner: one GPU process, durable per-slot journal (fsync pattern),
  120 s per fit, incremental storage accounting plus periodic full
  reconciliation, time/CUDA caps, stop-on-infrastructure-failure preserving
  evidence. Persist and drop each state; never retain 14,940 states in RAM.
- S4 context-local selection: partition by context, call `select_context`,
  persist 320 frozen decisions and `inherit_refit_epochs`; master-CV returns
  D0-M; fixed D3 is a control. Freeze before any outer-test prediction.
- S5 P05 refit kernel: fixed-epoch all-master recipe, no validation or test
  input in its training API; alias identical specifications; at most 2,880
  refits and 2,880 source-only scalar calibrations; freeze/checkpoint before
  outer prediction.
- S6 aggregation/comparison/figures: three-seed probabilities,
  instrument-balanced aggregation, P03/P04 alignment, coverage/failure report,
  native TikZ and offline HTML with matched semantic data.

## 5. Failure modes

- Existing lease ignored -> duplicate fitting; fail closed.
- Incomplete consumed slot retried -> prohibited; stop and preserve.
- Non-finite history or digest mismatch -> stop, keep evidence.
- Global rglob per epoch -> rejected; incremental + periodic only.
- Guard support unavailable -> reason-coded exclusion; disallow G3.
- Missing/failed fit -> candidate blocked; denominators retained.
- Predictive collapse -> recorded as complete diagnostic.

## 6. Owner approval vs implementation

- Approved: full benchmark, held evaluation, 48 h / 100 GiB / 4 GiB envelope.
- Still separate: any retry or amendment, P11 final inference, D4/D5,
  adaptation, P14, preprocessing selection, and any new uncertainty method
  pinned before reading new held outcomes.
- Implementation choices within locked scope: module boundaries, identity
  adapter, journal format, incremental storage accounting, synthetic tests,
  figures.

## 7. Unresolved scientific decisions

None may be resolved in this slice. No global winner, no test-informed choice,
and P11's hierarchical inference is neither replaced by a simpler interval nor
declared complete by this benchmark.
