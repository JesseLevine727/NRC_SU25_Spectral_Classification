# P05 frozen-reference comparison boundary

This assignment implements the approved descriptive comparison after all new predictions are frozen. It does not authorize another fit, calibration, model choice, preprocessing variant, prediction pass or scientific retry. The supervisor must review the implementation and synthetic tests before execution.

## Entry point and ordering

Implement `p05_comprehensive_comparison.run_comparison(*, project_root, artifact_root, contract_path, permit_path)`. This is a CPU stage with no device argument. The command does not launch another stage or export public results.

1. Record `time.perf_counter()` before input preparation. Use `inputs.prepare(..., require_unstarted=False)`. Read the finite cumulative bound from the completed aggregation receipt and set the remaining deadline within 172,800 seconds. Import the runtime, configure deterministic settings and set one torch CPU thread before provenance capture. No CUDA call is needed.
2. Call `p05_aggregation_authority.authenticate_aggregation(bundle, deadline=deadline)`. This authenticates all completed prediction prerequisites and re-derives the five aggregation tables. Its `prior_seconds` is the aggregation cumulative bound, not the earlier evaluation bound. Require exact agreement with the receipt read initially. Do not open legacy results before this gate.
3. Capture provenance. Exclusively consume `comparison/` and reject an existing `comparison_receipt.json`. Instantiate the existing `StorageBudget` and persist before-provenance. A consumed-stage exception must preserve a failure summary and manifest. An authentication-only failure must not consume the output stage.
4. Call `p05_legacy_references.load_references(bundle, authenticated=auth, deadline=deadline)`. It opens only the pinned P03/P04 aggregation shards, verifies their inventories and hashes, and returns two prediction tables plus identifier-free hash bindings. Do not construct a legacy execution context, use a `LATEST` pointer or read P13 results.
5. Load the pinned full context registry through `p05_outer_inputs.load_context_rows(bundle)`. Convert its records to a DataFrame for `p05_comparison.compare_predictions`. Supply the authenticated P05 ensemble table and the two authenticated legacy tables. No test outcome changes the frozen source-selection plan.
6. Persist exactly four private Parquet tables: `endpoint_metrics`, `paired_metrics`, `coverage`, and `summary`. The existing aggregation `_write_table` helper provides exact stored-byte and DataFrame round-trip checks with storage accounting. Persist the reference bindings separately as canonical JSON. Do not copy legacy row predictions into another directory or export them publicly.
7. Reauthenticate protected inputs and provenance. Re-read the legacy references through the same gated loader and require identical bindings. Verify that the original aggregation receipt and its pinned stage manifest remain unchanged, and verify manifest contents. Recheck the evaluation receipt/manifest through the loader. No new predictions or optimization are permitted during these checks.
8. Write a completed summary, verify its stage manifest and reconcile storage before writing the final receipt. Charge all authentication, comparison, persistence and acceptance time against the inherited bound. Late closure failures must leave a failed summary; a retained earlier receipt cannot authenticate that changed stage.

## Reconciliation

The comparison population is the registered `held_evaluation` contexts with parent experiment `EXP-N00-T3`. There are 260 in the scientific registry. Synthetic fixtures may contain fewer contexts; the production prerequisite gate authenticates the full registry.

The three P05 strategies are `D0-M`, `P05-SELECTED`, and fixed `D3`. The five references are historical `D0-ERM` and the four frozen classical procedures. Coverage contains eight rows per held context. Each complete model/context contributes two endpoints, M01 and M06. The 17 predeclared model/reference pairs contribute two rows per held context, including incomplete pairs with explicit coverage flags. Pairwise effects use only exactly aligned, complete test-observation sets; incomplete reference subsets are not scored.

Required integer counters are rows in each of the four tables, held contexts, complete and incomplete model/context combinations, and zero fits, calibrations, outer predictions and optimizer updates. Check their exact values against the returned tables. Retain prior source/refit optimizer counts in identity metadata without charging them twice.

Receipt and summary identity include schema/protocol, command/stage, permit, core contract/plan, ledger, frozen selection plan, aggregation receipt digest and the reference-binding digest. Completion requires `status=complete` and a strict Boolean `comparison_complete=true`. Both records contain measured stage time, prior cumulative bound, total bound, the unchanged 3,600-second prelaunch reserve and the unchanged 172,800-second ceiling. Storage remains bounded by 107,374,182,400 bytes. Errors use stable path-free reason codes.

## Evidence boundary

The comparison is descriptive. Balanced accuracy retains the existing observed-test-class definition; macro-F1 retains the three-class vocabulary. Report sparse class support alongside results. Equal-context and equal-domain means are different summaries and must be named separately. Missing-as-zero sensitivity summaries are labelled as a coverage penalty, not imputed valid scores. Repeated splits and seeds are not independent chemical samples. No bootstrap, significance test, definitive superiority claim, P11 decision or P13 substrate analysis belongs to this stage.
