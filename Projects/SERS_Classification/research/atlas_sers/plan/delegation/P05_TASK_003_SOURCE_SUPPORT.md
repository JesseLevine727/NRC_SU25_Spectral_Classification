# P05-T003 — inherited-role source-support audit

**Date:** 2026-09-24. **Gate:** bounded no-fit Gate B slice.

**Author:** OpenCode Go / `opencode-go/deepseek-v4.1-flash`.

**Purpose:** determine what positive pairs and class/instrument cells actually exist inside the frozen source-fitting roles before deciding loss details or an executable budget. This audits inherited P04 roles; it does not create approved P05 training roles or advance G3.

## Delivery and access

Return `apply_patch` patches adding exactly the following files. The supervisor may split delivery into module, wrapper, and tests when the provider output limit prevents a complete patch; the acceptance scope stays unchanged.

1. `src/atlas_sers/evaluation/p05_support.py`;
2. `scripts/audit_p05_support.py`;
3. `tests/test_p05_support.py`.

Read this assignment, `src/atlas_sers/evaluation/p05_readiness.py`, and `src/atlas_sers/evaluation/p04_plan.py`. Inspect the existing test conventions if useful. Maximum 10 tool calls; no shell, direct edits, other agents, data access, installs, training, commits, or push. The supervisor applies and independently tests the returned patch. Do not claim tests were run.

Implementation is standard-library-only; pytest is available for tests. Do not change existing implementation, original contracts, or scientific outputs. No torch, numpy, pandas, numerical arrays, automatic governed-root discovery, artifact-pointer mutation, or output-file writer. CLI emits JSON to stdout; redirection into a local audit file is supervisor-controlled.

## Inputs and trust boundary

Accept explicit manifest, context-registry, and role-registry CSV paths, each accompanied by a required expected SHA-256 supplied by the supervisor from the verified prerequisite artifact state. CLI flags: `--manifest`, `--manifest-sha256`, `--contexts`, `--contexts-sha256`, `--roles`, `--roles-sha256`, and optional `--project-root` for the existing public readiness contracts. Reject hash mismatch before parsing. Report hashes with logical keys only, not absolute paths. This is a caller-pinned integrity check, not independent authentication of a forged upstream registry; say so in module/help/report.

Project input columns:

- Manifest: `observation_uid`, `master_sample_id`, `station`, `target_analyte`, `instrument`, `sensor_family`. Ignore all other columns; never serialize filenames, operator fields, source paths, intensity/QC fields, or raw manifest records.
- Contexts: `context_id`, `station`, `task_id`, `domain`, `held_instrument`, `selection_mode`, `phase_gate`, `selection_unit_count`, `outer_fit_rows`, `outer_fit_masters`, `outer_test_rows`, `outer_test_masters`, `outer_fit_uid_sha256`, `outer_test_uid_sha256`. P04 context files actually omit `outer_test_masters`; do not require that optional field. All other named columns exist.
- Roles: `context_id`, `role_id`, `role`, `selection_unit_id`, `observation_uid`, `master_sample_id`, `target_analyte`, `instrument`.
- Role names: `outer_fit`, `outer_test`, `selection_fit`, `selection_validation`. Outer unit IDs equal their role names (`outer_fit` / `outer_test`). T1 units start `outer_fold_as_inner:`; T3 pseudo units start `pseudo:`; T3 fallback units start `master_cv:`.

Provide a pure builder for synthetic tests in addition to the hashed CSV loader/CLI. Validate:

- Required nonempty identifiers and columns, manifest UID uniqueness, master-to-station/chemical consistency, context uniqueness, all roles reference known contexts and UIDs, role metadata agrees exactly with the manifest, row station agrees with context.
- Each role ID has one context/type/unit identity; each context/type/unit has only one role ID; no duplicate UID within a role. Reject unknown role names.
- Exactly one nonempty outer-fit and outer-test role per context; source/test masters disjoint; outer count and UID-set hashes agree with the context. Hash convention is existing `sha256_value(sorted_UID_list)`; use the standard-library canonical module or its exact convention.
- Each selection unit has exactly one nonempty fit and validation role. Both UID sets are subsets of outer-fit, with disjoint masters, and the declared selection-unit count agrees. Do not demand full union for pseudo-domain units, because matched masters are intentionally excluded from fitting.
- For T3, held instrument absent from outer-fit and every inner role; outer-test contains only the held instrument. For pseudo units, validation contains only that pseudo instrument, fitting excludes it. T1 held instrument is `not_applicable`.
- Reject missing/malformed counts, empty input tables, duplicate CSV headers, missing row fields, and inconsistent context/task/selection-mode semantics rather than silently skipping records. A compact clear ValueError-derived custom error is appropriate.

All held-row metadata is read only to check boundaries; it cannot contribute to pair/support statistics or fit selection. No observed performance outcomes are accessed.

## Source-role audit

Audit `selection_fit` and `outer_fit` separately, in deterministic context/type/unit order. Retain parent P04 context/role IDs as provenance; assign a distinct `P05AUDIT-...` ID. These are audit IDs, not executable fit IDs.

For each fitting role, report context/station/domain/phase/role/unit, observation/master/instrument/class counts, master counts per chemical, and:

- All unordered distinct-UID same-chemical pairs, partitioned into the four mutually exclusive same/different master × same/different instrument categories. These are availability counts, not sampled training pairs or independent replicates. Same-master/same-instrument repeats are explicitly separated; do not invent their training weight.
- Different-chemical unordered pair count (within the already validated station); same-chemical cross-substrate pair count (overlaps the four categories and is labelled as such).
- Number of fitting masters with at least two instruments, number with at least two substrate families, and number of anchors with no other same-chemical UID in this fitting role.
- Number of chemicals with at least two distinct fitting masters; whether at least two such chemicals exist; whether all represented chemicals meet that threshold. These are necessary support indicators only, not proof that every minibatch will be feasible.
- Class × instrument cells: each cell's spectra and distinct-master counts; counts of cells with >=2 spectra and with >=2 masters. Covariance support from two repeated spectra does not imply two independent specimens.
- Deterministic unordered pair identity helper using version, parent context, parent role, and sorted constituent UIDs; reject identical UIDs. Include a streaming SHA-256 digest of the lexically UID-ordered positive-pair IDs per role rather than materializing millions of pair records. Never serialize constituent UIDs or master IDs in the returned report. Pair identity is only an audit proposal, not a frozen sampler specification.

Summaries group by station × phase_gate × fitting-role kind. Give denominators and counts of roles lacking same-master/cross-instrument pairs, lacking different-master/cross-instrument positives, lacking the two-chemical/two-master support condition, and containing zero-positive anchors. Include min/median/max of observations, masters, and same-master/cross-instrument pair counts. Empty categories must remain explicit zero, not omitted.

## Cost arithmetic and authorization

Use `build_readiness_report` for the validated loss/optimizer/seed grids. Count distinct inherited contexts and selection units, separately for development and held-evaluation source roles and overall. Multiply actual selection-unit counts by the illustrative per-unit crossing (currently 2646). Label this explicitly as an illustrative full-grid Cartesian scenario, **not** an approved P05 plan, total required fits, or compute-time estimate. Exclude final refits, calibration, D0 controls, and retries; list these exclusions. Do not call a source-role audit a held performance evaluation. Training remains unauthorized regardless of the inherited P04 flags.

Report `audit_status = "pass"`, `scientific_execution_authorized = false`, `scientific_fits_performed = 0`, `total_required_fits = null`, `scope = "inherited_role_metadata_only"`, and a concise claim boundary. Readiness's 14 decisions remain unresolved; copy their IDs, not a claim to have resolved them. Include public-contract hashes and explicit-input hashes. Output deterministic canonical JSON without timestamps, absolute paths, raw master IDs, or observation UIDs.

CLI exit 0 means only metadata audit passed; invalid input exits 1 with a concise error, no traceback or accidental source record/path dump. No train/run/execute option.

## Required tests and stopping boundary

Use small synthetic tables with hand-calculated positive categories, class/instrument support, master counts, and pair digest. Test determinism under input-row permutation; pair symmetry and context/role sensitivity; duplicate/missing/contradictory metadata; outer and inner master leakage; target/pseudo-instrument contamination; held-only changes never entering fitting statistics; wrong role parentage; orphan/missing roles; selection counts and source hashes; invalid CSV/hash input; CLI success/error; no numeric/training-runtime imports and no input mutation. Reuse the repository public contracts for readiness where appropriate without depending on private files.

Return only complete, independently applicable patches plus a short test recommendation. Stop; loss runtime, sampler, G3 design, full P05 registries, training, figures, and publication are outside this worker slice.
