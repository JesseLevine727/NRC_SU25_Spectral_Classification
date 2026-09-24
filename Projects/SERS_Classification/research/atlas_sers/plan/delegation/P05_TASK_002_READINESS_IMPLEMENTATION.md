# P05-T002 — contract-only readiness inventory

**Gate:** a small prerequisite slice within B, not completion of the full P05 no-fit audit.

**Assignee:** OpenCode Go / DeepSeek v4.1 Flash.

**Scientific fits/data access:** none.

**Delivery:** author an `apply_patch`-format patch; the supervisor will inspect paths, apply it, and independently execute tests. No direct edits or commands in this assignment.

## Read first

Read this assignment, `P05_TASK_001_REVIEW.md`, and these existing contracts:

- `plan/contracts/hyperparameter_registry.json`;
- `plan/contracts/p04_execution_contract.json`;
- `plan/contracts/compute_budget.json`.

The preceding audit's code context is available in this session. Do not repeat the broad exploration. Maximum eight tool calls; final output is the patch plus a short explanation. No further agents or model substitution.

## Exact allowed additions

1. `src/atlas_sers/evaluation/p05_readiness.py`;
2. `scripts/run_p05.py` (thin wrapper only);
3. `tests/test_p05_readiness.py`.

Do not update existing files. All implementation is standard-library-only; tests may use existing pytest. No torch, numpy, pandas, training runtime, package installation, or governed-root lookup.

## Required behavior

- Build a deterministic JSON-serializable readiness report from the three named public contracts, given an explicit package root or the installed module's package-root default.
- Enumerate applicable loss combinations in stable declared order: D1 temperature × lambda_supcon; D2 lambda_pair; D3 temperature × lambda_supcon × lambda_pair; D4 adds lambda_coral; D5 adds lambda_domain. Inapplicable fields must be absent or null, not spuriously multiplied. Stable unique IDs and complete parameter records are required.
- Obtain reference optimizer and seed counts from the P04 contract, not hardcoded 6 and 3. Check the overlapping optimizer grids agree with the original hyperparameter registry; validate nonempty, unique, finite, strictly positive numerical grids, rejecting booleans and malformed inputs. Seeds must be unique integers. Reject inconsistent or missing required contract structure with a clear error.
- Include hashes of the three input files under relative path keys. Do not expose the absolute package root, workstation identity, timestamps, or raw data.
- Expected current counts: 9/3/27/54/54, total 147; illustrative full optimizer×seed crossing gives 2646 per selection unit. Label this assumption explicitly. Total required fits and resource estimates remain unknown/null; do not multiply by an invented context count or assert an approved maximum.
- Report `status = "blocked_pending_design_decisions"`, `scientific_execution_authorized = false`, and `scientific_fits_performed = 0`. Separate successful report generation from readiness to train.
- List stable unresolved-decision IDs covering loss/optimizer nesting and exact budget; auxiliary-head architecture/parameter accounting (dimension 64 is already declared); exact loss normalization and KL/cosine combination; master/pair sampling, UID-level pair identity, and batch fallback; conditional alignment/adversarial details; D0 comparison and G3 support/aggregation; calibration/epoch inheritance mapping. Do not invent values that resolve them.
- The historical P05 300–700 estimate may be shown as a nonauthorizing reference, not as a cap. Preserve frozen files.
- CLI accepts `readiness` (emit report, exit 0 if generation succeeds) and `check` (emit report, exit 2 because design remains unresolved). Support `--project-root`; `--help` works. Invalid input should exit 1 with a concise error. No execute/train/run command, no output-file writer, no scientific-data access. A `check` exit 2 is expected and must be explained in help.
- Tests cover exact counts and conditional dimensions; unique deterministic recipes; repeatable byte-equivalent JSON; input hashes and no absolute paths; missing/inconsistent/malformed/nonfinite/duplicate/boolean grids; seed validation; both CLI exit codes and invalid inputs; absence of training/numeric-runtime imports. Use isolated temporary public-contract fixtures. Assert existing inputs are unchanged. The CLI must not touch scientific data or write result artifacts.

## Patch and stopping rules

Return one complete patch beginning `*** Begin Patch` and ending `*** End Patch`, with exactly the three `*** Add File:` paths above relative to this package. Do not include updates/deletions, executable shell setup, generated files, or unrelated formatting. Never claim tests passed: the worker is not running them. State which tests the supervisor should run and then stop.

This is real implementation of a fail-closed readiness tool, not the full P05 pair/support/role planner or any new neural training. Further coding requires another reviewed assignment.
