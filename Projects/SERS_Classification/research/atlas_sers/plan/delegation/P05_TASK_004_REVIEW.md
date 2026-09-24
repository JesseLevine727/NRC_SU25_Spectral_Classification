# Supervisor review — P05-T004 CI diagnostic follow-up

**Date:** 2026-09-24. **Status:** test-only diagnostic patch accepted locally; remote root cause unresolved.

## Trigger and evidence

The [first P05 support-audit CI run](https://github.com/JesseLevine727/NRC_SU25_Spectral_Classification/actions/runs/36067890541) failed an existing P01 repeat-build assertion. It recorded 316 passes, one failure, and one skipped module (torch is not installed by this CI workflow). All 85 new support-audit tests passed. The failed comparison had the same P01 run ID and `status=pass` but `action=new` rather than the required `verified_skip`.

The 321-test local run passed before that push. Investigation then ran the isolated P01 integration test successfully and obtained successful reuse in 12 consecutive subsequent builds of a fresh synthetic fixture. Twenty fresh-interpreter probes, using distinct Python hash seeds, produced identical sanitized BLAS inventories locally. None of these observations identifies the cause of the remote failure; they do not justify relaxing the check or claiming a fix.

The ranked working explanations are a changed compute-environment fingerprint, a changed repository/dependency fingerprint, or changed/missing cached payloads. The test-only snapshot and quarantine comparisons distinguish them. No source dataset, model output, preprocessing policy, production hash rule, or prerequisite artifact was changed.

## Supervision and review corrections

DeepSeek v4.1 Flash authored the patch under [T004](P05_TASK_004_CI_DIAGNOSTICS.md). It made five reads rather than the four assigned: the extra read listed the package directory before reading the task. All reads stayed within the package; the deviation is recorded, not described as compliance with the call limit. Subsequent correction authoring has no tool access.

The first diagnostic patch was not accepted. Independent review identified an incorrect quarantine-parent lookup, an overly broad metadata fallback, insufficient defensive handling of diagnostic paths, one incorrect missing-file test expectation, and a Ruff line-length violation. Corrections must exercise the full diagnostic call at the actual `phase/runs/run_id` layout, not merely an isolated helper.

The revised implementation exercises that full call path, confines payload reads to the matching quarantined run, rejects path traversal and escaping symlinks, and requires matching run IDs when recorded. Changed BLAS contents report sanitized field-level differences; pure record reordering is identified separately. Missing or unreadable diagnostic metadata does not relax the original reuse assertion. The integration test's expected equality is unchanged and the failure message is evaluated only when that assertion fails.

The diagnosis skill informed the reproducibility-first approach and the decision to collect discriminating evidence before any runtime fix. Its separate repository-setup prerequisite requires user-selected issue-tracker conventions; that administrative setup was not performed or silently assumed. Existing master-plan/governance records supplied project context. No new GitHub issue or repository-wide agent configuration was created.

## Acceptance boundary

This is an observability improvement for a failing test, not a relaxation of its success condition and not a demonstrated fix for the remote failure. Passing a later run would show that that execution succeeded, not establish why the earlier run failed. Test-only `[DEBUG-p01-reuse]` evidence is intentionally retained while the cause remains unresolved. Scientific training remains unauthorized under the unchanged P05 gates.

## Independent validation before publication

- Diagnostic helper tests: **11 passed**.
- Package-wide Ruff: **passed**.
- Full regression suite with the package held unchanged during execution: **332 passed in 129.48 seconds**, including the original P01 integration path.
- Public-package validator on a clean scoped release copy: **passed, 423 files checked**.
- No production module, dependency declaration, frozen scientific registry, or source dataset was changed by T004.
- A subsequent GitHub run is required to check the remote environment. Its outcome must not be substituted for a demonstrated root cause.
