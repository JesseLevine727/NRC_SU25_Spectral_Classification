# Supervisor review — P05-T004 CI diagnostic follow-up

**Date:** 2026-09-24. **Status:** diagnostic patch accepted; subsequent remote evidence identified an inventory-ordering defect. Repair is tracked in T005.

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

This was an observability improvement for a failing test, not a relaxation of its success condition or a runtime fix. Passing a later run alone would not establish why the earlier run failed. Test-only diagnostic evidence was retained while the cause was unresolved. Scientific training remains unauthorized under the unchanged P05 gates.

## Independent validation before publication

- Diagnostic helper tests: **11 passed**.
- Package-wide Ruff: **passed**.
- Full regression suite with the package held unchanged during execution: **332 passed in 129.48 seconds**, including the original P01 integration path.
- Public-package validator on a clean scoped release copy: **passed, 423 files checked**.
- No production module, dependency declaration, frozen scientific registry, or source dataset was changed by T004.
- A subsequent GitHub run is required to check the remote environment. Its outcome must not be substituted for a demonstrated root cause.

## Subsequent remote evidence

[Run 36070875804](https://github.com/JesseLevine727/NRC_SU25_Spectral_Classification/actions/runs/36070875804) recorded **327 passed, one failed, one skipped**. The original P01 assertion failed again, and all eleven diagnostic tests passed. The before/after comparison identified exactly one protected-environment difference: the same three sanitized BLAS records were enumerated in a different order. The matching quarantined run retained all original payload hashes. Different hashes in the rebuilt metadata were consequences of the changed environment fingerprint, not evidence that the original files had been corrupted.

This discriminates the leading hypothesis from repository/dependency changes and damaged payloads. [T005](P05_TASK_005_BLAS_CANONICALIZATION.md) authorizes a minimal canonical-ordering repair and permanent regression tests before removing the temporary diagnostic code. Genuine recorded library or thread-setting changes must continue to invalidate reuse. The temporary implementation remains recoverable from commit `21cc148b`.

The [T005 review](P05_TASK_005_REVIEW.md) records the successful local red/green reproduction and retirement of T004's temporary code. This document is the historical diagnostic record, not a claim that the temporary instrumentation remains installed.
