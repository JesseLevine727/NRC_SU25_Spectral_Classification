# P05-T004 — diagnose the older P01 CI reuse failure

**Date:** 2026-09-24. **Boundary:** test diagnostics only; no scientific implementation or contract change.

Commit `3eead2b1` passed 321 local tests. GitHub run `36067890541` passed all 85 new P05 tests but failed `test_synthetic_p00_to_p01_build_validates_and_verified_skips`: the repeated P01 build returned `action=new` with the same run ID, instead of `verified_skip`. A fresh isolated local integration run and 12 successive repeat builds passed. The CI failure's cause is not yet established; this is not evidence that P05 support results changed.

The diagnosis skill calls for reproducing and collecting discriminating evidence before modifying behavior. Existing master-plan/governance documents supply domain context; new issue-tracker/AGENTS setup is outside this task and is not authorized.

## Worker scope

Use OpenCode Go / DeepSeek v4.1 Flash. Read this assignment, `tests/test_p01_integration.py`, `src/atlas_sers/governance/artifacts.py`, and `src/atlas_sers/governance/provenance.py` (at most four read calls). Return small `apply_patch` updates/additions, not complete unrelated rewrites. Supervisor applies patches and runs tests.

Allowed paths:

- update `tests/test_p01_integration.py`;
- add `tests/test_p01_repeatability_diagnostics.py`.

No shell, direct edits, additional agents, raw data, results, model training, dependency changes, Git operations, changes to runtime provenance or artifact code, or weakened/removed assertions.

## Required diagnostic behavior

Before the second P01 build, retain a read-only snapshot of the first build's `_STATE.json`, `protected_state.json`, and the protected components of `environment_lock.json`. After a mismatch, append a compact `[DEBUG-p01-reuse]` diagnostic to the existing equality assertion. The equality requirement stays exactly the same and a mismatch still fails.

Evidence must distinguish changed protected identity, environment components, and mutated/missing cached payloads. Compare the protected repository/runtime/compute/dependency-lock/filesystem-total state, not free disk space or other intentionally volatile measurements. Report changed logical field names and sanitized protected values where informative, especially ordered BLAS records versus identical records in a different order. Inspect matching quarantined test runs when necessary to check the original payload hashes; never inspect production artifacts. Preserve quarantine and all original test evidence.

Do not emit workstation paths, source spectra, credentials, unrelated environment variables, or raw dataset records. Handle missing diagnostic files without masking the original assertion. Helpers are test-only and side-effect-free; no file writes outside synthetic test fixture creation. No diagnostics are printed on successful reuse.

Tests must construct small synthetic metadata/quarantine cases and demonstrate: no protected difference; changed compute/BLAS ordering detected; changed BLAS value remains a real difference; volatile free disk change excluded; payload missing/mutated identified; diagnostic missing-file handling; no input mutation or absolute-path leakage. The supervisor also reruns the original full P01 integration path.

This patch improves the failure signal, not the scientific implementation. Do not label the older bug fixed, normalize away a real environment change, retry until success, or substitute a weaker success criterion. Further runtime fixes require the captured evidence and another supervisor assignment.
