# P05-T005 — canonicalize the observed BLAS inventory ordering

**Date:** 2026-09-24. **Scope:** narrow reproducibility repair; no scientific experiment or contract change.

## Evidence and mechanism

[GitHub run 36070875804](https://github.com/JesseLevine727/NRC_SU25_Spectral_Classification/actions/runs/36070875804) reproduced the older P01 failure with the T004 diagnostics. It passed 327 tests and skipped the torch module. The only protected-environment difference reported was `compute.blas`: identical three records in a different order. The matching quarantined run retained every original payload hash. The ordered inventory therefore changed the protected environment fingerprint and correctly triggered the artifact store's mismatch handling, although no recorded library property had changed.

The fix must canonicalize this inventory's enumeration order while preserving all recorded values and multiplicities. It must not sort arbitrary scientific arrays, delete recorded environment fields, suppress real changes, weaken artifact verification, or reinterpret an actual change in library version/thread count as equivalent.

## Worker assignment

Use the existing OpenCode Go / DeepSeek v4.1 Flash implementation worker. Return separate `apply_patch` blocks in this order: new regression tests, minimal production fix, temporary diagnostic cleanup. The supervisor first applies/runs only the tests against the old implementation, observes the expected failure, then applies the remaining patches and reruns all checks. No worker execution claims.

Allowed paths:

- `src/atlas_sers/governance/provenance.py`: import the existing canonical JSON serializer and sort the already-sanitized BLAS records by its complete canonical byte representation. Keep duplicate records and all existing allowed fields. Do not mutate `threadpool_info()` inputs.
- `tests/test_governance_provenance.py`: add focused permanent regressions, including the actual `capture_provenance` protected-hash and `ArtifactStore.begin` reuse path under a controlled synthetic environment.
- `tests/test_p01_integration.py`: remove the temporary diagnostic imports/snapshot/message and restore the original strict equality assertion. Keep the real end-to-end build/reuse test.
- `tests/test_p01_repeatability_diagnostics.py`: remove this temporary T004 instrumentation once the permanent regression demonstrates the cause and fix. Its source remains recoverable from commit `21cc148b`.

No other implementation paths, tools, reads, shell commands, raw data, external agents, scientific runs, dependency changes, frozen contracts, existing artifacts, or Git operations are authorized for the worker. The necessary source context is already in the worker session and the supervisor supplies the canonical serializer interface. Planning and review documents are supervisor-owned.

## Regression requirements

1. Permutations of the same sanitized inventory yield equal inventory and protected-environment hashes; duplicate records are retained and input records are not mutated.
2. Through `capture_provenance` and the real artifact store, a completed synthetic run is `verified_skip` after an inventory-only permutation.
3. Changing a recorded version, thread count, architecture, or thread API remains a changed protected environment; adding/removing/duplicating a library also remains a real change. At least one such change exercises the real quarantine/new-run path.
4. Different key insertion order and optional null/missing fields have deterministic ordering. Fields already excluded for privacy (such as the absolute library path) remain excluded.
5. The original P01 end-to-end reuse test remains unchanged in meaning and is rerun. No retry-until-green or `xfail`/skip replacement is permitted.

Historical artifact directories and pointers remain untouched. New captures use canonical order; this is not a migration or retroactive alteration of any P00–P04 result. Source-version identity continues to distinguish the updated implementation.
