# Supervisor review — P05-T005 BLAS inventory canonicalization

**Date:** 2026-09-24. **Status:** accepted after deterministic reproduction and local regression checks; remote validation follows publication.

The [T004 remote evidence](P05_TASK_004_REVIEW.md#subsequent-remote-evidence) identifies an order-sensitive BLAS inventory hash, not a change to spectra or models. [T005](P05_TASK_005_BLAS_CANONICALIZATION.md) limits the implementation worker to canonicalizing that inventory, permanent regression tests, and retirement of the temporary diagnostics.

The supervisor applied the regression tests before the production patch, verified that the existing implementation failed on reordered but otherwise identical records, and then verified the repair. The real artifact-store reuse/quarantine seam preserves rejection of actual environment changes. Historical scientific artifacts, result files, and pointers remain immutable.

## Independent reproduction and review

The tests were applied before the production change. The old implementation produced **three expected failures and five passes**: inventory permutations differed, six permutations produced six protected-environment hashes, and the real artifact store returned `new` instead of `verified_skip`. This reproduces the remote mechanism in a fast synthetic test without depending on spontaneous library enumeration changes.

The first test draft required review corrections: duplicate preservation assumed a particular sorted position; the mocked inventory copied records and weakened the mutation check; mixed null/missing-field ordering needed a multi-record case; and Ruff identified a constant-attribute `setattr`. These were returned to the implementation worker. The production proposal is limited to sorting the sanitized BLAS records by the existing canonical serializer.

The corrected tests reproduced the same three failures before the fix and passed after it. The implementation retains duplicate records, existing privacy filtering, and all previously recorded allowed fields. Version, thread-count, architecture, API, threading-layer, library additions/removals, and multiplicity changes still alter the protected hash; a genuine thread-count change still triggers the real artifact store's quarantine/new-run path.

## Validation and cleanup

- Permanent provenance regressions: **8 passed** after the production fix; **3 failed, 5 passed** before it.
- Independent supervisor check: **24 mixed optional-field permutations** produced identical ordered output while retaining null, missing, and duplicate records.
- Full package regression suite with no concurrent package edits: **329 passed in 182.05 seconds**, including the original P01 end-to-end reuse assertion and local torch tests.
- Package-wide Ruff: **passed**.
- Public-package validator on a clean scoped release copy: **passed, 425 files checked**.
- Temporary T004 diagnostic imports, assertion message, and helper/test file were removed. Searching production and test sources found no remaining temporary diagnostic symbols or prefix. The removed helper is recoverable from commit `21cc148b`; no dataset or scientific artifact was deleted.
- The test-count change from 332 to 329 is eleven temporary tests removed and eight permanent regressions added, not skipped failing tests.
- All implementation-worker turns are finished. No new scientific training was started.

The diagnosis skill's evidence-first, failing-regression-first, and cleanup requirements determined this sequence. A later successful remote run will validate the originally failing integration scenario on that environment; the captured difference and deterministic regression establish the mechanism independently of that outcome.

The prevention lesson is to test permutation invariance explicitly for inventory-like metadata while preserving order in scientific arrays and other genuinely ordered inputs. No broader architecture rewrite is required for this defect.
