# Supervisor review — P05-T003 source-support audit

**Date:** 2026-09-24.

**Implementation author:** OpenCode Go / `opencode-go/deepseek-v4.1-flash`.

**Status:** accepted for the bounded inherited-role metadata audit. Focused, full-package, and publication checks passed. Scientific training remains unauthorized.

## Scope and supervision

The [assignment](P05_TASK_003_SOURCE_SUPPORT.md) authorizes only a metadata audit of inherited P04 fitting roles. The worker has no raw-data, shell, direct-edit, model-training, Git, or delegation access. It returns patches; the supervisor reviews exact paths, applies them, executes tests, and checks actual metadata. Planning, interpretation, and this review are supervisor-authored.

The initial worker made nine read/search tool calls, within the ten-call bound, but hit a client response limit without returning code. Delivery was split into independently applicable module, correction, wrapper, and test patches; the authorized scope did not grow. A subsequent test-authoring turn also ended with exactly 32,000 reasoning tokens and no code. The supervisor verified the installed client's output-limit flag and increased the allowance to 64,000 for bounded retry/revision processes, without changing global configuration, permissions, the model, or scientific compute authority. This is consistent with the limitation reported in the [OpenCode project tracker](https://github.com/anomalyco/opencode/issues/29363); local event logs, not a successful process exit alone, determined whether code was actually delivered. Subsequent patch-authoring turns have no tool access. No substitute model was used. Local transcripts and unaccepted diagnostic outputs are not publication artifacts.

## Rejected first implementation

The first complete module was not accepted. Independent review found:

- The cost loop counted both outer fitting roles and inner selection fitting roles, reporting 1,181 selection units instead of 861. The supervisor's independent calculation exposed this; a real-metadata diagnostic confirmed the erroneous 240/941 development/held split counts.
- Omitting `--project-root` bypassed public-contract validation and fell back to a hardcoded multiplier.
- An orphan inner fit/validation role could raise an unhandled `KeyError` rather than a controlled audit failure.
- The pair digest materialized and sorted pair IDs instead of streaming the specified UID-pair ordering.
- CSV parsing, identifier normalization, station/task/domain consistency, and sanitized error handling needed tightening. Ruff also identified a missing explicit `zip(strict=...)` argument.

Corrections were returned to the implementation worker rather than silently treating plausible output as valid. The support counts themselves were independently calculated from the pinned metadata before inspecting the worker's report.

The worker returned the corrected module as a same-path delete/add replacement. The supervisor converted that mechanically to an update patch; the implementation content was unchanged and no existing user file was deleted.

The first test delivery produced 66 passes and one failure: its malformed-CSV test inadvertently regenerated a valid CSV before invoking the audit. The worker corrected the fixture rather than changing valid loader behavior, removed a vacuous assertion, and added missing leakage, held-only-invariance, repeated-spectrum/master-support, input-preservation, and whitespace-identity cases. The final focused run passed **85 new support-audit tests plus 46 readiness tests (131 total)**. Package-wide Ruff passed.

## Independent metadata checks

- Reconciled pair categories and cross-substrate counts for all 1,181 fitting roles using separate combinatorial count formulas, rather than the worker's pair-enumeration loop.
- Checked every chemical/instrument cell's spectrum and distinct-master counts against the actual pinned metadata.
- Confirmed 598 observations, 69 masters, 320 contexts, 861 inner selection units, 2,362 total roles, and 132,392 role-row assignments.
- Confirmed the 87 single-instrument, 36 insufficient-two-chemical/two-master, and 177 insufficient-all-class/two-master surface inner-role counts.
- Repeated the complete audit and obtained byte-identical serialized output.
- Injected an orphan validation role, duplicate manifest UID, and contradictory role label; each failed with a controlled audit error.
- Confirmed actual report generation imports none of torch, numpy, pandas, sklearn, or scipy. Scientific fit count remains zero.

## Final validation

- New source-support tests: **85 passed**; combined with readiness: **131 passed**.
- Full package regression suite, with the workspace held fixed: **321 passed** in 131.23 seconds.
- Package-wide Ruff: **passed**.
- Public-package validator on a clean scoped release copy: **passed, 420 files checked**.
- Actual thin-wrapper execution on the pinned metadata: **exit 0**, with unchanged report bytes after the final identity-validation correction.
- Full local report SHA-256: `507349cc4324f7a9e0b4830da761f28dd8a75fb2d5705c9d998021db26d014e5`.
- Input CSVs remain unchanged; no prerequisite artifact pointer, frozen contract, P04/P13 result, or source spectrum was modified. New aggregate results are in `plan/registries/p05_source_support_summary.json`; the detailed audit is retained separately under the local artifact root's `p05support/audits/` namespace, without a training-authorizing `LATEST` pointer.
- All assigned worker turns are stopped. No new scientific run is left unattended.

The full suite reused the temporary validation-only `pyarrow` dependency described in the T002 review, with bounded native thread counts. No project/global dependency or client configuration was changed. The implementation remains standard-library-only.

## Scientific acceptance boundary

The [support findings](../P05_SOURCE_SUPPORT_AUDIT.md) and [design handoff](../P05_DESIGN_HANDOFF.md) distinguish availability from feasibility, repeated pairs from independent specimens, source-selection metadata from held predictive outcomes, and illustrative cost arithmetic from an approved budget.

Even after this slice passes, full P05 Gate B remains open: exact P05 executable registries, loss/sampler choices, source-only G3 denominators, and an approved finite compute schedule are still required. No D1–D5 model is fitted, no G3 outcome is claimed, and the old P04 training authorization is not inherited.

## Post-push CI follow-up

The first remote run passed all 85 support-audit tests but failed an older P01 repeat-build assertion. This does not invalidate the metadata counts or establish the cause of the reuse mismatch. [T004's review](P05_TASK_004_REVIEW.md) records the independent reproduction attempts and test-only diagnostic follow-up; the runtime reuse requirement remains unchanged.
