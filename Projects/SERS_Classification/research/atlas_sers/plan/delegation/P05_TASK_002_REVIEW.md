# Supervisor review — P05-T002 readiness implementation

**Date:** 2026-09-24.

**Implementation author:** OpenCode Go / `opencode-go/deepseek-v4.1-flash`.

**Review method:** supervisor inspected and applied the exact allowed patches, ran independent tests, requested corrections, and checked CLI behavior.

**Status:** accepted for this bounded implementation slice; full regression suite and publication-boundary checks passed. Scientific execution remains unauthorized.

## Delivered implementation

- `src/atlas_sers/evaluation/p05_readiness.py`: standard-library-only contract reader, deterministic conditional loss-grid inventory, input hashes, illustrative cost multiplier, unresolved-decision list, and CLI implementation.
- `scripts/run_p05.py`: thin CLI wrapper with `readiness` and `check` only.
- `tests/test_p05_readiness.py`: 46 tests, including malformed-input and regression cases.

No P04 implementation, original frozen contract, completed result, or scientific dataset was changed. The worker authored the code; the supervisor applied its patches rather than granting unrestricted write or shell access. Local model-session transcripts are not release artifacts.

## Review corrections

The initial patch passed its 30 supplied tests but was not accepted. Independent review found that six-significant-digit identifier formatting merged distinct allowed loss values, and that conflicting optimizer identifiers were accepted. The revised worker patch adds full-precision IDs, checks optimizer agreement, requires protocol/projection metadata, handles unrepresentable numerical inputs, validates historical budget bounds, and tests those cases.

This is why worker test success is not the sole acceptance criterion. Initial defects and their corrections remain recorded rather than describing the first patch as valid.

## Verified behavior

- Package-wide Ruff check: passed.
- Focused P05 tests: **46 passed**.
- Final full package regression suite, with the workspace held fixed: **236 passed**.
- Public-package validator on a clean scoped release copy: **passed, 412 files checked**; local documentation/dashboard links also passed.
- Actual subprocess CLI calls: `readiness` exits 0; `check` exits 2 as documented.
- Both commands report `blocked_pending_design_decisions`, `scientific_execution_authorized = false`, and zero scientific fits.
- Input SHA-256 values match the actual three public contracts; serialized output contains no absolute project path.
- A fresh-interpreter check confirms report generation does not import torch, numpy, pandas, sklearn, or scipy.
- Current loss-grid counts are D1=9, D2=3, D3=27, D4=54, D5=54: 147 configurations. The six-optimizer/three-seed full crossing gives **2646 per selection unit only under that explicitly stated assumption**. Total fits and resource estimates remain unknown.

An initial full regression run, before this implementation, produced 182 passes and eight failures caused by missing `pyarrow` in the local project environment. The declared dependency was installed into a temporary validation-only directory; the project environment and dependency declarations were unchanged. All 15 tests in the affected files then passed. Final full-suite validation uses that temporary dependency location and bounded native thread counts.

An intermediate run produced 235 passes and one reproducibility failure because the supervisor edited `MASTER_PLAN.md` during validation. Comparing the two protected manifests confirmed changed plan/configuration hashes, so creating a new run identity was correct safeguard behavior. No production code was changed to suppress this failure. Repeating the full suite without concurrent edits produced the final 236 passes above. Governed validation and execution therefore require a fixed public package, including planning text, not merely fixed Python code.

## Meaning of the exit codes

From the maintained package with its dependencies available:

```bash
python scripts/run_p05.py readiness
python scripts/run_p05.py check
```

`readiness` returning 0 means the inventory was generated successfully, not that training is allowed. `check` returning 2 is the expected unresolved-design state, not a regression-test failure. Malformed input returns 1. No training or execution subcommand exists.

## Next milestone

Resolve and version the remaining P05 implementation choices and source-only selection/budget design. Then implement the exact metadata/role/pair/support expansion and its leakage tests. This tool is a contract-only prerequisite, **not** completion of the full P05 no-fit audit, new CNN training, or a successful G3 outcome. P08 preprocessing and P14 prototype experiments remain separate.
