# P05-T001 — read-only implementation readiness audit

**Issuer:** supervising research assistant, 2026-09-24.

**Assignee:** OpenCode Go / DeepSeek v4.1 Flash.

**Gate:** A only, under [the orchestration protocol](../ORCHESTRATION_PROTOCOL.md).

**Working scope:** the maintained `research/atlas_sers` package.

**Repository modifications allowed:** none. Return findings in the session response.

## Objective

Identify exactly how to implement P05's no-fit audit and subsequent acquisition-aware CNN work while preserving completed P04 and frozen evaluation rules. Produce a concrete implementation handoff and identify scientific details that require a supervisor decision. Do not implement or run models in this task.

P04 is complete. P05 is next; P08 preprocessing and P14 RBF/SOM are not the current implementation target. Keep the exact D0 backbone, primary `PP-U-MIN`, physical-master isolation, held-instrument exclusion, and source-only model selection. Do not infer a better loss or architecture from previously observed scores.

## Required inspection

Read, with targeted follow-up searches rather than loading the entire repository:

1. `PUBLICATION_POLICY.md`, `CONTRIBUTING.md`, and `plan/ORCHESTRATION_PROTOCOL.md`.
2. `plan/MASTER_PLAN.md` sections 12–14 (ordinary D0, P05, P06) and the G3 advancement conditions in section 13.7. Do not load historical results merely to choose an implementation.
3. `plan/P04_EXECUTION.md` and `plan/contracts/p04_execution_contract.json` for exact inherited architecture, optimization, roles, calibration, failure handling, and semantics.
4. Relevant P02 split/selection contracts and original experiment/model registries.
5. `src/atlas_sers/models/deep.py`; `evaluation/p04_plan.py`, `p04_runtime.py`; relevant P04 governance/CLI modules and `scripts/run_p04.py`.
6. Corresponding P02/P04 tests and fixtures, as needed to establish reusable interfaces and missing tests.

Read existing code and metadata specifications only. Do not open governed data roots, raw spectra, checkpoints, row predictions, held-result tables, credential files, or unrelated files. No shell execution, edits, additional agents, package installation, or web research is needed or permitted in this task.

## Questions to resolve

1. Which existing functions/classes can be reused without changing frozen P04 behavior? Give exact paths and symbol names.
2. What new P05 modules, CLI entry points, contracts, and tests are needed? Separate pure no-fit planning from PyTorch training/runtime imports.
3. How will exact P02 source-fitting, pseudo-domain validation, outer-test, and fallback roles be inherited? How will held-instrument and physical-master exclusion be checked for pairs as well as individual rows?
4. Which positive/negative masks, cross-station exclusions, sensor weighting, master-balanced sampling, and infeasible-batch fallbacks must be specified and tested?
5. Which choices remain unspecified: contrastive projection head, auxiliary parameter accounting, exact normalization of losses, relative KL/cosine weights, probability stabilization, memory-bank behavior, adversarial gradient paths/schedule, optimizer/loss-grid nesting, calibration, and G3 comparison denominator? Identify a choice as unresolved rather than silently making it.
6. Can the exact fit budget be derived without data access? Give only symbolic or registry-supported candidate counts, showing temperature/loss/optimizer/seed factors. Mark unknown role multiplicities and do not reuse P04 total fits as a P05 estimate without justification.
7. Which synthetic tests must fail if leakage, invalid negatives, duplicate rows, zero-positive anchors, nonfinite gradients, nondeterminism, or accidental training in a plan command occurs?
8. What is the smallest useful first coding slice after supervisor review, with a file allowlist and precise acceptance commands? Proposed commands must be labelled not executed.

## Required response

Return a concise but detailed audit with:

- reusable implementation interfaces, with evidence references;
- a requirements-versus-unresolved-decisions table;
- a proposed no-fit file/test slice and its acceptance conditions;
- a symbolic budget and required metadata inputs;
- ranked risks and explicit stop conditions;
- files changed, commands run, tests run, and scientific fits performed, all accurately reported.

Limit the final response to approximately 1800 words. Use at most 35 file/search tool calls; prioritize high-value inspection and disclose unread material rather than inventing evidence. Stop after the response. No autonomous continuation into Gate B, no training, no edits, and no Git operations.
