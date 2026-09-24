# Supervisor review — P05-T001

**Date:** 2026-09-24.

**Worker:** OpenCode Go / `opencode-go/deepseek-v4.1-flash`.

**Disposition:** readiness audit accepted with corrections below; scientific training remains unauthorized.

## Findings independently checked

- `CompactSERSClassifier.forward(return_embedding=True)` supplies logits and the 64-dimensional classifier embedding. P05 can wrap the existing encoder without altering frozen P04 implementation.
- `evaluation/p04_plan.py` reconstructs P02 source roles and checks master/instrument exclusions. Its role table omits the substrate-family field needed by P05's different-sensor weighting, so joining audited metadata is necessary.
- The declared loss grids imply D1=9, D2=3, D3=27, D4=54, and D5=54 configurations when all applicable loss/temperature factors are crossed: 147 total. Crossing the six inherited optimizer settings and three seeds would give 2646 fits per source-selection unit. This is an illustrative Cartesian expansion, not an authorized P05 budget or a count of executed fits.
- The historical compute file explicitly labels its 300–700-fit P05 estimate as nonauthorizing. The exact selection design, roles, conditional ladder, refits, and costs must be reconciled before training.
- Loss normalization, paired KL/cosine combination, batch fallback, auxiliary-head implementation, and G3 aggregation require precise versioned treatment. Historical held outcomes cannot decide them.

## Corrections to the worker's report

1. The session contains **40 read/search tool calls**, not the reported 27, and exceeded the assignment's requested 35-call soft limit. No shell command, file edit, or scientific fit occurred. Future assignments use narrower readings and a lower iteration cap; worker-reported counters are not accepted without log evidence.
2. `hyperparameter_registry.json` already specifies `projection_dimension = 64`. The contrastive head's layers, activation, normalization, and parameter accounting remain unspecified; the dimension is not wholly absent.
3. A pair key containing only master and instrument names is not sufficient when multiple spectra share a view. Any eventual pair identifier must distinguish its constituent observation UIDs and fitting-role/context provenance. The exact scheme is not frozen by this audit.
4. A zero-positive anchor must not cause an undefined loss. Whether to omit that anchor with accounting or reject the batch is an unresolved policy; the worker's suggestion to always raise is not an accepted scientific rule.
5. The first readiness inventory does not need to load intensity arrays. Later governed tensor validation is separate from metadata-only candidate enumeration.
6. Proposed full P05 plan commands are not implemented or executed. A successful readiness inventory must not be labelled a passed fit-authorization gate.

## Independent validation

Using the existing project virtual environment, not newly installed packages:

- package-wide Ruff check: passed;
- `tests/test_p04_deep.py`, `tests/test_p02_splits.py`, `tests/test_p02_contracts.py`: **10 passed**;
- no scientific model was trained by these synthetic/unit checks.

The system Python lacks pytest; validation used the already available project environment. This is an environment distinction, not a failed scientific result.

## Next allowed slice

[P05-T002](P05_TASK_002_READINESS_IMPLEMENTATION.md) implements a pure contract/readiness inventory. It may enumerate the already declared loss grid and list unresolved decisions, but it may not resolve those decisions, construct a training runtime, read scientific data, or claim that the full P05 no-fit audit is complete.
