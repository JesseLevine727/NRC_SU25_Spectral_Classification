# P08 supervision and acceptance record

**Opened:** 2026-09-30. **Current state:** input-audit and accounting milestone accepted locally; the full readiness package remains incomplete. Publication acceptance is recorded by the associated Git commit and remote CI, separately from the local checks below.

## Authority and implementation boundary

The owner requested the next persistent goal after the reviewed P06/P11 release. The goal is a no-fit P08 readiness package, not a preprocessing benchmark execution. [Readiness](../P08_READINESS.md) records the evidence, pending decisions and remaining gates. The [orchestration protocol](../ORCHESTRATION_PROTOCOL.md) remains authoritative.

DeepSeek V4.1 Flash, explicitly `opencode-go/deepseek-v4.1-flash`, authored patches from scoped public source snapshots and synthetic fixtures. OpenCode tools, sharing, snapshots and external plugins were disabled. The worker received no actual spectra, private row metadata or prediction records and had no Git-write authority. The supervisor controlled patch application and testing.

## T001: frozen-action auditor

Allowed files: `src/atlas_sers/evaluation/p08_actions.py` and `tests/test_p08_actions.py` only. The module validates already-loaded arrays and metadata. It performs no file access, transformations, fitting, selection, prediction or random sampling. The caller must authenticate the source artifacts independently.

The first patch was not accepted: 45 tests passed and one failed because a test helper replaced an explicit `None` manifest with a valid default. Review also identified missing duplicate-column guards, insufficient representation-ID type checks and an unhandled numeric-overflow case. These defects were returned to the same worker; production validation was not weakened to satisfy the test.

## T002: correction and focused acceptance

The worker supplied a surgical correction within the same two files. The helper now distinguishes omitted arguments from explicit `None`. Both metadata frames reject duplicate columns and malformed representation IDs with fixed, data-free reason codes. Numeric bounds reject overflow. Tests cover these cases, invalid array types/shapes, altered hashes, row order, normalization, QC validity, nonmutation and identifier-free output.

Independent local checks: **88 tests passed in 0.30 seconds**; Ruff passed on both files. The supervisor also ran the module against authenticated frozen inputs, separately from the synthetic tests. No worker self-assessment was used as acceptance evidence. Both worker sessions ended successfully; no worker or scientific training process is left running by this slice.

Full-package local regression at this checkpoint passed **3,045 tests**, with **four CUDA-only tests skipped** because CUDA was disabled for the test process (411.92 seconds). The 2,026 warnings arise from sparse-class metric fixtures. Full-package Ruff passed. This regression includes the new action auditor but predates the next accounting slice.

Focused acceptance does not complete role/routing/ledger implementation or establish a scientific preprocessing result. Public-boundary validation, exact-path release review, `main` push and remote CI remain separate checks. Existing unrelated modifications are preserved. The live-tree public-boundary scan found pre-existing ignored P03 LaTeX logs/auxiliary files; these are not tracked release files. They are preserved, excluded from publication and must not be used to justify relaxing the validator. An exact clean publication snapshot is required for release validation.

## Owner decisions

The owner approved the D0-M/context-local selected-recipe mapping and the scoring/weighted-uncertainty amendment, and explicitly included Extra Trees in the universal panel. The owner then approved complete minimal-pipeline fallback for unsupported held families, including the minimal-trained estimator. These four decisions are recorded in the [new readiness contract](../contracts/p08_readiness_contract.json). They do not authorize a scientific fit or reuse prior execution budgets. Mixed family-routed source training remains a separate unapproved experiment.

Lenarizer guided the planning prose: operating conditions, whole-pipeline versus component effects, conditional estimates and unsupported claims are explicit. Its preservation checker found only added numeric tokens in the readiness revision for the approved Extra Trees accounting and dated approvals; no original numeric token or citation was removed. The added counts were checked against inherited grid sizes and audited context/unit counts.

## T003/T004: universal-policy accounting

DeepSeek implemented `p08_accounting.py` and its synthetic tests as a pure, non-executable aggregate counter. The supervisor rejected the initial shorthand action identifiers despite all 26 initial accounting tests passing. The correction restored the exact frozen representation IDs, explained that selected-CNN counts are incremental unique work beyond D0-M, and added full-size synthetic arithmetic, tuple/context-unit reuse and private-identifier error checks.

After correction, **119 focused tests passed in 0.32 seconds** (88 action checks and 31 accounting checks); full-package Ruff passed. The earlier 3,045-test full regression predates this accounting module. The supervisor independently applied the counter to authenticated held-context/unit metadata and the frozen source-selected recipe map. It reproduces the preliminary five-model counts in the [aggregate accounting report](../../results/p08_readiness/universal_accounting.json), including conditional calibration-cache savings and separate scalar calibration operations. No sample identities enter the published report. Neither source-cache reuse nor historical MIN reuse is declared authenticated by this counter.

The counter always returns execution authorization false. Its model/prediction job counts exclude policy selection, family/QC, robustness and orchestration work and do not constitute an exact launch ledger. All four worker sessions have ended; no training was launched. Budget estimates and scientific permissions remain separate.

## Evidence review correction details

The held-family support count alone cannot establish whole-pipeline reuse. The supervisor extended the audit to source families and found 202 supported family–context combinations in 182 held contexts, despite zero supported held-family contexts. The plan now distinguishes deployment-row fallback from source-training routing. Any future reuse claim must compare the full routed inputs and fitting specification.

Two early supervisor-only audit attempts stopped on diagnostic assumptions: a development sentinel was incorrectly treated as a held-instrument name, and the P13 aggregation state was incorrectly assumed to contain the P01-style scientific-status field. The read-only diagnostic was corrected to use the explicit phase and the separate P13 validation report. No scientific input, split, result or historical contract changed.

The preservation checks for README and the handoff passed. Checks for the master plan, orchestration protocol and HTML navigation reported only added date tokens for the new dated status; original numeric tokens and citations were retained. These additions document the owner-approved next phase rather than revising historical outcomes.

The clean publication snapshot passed the public-boundary/plan validator with **872 files checked**, and full-package Ruff passed. The new readiness contract initially lacked the existing validator's required `protocol_version` field; that field was added in the recognized namespace before acceptance. No validator rule was weakened, and ignored local build logs were not included.

The final snapshot's focused P08 and inherited planning/governance contract checks passed **143 tests in 5.02 seconds**. This supplements, rather than relabels, the earlier full regression. Publication must include only the reviewed P08 files and scoped navigation edits; unrelated working-tree changes remain excluded.
