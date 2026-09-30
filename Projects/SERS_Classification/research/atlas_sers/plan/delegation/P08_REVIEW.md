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

## First milestone publication

The initial input/accounting milestone was committed and pushed to `main` as `0ea4202406eb31280c6626f1ea23c4abf8932561`. The remote SHA matched. GitHub Actions run `36742925891` completed successfully. This verifies that milestone, not the later files described below or completion of the P08 readiness goal.

## T005/T006: universal dependency planner

DeepSeek authored the two scoped planner/test files. The first response contained an incomplete draft followed by one complete patch; only the complete patch was applied under review. The first test run passed 27 tests and failed the order-invariance test because its supposedly reordered fixture changed the selected recipe. Review also rejected an invented selection-mode identifier, shortened policy IDs, identifier-bearing error messages and the proposed classical calibration order. Twenty-nine lint findings remained. None of this code was used to fit or predict.

The supervisor checked the inherited runtime and corrected the implementation specification: classical forest seed probabilities are averaged before applying one temperature, whereas neural probabilities are calibrated per seed before averaging. DeepSeek supplied the correction. Its response used delete/add operations for the same files, which the patch tool rejected without changing files. The supervisor mechanically converted that response to full-file update operations, preserving its source content, and corrected two remaining mechanical lint findings. No historical scientific result was rewritten.

The corrected planner uses exact policy/representation/mode IDs, fixed data-free error codes, full-length content hashes, explicit dependencies and an unconditional execution-denial entry point. Tests check classical versus neural calibration order, source-only dynamic dependencies, candidate/seed counts, alias identity, malformed/private inputs, exact hashes, permutation invariance and nonmutation. The independent two-context fixture totals 725 model-fit slots per policy (660 classical source, 21 neural source, 21 calibration-model and 23 final fits). The supervisor's correction request accidentally wrote 765 while giving the correct addends; the worker retained the correct sum, which review confirmed.

The three P08 modules passed **160 tests and 16 subtests in 1.52 seconds** before the two mechanical lint fixes. Full regression and release-boundary checks for this new slice remain separate. Both worker sessions ended; no training process was launched.

## Extended evidence and numerical specification

The read-only classical/neural role bridge passed on 1,721 distinct definitions, including 396 exact fitting/validation hash matches for master-CV calibration reuse. The initial audit assumed identical role labels and stopped; matching the actual role hashes resolved the historical prefix difference without changing any split.

The classical minimal-evidence audit passed for all three fixed families and all 260 contexts each. It rehashed 74 final-bundle files and verified selected candidates, registered test rows, source/refit evidence, seed identities and calibration bindings. The aggregate calibration JSONL uses rounded floating-point serialization; exact calibration-state verification therefore used the original canonical per-outer-shard JSON, not recalculation from the rounded summary. A diagnostic master hash also needed the original string-ID convention rather than CSV-inferred integers. These were corrections to the new audit, not scientific reruns or alterations of saved evidence.

The new [statistical protocol](../P08_STATISTICAL_PROTOCOL.md) and [contract](../contracts/p08_statistics_contract.json) specify prospective policy effects, interactions, 10,000 shared-weight draws, hierarchy feasibility, missing cells and conditional interpretation. No draws or new scores have been computed. Lenarizer guided the separation of measured support, prospective calculations and unsupported chemical claims. Continuous preservation diagnostics are not converted into an invented chemistry-preservation threshold.

The supervisor expanded the corrected planner against authenticated metadata, independently checked counts, unique identities, dependency existence and context/policy isolation, and tested the execution-denial entry. It produced 607,221 slots, 1,560 strategy aliases and the expected two-action 195,202 model-fit/3,354 scalar-calibration counts. The compressed private graph is 152,004,617 bytes; no graph identities or raw observations enter its public aggregate audit. Historical MIN entries still require a per-artifact bridge; no executor consumes this graph.

Preservation checks for the readiness and review additions found added numeric tokens only; original numeric tokens and citations were retained. Added counts and dates were checked against the read-only audits, tests and CI. An overly broad lint command also inspected the pre-existing report renderer and found 54 out-of-scope style errors. That renderer was not edited; the release uses the existing CI scope (`src`, `tests`, `scripts`) and records that scope explicitly.

## Continuation release checks

The exact clean publication snapshot passed the public-boundary/contract scan with **879 files checked**, and CI-scope Ruff passed. Its full regression produced **3,116 passed, four CUDA-only skips, 16 subtests passed and one failure in 385.80 seconds**. The failure occurred in the existing P02 integration test before any build: the exported Git archive had no repository metadata, and the test requires `git rev-parse --show-toplevel` for provenance. A targeted diagnostic reproduced that setup failure.

After the full run ended, the supervisor initialized a Git repository and committed the public snapshot **only in the temporary validation directory**. No user-repository configuration, source or historical evidence changed. Rerunning the P02 integration test together with all P08 and selected planning/contract checks then passed **176 tests and 16 subtests in 26.43 seconds**. This resolves the identified environment failure; it is not relabelled as a second full-suite run. The 2,026 full-run warnings are inherited sparse-class metric warnings. Final review-log additions are documentation only and receive a repeated boundary/whitespace check before the exact-path push. Remote CI for the new commit remains a separate acceptance check.

The continuation remains a no-fit milestone. Complete neural/historical job-to-artifact reuse, nested QC routing/selection roles, branch accounting, finite resources and a separately approved scientific smoke permit are still required. The full persistent readiness goal is not complete.

## T007/T008: nested QC support and owner amendment

The metadata audit found a further nesting constraint before any QC outcome was computed. Of 128 contexts with source pseudo-instrument support, 74 have a limiting class represented by one physical master in their policy-validation fitting roles. The owner approved P08-A05: three-fold master-separated inner estimator selection in the 54 fully supported contexts, complete MIN fallback in the other 206, operational reporting on all 260, and a separately labelled supported subset. The frozen 285 pseudo-units have minimum class-master counts of one, three and four in 177, 75 and 33 units respectively. All 54 eligible contexts are at CWA; pills and surfaces cannot support an adaptive-effect claim under this rule.

DeepSeek authored a pure metadata auditor and synthetic tests, with the same public-only, tools-denied boundary. The first run passed 34 tests and six subtests, but two assertions failed: an unknown well-formed UID was incorrectly expected to be malformed, and a leakage fixture changed only one validation instrument rather than the whole validation role. Three lint findings also remained. These defects were returned for correction.

The supervisor also corrected an overly restrictive condition in the original assignment. Actual frozen held tests comprise 219 three-class, 40 two-class and one one-class context. Held tests must retain that original support; they are not required to contain every task class. Source fitting and validation still require every class. The correction adds sparse-held-test acceptance tests and preserves all master/instrument exclusion checks. Additional review required governance-compatible Unicode hashes, generic mapping support and separate counts for individually feasible pseudo-units versus units in fully eligible contexts.

The corrected response placed two test-file hunks out of source order, so the first patch application failed without changing files. The supervisor mechanically sorted the hunks by their original source positions and applied the unchanged edits. No model code or scientific evidence was altered. All four P08 modules then passed **202 tests and 23 subtests in 1.53 seconds**; Ruff passed on the new files. The independent private-data application matched the support counts above, verified concrete UID/master/instrument exclusions and wrote a hash-bound private report. It constructed no nested folds and computed no thresholds, scores or predictions. Both DeepSeek sessions have ended.

## Neural minimal-evidence bridge

The supervisor authenticated 2,304 inherited source fits and 897 relevant unique final fits, their per-seed calibrations and their held predictions. All 16,416 distinct file checks passed. The audit binds the accepted P05 receipts, exact selected recipes, immutable MIN representation, source/test role hashes, three seeds, saved logits, checkpoints, original calibration states and final prediction files. The final duration matches the clipped, rounded median of the saved inner best epochs. A second read-only pass reproduced every calibration-input semantic hash from the original source-logit payload, without optimizing a temperature. Both passes produced the same private evidence-bridge digest.

Two exploratory reads initially guessed nonexistent legacy receipt locations; the audit now uses the existing read-only recovery resolver. Its first complete attempt also compared a structured manifest entry directly with a hash string; the corrected verifier checks both byte size and SHA-256. These diagnostic corrections did not change or rerun scientific evidence. The published aggregate contains no observation, master, operator or source-path records.

These checks complete the relevant neural evidence bridge, not the full runtime job-cache authority or the final P08 readiness goal. Classical per-operation bindings, the concrete nested-QC ledger, finite resources and reviewed execution guards remain open. Lenarizer guided the amendment wording to distinguish metadata feasibility, station-restricted evidence and future predictive results. Final clean-snapshot validation and publication checks follow separately.

The subsequent classical source-artifact audit passed for 89,892 completed grid fits. It checked 131 original selection shards, 524 file hashes and exact validation UID sets, candidate hashes and seeds for every relevant fit. The aggregate reports 751,104 SVM prediction appearances and 1,001,472 for each tree family; these repeat stored observations across candidates and splits. No estimator or prediction was recomputed. Calibration/selection/endpoint operation bindings remain distinct from this source-fit bridge.

The P03 artifact inventory and original serialization code also establish that fitted classical estimators were not persisted. Stored predictions remain valid reusable evidence, but cannot evaluate a perturbed spectrum. Reconstruction fits for the later robustness branch require a separately documented scope and budget; their need must not be hidden as prediction-file reuse or authorized by this audit.

## Nested-support milestone release checks

The exact ten-file-overlay publication snapshot passed the public-boundary scan with **883 files checked** and CI-scope Ruff. With temporary Git metadata present from the start, the full CPU regression passed **3,159 tests and 23 subtests**, with **four CUDA-only skips**, in **413.61 seconds**. The **2,026 warnings** are inherited sparse-class metric fixture warnings. No test rule was weakened. The later classical source-audit JSON and these final evidence/test notes are data/documentation-only additions; the final eleven-file release receives another public-boundary, whitespace and focused-contract check before publication.

Lenarizer preservation checks reported added numeric tokens only for the readiness, statistical protocol and review revisions; no original numeric token or citation was removed. The new values were checked against the metadata audits and test output. The changed statistical support fields explicitly implement owner-approved P08-A05, not an undocumented change to the initial 128-context eligibility count. The historical contracts remain untouched.

The preceding milestone `33c4a68ceba89feed1cf488b5b29664aa2e758e5` passed remote GitHub Actions run `36747121742`. This does not attest to the new commit; its exact remote revision and CI status require their own check. The full no-fit goal remains active until the remaining roles, resource budgets, guards and final execution-approval request are ready.

Final eleven-file snapshot checks passed: the public-boundary scan checked **884 files**, and the focused support/planning/preprocessing/input-contract tests passed **62 tests and seven subtests in 4.30 seconds**. Whitespace validation passed. The master-plan preservation check likewise found only the added amendment date and support counts, with original numeric tokens and citations retained. These checks support the scoped milestone publication, not authorization for a scientific run.

## T009/T010: concrete nested QC roles

The preceding support/evidence milestone was published on `main` as `f17f1f3e96eaa1bb988077dc0978305dd5a60738`. Its exact remote SHA matched and GitHub Actions run `36750480085` completed successfully.

DeepSeek authored only the scoped nested-role module and synthetic tests. The initial check passed 25 tests and five subtests but failed one test and three subtests because tests expected an output `classes` field absent from the specified schema. Review also found that the class-specific master-count test pooled masters from every class, and one import-order violation remained. These were test defects, not reasons to expand or weaken the production schema. The supervisor returned them to the same worker.

The correction uses input class metadata, verifies separate per-class master coverage, adds a literal independent assignment fixture, checks reversed membership-list order and covers a synthetic 260-context corpus with 54 eligible contexts and 206 fallbacks. The source description now says classifier-outcome-blind; source labels legitimately stratify the inner roles. No worker received private data or used tools, and neither worker performed scientific fitting.

After correction, all five P08 modules passed **230 tests and 31 subtests in 1.58 seconds**. CI-scope Ruff passed. The supervisor independently applied the reviewed builder to the authenticated actual metadata, reproduced every assignment and canonical hash, and verified all 324 folds: exact parent partition, cross-fold validation coverage, class support, master/instrument exclusion, source-only quantile roles and refusal of a forged execution flag. The exact private registry is hash-bound in the public aggregate audit. This is a role-construction result, not evidence that adaptive preprocessing improves classification.

The supervisor specified the nested gate-selection/calibration sequence and independently checked its literal 1,630,980-fit and 53,880-scalar ceiling. The prospective resource analysis shows that the old checkpoint-retention pattern would require approximately 223.33 GiB for QC neural checkpoints alone, exceeding the observed free storage. Universal and adaptive execution therefore have separate proposed ceilings and approval gates; no retention rule was silently relaxed.

The universal smoke proposal selects 78 existing fit slots and their 78 source-validation prediction slots by fixed metadata/candidate order. Its audit authenticates the parent graph, chosen specifications and complete seed counts without computing an outcome. It is a planning artifact, not a scientific permit or proof that the future runtime is ready. Full regression and publication checks for this new slice remain separate.

## Concrete-role milestone regression and outer-evidence audit

The exact eleven-file-overlay, Git-backed temporary snapshot passed the public-boundary validator with **891 files checked** and CI-scope Ruff. The full CPU regression passed **3,187 tests and 31 subtests**, with **four CUDA-only skips**, in **397.38 seconds**. The **2,026 warnings** are inherited sparse-class metric fixtures. Both DeepSeek sessions and all test processes ended; no scientific run was launched.

The private classical outer audit passed all 780 fixed-family model–context cells after checking 3,979 file hashes. It binds source-selected candidates, fitting/test roles, final-fit records, original scalar states and saved calibration/held predictions. Its first attempt assumed that every calibration file used the same seed-aggregated storage format and stopped on duplicate observation IDs. Inspection of the original runtime established the actual distinction: fresh pseudo-domain calibration files retain individual-seed predictions, whereas cached master-CV calibration files retain their seed aggregate. The corrected auditor verifies the appropriate format in 384 and 396 cells respectively. No historical file, model or score changed.

All held-prediction files contain the technical-seed aggregate. The audit therefore supports reuse of complete MIN endpoints, not nonexistent individual-seed held artifacts or estimator objects. This storage boundary is explicit in the public report and must remain explicit in the final operation-reuse ledger.

Lenarizer preservation checks for the master, readiness and review additions found added numeric tokens only; original numeric tokens and citations were retained. New values were checked against test outputs, the role/smoke audits and prospective arithmetic. The outer-audit JSON and these final notes are data/documentation-only additions after the full regression; final boundary, focused-contract and whitespace checks precede the scoped push. The complete P08 readiness goal and separate scientific execution gate remain open.

Final twelve-file snapshot checks passed: the public-boundary validator checked **892 files**, CI-scope Ruff passed, and focused role/support/planning/preprocessing/input checks passed **98 tests and 15 subtests in 0.89 seconds**. The final release includes no private identifiers, source paths, observations or checkpoints. Exact remote SHA and the new commit's CI require verification after publication.
