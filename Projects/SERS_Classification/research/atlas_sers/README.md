# NATO field-trial SERS research

This directory is the maintained execution and evidence package for the public
NATO multi-instrument SERS field-trial dataset. The study asks whether acquisition-aware deep
representations improve chemical identification when the test instrument and
physical samples are absent from model fitting. Parallel questions test whether
universal, acquisition-platform-family-aware, or identity-blind row-QC
preprocessing changes that conclusion without leaking held-test outcomes.

The wider repository contains the source archive. This maintained research
package contains methods, contracts, registries, curated aggregate results, and
publication figures. It intentionally excludes temporary caches, unrestricted
checkpoint sweeps, and local workstation paths.

## Scientific direction

The primary comparison is a leakage-controlled, station-conditioned,
source-only unseen-instrument evaluation:

1. establish rigorously tuned classical chemometric and machine-learning
   baselines;
2. train a compact acquisition-aware one-dimensional encoder;
3. compare the methods on identical physical-master-grouped partitions;
4. measure domain-balanced performance, calibration, robustness, and nuisance
   predictability; and
5. route the paper according to predefined evidence gates, including a valid
   classical-only or negative deep-learning result.

Preprocessing policy and learning strategy are orthogonal axes. The primary
classical/deep comparison remains fixed under universal minimal min–max.
Secondary experiments cross the same model panel and splits with universal SG
or arPLS, a source-selected platform-family rule, and a source-selected
row-local QC gate. Target-data adaptation remains a separate information
regime. Arbitrary post-test per-instrument preprocessing is prohibited.

VAE and disentanglement analyses remain diagnostic. Without paired clean
chemical spectra or factorial chemical/nuisance interventions, reconstruction
must not be described as denoising and latent factors must not be interpreted
as causal chemical/nuisance separation.

The planned [P14 RBF-network and self-organizing-map extension](plan/P14_RBF_SOM_EXTENSION.md)
adds compact prototype classification, acquisition-structure maps, repeated-view
diagnostics, preprocessing comparisons, and physical-master learning curves.
It is a dated post-outcome secondary/exploratory branch, not a replacement for
the original primary question. No P14 models or result figures exist yet.
Its [publication strategy](plan/P14_PUBLICATION_STRATEGY.md) separates the core
study from optional reliability and hybrid-model experiments.

The master analysis specification and dated amendments are in [plan/MASTER_PLAN.md](plan/MASTER_PLAN.md).
The concise question-to-experiment map is in
[plan/RESEARCH_QUESTION_MAP.md](plan/RESEARCH_QUESTION_MAP.md).
For quick browsing, open [plan/index.html](plan/index.html) locally.

Implementation follows the [supervised OpenCode workflow](plan/ORCHESTRATION_PROTOCOL.md):
DeepSeek v4.1 Flash implements bounded tasks; the supervising assistant plans,
reviews, verifies, and controls advancement. The first P05 readiness audit and
[contract-only inventory implementation](plan/delegation/P05_TASK_002_REVIEW.md)
are documented. The [source-support audit](plan/P05_SOURCE_SUPPORT_AUDIT.md)
checks all 1,181 inherited fitting roles and identifies sparse surface-training
groups. The dated [core protocol](plan/P05_CORE_PROTOCOL.md) and
[contract](plan/contracts/p05_core_contract.json) now lock the sampler, losses,
nested source-only selection and finite budgets. The [numerical smoke passed](results/p05_smoke/P05_SMOKE_RESULTS.md):
32 primary fits and two planned replays, eight epochs each, with all checkpoints
verified. The [core review](plan/delegation/P05_CORE_REVIEW.md) records 523 passing
local tests and independent acceptance checks. The [failure/recovery record](plan/P05_SMOKE_STATUS.md)
preserves the original checkpoint-saving failure and the owner-approved recovery:
35 total executions including that failed-save attempt. These historical smoke
diagnostics are not generalization results; the completed comprehensive benchmark
is linked below.
View the [actual loss curves](results/p05_smoke/figures/P05S01_training_ce.html)
and [gradient curves](results/p05_smoke/figures/P05S02_gradient_norm.html)
(offline HTML; native TikZ, PDF and PNG are alongside).
The subsequent **[36-fit source-validation pilot](results/p05_pilot/P05_PILOT_RESULTS.md)**
is also complete: 30–122 epochs, 6,880 updates, 72 audited checkpoints, and
station-dependent results, including three single-class CWA checkpoints.
Start with its [training–validation scatter](results/p05_pilot/figures/P05P02_best_checkpoints.html)
and [learning curves](results/p05_pilot/figures/P05P01_learning_curves.html).
The [development review](plan/delegation/P05_DEVELOPMENT_REVIEW.md) records
718 passing local pre-launch tests and independent acceptance. The 36-fit pilot
permit is exhausted; its original fits were subsequently reused in the separately
approved comprehensive benchmark.

The **[completed four-recipe core benchmark](results/p05_comprehensive/P05_RESULTS.md)**
now includes 14,940 source-evidence fits, 320 frozen context-local decisions,
1,995 distinct final refits/calibrations and complete held evaluation.
The [35-figure browser](results/p05_comprehensive/index.html) includes native
TikZ, offline HTML and PDF/PNG versions. Fixed D3 gained 1.68 percentage points
over the matched CNN on individual spectra, but gains varied across domains;
the source-selected procedure gained only 0.17 points. Classical tree ensembles
remain strong comparators. See the [completion audit](plan/P05_COMPLETION_AUDIT.md)
for coverage, recovery accounting and the remaining P06/P11 boundary.
The **[P06/P11 frozen-prediction analysis](reports/NATO_SERS_UNCERTAINTY_REPORT.md)**
now includes the approved 10,000-draw weighted addition, the original hierarchical
feasibility analysis, deletion checks and supplementary probability metrics.
The selected CNN gains 4.36 percentage points over selected classical at M01
(conditional interval 0.68–7.79), but added-loss and tree-ensemble comparisons
remain inconclusive. The original G4 gate is unpassed: four criteria are supported
and two unassessable. See the [four-figure browser](results/p06p11/index.html)
and [review record](plan/delegation/P06P11_REVIEW.md).
No models were retrained. The next boundary is the **[P08 no-fit preprocessing
planning gate](plan/P08_HANDOFF.md)**, not an automatically authorized sweep.
The **[P08 readiness audit](plan/P08_READINESS.md)** now verifies the three frozen
input pipelines and their held-split support. The owner approved the matched-CNN
mapping, explicit Extra Trees inclusion in the universal panel, the statistical
amendment and the complete minimal-pipeline fallback for unsupported families.
The approved nested-support rule retains 54 QC-adaptive contexts, all at CWA,
and 206 complete minimal-pipeline fallbacks. All 260 remain in operational
results, with the supported subset reported separately. The
[current requirement audit](plan/P08_COMPLETION_AUDIT.md) reconciles upstream
evidence, stage accounting, reporting ownership and the reviewed invented-data
U0 launcher. [Resource ceilings](plan/P08_RESOURCE_PROPOSAL.md) remain proposals.
The separate filtered-population Extra Trees choice, final release checks and
a distinct U0 execution request remain. Later numerical runtimes retain their
own acceptance gates; they are not all prerequisites for this planning lock.
No new P08 models have been trained.
Reviewed and validated substantive milestones are pushed to
`main` under the project owner's authorization.

## Repository map

```text
research/atlas_sers/
├── README.md                     Project entry point
├── PUBLICATION_POLICY.md          Public/private boundary
├── CONTRIBUTING.md                Safe contribution workflow
├── pyproject.toml                 Python package and tool configuration
├── data/README.md                 Private-data mount contract; no data
├── artifacts/README.md            Local output contract; no outputs
├── plan/
│   ├── MASTER_PLAN.md             Research questions, phases, gates, claims
│   ├── RESEARCH_QUESTION_MAP.md    Precise RQ comparisons and interpretations
│   ├── P00_EXECUTION.md           Governance procedure and phase boundary
│   ├── P01_EXECUTION.md           Data/representation freeze and validation
│   ├── P02_EXECUTION.md           Evaluation-design freeze and leakage audit
│   ├── P03_HANDOFF.md             Immutable classical consumer contract
│   ├── P03_EXECUTION.md           No-fit expansion and protected run boundary
│   ├── P03_DECISION_MEMO.md       Pre-fit compute/control decisions
│   ├── P03_COMPLETION_AUDIT.md    Requirement-to-evidence completion matrix
│   ├── P04_EXECUTION.md           Compact D0 architecture and source-only training
│   ├── P04_COMPLETION_AUDIT.md    D0 reconciliation, comparison, and limitations
│   ├── P13_PROTOCOL.md            Locked substrate-portability amendment
│   ├── P13_EXECUTION.md           Deterministic no-fit execution expansion
│   ├── P13_COMPLETION_AUDIT.md    Classical portability completion evidence
│   ├── P14_RBF_SOM_EXTENSION.md  Planned prototype methods, questions, and figures
│   ├── P14_PUBLICATION_STRATEGY.md Literature basis and evidence-to-paper routes
│   ├── FIGURE_STYLE_AND_REGENERATION.md
│   ├── index.html                 Standalone plan dashboard
│   ├── contracts/                 Machine-readable frozen protocols
│   ├── registries/                Phase/task/metric/experiment/figure tables
│   └── figures/                   Data-free TikZ, HTML, and vector plan figures
├── src/atlas_sers/
│   ├── governance/                P00/P01 audit, provenance, hashes, restart
│   ├── data/                      Private ingestion interfaces and QC
│   ├── preprocessing/             Frozen and sensitivity representations
│   ├── exploration/               PCA, clustering, UMAP, and t-SNE analyses
│   ├── splits/                    Master-grouped leakage-safe partitions
│   ├── models/                    Classical and deep model families
│   ├── evaluation/                Metrics, calibration, bootstrap, robustness
│   └── visualization/             Paired TikZ/HTML figure generation
├── scripts/run_p00.py             No-training governance audit/dry run
├── scripts/run_p01.py             Private data/representation freeze
├── scripts/run_p02.py             Private evaluation-design freeze
├── scripts/run_p03.py             Classical planning and gated shard runner
├── scripts/run_p04.py             Compact D0 planning, training, and comparison
├── scripts/publish_p04_results.py Aggregate D0 report and four-format figures
├── scripts/validate_public_scaffold.py
└── tests/                         Contract and privacy regression tests
```

See [REPOSITORY_ARCHITECTURE.md](REPOSITORY_ARCHITECTURE.md) for module
boundaries, artifact flow, and implementation order.

## Governed data and artifact locations

The existing execution code retains the original compatibility variables
`ATLAS_PRIVATE_ROOT`, `ATLAS_NATIVE_ROOT`, and `ATLAS_ARTIFACT_ROOT` because
renaming them would invalidate frozen commands and artifact identities. They
identify immutable inputs and governed run outputs; they are not a public
pseudonym. Keep generated run stores outside this maintained package even when
the underlying source dataset is public.

```bash
export ATLAS_PRIVATE_ROOT=/path/outside/the/repository/atlas_inputs
export ATLAS_NATIVE_ROOT=/different/path/outside/the/repository/atlas_native_sources
export ATLAS_ARTIFACT_ROOT=/different/path/outside/the/repository/atlas_artifacts
```

The expected private files and their frozen checksums are identified in
`plan/contracts/research_contract.json`. Their contents are not published.

## Quick start

From this directory:

```bash
python3 scripts/validate_public_scaffold.py
python3 -m pip install -e '.[dev]'
python3 scripts/run_p00.py audit
pytest -q
python3 scripts/run_p00.py dry-run
python3 scripts/run_p01.py audit
python3 scripts/run_p01.py dry-run
python3 scripts/run_p02.py audit
python3 scripts/run_p02.py dry-run
```

Install the `deep` extra only for neural experiments:

```bash
python3 -m pip install -e '.[deep]'
```

The P00 dry run verifies the private inputs and governance state but imports no
training modules, authorizes no fit, and materializes no representation. It
writes twelve private governance artifacts beneath
`${ATLAS_ARTIFACT_ROOT}/p00/runs/<run_id>/` and updates a sanitized private
`p00/LATEST.json` pointer. A repeated, unchanged successful invocation must
return `verified_skip`. See [plan/P00_EXECUTION.md](plan/P00_EXECUTION.md) for
the exact outputs, statuses, and failure behavior.

P01 then creates the source-reversible 598-row primary manifest, two frozen
sensitivity populations, eight row-local representations, preservation and
descriptive structure analyses, and paired F02–F09 TikZ/PDF/HTML figures. It
performs no predictive fit and constructs no split. See
[plan/P01_EXECUTION.md](plan/P01_EXECUTION.md) for the exact build, restart,
validation, outputs, and failure behavior.

P02 freezes five four-fold physical-master repeats, all 13 primary
held-instrument domains, exact source/target/exclusion roles, source-only inner
selection routes, platform-family support/fallback, the finite identity-blind
QC gate library, target-access draws, and held-chemical roles. It performs zero
predictive fits. See [plan/P02_EXECUTION.md](plan/P02_EXECUTION.md) for the
validated design and [plan/P03_HANDOFF.md](plan/P03_HANDOFF.md) for the next
phase's immutable consumer contract.

P03 completed its governed classical benchmark after a deterministic no-fit
plan and explicit approval of the 250,000-fit ceiling, source-to-source
covariance control, and frozen negative controls. All 225 selection shards and
8,082 executable outer/control shards reached validated terminal states; the
260,356-row fit ledger, expected endpoints, predictions, diagnostics, eight
four-format figures, report, and exact 260-cell P04 comparator freeze then
passed independent final validation. The disclosure-reviewed
[aggregate P03 report](results/p03_classical/P03_CLASSICAL_RESULTS.md), tables,
and F12/F13/F38–F43 figure set are published under `results/p03_classical/` and
`plan/figures/`. Row predictions, fit caches, and the full terminal ledger
remain outside the maintained publication package.
See [plan/P03_EXECUTION.md](plan/P03_EXECUTION.md) and
[plan/P03_DECISION_MEMO.md](plan/P03_DECISION_MEMO.md).

## Reproducibility rules

- Split by physical `master_sample_id`; spectrum rows are not independent.
- Fit preprocessing, feature selection, calibration, and thresholds on the
  permitted training roles only.
- Keep zero-shot, unlabeled adaptation, paired calibration, and supervised
  few-shot regimes separate.
- Record preprocessing policy, actual action, policy-access regime, platform
  family, fallback, and policy hash independently from model identity.
- Never select a transform from held-test labels, scores, or target-batch QC in
  a zero-shot regime.
- Preserve failed and collapsed neural runs in denominators.
- Save row-level predictions privately and publish only disclosure-approved
  aggregate tables.
- Generate every scientific figure as native TikZ/PGFPlots and standalone
  self-contained HTML from the same frozen aggregate table.

## Status

The research plan is execution-ready but is not a prospective preregistration:
pilot and P01 descriptive results informed its design. P00 governance, P01
data/representation freeze, P02 evaluation-design freeze, and P03 classical
benchmark are complete. The post-P03 P13 substrate-portability amendment was
locked and its classical experiments C01–C04 completed on 2026-09-04. No
substrate family met the locked instrument-portability decision across every
confirmatory domain; performance and the benefit of baseline correction were
condition-dependent. See the
[P13 results](results/p13_portability/P13_RESULTS.md) and
[completion audit](plan/P13_COMPLETION_AUDIT.md).

P04 compact D0 execution is complete: 16,458 fits, 320 complete evaluation
contexts, and 960 final checkpoints. The 208,691-parameter ordinary residual
classifier achieved mean unseen-instrument spectrum balanced accuracy 0.711
(worst domain 0.379). Its pooled paired gain over C-SELECTED was +0.050
(conditional 95% interval +0.022 to +0.078), but it showed no clear advantage
over fixed Random Forest or Extra Trees. Probability calibration remains a
limitation. See the [P04 results](results/p04_deep/P04_RESULTS.md),
[completion audit](plan/P04_COMPLETION_AUDIT.md), and
[interactive comparison](plan/figures/html/F48_deep_classical_comparison.html).

**P05 scientific completion, 2026-09-29:** the approved four-recipe core benchmark
completed source training, nested selection, refits, calibration, held prediction,
aggregation, frozen-reference comparison and reporting. Independent publication
authentication passed for 159 public files, including 35 four-format figures.
All three new strategies cover all 260 held contexts; eight historical selected-
classical references remain incomplete/missing and are explicitly retained.
The [results](results/p05_comprehensive/P05_RESULTS.md) are descriptive, not final
P11 inference or completion of the wider P06 programme.

The matched D0-M control shares the all-master sampler with D1–D3; historical
P04 D0 remains immutable. Fixed D3 has a small average gain over D0-M, but the
source-selected procedure changes little and no universal deep superiority is
established. Remaining synthesis/uncertainty must preserve these frozen outcomes;
P08 preprocessing and P14 prototype experiments require separate bounded scope.
D4/D5 are deferred with zero allocated fits. The P04 reuse of P13 held test views is
descriptive only: a controlled P13 deep comparison still needs exact
substrate-restricted source refits, matched-source loss, and preprocessing
sensitivities. Neither P13 support nor its portability margins has changed.
