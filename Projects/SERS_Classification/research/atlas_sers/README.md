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

The full locked analysis specification is in [plan/MASTER_PLAN.md](plan/MASTER_PLAN.md).
The concise question-to-experiment map is in
[plan/RESEARCH_QUESTION_MAP.md](plan/RESEARCH_QUESTION_MAP.md).
For quick browsing, open [plan/index.html](plan/index.html) locally.

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

P05 is next: expand and audit the predeclared source-only supervised-contrastive
and paired-consistency experiments before fitting. D0 is now their frozen
control; no D1–D5 model has been trained. The P04 reuse of P13 held test views is
descriptive only: a controlled P13 deep comparison still needs exact
substrate-restricted source refits, matched-source loss, and preprocessing
sensitivities. Neither P13 support nor its portability margins has changed.
