# ATLAS SERS research

This directory is the public, data-free execution scaffold for the ATLAS
multi-instrument SERS study. The study asks whether acquisition-aware deep
representations improve chemical identification when the test instrument and
physical samples are absent from model fitting.

The repository contains methods, contracts, registries, tests, and figure
specifications. It intentionally contains no spectra, row-level metadata,
derived feature arrays, fitted models, or local source paths.

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

VAE and disentanglement analyses remain diagnostic. Without paired clean
chemical spectra or factorial chemical/nuisance interventions, reconstruction
must not be described as denoising and latent factors must not be interpreted
as causal chemical/nuisance separation.

The full locked analysis specification is in [plan/MASTER_PLAN.md](plan/MASTER_PLAN.md).
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
│   ├── P00_EXECUTION.md           Governance procedure and phase boundary
│   ├── P01_EXECUTION.md           Data/representation freeze and validation
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
├── scripts/validate_public_scaffold.py
└── tests/                         Contract and privacy regression tests
```

See [REPOSITORY_ARCHITECTURE.md](REPOSITORY_ARCHITECTURE.md) for module
boundaries, artifact flow, and implementation order.

## Private data boundary

Set `ATLAS_PRIVATE_ROOT` to the immutable input directory and
`ATLAS_NATIVE_ROOT` to the immutable native vendor-export directory, and set
`ATLAS_ARTIFACT_ROOT` to a separate private output directory outside this
public project. The roots must not overlap one another or this project. Never
copy source spectra into this directory.

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

## Reproducibility rules

- Split by physical `master_sample_id`; spectrum rows are not independent.
- Fit preprocessing, feature selection, calibration, and thresholds on the
  permitted training roles only.
- Keep zero-shot, unlabeled adaptation, paired calibration, and supervised
  few-shot regimes separate.
- Preserve failed and collapsed neural runs in denominators.
- Save row-level predictions privately and publish only disclosure-approved
  aggregate tables.
- Generate every scientific figure as native TikZ/PGFPlots and standalone
  self-contained HTML from the same frozen aggregate table.

## Status

The research plan is execution-ready but is not a prospective preregistration:
pilot results informed its design. P00 governance and P01 data/representation
freeze are executable. P02 split construction and all predictive model/result
phases remain outside the current boundary.
