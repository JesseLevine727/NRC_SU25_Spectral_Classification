# NATO SERS publication and repository-boundary policy

## Public identity and data status

This is a public research repository for the NATO field-trial SERS dataset.
`NATO SERS` may be used directly in filenames, prose, figures, reports, commit
messages, and release notes. The former `ATLAS` label is retained only where
changing a frozen protocol ID, environment variable, Python import path, or
artifact hash would break reproducibility.

The source archive under the repository's top-level `2026July21/` directory is
public by project-owner decision. Do not duplicate that archive inside
`research/atlas_sers/`. This research package contains maintained code,
contracts, aggregate results, and publication figures; large caches and
checkpoints remain local unless explicitly selected as release artifacts.

## Content suitable for GitHub

- source data already designated public by the project owner;
- source code, schemas, split rules, hyperparameter grids, and decision gates;
- aggregate metrics and figures that have a documented publication review;
- native TikZ/PGFPlots, vector PDF, PNG review copies, and standalone HTML;
- support matrices containing recorded master IDs when intentionally released;
- compact reports that state denominators, missing endpoints, and limitations.

## Content excluded from routine commits

- credentials, tokens, private keys, or access-controlled source locations;
- absolute workstation paths and usernames;
- temporary fit caches, incomplete shards, lock files, and quarantine content;
- large model checkpoints unless they are a deliberate, documented release;
- LaTeX build products such as `.aux`, `.log`, `.fls`, and `.synctex.gz`;
- redundant clean-rebuild trees and orchestration logs;
- exploratory outputs that have not been reconciled with an authoritative run.

Row-level predictions and fold assignments may be published only when the
commit explicitly identifies them as intended research data. Aggregate P03
tables and F12/F13/F38–F43 are approved for publication; fit caches,
checkpoints, and the full terminal ledger are not part of that release.

## Required pre-commit checks

Run from `research/atlas_sers/`:

```bash
python3 scripts/validate_public_scaffold.py
ruff check src scripts tests
pytest
git diff --cached --check
```

Review every staged path. Never use `git add -A` from this repository while
large generated workspaces are present.

## Historical compatibility

Existing names such as `research/atlas_sers`, the `atlas_sers` Python package,
`${ATLAS_PRIVATE_ROOT}`, `${ATLAS_NATIVE_ROOT}`, `${ATLAS_ARTIFACT_ROOT}`, and
`atlas-sers-*` protocol identifiers are compatibility interfaces. Their
continued presence does not imply anonymization. New public-facing titles and
new protocol prose should use NATO SERS.
