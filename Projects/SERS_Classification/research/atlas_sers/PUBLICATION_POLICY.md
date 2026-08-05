# ATLAS publication and data-boundary policy

## Public identifier

`ATLAS` is the sole public project identifier. Public filenames, prose,
configuration identifiers, branches, commits, issues, pull requests, figures,
and release notes must use it consistently. Do not expand the code word or
identify the source organization, partners, event, or archive location.

## Allowed public content

- source code that operates on abstract data contracts;
- method descriptions and research hypotheses;
- schemas, split rules, hyperparameter grids, experiment registries, and
  decision gates;
- synthetic fixtures that cannot be mistaken for observations;
- aggregate, disclosure-reviewed metrics and figures;
- native TikZ/PGFPlots and standalone HTML generated from approved aggregates.

## Content that must remain private

- raw spectra or instrument export files;
- interpolated, normalized, smoothed, baseline-corrected, or otherwise derived
  row-level spectra;
- row-level manifests, labels, notes, timestamps, serial numbers, filenames,
  or acquisition paths;
- NumPy arrays, model checkpoints, embeddings, predictions, or fold membership
  that can be joined back to observations;
- original reports, recording logs, PDFs, and source spreadsheets;
- absolute workstation paths, usernames, credentials, tokens, or remote data
  locations;
- text that identifies the organization, partners, event, or archive behind
  the code word.

## Required pre-commit checks

Run:

```bash
python3 scripts/validate_public_scaffold.py
git diff --cached --name-only
git diff --cached --check
```

Review every staged path. Generated artifacts are private by default; an
aggregate result becomes public only after a deliberate disclosure review.

## Incident response

If restricted content is committed, stop pushing immediately. If it has
already reached a remote, deleting it in a later commit is insufficient because
Git history retains the object. Revoke exposed credentials, preserve a local
backup, and coordinate a dedicated history-rewrite and clone-rotation process.
