# Supervisor meeting deck, 21 October 2026

Beamer slides with native TikZ figures for the progress meeting with Li-Lin: dataset reminder,
research questions, classical vs CNN and preprocessing results, substrate portability, and two
paper routes with verified literature.

## Contents

- `deck.tex` / `deck.pdf` - the slides (15 main + 11 backup) and the compiled deck.
- `build_figure_data.py` - extracts aggregate tables into `data/` from the released P08-U1 and P13
  tables, plus coverage counts and a descriptive PCA from the private P01 run.
- `make_figures.py` - writes the TikZ figures in `figures/` from `data/`.
- `literature/` - verified literature checks for Paper 1 (this dataset) and Paper 2 (designed
  dataset for disentangled VAEs).

Scores use the pooled four-fold estimand so they match the 14 September classical report
(Extra Trees 68.5% with minimal preprocessing).

## Build

```bash
export ATLAS_ARTIFACT_ROOT=/path/outside/the/repository/atlas_artifacts
./build.sh
```

## Published and local-only files

`deck.pdf` is the compiled deck as presented, published at the project owner's request
(2026-10-20); its slide-5 PCA panel plots one point per spectrum.

`figures/pca_*.tex` (the per-spectrum PCA coordinates as text) and `private_data/` stay local under
`PUBLICATION_POLICY.md`. Rebuilding them needs access to the private artifact root.
