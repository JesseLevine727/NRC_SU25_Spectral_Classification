# Classical ML supervisor report

This directory contains the concise Li-Lin briefing report for the completed
classical machine-learning benchmark.

## Files

- [NATO_SERS_BRIEF_REPORT.pdf](../NATO_SERS_BRIEF_REPORT.pdf) — one title page plus four content pages.
- [NATO_SERS_BRIEF_REPORT.tex](../NATO_SERS_BRIEF_REPORT.tex) — editable LaTeX source.
- [index.html](index.html) — local index for the interactive figures.
- `figures/tikz/` — native TikZ/PGFPlots sources and generated vector PDFs.
- `figures/html/` — standalone, self-contained HTML quick views with hover data.
- `figures/data/` — report-level plot tables shared by the TikZ and HTML outputs.
- `evidence_manifest.json` — source and output SHA-256 hashes.
- `validation.json` — final page-count, vector, and layout checks.

## Rebuild

From this directory, run:

```bash
/home/elfo/Documents/NRC/ATLAS_venv/bin/python build_report.py
```

The builder reads only the validated aggregate tables under
`research/atlas_sers/results/p03_classical/tables` and the existing public F13
and F38 plot tables. It does not fit models, read row-level predictions, or
modify the frozen P03 execution. It generates the four native-TikZ/HTML figure
pairs and compiles the report.

The final validation requires exactly five pages, black report text, no raster
objects in the report or figure PDFs, no overfull boxes, and no undefined
references. Missing test results remain missing in the plot data and are not
fabricated as zero scores.
