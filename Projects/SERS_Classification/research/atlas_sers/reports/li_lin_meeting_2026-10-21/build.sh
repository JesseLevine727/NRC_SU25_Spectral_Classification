#!/usr/bin/env bash
# Rebuild the 2026-10-21 supervisor deck: data -> TikZ figures -> PDF.
set -euo pipefail
cd "$(dirname "$0")"
: "${ATLAS_ARTIFACT_ROOT:?set ATLAS_ARTIFACT_ROOT to the private artifact root (see atlas_sers/README.md)}"
PY=../../../../.venv/bin/python
"$PY" build_figure_data.py
"$PY" make_figures.py
pdflatex -interaction=nonstopmode deck.tex >/dev/null
pdflatex -interaction=nonstopmode deck.tex >/dev/null
echo "built deck.pdf"
