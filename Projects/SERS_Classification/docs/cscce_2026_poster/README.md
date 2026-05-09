# CSCCE 2026 SERS Poster

Poster title: **Toward Substrate-Agnostic SERS Classification Using Siamese Metric Learning**

Compile from this directory with:

```bash
tectonic cscce_2026_sers_poster.tex
```

The poster is configured as a 48 in by 48 in LaTeX page.

## Contents

- `cscce_2026_sers_poster.tex`: main poster source.
- `cscce_2026_sers_poster.pdf`: compiled poster.
- `figures/`: generated poster figures.
- `poster_asset_manifest.json`: generation metadata, including CUDA/GPU details for the representative loss trace.
- `image_generation_prompts.md`: optional prompts for generating polished illustrative figures in browser ChatGPT.

## Regenerating Figures

Run from the repository root:

```bash
./.venv/bin/python scripts/generate_cscce_poster_assets.py
```

The asset generator requires CUDA and intentionally does not allow CPU training for the representative substrate-agnostic loss trace.
