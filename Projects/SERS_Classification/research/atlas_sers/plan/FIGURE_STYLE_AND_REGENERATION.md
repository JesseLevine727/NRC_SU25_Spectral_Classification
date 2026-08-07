# ATLAS SERS figure style and regeneration contract

This document is normative. It applies to every plot, schematic, diagnostic, manuscript figure, and supplementary figure produced by the research master plan.

## 1. Required output pair

Every registered figure ID produces the same analysis twice:

```text
figures/data/Fxx_slug.csv or json       frozen plot-level data
figures/tikz/Fxx_slug.tex               native editable TikZ/PGFPlots
figures/pdf/Fxx_slug.pdf                compiled vector verification
figures/html/Fxx_slug.html              standalone interactive quick view
figures/logs/Fxx_slug.pdflatex.log      compilation log
```

A figure is incomplete if either the native TikZ or standalone HTML counterpart is absent.

## 2. Meaning of native TikZ

The `.tex` file must draw marks, lines, boxes, heatmap cells, axes, labels, annotations, uncertainty intervals, and legends using TikZ, PGFPlots, or PGFPlotstable commands. It may read a frozen CSV/TSV table. It must not use `\includegraphics` to place a pre-rendered plot, screenshot, PDF, SVG, PNG, JPEG, or bitmap inside a nominal TikZ wrapper.

Scientific schematics use TikZ nodes and paths. Quantitative plots use PGFPlots. Tables intended as figure panels use TikZ/PGFPlotstable rather than rasterized tables.

The compiled PDF is a validation artifact. The `.tex` source is the publication source of record.

## 3. Meaning of standalone HTML

The HTML must:

- open locally without a server;
- contain a complete HTML document;
- embed all figure data;
- embed Plotly JavaScript or use native inline SVG/JavaScript;
- contain no CDN or remote-script dependency;
- provide hover details for quantitative marks;
- provide zoom, pan, trace toggling, or filtering where meaningful;
- state research-question ID, preprocessing/access regime, population,
  independent unit, and scope in visible text;
- include a visible caption or description;
- expose the frozen data hash.

The HTML may show extra hover metadata or allow traces to be toggled. It may not change the underlying population, aggregation, statistic, interval, or scale.

## 4. Shared plot-level data

This shared table is the enforceable **semantic parity** boundary between publication and quick-view outputs.

TikZ and HTML are generated from one plot-level table. The figure manifest stores its SHA-256. Plotting code must not independently recompute statistics for each output format.

Each plot-level table includes as applicable:

- `figure_id`;
- `research_question_id`;
- `scope`;
- population and representation IDs;
- preprocessing policy, actual action or action summary, policy access, support,
  fallback denominator, model, and task information regime;
- station/domain/target;
- point estimate;
- lower and upper interval;
- independent-unit count;
- aggregation label;
- display order;
- color key;
- line/marker key.

## 5. Visual language

Use the Okabe–Ito palette:

| Key | Hex | Principal use |
|---|---|---|
| blue | `#0072B2` | selected classical model |
| vermillion | `#D55E00` | acquisition-aware deep model |
| green | `#009E73` | compact ERM deep control |
| orange | `#E69F00` | preprocessing or adaptation sensitivity |
| sky blue | `#56B4E9` | calibration/reference |
| purple | `#CC79A7` | open-set or uncertainty result |
| yellow | `#F0E442` | caution/low-support with black boundary |
| black | `#000000` | chance/null/reference text |

Color is always redundant with marker or line style:

- selected classical: blue circle, solid line;
- acquisition-aware deep: vermillion square, solid line;
- ERM deep: green triangle, dashed line;
- sensitivity: orange diamond, dotted line;
- low-support: yellow open marker with black edge;
- chance/null: black dash-dot line.

Every figure must remain interpretable in grayscale.

## 6. Typography and geometry

- Target final width: IEEE double column, 7.16 inches, unless the registry declares single column.
- Minimum final-size text: 8 pt.
- Panel labels: bold uppercase, 10–12 pt.
- Minimum stroke: 0.5 pt.
- Principal data strokes: 0.8–1.2 pt.
- Reference lines: 0.6–0.8 pt.
- Error bars: visible caps and at least 0.6 pt.
- Font: Helvetica-compatible sans serif where available; math in standard LaTeX math.
- Axis label format: sentence case with units in parentheses.
- Avoid rotated tick labels when a horizontal layout or abbreviation can solve the problem.

## 7. Plot-specific rules

### Spectra

- Wavenumber increases left to right unless a journal explicitly requires the reverse.
- Label x axis `Raman shift (cm⁻¹)`.
- State whether intensity is raw, min–max, SNV, vector, area, or baseline corrected.
- Do not vertically offset spectra without labelling the offset.
- Private interactive hover may include UID/master fields. Public-after-review
  HTML uses disclosure-approved aggregate domain/action/support fields only.
- Dense all-spectrum TikZ figures use robust quantile ribbons plus prespecified representative rows; the HTML may expose all spectra.

### Performance comparisons

- Prefer paired-dot, interval, forest, or heatmap displays over unadorned bars.
- Show chance/reference lines.
- Display all 13 domain effects in the primary figure.
- Do not hide failed or chance-level domains in an average.
- Distinguish spectrum and master aggregation in both title and caption.

### Preprocessing-policy comparisons

- Show `PP-U-MIN` as the common paired reference.
- Universal, family-aware, and QC-adaptive policies use distinct line/marker
  encodings; never label all three simply `preprocessed`.
- Family figures show all-domain fallback-inclusive and supported-family
  estimands together, with support and fallback counts.
- QC figures show selected gate/action distributions, missing/invalid fallback,
  and stability in addition to accuracy.
- Interaction figures show paired difference-in-differences with a zero line.
- Preservation violations and worst-domain changes remain visible.

### Calibration and risk–coverage

- Reliability plots include the identity line.
- State binning rule and sample unit.
- Risk–coverage uses realized rather than requested test coverage on the x axis.
- Thresholds are annotated as development-known thresholds.

### Embeddings and clustering

- PCA axes state explained variance.
- UMAP/t-SNE figures say `exploratory projection` in the visible subtitle.
- Report random seed and parameters in caption/hover.
- Never use visual separation as a hypothesis test.
- Use identical coordinates when recoloring the same embedding by target/instrument/sensor.

### Heatmaps

- Use sequential or scientifically centered diverging scales.
- Never use a rainbow palette.
- Missing/unsupported cells are hatched in TikZ and explicitly labelled `NA` in HTML.
- Cell text includes value when legible.

### Neural architecture and workflow diagrams

- Nodes are aligned to a grid.
- Different arrow styles have declared meanings.
- No decorative 3-D layers.
- Tensor sizes, losses, frozen stages, and information boundaries are labelled.
- Training-only and test-only data paths are visually distinct and grayscale redundant.

## 8. Captions

Every caption includes:

1. scope `(P)`, `(S)`, or `(E)`;
2. stable research-question ID and the question shown;
3. preprocessing policy, actual-action meaning, and target-information access;
4. population and representation;
5. independent unit and sample size;
6. adaptive-policy support, coverage, and fallback denominator where relevant;
7. aggregation rule;
8. interval or error-bar definition;
9. chance/null reference;
10. explanation of colors, shapes, and abbreviations;
11. interpretation limit.

Example:

> **(P) Unseen-instrument chemical identification.** Balanced accuracy is shown for the 13 support-qualified station/instrument domains using the 598-spectrum primary population and minimal 400–1,800 cm⁻¹ representation. Points are domains; intervals resample physical masters within target. The dashed line denotes three-class chance. Domain means are not weighted by spectrum count. These results support only the tested field instruments and targets.

## 9. Generation API

Every figure generator accepts:

```text
--input-table PATH
--figure-id Fxx
--tikz-output PATH
--html-output PATH
--pdf-output PATH
--manifest-output PATH
--style-config contracts/figure_contract.json
```

The generator first writes or reads the single frozen plot table, then renders both formats. It records:

- input and plot-data SHA-256;
- command line;
- code/config hashes;
- timestamp;
- row/filter counts;
- semantic axis/legend specification;
- output hashes.

## 10. Validation

The figure validator must:

1. verify every registry path exists;
2. reject `\includegraphics` in native TikZ sources;
3. compile every `.tex` with `pdflatex -halt-on-error`;
4. confirm the PDF is vector and contains no raster image object where inspection tools permit;
5. verify every HTML has `<html>` and closing `</html>`;
6. reject `cdn.plot.ly`, external `<script src=http...>`, or other remote assets;
7. compare the TikZ and HTML plot-data hash stored in both outputs;
8. verify axis labels, units, RQ ID, policy/access/fallback labels, scope, and independent-unit count;
9. verify registered colors and redundant marker/line styles;
10. verify captions are nonempty;
11. verify all output hashes.

## 11. Accessibility review

Before manuscript freeze:

- print PDFs at final size;
- inspect at 25%, 100%, and 400% zoom;
- simulate grayscale and common color-vision deficiencies;
- verify a minimum 4.5:1 text contrast and 3:1 graphical contrast where applicable;
- verify HTML keyboard access and visible focus;
- add plain-language descriptions for all primary figures.

Raster previews may be generated temporarily for accessibility testing, but they are diagnostic fallbacks and never the publication source or required quick-view artifact.
