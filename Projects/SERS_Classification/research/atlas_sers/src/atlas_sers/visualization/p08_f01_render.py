"""Pure renderer for the P08-F01 descriptive spectral aggregate figures.

The module turns the authenticated public spectral bundle into a frozen
semantic specification plus native TikZ/PGFPlots, standalone offline HTML and
approved-aggregate CSV strings.  It performs no file I/O, no fitting, no
averaging, no normalisation, no offsetting, no smoothing and no uncertainty
calculation, and it never publishes or approves a figure.
"""

# Percent-format templates keep literal TikZ/JavaScript braces unchanged.
# ruff: noqa: UP031
from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import math

SCHEMA_VERSION = "nato-sers-p08-f01-spectral-bundle-v1"
SEMANTIC_SCHEMA_VERSION = "nato-sers-p08-f01-semantic-v1"
FIGURE_ID = "P08-F01"
RESEARCH_QUESTION_ID = "RQ-S01"
AXIS_N = 1401
AXIS_MIN = 400.0
AXIS_MAX = 1800.0
ACTION_MIN = "R_MIN_400_1800"
ACTION_SG = "R_SG_400_1800"
ACTION_ARPLS = "R_ARPLS_400_1800"
ACTION_ORDER = (ACTION_MIN, ACTION_SG, ACTION_ARPLS)
MAX_CELLS_PER_DOMAIN = 3
UNAVAILABLE_REASON = "fewer_than_two_physical_masters"
SCOPE = "descriptive primary-population display"
ACCESS = "fixed universal preprocessing; no target fitting"
POPULATION = "598 stored spectra from 69 physical masters; per-cell counts shown"
INDEPENDENT_UNIT = "physical master"
AGGREGATION = (
    "within each physical master and domain, stored views are averaged first; "
    "master-level curves then receive equal weight; no further normalisation "
    "or offset is applied"
)
CAPTION = (
    "MIN is the minimally processed, min-max-scaled representation, not raw "
    "instrument counts. Curves average repeated views within each physical sample "
    "and then give samples equal weight. These are descriptive displays, not "
    "averaged inputs to the classifier; they do not establish recovery of pure "
    "chemical spectra or peak preservation. No uncertainty interval or significance "
    "test is shown."
)

BUNDLE_KEYS = ("schema_version", "figure_id", "axis_cm1", "action_order", "cells", "caption")
CELL_KEYS = (
    "cell_id",
    "station",
    "instrument",
    "analyte",
    "n_spectra",
    "n_masters",
    "held_comparison_domain",
    "available",
    "reason",
    "curves",
)
CSV_HEADER = (
    "cell_id",
    "station",
    "instrument",
    "analyte",
    "n_spectra",
    "n_masters",
    "held_comparison_domain",
    "available",
    "reason",
    "representation_id",
    "raman_shift_cm1",
    "intensity",
)
X_TICKS = (400, 600, 800, 1000, 1200, 1400, 1600, 1800)
Y_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
SVG_W, SVG_H = 960, 300
SVG_ML, SVG_MR, SVG_MT, SVG_MB = 92, 22, 36, 48

STYLES = {
    ACTION_MIN: {
        "label": "MIN",
        "color": "#000000",
        "html_dash": "none",
        "line_style": "solid",
        "marker": "none",
        "tex": "black, solid, line width=1pt",
    },
    ACTION_SG: {
        "label": "SG",
        "color": "#0072B2",
        "html_dash": "9 5",
        "line_style": "dashed",
        "marker": "none",
        "tex": "SGblue, dashed, line width=1pt",
    },
    ACTION_ARPLS: {
        "label": "arPLS",
        "color": "#D55E00",
        "html_dash": "12 4 2 4",
        "line_style": "dashdot",
        "marker": "none",
        "tex": "arPLSvermillion, dash pattern=on 4pt off 2pt on 1pt off 2pt, line width=1pt",
    },
}


class P08F01Error(ValueError):
    """Raised when the public spectral bundle is malformed or carries private extras."""


def _exact_keys(obj, keys, where):
    if not isinstance(obj, dict):
        raise P08F01Error("%s: expected a mapping" % where)
    got = set(obj)
    expected = set(keys)
    if got != expected:
        raise P08F01Error(
            "%s: unexpected keys %s; missing keys %s"
            % (where, sorted(got - expected), sorted(expected - got))
        )


def _pos_int(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise P08F01Error("%s: expected a positive integer (bool refused)" % where)
    return value


def _label(value, where):
    if not isinstance(value, str) or not value.strip():
        raise P08F01Error("%s: expected a non-empty string" % where)
    return value


def _axis(axis):
    if not isinstance(axis, list) or len(axis) != AXIS_N:
        raise P08F01Error("axis_cm1: expected %d values" % AXIS_N)
    clean = []
    for i, value in enumerate(axis):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise P08F01Error("axis_cm1[%d]: non-numeric" % i)
        value = float(value)
        expected = AXIS_MIN + i
        if not math.isfinite(value) or value != expected:
            raise P08F01Error("axis_cm1[%d]: expected exactly %.1f" % (i, expected))
        clean.append(value)
    return clean


def _actions(action_order):
    if not isinstance(action_order, (list, tuple)) or list(action_order) != list(ACTION_ORDER):
        raise P08F01Error("action_order: expected %s" % (list(ACTION_ORDER),))
    return list(ACTION_ORDER)


def _curve(values, where):
    if not isinstance(values, list) or len(values) != AXIS_N:
        raise P08F01Error("%s: expected %d intensity values" % (where, AXIS_N))
    clean = []
    for i, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise P08F01Error("%s[%d]: non-numeric intensity" % (where, i))
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise P08F01Error("%s[%d]: intensity outside [0,1]" % (where, i))
        clean.append(value)
    return clean


def _cells(cells, actions):
    if not isinstance(cells, list) or not cells:
        raise P08F01Error("cells: a non-empty list is required")
    expected_actions = set(actions)
    clean = []
    seen_ids = set()
    seen_keys = set()
    previous = None
    for index, cell in enumerate(cells):
        where = "cells[%d]" % index
        _exact_keys(cell, CELL_KEYS, where)
        cell_id = _label(cell["cell_id"], where + ".cell_id")
        if cell_id in seen_ids:
            raise P08F01Error("%s.cell_id: duplicate %r" % (where, cell_id))
        seen_ids.add(cell_id)
        station = _label(cell["station"], where + ".station")
        instrument = _label(cell["instrument"], where + ".instrument")
        analyte = _label(cell["analyte"], where + ".analyte")
        n_spectra = _pos_int(cell["n_spectra"], where + ".n_spectra")
        n_masters = _pos_int(cell["n_masters"], where + ".n_masters")
        if n_masters > n_spectra:
            raise P08F01Error("%s: n_masters must not exceed n_spectra" % where)
        held = cell["held_comparison_domain"]
        available = cell["available"]
        if not isinstance(held, bool):
            raise P08F01Error("%s.held_comparison_domain: expected bool" % where)
        if not isinstance(available, bool):
            raise P08F01Error("%s.available: expected bool" % where)
        reason = cell["reason"]
        if not isinstance(reason, str):
            raise P08F01Error("%s.reason: expected string" % where)
        curves = cell["curves"]
        if not isinstance(curves, dict):
            raise P08F01Error("%s.curves: expected a mapping" % where)
        if available:
            if n_masters < 2:
                raise P08F01Error("%s: available cell needs at least two physical masters" % where)
            if set(curves) != expected_actions:
                raise P08F01Error(
                    "%s.curves: expected exactly the three registered actions" % where
                )
            if reason != "":
                raise P08F01Error("%s.reason: available cells must have an empty reason" % where)
            clean_curves = {a: _curve(curves[a], "%s.curves[%s]" % (where, a)) for a in actions}
        else:
            if n_masters >= 2:
                raise P08F01Error("%s: unavailable cell must have fewer than two masters" % where)
            if curves:
                raise P08F01Error("%s.curves: unavailable cells must have empty curves" % where)
            if reason != UNAVAILABLE_REASON:
                raise P08F01Error("%s.reason: expected %r" % (where, UNAVAILABLE_REASON))
            clean_curves = {}
        key = (station, instrument, analyte)
        if key in seen_keys:
            raise P08F01Error("%s: duplicate domain-analyte key %r" % (where, key))
        seen_keys.add(key)
        if previous is not None and key <= previous:
            raise P08F01Error("%s: cells must be sorted by station, instrument, analyte" % where)
        previous = key
        clean.append(
            {
                "cell_id": cell_id,
                "station": station,
                "instrument": instrument,
                "analyte": analyte,
                "n_spectra": n_spectra,
                "n_masters": n_masters,
                "held_comparison_domain": held,
                "available": available,
                "reason": reason,
                "curves": clean_curves,
            }
        )
    return clean


def _validate_bundle(spectral):
    _exact_keys(spectral, BUNDLE_KEYS, "spectral")
    if spectral["schema_version"] != SCHEMA_VERSION:
        raise P08F01Error("spectral.schema_version: unsupported")
    if spectral["figure_id"] != FIGURE_ID:
        raise P08F01Error("spectral.figure_id: expected %s" % FIGURE_ID)
    axis = _axis(spectral["axis_cm1"])
    actions = _actions(spectral["action_order"])
    caption = spectral["caption"]
    if not isinstance(caption, str):
        raise P08F01Error("spectral.caption: expected string")
    cells = _cells(spectral["cells"], actions)
    return axis, actions, caption, cells


def _canonical_json(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "^": r"\textasciicircum{}",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
}


def _tex_escape(text):
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in text)


def _html_escape(text):
    return html.escape(str(text), quote=True)


def _json_for_script(obj):
    return (
        _canonical_json(obj).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )


def _fmt(value):
    return format(float(value), ".17g")


def _tex_coords(axis, values):
    parts = []
    for i in range(len(values)):
        parts.append("(" + _fmt(axis[i]) + "," + _fmt(values[i]) + ")")
    return "".join(parts)


_TEX_PREAMBLE = (
    "\\documentclass[10pt,border=0pt]{standalone}\n"
    "\\usepackage[T1]{fontenc}\n"
    "\\usepackage[utf8]{inputenc}\n"
    "\\usepackage{pgfplots}\n"
    "\\pgfplotsset{compat=1.18}\n"
    "\\usepgfplotslibrary{groupplots}\n"
    "\\definecolor{SGblue}{HTML}{0072B2}\n"
    "\\definecolor{arPLSvermillion}{HTML}{D55E00}\n"
    "\\renewcommand{\\familydefault}{\\rmdefault}\n"
    "\\begin{document}\n"
)


def _render_tex(domain, cells, axis, sha, source_caption):
    esc = _tex_escape
    n = len(cells)
    first_available = None
    for index, cell in enumerate(cells):
        if cell["available"]:
            first_available = index
            break
    # Preserve the registered outer width and fonts while clearing glyph edges.
    out = [
        _TEX_PREAMBLE,
        "\\begin{minipage}{181.86mm}\n",
        "\\vspace*{6pt}\\noindent\\hspace*{3pt}\n",
        "\\begin{tikzpicture}\n",
    ]
    out.append("\\begin{groupplot}[\n")
    out.append("  group style={group size=1 by %d, vertical sep=1.65cm},\n" % n)
    out.append("  width=17cm, height=4.6cm,\n")
    out.append("  xmin=400, xmax=1800, ymin=0, ymax=1,\n")
    out.append("  xtick={400,600,800,1000,1200,1400,1600,1800},\n")
    out.append("  ytick={0,0.25,0.5,0.75,1},\n")
    out.append("  tick label style={font=\\fontsize{8}{9.6}\\selectfont, text=black},\n")
    out.append("  label style={font=\\fontsize{8}{9.6}\\selectfont, text=black},\n")
    out.append(
        "  title style={font=\\bfseries\\fontsize{10}{12}\\selectfont, text=black, align=center},\n"
    )
    out.append("  axis line style={black, line width=0.6pt},\n")
    out.append("  tick style={black, line width=0.6pt},\n")
    out.append("  ylabel={Mean scaled intensity},\n")
    out.append("]\n")
    for index, cell in enumerate(cells):
        label = "ABC"[index] if index < 3 else "X%d" % (index + 1)
        title = esc(
            "%s. %s (physical masters: %d; stored spectra: %d)"
            % (label, cell["analyte"], cell["n_masters"], cell["n_spectra"])
        )
        options = "title={%s}" % title
        if index == n - 1:
            options = options + ", xlabel={Raman shift (cm$^{-1}$)}"
        out.append("\\nextgroupplot[%s]\n" % options)
        if cell["available"]:
            for action in ACTION_ORDER:
                out.append(
                    "\\addplot[%s] coordinates {%s};\n"
                    % (STYLES[action]["tex"], _tex_coords(axis, cell["curves"][action]))
                )
        else:
            out.append(
                "\\node[anchor=center, text=black, "
                "font=\\fontsize{8}{9.6}\\selectfont, text width=5.5cm, align=center] "
                "at (axis description cs:0.5,0.5) "
                "{Unavailable: fewer than two physical samples};\n"
            )
    out.append("\\end{groupplot}\n")
    tag = (
        "EXPLORATORY (outside held comparison; displayed, not part of held tests)"
        if domain["exploratory"]
        else "held comparison domain"
    )
    heading = (
        "\\textbf{%s}\\quad %s\\quad %s\\quad %s\\\\"
        "%s\\\\"
        "Population: %s\\\\"
        "Access: %s\\\\"
        "Independent unit: %s\\quad Scope: %s\\\\"
        "semantic sha256: {\\fontsize{8}{9.6}\\selectfont\\texttt{%s}}"
        % (
            esc(domain["domain_id"]),
            RESEARCH_QUESTION_ID,
            esc(domain["station"]),
            esc(domain["instrument"]),
            esc(tag),
            esc(POPULATION),
            esc(ACCESS),
            esc(INDEPENDENT_UNIT),
            esc(SCOPE),
            sha,
        )
    )
    out.append(
        "\\node[anchor=south west, align=left, inner sep=0pt, "
        "font=\\fontsize{8}{9.6}\\selectfont, text=black, text width=166mm] "
        "at ([yshift=22mm]group c1r1.north west) {%s};\n" % heading
    )
    if first_available is not None:
        legend = (
            "\\tikz[baseline=-0.6ex]{"
            "\\draw[black, solid, line width=1pt] (0,0) -- (0.7,0);"
            "\\node[anchor=west, font=\\fontsize{8}{9.6}\\selectfont, text=black, "
            "inner sep=1pt] at (0.75,0) {MIN};"
            "\\draw[SGblue, dashed, line width=1pt] (1.9,0) -- (2.6,0);"
            "\\node[anchor=west, font=\\fontsize{8}{9.6}\\selectfont, text=black, "
            "inner sep=1pt] at (2.65,0) {SG};"
            "\\draw[arPLSvermillion, dash pattern=on 4pt off 2pt on 1pt off 2pt, "
            "line width=1pt] (3.9,0) -- (4.6,0);"
            "\\node[anchor=west, font=\\fontsize{8}{9.6}\\selectfont, text=black, "
            "inner sep=1pt] at (4.65,0) {arPLS};}"
        )
        out.append(
            "\\node[anchor=south west, inner sep=0pt] "
            "at ([yshift=12mm]group c1r1.north west) {%s};\n" % legend
        )
    caption = CAPTION
    if source_caption:
        caption = caption + " Source caption: " + source_caption
    out.append(
        "\\node[anchor=north west, align=left, inner sep=0pt, "
        "font=\\fontsize{8}{9.6}\\selectfont, text=black, text width=166mm] "
        "at ([yshift=-14mm]group c1r%d.south west) {%s};\n" % (n, esc(caption))
    )
    out.append("\\end{tikzpicture}\\par\\vspace*{6pt}\n\\end{minipage}\n\\end{document}\n")
    return "".join(out)


def _tick_label(value):
    if value == int(value):
        return str(int(value))
    return "%g" % value


def _svg_cell(cell, axis, index):
    esc = _html_escape
    w, h = SVG_W, SVG_H
    pw, ph = w - SVG_ML - SVG_MR, h - SVG_MT - SVG_MB
    x0, x1 = float(axis[0]), float(axis[-1])

    def px(value):
        return SVG_ML + (float(value) - x0) / (x1 - x0) * pw

    def py(value):
        return SVG_MT + (1.0 - float(value)) * ph

    label = "ABC"[index] if index < 3 else "X%d" % (index + 1)
    title = "%s. %s (physical masters: %d; stored spectra: %d)" % (
        label,
        cell["analyte"],
        cell["n_masters"],
        cell["n_spectra"],
    )
    aria = "%s %s %s %s" % (cell["cell_id"], cell["station"], cell["instrument"], cell["analyte"])
    out = [
        '<svg class="cell" role="img" aria-label="%s" viewBox="0 0 %d %d" '
        'data-cell-id="%s" data-ml="%d" data-mr="%d" data-w="%d" '
        'xmlns="http://www.w3.org/2000/svg">\n'
        % (esc(aria), w, h, esc(cell["cell_id"]), SVG_ML, SVG_MR, w)
    ]
    out.append('<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>\n' % (w, h))
    out.append(
        '<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#000000" '
        'stroke-width="0.8"/>\n' % (SVG_ML, SVG_MT, pw, ph)
    )
    for yv in Y_TICKS:
        y = py(yv)
        out.append(
            '<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#000000" '
            'stroke-width="0.8"/>\n' % (SVG_ML - 5, y, SVG_ML, y)
        )
        out.append(
            '<text class="tick" x="%d" y="%.2f" text-anchor="end">%s</text>\n'
            % (SVG_ML - 8, y + 4, _tick_label(yv))
        )
    for xv in X_TICKS:
        x = px(xv)
        out.append(
            '<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#000000" '
            'stroke-width="0.8"/>\n' % (x, SVG_MT + ph, x, SVG_MT + ph + 5)
        )
        out.append(
            '<text class="tick" x="%.2f" y="%d" text-anchor="middle">%d</text>\n'
            % (x, SVG_MT + ph + 20, xv)
        )
    out.append(
        '<text class="axis" x="%.2f" y="%d" text-anchor="middle">'
        "Raman shift (cm\u207b\u00b9)</text>\n" % (SVG_ML + pw / 2.0, h - 8)
    )
    out.append(
        '<text class="axis" x="18" y="%.2f" text-anchor="middle" '
        'transform="rotate(-90 18 %.2f)">Mean scaled intensity</text>\n'
        % (SVG_MT + ph / 2.0, SVG_MT + ph / 2.0)
    )
    out.append('<text class="title" x="%d" y="22">%s</text>\n' % (SVG_ML, esc(title)))
    if cell["available"]:
        for action in ACTION_ORDER:
            style = STYLES[action]
            values = cell["curves"][action]
            points = []
            for i in range(len(values)):
                points.append("%.2f,%.2f" % (px(axis[i]), py(values[i])))
            trace_title = "%s | %s | %s | %s | %s | physical masters: %d; stored spectra: %d" % (
                cell["cell_id"],
                cell["station"],
                cell["instrument"],
                cell["analyte"],
                action,
                cell["n_masters"],
                cell["n_spectra"],
            )
            out.append(
                '<path fill="none" stroke="%s" stroke-width="1.33" '
                'stroke-linejoin="round" stroke-dasharray="%s" data-action="%s" '
                'd="M%s"><title>%s</title></path>\n'
                % (
                    style["color"],
                    style["html_dash"],
                    esc(action),
                    " L".join(points),
                    esc(trace_title),
                )
            )
    else:
        out.append(
            '<text class="unavail" x="%.2f" y="%.2f" text-anchor="middle">'
            "Unavailable: fewer than two physical samples</text>\n"
            % (SVG_ML + pw / 2.0, SVG_MT + ph / 2.0)
        )
    out.append("</svg>\n")
    return "".join(out)


def _render_csv(cells, axis):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for cell in cells:
        base = [
            cell["cell_id"],
            cell["station"],
            cell["instrument"],
            cell["analyte"],
            cell["n_spectra"],
            cell["n_masters"],
            cell["held_comparison_domain"],
            cell["available"],
            cell["reason"],
        ]
        if cell["available"]:
            for action in ACTION_ORDER:
                values = cell["curves"][action]
                for i in range(AXIS_N):
                    writer.writerow(base + [action, _fmt(axis[i]), _fmt(values[i])])
        else:
            writer.writerow(base + ["", "", ""])
    return buf.getvalue()


_HTML_CSS = (
    "body{font-family:'Times New Roman',Times,serif;color:#000;background:#fff;"
    "margin:24px;line-height:1.45}"
    "h1{font-size:22px} .meta{font-size:16px} .exploratory{font-size:17px;font-weight:bold}"
    ".hash{font-size:16px} .hash code{font-family:'Courier New',Courier,monospace;"
    "font-size:16px;overflow-wrap:anywhere}"
    ".caption{font-size:16px;max-width:960px} .panel{max-width:960px}"
    "svg.cell{display:block;width:100%;height:auto;margin:6px 0 20px 0}"
    "text{fill:#000;font-family:'Times New Roman',Times,serif}"
    ".tick{font-size:16px} .axis{font-size:17px} .title{font-size:18px;font-weight:bold}"
    ".unavail{font-size:16px;font-style:italic}"
    ".swatch{display:inline-flex;align-items:center;gap:6px;margin:0 18px 6px 0;font-size:16px}"
    ".controls label{margin-right:18px;font-size:16px}"
    "#p08-f01-tip{position:fixed;display:none;background:#fff;border:1px solid #000;"
    "padding:6px 8px;font-size:16px;white-space:pre-line;pointer-events:none;z-index:10}"
    "a{color:#000}"
)

_HTML_JS = r"""<script>
(function(){
  var SEM = JSON.parse(document.getElementById('p08-f01-semantic').textContent);
  var byId = {};
  SEM.cells.forEach(function(c){ byId[c.cell_id] = c; });
  var AXIS = SEM.axis_cm1;
  var tip = document.getElementById('p08-f01-tip');
  document.querySelectorAll('svg.cell').forEach(function(svg){
    svg.addEventListener('mousemove', function(ev){
      var cell = byId[svg.getAttribute('data-cell-id')];
      if (!cell || !cell.available) { tip.style.display = 'none'; return; }
      var ml = parseFloat(svg.getAttribute('data-ml'));
      var mr = parseFloat(svg.getAttribute('data-mr'));
      var w = parseFloat(svg.getAttribute('data-w'));
      var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      var loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      var frac = (loc.x - ml) / (w - ml - mr);
      if (frac < 0 || frac > 1) { tip.style.display = 'none'; return; }
      var idx = Math.round(frac * (AXIS.length - 1));
      if (idx < 0) { idx = 0; }
      if (idx > AXIS.length - 1) { idx = AXIS.length - 1; }
      var lines = [cell.station + ' | ' + cell.instrument + ' | ' + cell.analyte];
      lines.push('physical masters: ' + cell.n_masters + '; stored spectra: ' + cell.n_spectra);
      SEM.action_order.forEach(function(a){
        var cb = document.querySelector('input[data-action="' + a + '"]');
        if (cb && !cb.checked) { return; }
        if (cell.curves[a]) {
          lines.push(SEM.style.actions[a].label + ' (' + a + '): '
                     + cell.curves[a][idx].toFixed(4));
        }
      });
      lines.push('Raman shift: ' + AXIS[idx] + ' cm\u207b\u00b9');
      tip.textContent = lines.join('\n');
      tip.style.display = 'block';
      tip.style.left = (ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
    });
    svg.addEventListener('mouseleave', function(){ tip.style.display = 'none'; });
  });
  document.querySelectorAll('input[data-action]').forEach(function(cb){
    cb.addEventListener('change', function(){
      var a = cb.getAttribute('data-action');
      document.querySelectorAll('path[data-action="' + a + '"]').forEach(function(p){
        p.style.display = cb.checked ? '' : 'none';
      });
    });
  });
})();
</script>
"""


def _render_html(domain, cells, axis, sha, semantic):
    esc = _html_escape
    tag = (
        "EXPLORATORY (outside held comparison; displayed, not part of held tests)"
        if domain["exploratory"]
        else "held comparison domain"
    )
    out = [
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8"/>\n',
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n',
        "<title>%s %s %s</title>\n"
        % (esc(FIGURE_ID), esc(domain["domain_id"]), esc(domain["station"])),
        "<style>%s</style>\n</head>\n<body>\n" % _HTML_CSS,
    ]
    out.append(
        "<h1>%s %s &#8212; %s | %s</h1>\n"
        % (
            esc(FIGURE_ID),
            esc(domain["domain_id"]),
            esc(domain["station"]),
            esc(domain["instrument"]),
        )
    )
    out.append(
        '<p class="meta">%s &#183; %s &#183; RQ-S01 &#183; station: %s &#183; '
        "instrument: %s &#183; population: %s &#183; independent unit: %s &#183; "
        "access: %s &#183; scope: %s</p>\n"
        % (
            esc(FIGURE_ID),
            esc(domain["domain_id"]),
            esc(domain["station"]),
            esc(domain["instrument"]),
            esc(POPULATION),
            esc(INDEPENDENT_UNIT),
            esc(ACCESS),
            esc(SCOPE),
        )
    )
    out.append('<p class="hash">semantic sha256: <code>%s</code></p>\n' % sha)
    out.append('<p class="exploratory">%s</p>\n' % esc(tag))
    out.append('<div class="controls" role="group" aria-label="trace visibility">\n')
    for action in ACTION_ORDER:
        out.append(
            '<label><input type="checkbox" data-action="%s" checked="checked"/> '
            "%s</label>\n" % (esc(action), esc(STYLES[action]["label"]))
        )
    out.append('</div>\n<div class="legend">\n')
    for action in ACTION_ORDER:
        style = STYLES[action]
        out.append(
            '<span class="swatch"><svg width="28" height="12" aria-hidden="true" '
            'xmlns="http://www.w3.org/2000/svg"><line x1="1" y1="6" x2="27" y2="6" '
            'stroke="%s" stroke-width="2" stroke-dasharray="%s"/></svg>%s</span>\n'
            % (style["color"], style["html_dash"], esc(style["label"]))
        )
    out.append('</div>\n<div class="panel">\n')
    for index, cell in enumerate(cells):
        out.append(_svg_cell(cell, axis, index))
    out.append("</div>\n")
    encoded = base64.b64encode(_render_csv(cells, axis).encode("utf-8")).decode("ascii")
    out.append(
        '<p><a download="%s.csv" href="data:text/csv;base64,%s">'
        "Download approved aggregate data (CSV)</a></p>\n" % (esc(domain["domain_id"]), encoded)
    )
    caption = CAPTION
    if semantic.get("source_caption"):
        caption = caption + " Source caption: " + semantic["source_caption"]
    out.append('<p class="caption">%s</p>\n' % esc(caption))
    out.append('<div id="p08-f01-tip" role="status" aria-live="polite"></div>\n')
    out.append(
        '<script type="application/json" id="p08-f01-semantic">%s</script>\n'
        % _json_for_script(semantic)
    )
    out.append(_HTML_JS)
    out.append("</body>\n</html>\n")
    return "".join(out)


def prepare_f01(spectral):
    """Return the frozen semantic spec, its hash and every domain panel."""
    axis, actions, caption, cells = _validate_bundle(spectral)
    groups = {}
    for cell in cells:
        groups.setdefault((cell["station"], cell["instrument"]), []).append(cell)
    for (station, instrument), group_cells in groups.items():
        if len(group_cells) > MAX_CELLS_PER_DOMAIN:
            raise P08F01Error(
                "domain %s/%s: at most %d cells are supported"
                % (station, instrument, MAX_CELLS_PER_DOMAIN)
            )
        if len({c["held_comparison_domain"] for c in group_cells}) != 1:
            raise P08F01Error(
                "domain %s/%s: mixed held_comparison_domain flags" % (station, instrument)
            )
    domains = []
    for index, ((station, instrument), group_cells) in enumerate(sorted(groups.items()), 1):
        domains.append(
            {
                "domain_id": "P08-F01-D%02d" % index,
                "station": station,
                "instrument": instrument,
                "exploratory": not group_cells[0]["held_comparison_domain"],
                "cell_ids": [c["cell_id"] for c in group_cells],
            }
        )
    semantic = {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "research_question_id": RESEARCH_QUESTION_ID,
        "scope": SCOPE,
        "access": ACCESS,
        "population": POPULATION,
        "independent_unit": INDEPENDENT_UNIT,
        "aggregation": AGGREGATION,
        "caption": CAPTION,
        "source_caption": caption,
        "claim_limits": CAPTION,
        "axis": {
            "x": {
                "label_tex": "Raman shift (cm$^{-1}$)",
                "label_text": "Raman shift (cm\u207b\u00b9)",
                "unit": "cm^-1",
                "min": AXIS_MIN,
                "max": AXIS_MAX,
                "ticks": list(X_TICKS),
                "scale": "linear",
            },
            "y": {
                "label_text": "Mean scaled intensity",
                "min": 0.0,
                "max": 1.0,
                "ticks": list(Y_TICKS),
                "scale": "linear",
            },
        },
        "style": {
            "font": {
                "tex": "Computer Modern Roman",
                "html": "Times New Roman, Times, serif",
                "color": "#000000",
            },
            "actions": {
                a: {
                    "label": STYLES[a]["label"],
                    "color": STYLES[a]["color"],
                    "line_style": STYLES[a]["line_style"],
                    "marker": STYLES[a]["marker"],
                }
                for a in ACTION_ORDER
            },
            "grayscale_redundancy": True,
        },
        "axis_cm1": axis,
        "action_order": list(ACTION_ORDER),
        "domains": domains,
        "cells": cells,
    }
    sha = _sha256(_canonical_json(semantic))
    by_id = {c["cell_id"]: c for c in cells}
    panels = []
    for domain in domains:
        domain_cells = [by_id[cid] for cid in domain["cell_ids"]]
        panels.append(
            {
                "slug": domain["domain_id"],
                "domain_id": domain["domain_id"],
                "station": domain["station"],
                "instrument": domain["instrument"],
                "exploratory": domain["exploratory"],
                "semantic_sha256": sha,
                "tex": _render_tex(domain, domain_cells, axis, sha, caption),
                "html": _render_html(domain, domain_cells, axis, sha, semantic),
                "csv": _render_csv(domain_cells, axis),
            }
        )
    manifest = {
        "figure_id": FIGURE_ID,
        "schema_version": "nato-sers-p08-f01-render-manifest-v1",
        "research_question_id": RESEARCH_QUESTION_ID,
        "semantic_sha256": sha,
        "panel_slugs": [p["slug"] for p in panels],
        "status": "prepared",
        "reviewed": False,
        "published": False,
        "note": "unreviewed pure-renderer output; compile and visual inspection are separate",
    }
    return {"semantic": semantic, "semantic_sha256": sha, "panels": panels, "manifest": manifest}
