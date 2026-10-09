# ruff: noqa: E501, UP031
# Literal TeX/HTML/JS templates retain percent formatting to avoid brace escaping errors.
"""P08-F07 renderer-only: accepted semantic -> native TeX + standalone HTML.

Pure in-memory strings. No fits, data access, network access or new statistics.
Helpers are reused from the reviewed sibling ``p08_f02_render`` module.
"""

from __future__ import annotations

import math
import re

from atlas_sers.visualization.p08_f02_render import (
    ENDPOINT_LABELS,
    ENDPOINTS,
    ESTIMANDS,
    MODEL_LABELS,
    MODELS,
    _canonical_sha,
    _html,
    _json_script,
    _tex,
)
from atlas_sers.visualization.p08_f07_data import (
    ACTION_BY_REPRESENTATION,
    ACTION_ORDER,
    METRIC_ORDER,
    POINT_FIELDS,
    SUMMARY_FIELDS,
)

FIGURE_ID = "P08-F07"
RESEARCH_QUESTION_ID = "RQ-S05"

POLICIES = ("PP-U-MIN", "PP-U-SG", "PP-U-ARPLS")
POLICY_LABELS = {"PP-U-MIN": "MIN", "PP-U-SG": "SG", "PP-U-ARPLS": "arPLS"}
POLICY_TEX_MARKS = {"PP-U-MIN": "triangle*", "PP-U-SG": "*", "PP-U-ARPLS": "square*"}
POLICY_TEX_COLORS = {"PP-U-MIN": "f07min", "PP-U-SG": "f07sg", "PP-U-ARPLS": "f07arpls"}
POLICY_HTML_MARKS = {"PP-U-MIN": "triangle", "PP-U-SG": "circle", "PP-U-ARPLS": "square"}
POLICY_FILLS = {"PP-U-MIN": "#000000", "PP-U-SG": "#0072B2", "PP-U-ARPLS": "#D55E00"}
POLICY_SYMBOLS = {"PP-U-MIN": "\u25b2", "PP-U-SG": "\u25cf", "PP-U-ARPLS": "\u25a0"}

AXES = {
    "A": {
        "x_field": "peak_displacement_median_cm1",
        "finite": "peak_displacement_finite_count",
        "undefined": "peak_displacement_undefined_count",
        "title": "Peak displacement",
        "tex_label": r"Peak displacement (cm$^{-1}$)",
        "html_label": "Peak displacement (cm\u207b\u00b9)",
    },
    "B": {
        "x_field": "peak_recall_median",
        "finite": "peak_recall_finite_count",
        "undefined": "peak_recall_undefined_count",
        "title": "Peak recall",
        "tex_label": r"Peak recall ($\pm$5 cm$^{-1}$)",
        "html_label": "Peak recall (\u00b15 cm\u207b\u00b9)",
    },
}

N_POINTS = 1020
N_SUMMARIES = 561
N_DOMAINS = 17
N_HELD = 13
N_EXPLORATORY = 4
N_PANELS = 20
N_RECORDS_PER_PANEL = 51

_TICK_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PREPARED_KEYS = frozenset({"semantic", "semantic_sha256", "manifest"})
_REQUIRED_SEMANTIC = (
    "figure_id",
    "research_question_id",
    "schema_version",
    "source_model_semantic_sha256",
    "caption",
    "summaries",
    "points",
)

_CSS = (
    'html,body{background:#fff;color:#000;font:16px "Times New Roman",Times,serif;margin:0;padding:16px}'
    "h1{font-size:20px}h2{font-size:16px}h3{font-size:16px}"
    ".f07-controls label{margin-right:12px}"
    ".f07-controls button{font-size:16px}"
    "svg{max-width:100%;height:auto;border:0}"
    ".f07-table{overflow:auto;max-height:360px;border:1px solid #000}"
    "table{border-collapse:collapse;font-size:14px}"
    "th,td{border:1px solid #999;padding:2px 4px;white-space:nowrap;text-align:left}"
    ".f07-pt:focus{outline:3px solid #000}"
    ".f07-tooltip{position:fixed;display:none;background:#fff;color:#000;border:1px solid #000;"
    'padding:4px 6px;font:14px "Times New Roman",Times,serif;white-space:pre-wrap;'
    "z-index:10;max-width:560px;pointer-events:none}"
    "code{font-family:monospace}"
)

_JS = (
    "(function(){"
    "var data=JSON.parse(document.getElementById('f07-data').textContent);"
    "var root=document.getElementById('f07-panel');"
    "var boxes=Array.prototype.slice.call(document.querySelectorAll('input[data-policy]'));"
    "function apply(){boxes.forEach(function(b){"
    "var sel='.f07-pt[data-policy=\"'+b.getAttribute('data-policy')+'\"]';"
    "Array.prototype.forEach.call(root.querySelectorAll(sel),function(el){"
    "el.style.display=b.checked?'':'none';});});hide();}"
    "boxes.forEach(function(b){b.addEventListener('change',apply);});"
    "var tip=document.getElementById('f07-tooltip');"
    "function move(x,y){if(x===undefined||x===null){return;}"
    "tip.style.left=(x+14)+'px';tip.style.top=(y+14)+'px';}"
    "function show(el,x,y){tip.textContent=el.getAttribute('aria-label');"
    "tip.style.display='block';move(x,y);}"
    "function hide(){tip.style.display='none';}"
    "document.addEventListener('mouseover',function(e){"
    "var el=e.target&&e.target.closest?e.target.closest('.f07-pt'):null;"
    "if(el){show(el,e.clientX,e.clientY);}});"
    "document.addEventListener('mousemove',function(e){"
    "var el=e.target&&e.target.closest?e.target.closest('.f07-pt'):null;"
    "if(el&&tip.style.display==='block'){move(e.clientX,e.clientY);}});"
    "document.addEventListener('mouseout',function(e){"
    "var el=e.target&&e.target.closest?e.target.closest('.f07-pt'):null;"
    "if(el){hide();}});"
    "document.addEventListener('focusin',function(e){"
    "var el=e.target&&e.target.closest?e.target.closest('.f07-pt'):null;"
    "if(el){var r=el.getBoundingClientRect();show(el,r.left,r.bottom);}});"
    "document.addEventListener('focusout',function(e){"
    "var el=e.target&&e.target.closest?e.target.closest('.f07-pt'):null;"
    "if(el){hide();}});"
    "function blob(text,name,type){var b=new Blob([text],{type:type});"
    "var url=URL.createObjectURL(b);var a=document.createElement('a');"
    "a.href=url;a.download=name;document.body.appendChild(a);a.click();"
    "document.body.removeChild(a);setTimeout(function(){URL.revokeObjectURL(url);},0);}"
    "document.getElementById('f07-download').addEventListener('click',function(){"
    "blob(data.csv,data.slug+'.csv','text/csv');});"
    "document.getElementById('f07-semantic-download').addEventListener('click',function(){"
    "blob(document.getElementById('f07-semantic').textContent,data.slug+'-semantic.json','application/json');});"
    "})();"
)


def _deepcopy(obj):
    if isinstance(obj, dict):
        return {k: _deepcopy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deepcopy(v) for v in obj]
    return obj


def _attr(text):
    return _html(text).replace('"', "&quot;").replace("\n", "&#10;")


def _check_fields(rows, fields, name):
    if not isinstance(rows, list):
        raise ValueError("%s table must be a list" % name)
    wanted = set(fields)
    out = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("%s[%d] must be a mapping" % (name, index))
        if set(row) != wanted:
            raise ValueError("%s[%d] field whitelist mismatch" % (name, index))
        if any(isinstance(value, (dict, list, tuple, set)) for value in row.values()):
            raise ValueError("%s[%d] non-scalar public field" % (name, index))
        out.append({key: _deepcopy(row[key]) for key in fields})
    return out


def _coord(value, name, lo=None, hi=None):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a real number or null" % name)
    num = float(value)
    if math.isnan(num) or math.isinf(num):
        raise ValueError("%s must be finite" % name)
    if lo is not None and num < lo:
        raise ValueError("%s below %r" % (name, lo))
    if hi is not None and num > hi:
        raise ValueError("%s above %r" % (name, hi))
    return num


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)


def _fmt_num(value):
    num = float(value)
    return str(int(num)) if num.is_integer() else "%.6g" % num


def _tex_numbers(values):
    return ",".join("%.17g" % float(value) for value in values)


def _slug(*parts):
    raw = "f07-" + "-".join(str(part).lower().replace("_", "-") for part in parts)
    return "".join(ch for ch in raw if ch.isalnum() or ch in "-.")


def _short_estimand(estimand):
    low = str(estimand).lower()
    if "pool" in low or "sens" in low:
        return "Pooled sensitivity (S)"
    return "Primary (P)"


def _heading(model, endpoint, estimand):
    return "P08-F07 / RQ-S05 | %s | %s | %s" % (
        MODEL_LABELS[model],
        ENDPOINT_LABELS[endpoint],
        _short_estimand(estimand),
    )


def _validate_semantic(semantic):
    if not isinstance(semantic, dict):
        raise ValueError("semantic must be a mapping")
    if set(semantic) != set(_REQUIRED_SEMANTIC):
        raise ValueError("semantic keys must be exactly %s" % (_REQUIRED_SEMANTIC,))
    for key in (
        "figure_id",
        "research_question_id",
        "schema_version",
        "source_model_semantic_sha256",
        "caption",
    ):
        if not isinstance(semantic[key], str):
            raise ValueError("semantic.%s must be a non-nested string" % key)
    if semantic["figure_id"] != FIGURE_ID:
        raise ValueError("unexpected figure_id %r" % (semantic["figure_id"],))
    if semantic["research_question_id"] != RESEARCH_QUESTION_ID:
        raise ValueError("unexpected research_question_id %r" % (semantic["research_question_id"],))
    if semantic["schema_version"] != "nato-sers-p08-f07-data-v1":
        raise ValueError("unexpected source schema")
    if not _HEX64.match(semantic["source_model_semantic_sha256"]):
        raise ValueError("semantic.source_model_semantic_sha256 must be a 64-character hex string")
    if not isinstance(semantic["points"], list) or not isinstance(semantic["summaries"], list):
        raise ValueError("semantic.points and semantic.summaries must be lists")


def _validate_points(rows):
    seen = set()
    for index, row in enumerate(rows):
        for key in ("station", "instrument", "model_id", "estimand", "endpoint", "policy_id"):
            if not isinstance(row[key], str):
                raise ValueError("point[%d].%s must be a string" % (index, key))
        if row["model_id"] not in MODELS:
            raise ValueError("point[%d] unknown model_id %r" % (index, row["model_id"]))
        if row["estimand"] not in ESTIMANDS:
            raise ValueError("point[%d] unknown estimand %r" % (index, row["estimand"]))
        if row["endpoint"] not in ENDPOINTS:
            raise ValueError("point[%d] unknown endpoint %r" % (index, row["endpoint"]))
        if row["policy_id"] not in POLICIES:
            raise ValueError("point[%d] unknown policy_id %r" % (index, row["policy_id"]))
        held = row["held_comparison_domain"]
        available = row["available"]
        if type(held) is not bool:
            raise ValueError("point[%d].held_comparison_domain must be an exact bool" % index)
        if type(available) is not bool:
            raise ValueError("point[%d].available must be an exact bool" % index)
        accuracy = _coord(row["balanced_accuracy"], "point[%d].balanced_accuracy" % index, 0.0, 1.0)
        if held != available:
            raise ValueError("point[%d] held_comparison_domain/available disagree" % index)
        if held and accuracy is None:
            raise ValueError("point[%d] held point missing balanced_accuracy" % index)
        if not held and accuracy is not None:
            raise ValueError(
                "point[%d] exploratory point must carry null balanced_accuracy" % index
            )
        _coord(
            row["peak_displacement_median_cm1"],
            "point[%d].peak_displacement_median_cm1" % index,
            0.0,
            None,
        )
        _coord(row["peak_recall_median"], "point[%d].peak_recall_median" % index, 0.0, 1.0)
        key = (
            row["station"],
            row["instrument"],
            row["policy_id"],
            row["estimand"],
            row["endpoint"],
            row["model_id"],
        )
        if key in seen:
            raise ValueError("duplicate point row %r" % (key,))
        seen.add(key)


def _validate_summaries(rows):
    representations = set(ACTION_BY_REPRESENTATION)
    metrics = set(METRIC_ORDER)
    seen = set()
    for index, row in enumerate(rows):
        for key in ("station", "instrument", "representation_id", "metric", "action"):
            if not isinstance(row[key], str):
                raise ValueError("summary[%d].%s must be a string" % (index, key))
        if row["action"] not in ACTION_ORDER:
            raise ValueError("summary[%d] unknown action %r" % (index, row["action"]))
        if row["representation_id"] not in representations:
            raise ValueError(
                "summary[%d] unknown representation_id %r" % (index, row["representation_id"])
            )
        if row["metric"] not in metrics:
            raise ValueError("summary[%d] unknown metric %r" % (index, row["metric"]))
        if ACTION_BY_REPRESENTATION[row["representation_id"]] != row["action"]:
            raise ValueError("summary[%d] representation/action identities disagree" % index)
        key = (row["station"], row["instrument"], row["representation_id"], row["metric"])
        if key in seen:
            raise ValueError("duplicate summary key %r" % (key,))
        seen.add(key)


def _axis_metadata(rows):
    """Freeze axis maxima once from ALL data, then derive the shared tick positions."""
    maximum = 0.0
    for row in rows:
        value = _coord(
            row["peak_displacement_median_cm1"], "peak_displacement_median_cm1", 0.0, None
        )
        if value is not None and value > maximum:
            maximum = value
    x_max = max(1.0, float(math.ceil(maximum)))
    return {
        "x_displacement_min": 0.0,
        "x_displacement_max": x_max,
        "x_recall_min": 0.0,
        "x_recall_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
        "ticks": {
            "A": [fraction * x_max for fraction in _TICK_FRACTIONS],
            "B": [fraction for fraction in _TICK_FRACTIONS],
            "y": [fraction for fraction in _TICK_FRACTIONS],
        },
    }


def _series(records, axis_key, xmax):
    spec = AXES[axis_key]
    out = {policy: [] for policy in POLICIES}
    na = 0
    for row in records:
        x = _coord(row[spec["x_field"]], spec["x_field"], 0.0, xmax)
        y = _coord(row["balanced_accuracy"], "balanced_accuracy", 0.0, 1.0)
        if x is None or y is None:
            na += 1
            continue
        out[row["policy_id"]].append({"x": x, "y": y, "rec": row})
    return out, na


def _tex_legend():
    lines = []
    for policy in POLICIES:
        lines.append(
            r"\addlegendimage{only marks, mark=%s, mark size=2.2pt, "
            r"mark options={fill=%s, draw=black, line width=0.8pt}}"
            % (POLICY_TEX_MARKS[policy], POLICY_TEX_COLORS[policy])
        )
        lines.append(r"\addlegendentry{%s}" % _tex(POLICY_LABELS[policy]))
    return lines


def _tex_axis(series):
    lines = []
    for policy in POLICIES:
        items = series[policy]
        if not items:
            continue
        coords = " ".join("(%.17g,%.17g)" % (item["x"], item["y"]) for item in items)
        lines.append(
            r"\addplot+[only marks, mark=%s, mark size=2.2pt, "
            r"mark options={fill=%s, draw=black, line width=0.8pt}] coordinates {%s};"
            % (POLICY_TEX_MARKS[policy], POLICY_TEX_COLORS[policy], coords)
        )
    return lines


def _panel_caption(base, na_a, na_b):
    return (
        "RQ-S05. Each plotted marker is one accepted point: a comparison domain crossed with an action "
        "(policy), not an individual spectrum. Panel A shows stored median peak displacement; "
        "panel B shows stored median reference-peak recall. Both use balanced accuracy on y. "
        "MIN (black triangle), SG (blue circle) and "
        "arPLS (vermillion square) are overlaid. MIN uses minimal min-max scaling; SG combines "
        "impulse replacement and smoothing; arPLS combines impulse replacement and baseline "
        "correction. All use 400-1800 cm^-1 and final [0,1] scaling. "
        "Per-axis NA points (A=%d, B=%d) remain in the table/CSV and "
        "are not plotted, so the two axes may carry different plotted denominators. Stored 10th/90th "
        "percentiles are summaries, not confidence intervals. " % (na_a, na_b)
    ) + base


def _tex_panel(slug, heading, series_a, series_b, axes, caption, sha):
    ticks = axes["ticks"]
    y_numbers = _tex_numbers(ticks["y"])
    a_numbers = _tex_numbers(ticks["A"])
    b_numbers = _tex_numbers(ticks["B"])
    y_labels = ",".join(_fmt_num(value) for value in ticks["y"])
    a_labels = ",".join(_fmt_num(value) for value in ticks["A"])
    b_labels = ",".join(_fmt_num(value) for value in ticks["B"])
    parts = [
        r"\documentclass[border=0pt]{standalone}",
        r"\usepackage{lmodern}",
        r"\usepackage{tikz}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.17}",
        r"\definecolor{f07min}{HTML}{000000}",
        r"\definecolor{f07sg}{HTML}{0072B2}",
        r"\definecolor{f07arpls}{HTML}{D55E00}",
        r"\begin{document}",
        r"\begin{minipage}{181.86mm}",
        r"\centering",
        r"%% figure: %s; panel: %s; semantic-sha256: %s" % (FIGURE_ID, slug, sha),
        r"{\bfseries\fontsize{10}{12}\selectfont %s\par}" % _tex(heading),
        r"\vspace{2mm}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=10mm, vertical sep=0mm}, "
        r"width=75mm, height=65mm, clip=false, axis line style={line width=0.6pt}, tick style={line width=0.6pt}, "
        r"tick label style={font=\fontsize{8}{9.6}\selectfont, black}, "
        r"label style={font=\fontsize{8}{9.6}\selectfont, black}, "
        r"title style={font=\bfseries\fontsize{10}{12}\selectfont}, "
        r"legend style={at={(0.5,1.18)}, anchor=south, draw=none, fill=none, legend columns=3, "
        r"font=\fontsize{8}{9.6}\selectfont}, "
        r"ymin=0, ymax=1, ytick={%s}, yticklabels={%s}]" % (y_numbers, y_labels),
        r"\nextgroupplot[title={(A) Peak displacement}, xlabel={%s}, ylabel={Balanced accuracy}, "
        r"xmin=%s, xmax=%s, xtick={%s}, xticklabels={%s}]"
        % (
            AXES["A"]["tex_label"],
            _fmt_num(axes["x_displacement_min"]),
            _fmt_num(axes["x_displacement_max"]),
            a_numbers,
            a_labels,
        ),
    ]
    parts.extend(_tex_legend())
    parts.extend(_tex_axis(series_a))
    parts.append(
        r"\nextgroupplot[title={(B) Peak recall}, xlabel={%s}, ylabel={Balanced accuracy}, "
        r"xmin=0, xmax=1, xtick={%s}, xticklabels={%s}]"
        % (AXES["B"]["tex_label"], b_numbers, b_labels)
    )
    parts.extend(_tex_axis(series_b))
    parts.append(r"\end{groupplot}")
    parts.append(
        r"\node[anchor=north, yshift=-6mm, text width=175mm, align=left, "
        r"font=\fontsize{8}{9.6}\selectfont] at (current bounding box.south) {%s};" % _tex(caption)
    )
    parts.extend([r"\end{tikzpicture}", r"\end{minipage}", r"\end{document}"])
    return "\n".join(parts)


def _tooltip(rec):
    return (
        "station=%s | instrument=%s | model=%s | policy=%s\n"
        "balanced accuracy=%s\n"
        "(A) peak displacement median=%s finite=%s undefined=%s\n"
        "(B) peak recall median=%s finite=%s undefined=%s\n"
        "original: n_spectra=%s n_masters=%s\n"
        "held model support: domain=%s contexts=%s unit_appearances=%s physical_masters=%s distinct_units=%s"
        % (
            rec["station"],
            rec["instrument"],
            rec["model_id"],
            POLICY_LABELS[rec["policy_id"]],
            _fmt(rec["balanced_accuracy"]),
            _fmt(rec["peak_displacement_median_cm1"]),
            rec["peak_displacement_finite_count"],
            rec["peak_displacement_undefined_count"],
            _fmt(rec["peak_recall_median"]),
            rec["peak_recall_finite_count"],
            rec["peak_recall_undefined_count"],
            rec["original_n_spectra"],
            rec["original_n_masters"],
            rec["model_domain"],
            rec["model_contexts"],
            rec["model_unit_appearances"],
            rec["model_physical_masters"],
            rec["model_distinct_units"],
        )
    )


def _svg_subplot(series, axis_key, xmin, xmax, xticks, yticks, na, offset):
    spec = AXES[axis_key]
    left, width, top, height = 64.0, 288.0, 52.0, 250.0
    span = (xmax - xmin) or 1.0
    out = ['<g transform="translate(%.1f,0)">' % offset]
    out.append(
        '<rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" fill="none" stroke="#000" stroke-width="1"/>'
        % (left, top, width, height)
    )
    for tick in xticks:
        xx = left + (float(tick) - xmin) / span * width
        out.append(
            '<line x1="%.2f" y1="%.0f" x2="%.2f" y2="%.0f" stroke="#bbb" stroke-width="0.6"/>'
            % (xx, top, xx, top + height)
        )
        out.append(
            '<text x="%.2f" y="%.1f" font-size="16" fill="#000" text-anchor="middle">%s</text>'
            % (xx, top + height + 20, _html(_fmt_num(tick)))
        )
    for tick in yticks:
        yy = top + height - float(tick) * height
        out.append(
            '<line x1="%.0f" y1="%.2f" x2="%.0f" y2="%.2f" stroke="#bbb" stroke-width="0.6"/>'
            % (left, yy, left + width, yy)
        )
        out.append(
            '<text x="%.1f" y="%.2f" font-size="16" fill="#000" text-anchor="end">%s</text>'
            % (left - 6, yy + 5, _html(_fmt_num(tick)))
        )
    out.append(
        '<text x="%.1f" y="%.1f" font-size="16" fill="#000" font-weight="bold" text-anchor="middle">'
        "(%s) %s</text>" % (left + width / 2, top - 18, axis_key, _html(spec["title"]))
    )
    out.append(
        '<text x="%.1f" y="%.1f" font-size="16" fill="#000" text-anchor="middle">%s</text>'
        % (left + width / 2, top + height + 44, _html(spec["html_label"]))
    )
    out.append(
        '<text x="18" y="%.1f" font-size="16" fill="#000" text-anchor="middle" '
        'transform="rotate(-90 18 %.1f)">Balanced accuracy</text>'
        % (top + height / 2, top + height / 2)
    )
    out.append(
        '<text x="%.1f" y="%.1f" font-size="16" fill="#000" text-anchor="end">NA=%d</text>'
        % (left + width, top - 18, na)
    )
    for policy in POLICIES:
        for item in series[policy]:
            x = left + (item["x"] - xmin) / span * width
            y = top + height - item["y"] * height
            tooltip = _tooltip(item["rec"])
            out.append(
                '<g class="f07-pt" data-policy="%s" tabindex="0" role="button" aria-label="%s" '
                'transform="translate(%.1f,%.1f)">' % (_attr(policy), _attr(tooltip), x, y)
            )
            out.append("<title>%s</title>" % _html(tooltip))
            fill = POLICY_FILLS[policy]
            mark = POLICY_HTML_MARKS[policy]
            if mark == "circle":
                out.append('<circle r="4.2" fill="%s" stroke="#000" stroke-width="0.8"/>' % fill)
            elif mark == "square":
                out.append(
                    '<rect x="-4.2" y="-4.2" width="8.4" height="8.4" fill="%s" stroke="#000" '
                    'stroke-width="0.8"/>' % fill
                )
            else:
                out.append(
                    '<polygon points="0,-5 -4.6,3.8 4.6,3.8" fill="%s" stroke="#000" stroke-width="0.8"/>'
                    % fill
                )
            out.append("</g>")
    out.append("</g>")
    return "".join(out)


def _table(records):
    head = "".join("<th>%s</th>" % _html(field) for field in POINT_FIELDS)
    body = []
    for row in records:
        body.append(
            "<tr>%s</tr>"
            % "".join("<td>%s</td>" % _html(_fmt(row[field])) for field in POINT_FIELDS)
        )
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (head, "".join(body))


def _json_block(obj, element_id):
    return '<script type="application/json" id="%s">%s</script>' % (
        _attr(element_id),
        _json_script(obj),
    )


def _csv_cell(value):
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in ',"\r\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _csv_panel(records):
    lines = [",".join(POINT_FIELDS)]
    for row in records:
        lines.append(",".join(_csv_cell(row[field]) for field in POINT_FIELDS))
    return "\r\n".join(lines) + "\r\n"


def _html_panel(
    slug, heading, records, series_a, series_b, na_a, na_b, axes, caption, sha, semantic, csv_text
):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 360" width="100%%" '
        'role="img" aria-label="%s">' % _attr(heading)
    )
    svg += _svg_subplot(
        series_a,
        "A",
        axes["x_displacement_min"],
        axes["x_displacement_max"],
        axes["ticks"]["A"],
        axes["ticks"]["y"],
        na_a,
        0.0,
    )
    svg += _svg_subplot(
        series_b,
        "B",
        axes["x_recall_min"],
        axes["x_recall_max"],
        axes["ticks"]["B"],
        axes["ticks"]["y"],
        na_b,
        400.0,
    )
    svg += "</svg>"
    data = {
        "slug": slug,
        "figure_id": FIGURE_ID,
        "semantic_sha256": sha,
        "axes": axes,
        "records": records,
        "csv": csv_text,
    }
    checks = "".join(
        '<label><input type="checkbox" checked data-policy="%s"/> '
        '<span style="color:%s;font-size:16px">%s</span> %s</label> '
        % (
            _attr(policy),
            POLICY_FILLS[policy],
            POLICY_SYMBOLS[policy],
            _html(POLICY_LABELS[policy]),
        )
        for policy in POLICIES
    )
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"/>',
        "<title>%s %s</title>" % (_html(FIGURE_ID), _html(slug)),
        "<style>",
        _CSS,
        "</style>",
        "</head><body>",
        "<h1>%s</h1>" % _html(FIGURE_ID),
        "<h2>%s</h2>" % _html(heading),
        "<p>%s</p>" % _html(caption),
        "<p>semantic SHA-256: <code>%s</code></p>" % _html(sha),
        (
            '<div class="f07-controls">%s '
            '<button id="f07-download" type="button">Download panel CSV</button> '
            '<button id="f07-semantic-download" type="button">'
            "Download semantic JSON (561 summaries, 1020 points)</button></div>"
        )
        % checks,
        '<div id="f07-panel">%s</div>' % svg,
        "<p>Plotted points only; axis A NA=%d and axis B NA=%d remain in the table/CSV and are not plotted.</p>"
        % (na_a, na_b),
        '<h3>Panel records (51 rows, all point fields)</h3><div class="f07-table">%s</div>'
        % _table(records),
        _json_block(data, "f07-data"),
        _json_block(semantic, "f07-semantic"),
        '<div id="f07-tooltip" class="f07-tooltip" role="tooltip"></div>',
        "<script>%s</script>" % _JS,
        "</body></html>",
    ]
    return "".join(parts)


def prepare_f07_render(prepared):
    """Render an accepted F07 prepared mapping to 20 native TeX + HTML panels.

    Returns ``{"semantic", "semantic_sha256", "panels", "manifest"}``.
    """
    # Authenticate the incoming canonical semantic SHA before trusting any content.
    if not isinstance(prepared, dict):
        raise ValueError("prepared must be a mapping")
    if set(prepared) != _PREPARED_KEYS:
        raise ValueError("prepared must have exactly semantic, semantic_sha256 and manifest")
    given_sha = prepared["semantic_sha256"]
    if not isinstance(given_sha, str) or not _HEX64.match(given_sha):
        raise ValueError("prepared.semantic_sha256 must be a 64-character hex string")
    raw_semantic = prepared["semantic"]
    if not isinstance(raw_semantic, dict):
        raise ValueError("prepared.semantic must be a mapping")
    if _canonical_sha(raw_semantic) != given_sha:
        raise ValueError("prepared.semantic_sha256 does not match prepared.semantic")

    _validate_semantic(raw_semantic)
    semantic_source = _deepcopy(raw_semantic)
    point_rows = _check_fields(raw_semantic["points"], POINT_FIELDS, "point")
    summary_rows = _check_fields(raw_semantic["summaries"], SUMMARY_FIELDS, "summary")
    _validate_points(point_rows)
    _validate_summaries(summary_rows)

    if len(point_rows) != N_POINTS:
        raise ValueError("expected %d points, found %d" % (N_POINTS, len(point_rows)))
    if len(summary_rows) != N_SUMMARIES:
        raise ValueError("expected %d summaries, found %d" % (N_SUMMARIES, len(summary_rows)))

    domains = {}
    for row in point_rows:
        key = (row["station"], row["instrument"])
        flag = row["held_comparison_domain"]
        if key in domains and domains[key] is not flag:
            raise ValueError("unstable held/exploratory status for domain %r" % (key,))
        domains[key] = flag
    if len(domains) != N_DOMAINS:
        raise ValueError("expected %d domains, found %d" % (N_DOMAINS, len(domains)))
    held = sum(1 for value in domains.values() if value)
    if held != N_HELD or len(domains) - held != N_EXPLORATORY:
        raise ValueError("expected %d held / %d exploratory domains" % (N_HELD, N_EXPLORATORY))
    if len(ESTIMANDS) != 2 or len(ENDPOINTS) != 2 or len(MODELS) != 5:
        raise ValueError("unexpected estimand/endpoint/model enumerations")

    # Full cartesian grid of the point table.
    point_grid = set()
    for row in point_rows:
        point_grid.add(
            (
                row["station"],
                row["instrument"],
                row["policy_id"],
                row["estimand"],
                row["endpoint"],
                row["model_id"],
            )
        )
    if len(point_grid) != N_POINTS:
        raise ValueError("point rows are not a distinct full grid")
    expected_points = set()
    for station, instrument in domains:
        for policy in POLICIES:
            for estimand in ESTIMANDS:
                for endpoint in ENDPOINTS:
                    for model in MODELS:
                        expected_points.add(
                            (station, instrument, policy, estimand, endpoint, model)
                        )
    if point_grid != expected_points:
        raise ValueError("point rows do not form the full cartesian grid")

    # Full cartesian grid of the summary table with matching representation/action identities.
    representations = set(ACTION_BY_REPRESENTATION)
    summary_grid = set()
    for row in summary_rows:
        summary_grid.add(
            (row["station"], row["instrument"], row["representation_id"], row["metric"])
        )
    if len(summary_grid) != N_SUMMARIES:
        raise ValueError("summary keys are not unique")
    expected_summaries = set()
    for station, instrument in domains:
        for representation in representations:
            for metric in METRIC_ORDER:
                expected_summaries.add((station, instrument, representation, metric))
    if summary_grid != expected_summaries:
        raise ValueError("summary rows do not form the full cartesian grid")

    axes = _axis_metadata(point_rows)

    render_semantic = {
        "figure_id": FIGURE_ID,
        "research_question_id": RESEARCH_QUESTION_ID,
        "schema_version": "nato-sers-p08-f07-render-v1",
        "source_model_semantic_sha256": semantic_source["source_model_semantic_sha256"],
        "source_semantic_sha256": given_sha,
        "caption": semantic_source["caption"],
        "summaries": _deepcopy(summary_rows),
        "points": _deepcopy(point_rows),
        "axes": _deepcopy(axes),
        "style": {
            "font_family": "lmodern / Times New Roman",
            "font_min_pt": 8,
            "heading_pt": 10,
            "canvas_mm": 181.86,
            "policy_fills": {policy: POLICY_FILLS[policy] for policy in POLICIES},
            "policy_marks": {policy: POLICY_HTML_MARKS[policy] for policy in POLICIES},
            "policy_labels": {policy: POLICY_LABELS[policy] for policy in POLICIES},
            "tex_marks": {policy: POLICY_TEX_MARKS[policy] for policy in POLICIES},
            "tex_colors": {policy: POLICY_TEX_COLORS[policy] for policy in POLICIES},
        },
        "labels": {
            "axis_a": AXES["A"]["html_label"],
            "axis_b": AXES["B"]["html_label"],
            "y": "Balanced accuracy",
            "heading": "P08-F07 / RQ-S05 | model | endpoint | Primary (P) or Pooled sensitivity (S)",
        },
    }
    render_sha256 = _canonical_sha(render_semantic)

    panels = []
    for estimand in ESTIMANDS:
        for endpoint in ENDPOINTS:
            for model in MODELS:
                records = [
                    _deepcopy(row)
                    for row in point_rows
                    if row["estimand"] == estimand
                    and row["endpoint"] == endpoint
                    and row["model_id"] == model
                ]
                if len(records) != N_RECORDS_PER_PANEL:
                    raise ValueError(
                        "panel %s/%s/%s expected %d records, found %d"
                        % (estimand, endpoint, model, N_RECORDS_PER_PANEL, len(records))
                    )
                if {row["policy_id"] for row in records} != set(POLICIES):
                    raise ValueError(
                        "panel %s/%s/%s missing a policy identity" % (estimand, endpoint, model)
                    )
                slug = _slug(estimand, endpoint, model)
                heading = _heading(model, endpoint, estimand)
                series_a, na_a = _series(records, "A", axes["x_displacement_max"])
                series_b, na_b = _series(records, "B", axes["x_recall_max"])
                caption = _panel_caption(semantic_source["caption"], na_a, na_b)
                csv_text = _csv_panel(records)
                panels.append(
                    {
                        "slug": slug,
                        "semantic_sha256": render_sha256,
                        "tex": _tex_panel(
                            slug, heading, series_a, series_b, axes, caption, render_sha256
                        ),
                        "html": _html_panel(
                            slug,
                            heading,
                            records,
                            series_a,
                            series_b,
                            na_a,
                            na_b,
                            axes,
                            caption,
                            render_sha256,
                            render_semantic,
                            csv_text,
                        ),
                        "csv": csv_text,
                    }
                )
    if len(panels) != N_PANELS:
        raise ValueError("expected %d panels, found %d" % (N_PANELS, len(panels)))

    manifest = {
        "figure_id": FIGURE_ID,
        "status": "prepared",
        "reviewed": False,
        "published": False,
        "semantic_sha256": render_sha256,
        "source_semantic_sha256": given_sha,
        "source_model_semantic_sha256": semantic_source["source_model_semantic_sha256"],
        "slugs": [panel["slug"] for panel in panels],
        "axes": _deepcopy(axes),
    }
    return {
        "semantic": render_semantic,
        "semantic_sha256": render_sha256,
        "panels": panels,
        "manifest": manifest,
    }
