# ruff: noqa: E501, UP031
# Literal TeX/HTML/JS templates retain percent formatting to avoid brace escaping errors.
"""P08-F03 concise renderer: copies reviewed estimates only.

This module performs no new statistics. It authenticates the canonical SHA of
the accepted model-figure adapter, whitelists the reviewed rows/metadata and
renders print (native TikZ/PGFPlots), offline HTML/SVG and CSV.
"""

from __future__ import annotations

import csv
import io
import re
from urllib.parse import quote

try:  # helpers reviewed in the same private directory
    from atlas_sers.visualization.p08_f02_render import (
        ENDPOINT_LABELS,
        ENDPOINTS,
        ESTIMAND_LABELS,
        ESTIMANDS,
        MODEL_LABELS,
        MODELS,
        POLICIES,
        POLICY_STYLE,
        _canonical_sha,
        _html,
        _json_script,
        _tex,
    )
except ImportError:  # pragma: no cover - package-relative fallback
    from .p08_f02_render import (
        ENDPOINT_LABELS,
        ENDPOINTS,
        ESTIMAND_LABELS,
        ESTIMANDS,
        MODEL_LABELS,
        MODELS,
        POLICIES,
        POLICY_STYLE,
        _canonical_sha,
        _html,
        _json_script,
        _tex,
    )

FIGURE_ID = "P08-F03"
RESEARCH_QUESTION_ID = "RQ-S01"
SCHEMA_VERSION = "p08-f03/1"
PRIMARY_ESTIMAND = "equal_context"

MODES = ("crossed", "master_only", "instrument_only")
MODE_LABELS = {
    "crossed": "crossed",
    "master_only": "master-only",
    "instrument_only": "instrument-only",
}
MODE_FIELDS = {
    "crossed": ("crossed_lower", "crossed_upper", "crossed_reason"),
    "master_only": ("master_only_lower", "master_only_upper", "master_only_reason"),
    "instrument_only": ("instrument_only_lower", "instrument_only_upper", "instrument_only_reason"),
}
EFFECT_KEYS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "model_id",
    "policy_id",
    "available",
    "reason",
    "family_size",
    "adjustment",
    "point_effect",
    "domain_raw_p",
    "domain_adjusted_p",
    "instrument_raw_p",
    "instrument_adjusted_p",
    "hierarchy_planned",
    "hierarchy_defined",
    "hierarchy_undefined",
    "hierarchy_lower",
    "hierarchy_upper",
    "hierarchy_reason",
    "crossed_lower",
    "crossed_upper",
    "crossed_reason",
    "master_only_lower",
    "master_only_upper",
    "master_only_reason",
    "instrument_only_lower",
    "instrument_only_upper",
    "instrument_only_reason",
)
DOMAIN_KEYS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "model_id",
    "policy_id",
    "domain",
    "station",
    "instrument",
    "point_effect",
    "contexts",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
)
CSV_PREFIX = ("rowtype", "semantic_sha256", "interval_mode")
CSV_UNION_KEYS = tuple(dict.fromkeys(EFFECT_KEYS + DOMAIN_KEYS))
POPULATION_KEYS = (
    "primary_spectra",
    "held_spectra",
    "masters",
    "instruments",
    "held_domains",
    "contexts",
)
POPULATION_FIXED = {
    "primary_spectra": 598,
    "held_spectra": 557,
    "masters": 69,
    "instruments": 10,
    "held_domains": 13,
    "contexts": 260,
}
META_KEYS = (
    "independent_unit",
    "selection",
    "interval_caption",
    "interaction_caption",
    "counts_reference",
    "display",
)
ROW_ORDER = tuple((m, p) for m in MODELS for p in POLICIES)
DOMAIN_OFFSETS = tuple(round(-0.18 + 0.03 * i, 4) for i in range(13))
XLIM = (-1.0, 1.0)
XTICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)
XTICKLABELS = ("-100", "-50", "0", "50", "100")
XLABEL = "Balanced-accuracy change (percentage points)"
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_POLICY_FALLBACK = {
    "PP-U-SG": {"color": "#0072B2", "mark": "circle", "short": "SG"},
    "PP-U-ARPLS": {"color": "#D55E00", "mark": "square", "short": "arPLS"},
}


def _fail(message):
    raise ValueError(message)


def _is_primary(estimand, mode):
    return mode == "crossed" and estimand == PRIMARY_ESTIMAND


def _num(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field}: not a number")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        _fail(f"{field}: not finite")
    return out


def _pvalue(value, field):
    if value is None:
        return None
    out = _num(value, field)
    if not (0.0 <= out <= 1.0):
        _fail(f"{field}: outside [0,1]")
    return out


def _count(value, field, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field}: bad count")
    if value == 0 and not allow_zero:
        _fail(f"{field}: not positive")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value:
        _fail(f"{field}: bad string")
    return value


def _optional_text(value, field):
    if value is None:
        return None
    return _text(value, field)


def _interval(lower, upper, reason, field):
    if lower is None and upper is None:
        _optional_text(reason, f"{field}.reason")
        if not reason:
            _fail(f"{field}: missing interval NA reason")
        return None, None
    if lower is None or upper is None:
        _fail(f"{field}: one-sided interval")
    lo, hi = _num(lower, field), _num(upper, field)
    if not (lo <= hi):
        _fail(f"{field}: unordered bounds")
    if not (-1.0 <= lo <= 1.0 and -1.0 <= hi <= 1.0):
        _fail(f"{field}: bounds outside [-1,1]")
    return lo, hi


def _rows(values, keys, name):
    if not isinstance(values, list):
        _fail(f"{name}: must be a list")
    allowed = set(keys)
    out = []
    for row in values:
        if not isinstance(row, dict) or set(row) != allowed:
            _fail(f"{name}: unexpected row keys")
        clean = {}
        for key in keys:
            value = row[key]
            if isinstance(value, (dict, list, tuple, set)):
                _fail(f"{name}.{key}: nested value")
            clean[key] = value
        out.append(clean)
    return out


def _check_effect(row):
    for key in ("estimand", "contrast_id", "family_id", "endpoint", "model_id", "policy_id"):
        _text(row[key], key)
    if row["available"] is not True:
        _fail("available: complete universal evidence required")
    if type(row["family_size"]) is not int or row["family_size"] != 20:
        _fail("family_size: must be integer 20")
    if row["adjustment"] != "holm":
        _fail("adjustment: must be holm")
    _optional_text(row["reason"], "reason")
    for key in (
        "hierarchy_reason",
        "crossed_reason",
        "master_only_reason",
        "instrument_only_reason",
    ):
        _optional_text(row[key], key)
    pe = _num(row["point_effect"], "point_effect")
    if not (-1.0 <= pe <= 1.0):
        _fail("point_effect: outside [-1,1]")
    for key in ("domain_raw_p", "domain_adjusted_p", "instrument_raw_p", "instrument_adjusted_p"):
        _pvalue(row[key], key)
    for key in ("hierarchy_planned", "hierarchy_defined", "hierarchy_undefined"):
        _count(row[key], key, allow_zero=True)
    _interval(row["hierarchy_lower"], row["hierarchy_upper"], row["hierarchy_reason"], "hierarchy")
    for mode, (lo, hi, reason) in MODE_FIELDS.items():
        _interval(row[lo], row[hi], row[reason], mode)


def _check_domain(row):
    for key in (
        "estimand",
        "contrast_id",
        "family_id",
        "endpoint",
        "model_id",
        "policy_id",
        "domain",
        "station",
        "instrument",
    ):
        _text(row[key], key)
    pe = _num(row["point_effect"], "domain point_effect")
    if not (-1.0 <= pe <= 1.0):
        _fail("domain point_effect: outside [-1,1]")
    for key in ("contexts", "unit_appearances", "physical_masters", "distinct_units"):
        _count(row[key], key)


def _validate_effects(rows):
    expected = {
        (e, en, m, p) for e in ESTIMANDS for en in ENDPOINTS for m in MODELS for p in POLICIES
    }
    seen = set()
    per_estimand = {}
    for row in rows:
        _check_effect(row)
        key = (row["estimand"], row["endpoint"], row["model_id"], row["policy_id"])
        if key in seen:
            _fail("duplicate effect row")
        seen.add(key)
        want = (
            "universal_policy::"
            + row["policy_id"]
            + "::"
            + row["model_id"]
            + "::"
            + row["endpoint"]
        )
        if row["contrast_id"] != want:
            _fail("contrast_id mismatch")
        triple = (row["endpoint"], row["model_id"], row["policy_id"])
        mapping = per_estimand.setdefault(row["estimand"], {})
        previous = mapping.setdefault(row["contrast_id"], triple)
        if previous != triple:
            _fail("contrast_id collision")
    if seen != expected:
        _fail("effect grid mismatch")
    for mapping in per_estimand.values():
        if len(mapping) != 20:
            _fail("expected 20 distinct contrast ids per estimand")


def _validate_domains(rows, effects_by_key):
    combos = {
        (e, en, m, p) for e in ESTIMANDS for en in ENDPOINTS for m in MODELS for p in POLICIES
    }
    by_combo, identity, order = {}, {}, None
    for row in rows:
        _check_domain(row)
        key = (row["estimand"], row["endpoint"], row["model_id"], row["policy_id"])
        if key not in combos:
            _fail("unexpected domain combo")
        bucket = by_combo.setdefault(key, {})
        if row["domain"] in bucket:
            _fail("duplicate domain row")
        bucket[row["domain"]] = row
        pair = (row["station"], row["instrument"])
        previous = identity.setdefault(row["domain"], pair)
        if previous != pair:
            _fail("domain station/instrument identity mismatch")
    if set(by_combo) != combos:
        _fail("domain grid mismatch")
    for key, bucket in by_combo.items():
        if len(bucket) != 13:
            _fail("expected 13 global domains")
        effect = effects_by_key[key]
        for row in bucket.values():
            for field in (
                "estimand",
                "endpoint",
                "model_id",
                "policy_id",
                "contrast_id",
                "family_id",
            ):
                if row[field] != effect[field]:
                    _fail("domain row does not match overall " + field)
        if order is None:
            order = sorted(bucket)
        elif sorted(bucket) != order:
            _fail("domain identity mismatch")
    return list(order or [])


def _clean_metadata(meta):
    if not isinstance(meta, dict):
        _fail("metadata: not a mapping")
    pop = meta.get("population")
    if not isinstance(pop, dict):
        _fail("population: not a mapping")
    clean_pop = {}
    for key, expected in POPULATION_FIXED.items():
        value = pop.get(key)
        if type(value) is not int or value != expected:
            _fail(f"population.{key}: expected {expected}")
        clean_pop[key] = value
    clean = {"population": clean_pop}
    for key in META_KEYS:
        clean[key] = _text(meta.get(key), "metadata." + key)
    if clean["independent_unit"] != "physical_master":
        _fail("metadata.independent_unit: expected physical_master")
    return clean


def _policy_style(policy):
    fallback = _POLICY_FALLBACK.get(policy, {"color": "#666666", "mark": "circle", "short": policy})
    base = dict(fallback)
    extra = POLICY_STYLE.get(policy) if isinstance(POLICY_STYLE, dict) else None
    if isinstance(extra, dict):
        color = extra.get("color")
        if isinstance(color, str) and _HEX_RE.match(color):
            base["color"] = color
        if isinstance(extra.get("mark"), str) and extra["mark"]:
            base["mark"] = extra["mark"]
        if isinstance(extra.get("short"), str) and extra["short"]:
            base["short"] = extra["short"]
    return base


def _policy_label(policy):
    return str(_policy_style(policy).get("short", policy))


def _is_square(style):
    return "square" in str(style.get("mark", "")).lower()


def _captions(meta):
    pop = meta["population"]
    caption = (
        "Positive values mean the universal preprocessing pipeline helps relative to the "
        "MIN baseline. MIN uses minimal min-max scaling; SG combines impulse replacement "
        "and Savitzky-Golay smoothing; arPLS combines impulse replacement and baseline "
        "correction. All three finish with min-max scaling to [0,1]. The compared "
        "policies differ by the whole preprocessing pipeline, not by an isolated smoothing "
        "or background operation. Gray points are the 13 station/instrument domains (not "
        "samples); colored points are the equal-domain effect with 95% marginal conditional "
        "percentile intervals (10,000 support-preserving weights). M01 uses individual "
        "predictions; M06 averages MODEL probabilities within sample/instrument then equal "
        "instruments, not mean spectra. No retraining, no new instrument uncertainty, no G4 "
        "interval, no superiority claim; source-only model selection/calibration, frozen CNN "
        "identity across policies. This figure is not clean-spectrum recovery. "
        f"{pop['primary_spectra']} original spectra, {pop['masters']} physical samples, "
        f"{pop['instruments']} instruments, {pop['held_spectra']} held spectra, "
        f"{pop['contexts']} contexts."
    )
    return {
        "header": "RQ-S01 / P08-F03 universal preprocessing effect",
        "caption": caption,
    }


def _build_semantic(effects, domains, order, meta, source_sha):
    return {
        "figure_id": FIGURE_ID,
        "research_question_id": RESEARCH_QUESTION_ID,
        "schema_version": SCHEMA_VERSION,
        "source_semantic_sha256": source_sha,
        "estimands": list(ESTIMANDS),
        "endpoints": list(ENDPOINTS),
        "models": list(MODELS),
        "policies": list(POLICIES),
        "modes": list(MODES),
        "mode_labels": dict(MODE_LABELS),
        "domain_order": list(order),
        "domain_offsets": list(DOMAIN_OFFSETS),
        "row_order": [{"model_id": m, "policy_id": p} for m, p in ROW_ORDER],
        "estimand_labels": {e: ESTIMAND_LABELS.get(e, e) for e in ESTIMANDS},
        "endpoint_labels": {e: ENDPOINT_LABELS.get(e, e) for e in ENDPOINTS},
        "model_labels": {m: MODEL_LABELS.get(m, m) for m in MODELS},
        "policy_labels": {p: _policy_label(p) for p in POLICIES},
        "policy_styles": {p: _policy_style(p) for p in POLICIES},
        "axes": {
            "xlim": list(XLIM),
            "xticks": list(XTICKS),
            "xticklabels": list(XTICKLABELS),
            "xlabel": XLABEL,
        },
        "captions": _captions(meta),
        "metadata": meta,
        "f03_effects": effects,
        "f03_domains": domains,
    }


def _panel_data(sem, estimand, endpoint, mode):
    eff = {
        (r["estimand"], r["endpoint"], r["model_id"], r["policy_id"]): r for r in sem["f03_effects"]
    }
    dom = {}
    for r in sem["f03_domains"]:
        dom.setdefault((r["estimand"], r["endpoint"], r["model_id"], r["policy_id"]), {})[
            r["domain"]
        ] = r
    lo_key, hi_key, reason_key = MODE_FIELDS[mode]
    n = len(ROW_ORDER)
    rows = []
    for index, (model, policy) in enumerate(ROW_ORDER):
        key = (estimand, endpoint, model, policy)
        effect, bucket = eff[key], dom[key]
        y = n - 1 - index
        points = []
        for offset_index, name in enumerate(sem["domain_order"]):
            domain = dict(bucket[name])
            domain["x"] = float(domain["point_effect"])
            domain["y"] = y + DOMAIN_OFFSETS[offset_index]
            points.append(domain)
        rows.append(
            {
                "model_id": model,
                "policy_id": policy,
                "y": y,
                "label": sem["model_labels"][model] + " " + sem["policy_labels"][policy],
                "point": float(effect["point_effect"]),
                "lower": effect[lo_key],
                "upper": effect[hi_key],
                "reason": effect[reason_key],
                "effect": effect,
                "domains": points,
            }
        )
    return {"estimand": estimand, "endpoint": endpoint, "mode": mode, "rows": rows}


def _policy_defs():
    lines = []
    for index, policy in enumerate(POLICIES):
        style = _policy_style(policy)
        name = f"Pol{chr(65 + index)}"
        lines.append(f"\\definecolor{{{name}}}{{HTML}}{{{style['color'].lstrip('#').upper()}}}")
        mark = "square*" if _is_square(style) else "*"
        lines.append(
            "\\pgfplotsset{pt%d/.style={only marks, mark=%s, "
            "color=%s, mark options={fill=%s, draw=black}}}" % (index, mark, name, name)
        )
        lines.append(
            "\\pgfplotsset{bar%d/.style={%s, line width=0.8pt, mark=|, "
            "mark options={color=%s}}}" % (index, name, name)
        )
    return "\n".join(lines)


def _f(value):
    return format(float(value), ".17g")


def _panel_tex(sem, panel, sha):
    tag = "P" if _is_primary(panel["estimand"], panel["mode"]) else "S"
    title = (
        tag
        + ": "
        + str(sem["endpoint_labels"][panel["endpoint"]])
        + " | "
        + str(sem["mode_labels"][panel["mode"]])
    )
    n = len(panel["rows"])
    ylabels = ", ".join("{" + _tex(r["label"]) + "}" for r in reversed(panel["rows"]))
    body = []
    for row in panel["rows"]:
        y = float(row["y"])
        points = " ".join(f"({_f(d['x'])},{_f(d['y'])})" for d in row["domains"])
        body.append("\\addplot[domainpts] coordinates {" + points + "};")
        index = POLICIES.index(row["policy_id"])
        if row["lower"] is None:
            body.append("\\node[intervalna] at (axis cs:0.98," + _f(y + 0.30) + ") {interval NA};")
        else:
            body.append(
                "\\addplot[bar"
                + str(index)
                + "] coordinates {("
                + _f(row["lower"])
                + ","
                + _f(y)
                + ") ("
                + _f(row["upper"])
                + ","
                + _f(y)
                + ")};"
            )
        body.append(
            "\\addplot[pt"
            + str(index)
            + "] coordinates {("
            + _f(row["point"])
            + ","
            + _f(y)
            + ")};"
        )
    caption = (
        _tex(sem["captions"]["header"])
        + "\\\\ "
        + _tex(
            "Aggregation: "
            + sem["estimand_labels"][panel["estimand"]]
            + ". Interval weighting: "
            + sem["mode_labels"][panel["mode"]]
            + ". "
        )
        + _tex(sem["captions"]["caption"])
    )
    axis = (
        "\\begin{axis}[width=150mm, height=105mm, clip=false, xmin=-1, xmax=1, "
        "xtick={-1,-0.5,0,0.5,1}, xticklabels={-100,-50,0,50,100}, "
        "xlabel={" + XLABEL + "}, ymin=-0.5, ymax=" + _f(n - 0.5) + ", "
        "ytick={" + ",".join(str(i) for i in range(n)) + "}, "
        "yticklabels={" + ylabels + "}, "
        "yticklabel style={font=\\fontsize{8}{10}\\selectfont}, "
        "tick label style={font=\\fontsize{8}{10}\\selectfont}, axis lines=left, "
        "title={" + _tex(title) + "}, "
        "title style={font=\\bfseries\\fontsize{10}{12}\\selectfont}, "
        "every axis plot/.append style={line width=0.6pt}]"
    )
    lines = [
        "% P08-F03 universal preprocessing renderer output -- copied estimates only",
        "%% source SHA256: " + str(sem["source_semantic_sha256"]),
        "%% semantic SHA256: " + sha,
        "\\documentclass[border=0pt]{standalone}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{lmodern}",
        "\\usepackage{tikz}",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.18}",
        _policy_defs(),
        "\\pgfplotsset{domainpts/.style={only marks, mark=o, mark options="
        "{draw=gray, fill=none, scale=0.7}}}",
        "\\tikzset{intervalna/.style={font=\\fontsize{8}{10}\\selectfont, anchor=east}}",
        "\\begin{document}",
        "\\begin{minipage}{181.86mm}\\centering",
        "\\begin{tikzpicture}",
        axis,
        "\\draw[black, dash dot, line width=0.6pt] (axis cs:0,-0.5) -- (axis cs:0,"
        + _f(n - 0.5)
        + ");",
        *body,
        "\\end{axis}",
        "\\node[anchor=north, align=justify, text width=170mm, "
        "font=\\fontsize{8}{10}\\selectfont] at "
        "([yshift=-6mm]current bounding box.south) {" + caption + "};",
        "\\end{tikzpicture}",
        "\\end{minipage}",
        "\\end{document}",
        "",
    ]
    return "\n".join(lines)


_CSS = (
    'body{font-family:"Times New Roman",Times,serif;font-size:16px;color:#000;margin:12px}'
    "figure{margin:0;position:relative}svg{background:#fff;display:block}"
    ".grid{stroke:#e6e6e6;stroke-width:1}"
    ".null{stroke:#000;stroke-width:1;stroke-dasharray:6 3 1 3}"
    ".dp{fill:none;stroke:#808080;stroke-width:0.8}"
    ".iv,.cap{stroke-width:2}"
    "circle.op,rect.op{stroke:#000;stroke-width:0.5}"
    ".tick,.ytick,.natick{font-size:16px;fill:#000}"
    "table{border-collapse:collapse;font-size:14px}"
    ".data-table{overflow:auto;max-height:420px}"
    "th,td{border:1px solid #999;padding:2px 4px;text-align:left}"
    "svg [tabindex]:focus{outline:2px solid #000;outline-offset:2px}"
    ".tip{display:none;position:fixed;background:#fff;color:#000;padding:3px 6px;"
    "font-size:16px;border:1px solid #000;pointer-events:none;z-index:10;max-width:420px}"
    ".scope{margin:2px 0 8px}.caption{margin:2px 0 8px}"
)
_JS = (
    "<script>(function(){"
    "var root=document.currentScript.closest('figure');"
    "if(!root){return;}"
    "var tip=root.querySelector('.tip');"
    "var d=root.querySelector('#dpoints');"
    "var i=root.querySelector('#intervals');"
    "function hide(){if(tip){tip.style.display='none';tip.textContent='';}}"
    "function show(el){var t=el.getAttribute('data-tip');if(!t||!tip){return;}"
    "tip.textContent=t;tip.style.display='block';"
    "var r=el.getBoundingClientRect();"
    "tip.style.left=Math.round(r.left+r.width/2)+'px';"
    "tip.style.top=Math.round(r.top-30)+'px';}"
    "function target(ev){var el=ev.target;"
    "if(el&&el.closest){el=el.closest('[data-tip]');}"
    "return (el&&root.contains(el))?el:null;}"
    "root.addEventListener('mouseover',function(ev){var el=target(ev);if(el){show(el);}});"
    "root.addEventListener('mouseout',function(ev){"
    "var el=target(ev);if(el&&(!ev.relatedTarget||!el.contains(ev.relatedTarget))){hide();}});"
    "root.addEventListener('focusin',function(ev){var el=target(ev);if(el){show(el);}});"
    "root.addEventListener('focusout',hide);"
    "function set(){"
    "root.querySelectorAll('.dp').forEach(function(e){"
    "e.style.display=(d&&d.checked)?'':'none';});"
    "root.querySelectorAll('.iv,.cap').forEach(function(e){"
    "e.style.display=(i&&i.checked)?'':'none';});"
    "hide();}"
    "if(d){d.addEventListener('change',set);}"
    "if(i){i.addEventListener('change',set);}"
    "set();"
    "})();</script>"
)


def _cell(value):
    return _html("" if value is None else str(value))


def _table(title, keys, rows):
    head = "".join(f"<th>{_html(k)}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(r[k])}</td>" for k in keys) + "</tr>" for r in rows
    )
    return (
        f'<h3>{_html(title)}</h3><div class="data-table"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _slug(estimand, endpoint, mode):
    return f"p08-f03-{estimand}-{endpoint}-{mode}"


def _json_block(element_id, payload):
    text = _json_script(payload)
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("</", "<\\/")
    return f'<script type="application/json" id="{_html(element_id)}">{text}</script>'


def _csv_value(value):
    if value is None:
        return ""
    return value


def _panel_csv(panel, sha):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(CSV_PREFIX) + list(CSV_UNION_KEYS))
    for row in panel["rows"]:
        effect = row["effect"]
        writer.writerow(
            ["overall", sha, panel["mode"]] + [_csv_value(effect.get(k)) for k in CSV_UNION_KEYS]
        )
    for row in panel["rows"]:
        for domain in row["domains"]:
            writer.writerow(
                ["domain", sha, panel["mode"]] + [_csv_value(domain.get(k)) for k in CSV_UNION_KEYS]
            )
    return buf.getvalue()


def _panel_html(sem, panel, sha, csv_text):
    width, height = 900, 500
    left, right, top, bottom = 240, 24, 34, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(panel["rows"])
    span = float(n)
    tag = "P" if _is_primary(panel["estimand"], panel["mode"]) else "S"
    endpoint_label = str(sem["endpoint_labels"][panel["endpoint"]])
    estimand_label = str(sem["estimand_labels"][panel["estimand"]])
    mode_label = str(sem["mode_labels"][panel["mode"]])
    slug = _slug(panel["estimand"], panel["endpoint"], panel["mode"])
    caption = sem["captions"]["caption"]

    def sx(x):
        return left + (float(x) + 1.0) / 2.0 * plot_w

    def sy(y):
        return top + ((n - 0.5) - float(y)) / span * plot_h

    scope = (
        f"{tag} panel: {n} model/policy rows x 13 station/instrument domains; "
        "original population; conditional per-row intervals"
    )
    parts = [
        f'<figure id="{_html(slug)}">',
        f"<figcaption><strong>{_html(sem['captions']['header'])}</strong></figcaption>",
        f"<h2>{_html(RESEARCH_QUESTION_ID)} / {_html(FIGURE_ID)} universal preprocessing</h2>",
        (
            '<p class="scope"><strong>Endpoint:</strong> '
            + _html(endpoint_label)
            + " &middot; <strong>Estimand:</strong> "
            + _html(estimand_label)
            + " &middot; <strong>Interval mode:</strong> "
            + _html(mode_label)
            + " &middot; <strong>Scope:</strong> "
            + _html(scope)
            + "</p>"
        ),
        f'<p class="caption">{_html(caption)}</p>',
        f"<p>Semantic SHA-256: <code>{_html(sha)}</code></p>",
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{_html(caption)}">'
        ),
    ]
    for tick in XTICKS:
        x = sx(tick)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 20}" '
            f'text-anchor="middle">{int(round(tick * 100))}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left + plot_w / 2:.1f}" '
        f'y="{top + plot_h + 46}" text-anchor="middle">{_html(XLABEL)}</text>'
    )
    x0 = sx(0.0)
    parts.append(f'<line class="null" x1="{x0:.1f}" y1="{top}" x2="{x0:.1f}" y2="{top + plot_h}"/>')
    for row in panel["rows"]:
        py = sy(row["y"])
        parts.append(
            f'<text class="ytick" x="{left - 8}" y="{py + 5:.1f}" '
            f'text-anchor="end">{_html(row["label"])}</text>'
        )
        for domain in row["domains"]:
            tip = (
                f"{row['model_id']} {row['policy_id']} domain {domain['domain']} "
                f"({domain['station']}/{domain['instrument']}): "
                f"{domain['point_effect']} contexts={domain['contexts']} "
                f"appearances={domain['unit_appearances']} "
                f"masters={domain['physical_masters']} "
                f"units={domain['distinct_units']}"
            )
            parts.append(
                f'<circle class="dp" cx="{sx(domain["x"]):.1f}" '
                f'cy="{sy(domain["y"]):.1f}" r="3" tabindex="0" '
                f'data-tip="{_html(tip)}" aria-label="{_html(tip)}"></circle>'
            )
        style = _policy_style(row["policy_id"])
        color = style["color"]
        if row["lower"] is not None:
            low, high = sx(row["lower"]), sx(row["upper"])
            itip = (
                f"{row['model_id']} {row['policy_id']} {panel['mode']} interval "
                f"[{row['lower']}, {row['upper']}]"
            )
            parts.append(
                f'<line class="iv" x1="{low:.1f}" y1="{py:.1f}" '
                f'x2="{high:.1f}" y2="{py:.1f}" stroke="{_html(color)}" '
                f'tabindex="0" data-tip="{_html(itip)}" '
                f'aria-label="{_html(itip)}"></line>'
            )
            for cap in (low, high):
                parts.append(
                    f'<line class="cap" x1="{cap:.1f}" y1="{py - 5:.1f}" '
                    f'x2="{cap:.1f}" y2="{py + 5:.1f}" '
                    f'stroke="{_html(color)}"/>'
                )
        else:
            rtip = (
                f"{row['model_id']} {row['policy_id']} {panel['mode']} "
                f"interval NA: {row['reason'] or ''}"
            )
            parts.append(
                f'<text class="natick" x="{sx(0.98):.1f}" '
                f'y="{sy(float(row["y"]) + 0.30):.1f}" text-anchor="end" '
                f'tabindex="0" data-tip="{_html(rtip)}" '
                f'aria-label="{_html(rtip)}">interval NA</text>'
            )
        otip = (
            f"{row['model_id']} {row['policy_id']}: point {row['point']} "
            f"({panel['estimand']}/{panel['endpoint']}/{panel['mode']})"
        )
        point_x = sx(row["point"])
        if _is_square(style):
            parts.append(
                f'<rect class="op" x="{point_x - 4:.1f}" y="{py - 4:.1f}" '
                f'width="8" height="8" fill="{_html(color)}" tabindex="0" '
                f'data-tip="{_html(otip)}" aria-label="{_html(otip)}"></rect>'
            )
        else:
            parts.append(
                f'<circle class="op" cx="{point_x:.1f}" cy="{py:.1f}" r="4.5" '
                f'fill="{_html(color)}" tabindex="0" '
                f'data-tip="{_html(otip)}" aria-label="{_html(otip)}"></circle>'
            )
    parts.append("</svg>")
    parts.append(
        '<div class="tools"><label><input type="checkbox" id="dpoints" '
        'checked> domain points</label> <label><input type="checkbox" '
        'id="intervals" checked> intervals</label> <a class="btn" download="'
        + _html(slug + ".csv")
        + '" href="data:text/csv;charset=utf-8,'
        + quote(csv_text)
        + '">CSV</a></div>'
    )
    parts.append(_table("Overall rows", EFFECT_KEYS, [r["effect"] for r in panel["rows"]]))
    parts.append(
        _table("Domain rows", DOMAIN_KEYS, [d for r in panel["rows"] for d in r["domains"]])
    )
    parts.append('<div class="tip" role="status" aria-live="polite"></div>')
    parts.append(
        _json_block(
            f"panel-data-{slug}",
            {
                "figure_id": FIGURE_ID,
                "research_question_id": RESEARCH_QUESTION_ID,
                "schema_version": SCHEMA_VERSION,
                "source_semantic_sha256": sem["source_semantic_sha256"],
                "semantic_sha256": sha,
                "slug": slug,
                "estimand": panel["estimand"],
                "endpoint": panel["endpoint"],
                "mode": panel["mode"],
                "scope": tag,
                "metadata": sem["metadata"],
                "csv": csv_text,
                "rows": panel["rows"],
            },
        )
    )
    parts.append(_JS)
    parts.append("</figure>")
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>'
        + _html(slug)
        + "</title><style>"
        + _CSS
        + "</style></head><body>"
        + "".join(parts)
        + "</body></html>"
    )


def _panel(sem, estimand, endpoint, mode, sha):
    data = _panel_data(sem, estimand, endpoint, mode)
    csv_text = _panel_csv(data, sha)
    return {
        "slug": _slug(estimand, endpoint, mode),
        "semantic_sha256": sha,
        "tex": _panel_tex(sem, data, sha),
        "html": _panel_html(sem, data, sha, csv_text),
        "csv": csv_text,
    }


def prepare_f03(prepared):
    """Authenticate and project a reviewed model-figure adapter for rendering."""
    if not isinstance(prepared, dict):
        _fail("prepared: not a mapping")
    source = prepared.get("semantic")
    if not isinstance(source, dict):
        _fail("semantic: not a mapping")
    source_sha = _canonical_sha(source)
    if source_sha != prepared.get("semantic_sha256"):
        _fail("semantic SHA mismatch")
    effects = _rows(source.get("f03_effects"), EFFECT_KEYS, "f03_effects")
    domains = _rows(source.get("f03_domains"), DOMAIN_KEYS, "f03_domains")
    _validate_effects(effects)
    effects_by_key = {
        (r["estimand"], r["endpoint"], r["model_id"], r["policy_id"]): r for r in effects
    }
    order = _validate_domains(domains, effects_by_key)
    meta = _clean_metadata(source.get("metadata"))
    semantic = _build_semantic(effects, domains, order, meta, source_sha)
    sha = _canonical_sha(semantic)
    panels = [
        _panel(semantic, e, en, mode, sha) for e in ESTIMANDS for en in ENDPOINTS for mode in MODES
    ]
    manifest = {
        "status": "prepared",
        "reviewed": False,
        "published": False,
        "figure_id": FIGURE_ID,
        "research_question_id": RESEARCH_QUESTION_ID,
        "schema_version": SCHEMA_VERSION,
        "source_semantic_sha256": source_sha,
        "semantic_sha256": sha,
        "slugs": [p["slug"] for p in panels],
    }
    return {"semantic": semantic, "semantic_sha256": sha, "panels": panels, "manifest": manifest}
