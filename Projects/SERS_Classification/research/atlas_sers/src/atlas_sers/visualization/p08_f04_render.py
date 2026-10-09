# ruff: noqa: E501, UP031
# Literal TeX/HTML/JS templates retain percent formatting to avoid brace escaping errors.
"""P08-F04 interaction renderer.

Copies reviewed interaction estimates only; performs no new statistics.
Authenticates the accepted model-figure adapter, whitelists the reviewed
tables/metadata and renders print (native TikZ/PGFPlots), offline HTML/SVG
and CSV.
"""

from __future__ import annotations

import csv
import io
from urllib.parse import quote

try:  # helpers reviewed in the same private directory
    from atlas_sers.visualization.p08_f02_render import (
        ENDPOINT_LABELS,
        ENDPOINTS,
        ESTIMAND_LABELS,
        ESTIMANDS,
        MODEL_LABELS,
        _canonical_json,
        _canonical_sha,
        _html,
        _json_script,
        _tex,
    )
    from atlas_sers.visualization.p08_f03_render import (
        _CSS,
        _JS,
        _clean_metadata,
        _f,
        _is_square,
        _policy_style,
    )
except ImportError:  # pragma: no cover - package-relative fallback
    from .p08_f02_render import (
        ENDPOINT_LABELS,
        ENDPOINTS,
        ESTIMAND_LABELS,
        ESTIMANDS,
        MODEL_LABELS,
        _canonical_json,
        _canonical_sha,
        _html,
        _json_script,
        _tex,
    )
    from .p08_f03_render import (
        _CSS,
        _JS,
        _clean_metadata,
        _f,
        _is_square,
        _policy_style,
    )

FIGURE_ID = "P08-F04"
RESEARCH_QUESTION_ID = "RQ-S01"
SCHEMA_VERSION = "p08-f04/1"
PRIMARY_ESTIMAND = "equal_context"
PRIMARY_MODE = "crossed"
DEEP_MODELS = ("D0-M", "P05-SELECTED")
COMPARATORS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
POLICIES = ("PP-U-SG", "PP-U-ARPLS")
QC_POLICY = "PP-QC-SRC"
MIN_POLICY = "PP-U-MIN"
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
OVERALL_KEYS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "policy_id",
    "deep_model_id",
    "comparator_model_id",
    "model_id",
    "available",
    "reason",
    "procedure_labels",
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
    "procedure_balanced_accuracies",
)
DOMAIN_KEYS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "policy_id",
    "deep_model_id",
    "comparator_model_id",
    "domain",
    "station",
    "instrument",
    "point_effect",
    "contexts",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
    "procedure_balanced_accuracies",
)
CSV_PREFIX = ("row_type", "semantic_sha256", "interval_mode")
CSV_UNION_KEYS = tuple(dict.fromkeys(OVERALL_KEYS + DOMAIN_KEYS))
ROW_ORDER = tuple((c, p) for c in COMPARATORS for p in POLICIES)
DOMAIN_OFFSETS = tuple(round(-0.18 + 0.03 * i, 4) for i in range(13))
XLIM = (-2.0, 2.0)
XTICKS = (-2.0, -1.0, 0.0, 1.0, 2.0)
XTICKLABELS = ("-200", "-100", "0", "100", "200")
XLABEL = "Difference-in-differences (percentage points)"


def _fail(message):
    raise ValueError(message)


def _num(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field}: not a number")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        _fail(f"{field}: not finite")
    return out


def _unit(value, field):
    out = _num(value, field)
    if not (0.0 <= out <= 1.0):
        _fail(f"{field}: outside [0,1]")
    return out


def _pvalue(value, field):
    if value is None:
        return None
    return _unit(value, field)


def _count(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field}: bad count")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value:
        _fail(f"{field}: bad string")
    return value


def _optional_text(value, field):
    if value is None:
        return None
    return _text(value, field)


def _procedure_labels(value, field):
    if not isinstance(value, list) or len(value) != 4:
        _fail(f"{field}: expected 4 label rows")
    out = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            _fail(f"{field}: expected 3 scalars per row")
        out.append([_text(x, field) for x in item])
    return out


def _accuracies(value, field, allow_none=False):
    if value is None:
        if allow_none:
            return None
        _fail(f"{field}: missing balanced accuracies")
    if not isinstance(value, list) or len(value) != 4:
        _fail(f"{field}: expected 4 balanced accuracies")
    return [_unit(v, field) for v in value]


def _interval(lower, upper, reason, field, available):
    if not available:
        if lower is not None or upper is not None or reason is not None:
            _fail(f"{field}: unavailable rows must carry no interval")
        return
    if lower is None and upper is None:
        if not reason:
            _fail(f"{field}: missing interval NA reason")
        _text(reason, f"{field}.reason")
        return
    if lower is None or upper is None:
        _fail(f"{field}: one-sided interval")
    lo, hi = _num(lower, field), _num(upper, field)
    if lo > hi:
        _fail(f"{field}: unordered bounds")
    if not (-2.0 <= lo <= 2.0 and -2.0 <= hi <= 2.0):
        _fail(f"{field}: bounds outside [-2,2]")
    _optional_text(reason, f"{field}.reason")


def _copy_row(row, keys, name):
    if not isinstance(row, dict) or set(row) != set(keys):
        _fail(f"{name}: unexpected row keys")
    copied = {key: row[key] for key in keys}
    if "procedure_labels" in copied:
        copied["procedure_labels"] = _procedure_labels(
            copied["procedure_labels"], "procedure_labels"
        )
    copied["procedure_balanced_accuracies"] = _accuracies(
        copied["procedure_balanced_accuracies"], "procedure_balanced_accuracies", allow_none=True
    )
    return copied


def _check_overall(row):
    for key in (
        "estimand",
        "contrast_id",
        "family_id",
        "endpoint",
        "policy_id",
        "deep_model_id",
        "comparator_model_id",
    ):
        _text(row[key], key)
    if row["model_id"] is not None:
        _fail("model_id: expected None")
    if row["family_id"] != "policy_model_interaction":
        _fail("family_id: expected policy_model_interaction")
    available = row["available"]
    if not isinstance(available, bool):
        _fail("available: not a bool")
    _procedure_labels(row["procedure_labels"], "procedure_labels")
    if type(row["family_size"]) is not int or row["family_size"] != 32:
        _fail("family_size: must be integer 32")
    if row["adjustment"] != "holm":
        _fail("adjustment: must be holm")
    if available:
        if row["reason"] is not None:
            _fail("reason: available row must not carry a reason")
        point = _num(row["point_effect"], "point_effect")
        if not (-2.0 <= point <= 2.0):
            _fail("point_effect: outside [-2,2]")
        acc = _accuracies(row["procedure_balanced_accuracies"], "procedure_balanced_accuracies")
        expected = acc[0] - acc[1] - acc[2] + acc[3]
        if abs(point - expected) > 1e-12:
            _fail("point_effect: does not equal a-b-c+d")
        for key in (
            "domain_raw_p",
            "domain_adjusted_p",
            "instrument_raw_p",
            "instrument_adjusted_p",
        ):
            _pvalue(row[key], key)
        for key in ("hierarchy_planned", "hierarchy_defined", "hierarchy_undefined"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
                _fail(f"{key}: bad count")
        _interval(
            row["hierarchy_lower"],
            row["hierarchy_upper"],
            row["hierarchy_reason"],
            "hierarchy",
            True,
        )
        for mode, (lo, hi, reason) in MODE_FIELDS.items():
            _interval(row[lo], row[hi], row[reason], mode, True)
    else:
        if row["policy_id"] != QC_POLICY:
            _fail("only the planned QC entries may be unavailable")
        if row["reason"] != "outside_universal_execution_scope":
            _fail("reason: unavailable rows need outside_universal_execution_scope")
        for key in (
            "point_effect",
            "domain_raw_p",
            "domain_adjusted_p",
            "instrument_raw_p",
            "instrument_adjusted_p",
            "hierarchy_planned",
            "hierarchy_defined",
            "hierarchy_undefined",
            "procedure_balanced_accuracies",
        ):
            if row[key] is not None:
                _fail(f"{key}: unavailable rows must be None")
        for mode, (lo, hi, reason) in MODE_FIELDS.items():
            _interval(row[lo], row[hi], row[reason], mode, False)
        _interval(
            row["hierarchy_lower"],
            row["hierarchy_upper"],
            row["hierarchy_reason"],
            "hierarchy",
            False,
        )


def _check_domain(row):
    for key in (
        "estimand",
        "contrast_id",
        "family_id",
        "endpoint",
        "policy_id",
        "deep_model_id",
        "comparator_model_id",
        "domain",
        "station",
        "instrument",
    ):
        _text(row[key], key)
    point = _num(row["point_effect"], "domain point_effect")
    if not (-2.0 <= point <= 2.0):
        _fail("domain point_effect: outside [-2,2]")
    for key in ("contexts", "unit_appearances", "physical_masters", "distinct_units"):
        _count(row[key], key)
    acc = _accuracies(row["procedure_balanced_accuracies"], "domain procedure_balanced_accuracies")
    if abs(point - (acc[0] - acc[1] - acc[2] + acc[3])) > 1e-12:
        _fail("domain point_effect: does not equal a-b-c+d")


def _validate(overall, domains):
    expected_available = set()
    for e in ESTIMANDS:
        for p in POLICIES:
            for d in DEEP_MODELS:
                for c in COMPARATORS:
                    for en in ENDPOINTS:
                        expected_available.add((e, p, d, c, en))
    expected_unavailable = set()
    for e in ESTIMANDS:
        for d in DEEP_MODELS:
            for c in ("C-RBF-SVM", "C-RANDOM-FOREST"):
                for en in ENDPOINTS:
                    expected_unavailable.add((e, d, c, en))
    by_combo = {}
    seen_contrast = set()
    available_keys = set()
    unavailable_keys = set()
    for row in overall:
        _check_overall(row)
        combo = (
            row["estimand"],
            row["policy_id"],
            row["deep_model_id"],
            row["comparator_model_id"],
            row["endpoint"],
        )
        if combo in by_combo:
            _fail("duplicate overall row")
        by_combo[combo] = row
        contrast = row["contrast_id"]
        want = "policy_model_interaction::%s::%s::%s::%s" % (
            row["policy_id"],
            row["deep_model_id"],
            row["comparator_model_id"],
            row["endpoint"],
        )
        if contrast != want:
            _fail("contrast_id mismatch")
        marker = (row["estimand"], contrast)
        if marker in seen_contrast:
            _fail("duplicate estimand/contrast")
        seen_contrast.add(marker)
        labels = row["procedure_labels"]
        want_labels = [
            [row["policy_id"], row["deep_model_id"], row["endpoint"]],
            [MIN_POLICY, row["deep_model_id"], row["endpoint"]],
            [row["policy_id"], row["comparator_model_id"], row["endpoint"]],
            [MIN_POLICY, row["comparator_model_id"], row["endpoint"]],
        ]
        if labels != want_labels:
            _fail("procedure_labels mismatch")
        if row["available"]:
            available_keys.add(combo)
        else:
            unavailable_keys.add((combo[0], combo[2], combo[3], combo[4]))
    if available_keys != expected_available:
        _fail("available overall grid mismatch")
    if unavailable_keys != expected_unavailable:
        _fail("unavailable overall grid mismatch")
    if len(domains) != len(expected_available) * 13:
        _fail("domain row count mismatch")
    by_domain_combo = {}
    identity = {}
    supports = {}
    order = None
    for row in domains:
        _check_domain(row)
        combo = (
            row["estimand"],
            row["policy_id"],
            row["deep_model_id"],
            row["comparator_model_id"],
            row["endpoint"],
        )
        effect = by_combo.get(combo)
        if effect is None or not effect["available"]:
            _fail("domain row for unknown or unavailable combo")
        for field in (
            "estimand",
            "contrast_id",
            "family_id",
            "endpoint",
            "policy_id",
            "deep_model_id",
            "comparator_model_id",
        ):
            if row[field] != effect[field]:
                _fail("domain row does not match overall " + field)
        bucket = by_domain_combo.setdefault(combo, {})
        if row["domain"] in bucket:
            _fail("duplicate domain row")
        bucket[row["domain"]] = row
        pair = (row["station"], row["instrument"])
        if identity.setdefault(row["domain"], pair) != pair:
            _fail("domain station/instrument identity mismatch")
        support = (
            row["contexts"],
            row["unit_appearances"],
            row["physical_masters"],
            row["distinct_units"],
        )
        support_key = (row["estimand"], row["endpoint"], row["domain"])
        if supports.setdefault(support_key, support) != support:
            _fail("domain support mismatch")
    for bucket in by_domain_combo.values():
        if len(bucket) != 13:
            _fail("expected 13 global domains")
        names = sorted(bucket)
        if order is None:
            order = names
        elif names != order:
            _fail("domain identity mismatch")
    if len(by_domain_combo) != len(expected_available):
        _fail("domain grid mismatch")
    return list(order or [])


def _policy_label(policy):
    return str(_policy_style(policy).get("short", policy))


def _policy_defs():
    lines = []
    for index, policy in enumerate(POLICIES):
        style = _policy_style(policy)
        name = "Pol" + chr(65 + index)
        lines.append("\\definecolor{%s}{HTML}{%s}" % (name, style["color"].lstrip("#").upper()))
        mark = "square*" if _is_square(style) else "*"
        lines.append(
            "\\pgfplotsset{pt%d/.style={only marks, mark=%s, color=%s, "
            "mark options={fill=%s, draw=black}}}" % (index, mark, name, name)
        )
        lines.append(
            "\\pgfplotsset{bar%d/.style={%s, line width=0.8pt, mark=|, "
            "mark options={color=%s}}}" % (index, name, name)
        )
    return "\n".join(lines)


def _captions(meta):
    pop = meta["population"]
    caption = (
        "Positive interaction means preprocessing helped the deep strategy more "
        "(or hurt it less) than the comparator, not that the deep absolute "
        "balanced accuracy is higher. Whole pipelines: MIN is minimal min-max "
        "scaling; SG is impulse replacement, smoothing and min-max; arPLS is "
        "impulse replacement, baseline correction and min-max; all use "
        "400-1800 cm^-1 and [0,1]. Source-only selection/calibration; frozen CNN "
        "recipe identity across policies. %d original spectra, %d physical "
        "masters, %d instruments, %d held spectra, %d contexts, %d domains. "
        "M01 uses individual predictions; M06 averages model probabilities "
        "within sample/instrument then equal instruments, not mean spectra. Gray "
        "points are domains, not independent samples. 95%% marginal conditional "
        "intervals use 10,000 support-preserving weights; no retraining, no "
        "new-instrument uncertainty, no causal nuisance removal, no G4, no "
        "superiority or winner selection. Multiplicity family 32 retained; "
        "symmetry p-values are descriptive and no stars are drawn."
        % (
            pop["primary_spectra"],
            pop["masters"],
            pop["instruments"],
            pop["held_spectra"],
            pop["contexts"],
            pop["held_domains"],
        )
    )
    return {
        "header": "RQ-S01 / P08-F04 model-by-preprocessing interaction",
        "caption": caption,
        "qc_note": ("Two planned QC contrasts are outside this universal run; no estimates."),
    }


def _build_semantic(overall, domains, order, meta, source_sha):
    return {
        "figure_id": FIGURE_ID,
        "research_question_id": RESEARCH_QUESTION_ID,
        "schema_version": SCHEMA_VERSION,
        "source_semantic_sha256": source_sha,
        "estimands": list(ESTIMANDS),
        "endpoints": list(ENDPOINTS),
        "deep_models": list(DEEP_MODELS),
        "comparators": list(COMPARATORS),
        "policies": list(POLICIES),
        "modes": list(MODES),
        "mode_labels": dict(MODE_LABELS),
        "domain_order": list(order),
        "domain_offsets": list(DOMAIN_OFFSETS),
        "row_order": [{"comparator_model_id": c, "policy_id": p} for c, p in ROW_ORDER],
        "estimand_labels": {e: ESTIMAND_LABELS.get(e, e) for e in ESTIMANDS},
        "endpoint_labels": {e: ENDPOINT_LABELS.get(e, e) for e in ENDPOINTS},
        "model_labels": {m: MODEL_LABELS.get(m, m) for m in DEEP_MODELS},
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
        "f04_interactions": overall,
        "f04_domains": domains,
    }


def _panel_data(sem, estimand, endpoint, deep, mode):
    overall = {
        (
            r["estimand"],
            r["policy_id"],
            r["deep_model_id"],
            r["comparator_model_id"],
            r["endpoint"],
        ): r
        for r in sem["f04_interactions"]
    }
    domains = {}
    for row in sem["f04_domains"]:
        domains.setdefault((row["estimand"], row["contrast_id"]), {})[row["domain"]] = row
    lo_key, hi_key, reason_key = MODE_FIELDS[mode]
    rows = []
    for index, (comparator, policy) in enumerate(ROW_ORDER):
        effect = overall[(estimand, policy, deep, comparator, endpoint)]
        y = len(ROW_ORDER) - 1 - index
        points = []
        for offset_index, name in enumerate(sem["domain_order"]):
            item = dict(domains[(estimand, effect["contrast_id"])][name])
            item["x"] = float(item["point_effect"])
            item["y"] = y + DOMAIN_OFFSETS[offset_index]
            points.append(item)
        rows.append(
            {
                "comparator_model_id": comparator,
                "policy_id": policy,
                "y": y,
                "label": MODEL_LABELS[comparator] + " " + _policy_label(policy),
                "point": float(effect["point_effect"]),
                "lower": effect[lo_key],
                "upper": effect[hi_key],
                "reason": effect[reason_key],
                "effect": effect,
                "domains": points,
            }
        )
    table = [row["effect"] for row in rows]
    for comparator in ("C-RBF-SVM", "C-RANDOM-FOREST"):
        table.append(overall[(estimand, QC_POLICY, deep, comparator, endpoint)])
    return {
        "estimand": estimand,
        "endpoint": endpoint,
        "deep_model_id": deep,
        "mode": mode,
        "rows": rows,
        "overall": table,
    }


def _panel_tex(sem, panel, sha):
    tag = "P" if (panel["estimand"] == PRIMARY_ESTIMAND and panel["mode"] == PRIMARY_MODE) else "S"
    header = (
        tag + ": " + _tex(str(MODEL_LABELS.get(panel["deep_model_id"], panel["deep_model_id"])))
    )
    n = len(panel["rows"])
    ylabels = ", ".join("{" + _tex(row["label"]) + "}" for row in reversed(panel["rows"]))
    body = []
    for row in panel["rows"]:
        y = float(row["y"])
        points = " ".join("(%s,%s)" % (_f(d["x"]), _f(d["y"])) for d in row["domains"])
        body.append("\\addplot[domainpts] coordinates {" + points + "};")
        index = POLICIES.index(row["policy_id"])
        if row["lower"] is None:
            body.append("\\node[intervalna] at (axis cs:1.96,%s) {interval NA};" % _f(y + 0.30))
        else:
            body.append(
                "\\addplot[bar%d] coordinates {(%s,%s) (%s,%s)};"
                % (index, _f(row["lower"]), _f(y), _f(row["upper"]), _f(y))
            )
        body.append("\\addplot[pt%d] coordinates {(%s,%s)};" % (index, _f(row["point"]), _f(y)))
    caption = _tex(
        sem["captions"]["header"]
        + ". Endpoint: "
        + str(sem["endpoint_labels"][panel["endpoint"]])
        + ". Mode: "
        + sem["mode_labels"][panel["mode"]]
        + ". Estimand: "
        + sem["estimand_labels"][panel["estimand"]]
        + ". "
        + sem["captions"]["caption"]
        + " "
        + sem["captions"]["qc_note"]
    )
    axis = (
        "\\begin{axis}[width=145mm, height=80mm, clip=false, xmin=-2, xmax=2, "
        "xtick={-2,-1,0,1,2}, xticklabels={-200,-100,0,100,200}, "
        "xlabel={" + XLABEL + "}, ymin=-0.5, ymax=" + _f(n - 0.5) + ", "
        "ytick={" + ",".join(str(i) for i in range(n)) + "}, "
        "yticklabels={" + ylabels + "}, "
        "yticklabel style={font=\\fontsize{8}{10}\\selectfont}, "
        "tick label style={font=\\fontsize{8}{10}\\selectfont}, axis lines=left, "
        "title={" + header + "}, "
        "title style={font=\\bfseries\\fontsize{10}{12}\\selectfont}, "
        "every axis plot/.append style={line width=0.6pt}]"
    )
    lines = [
        "% P08-F04 bounded interaction renderer -- copied estimates only",
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
        "\\draw[black, dash dot, line width=0.6pt] (axis cs:0,-0.5) -- "
        "(axis cs:0," + _f(n - 0.5) + ");",
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


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return _canonical_json(value)
    return value


def _cell(value):
    return _html("" if value is None else str(value))


def _table(title, keys, rows):
    head = "".join("<th>%s</th>" % _html(key) for key in keys)
    body = "".join(
        "<tr>" + "".join("<td>%s</td>" % _cell(row.get(key)) for key in keys) + "</tr>"
        for row in rows
    )
    return (
        '<h3>%s</h3><div class="data-table"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>'
        % (_html(title), head, body)
    )


def _json_block(element_id, payload):
    text = _json_script(payload)
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("</", "<\\/")
    return '<script type="application/json" id="%s">%s</script>' % (_html(element_id), text)


def _slug(estimand, endpoint, deep, mode):
    return "p08-f04-%s-%s-%s-%s" % (estimand, endpoint, deep, mode)


def _panel_csv(panel, sha):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(CSV_PREFIX) + list(CSV_UNION_KEYS))
    for row in panel["overall"]:
        writer.writerow(
            ["overall", sha, panel["mode"]] + [_csv_value(row.get(key)) for key in CSV_UNION_KEYS]
        )
    for row in panel["rows"]:
        for domain in row["domains"]:
            writer.writerow(
                ["domain", sha, panel["mode"]]
                + [_csv_value(domain.get(key)) for key in CSV_UNION_KEYS]
            )
    return buf.getvalue()


def _panel_html(sem, panel, sha, csv_text):
    width, height = 900, 450
    left, right, top, bottom = 240, 24, 34, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(panel["rows"])
    span = float(n)
    tag = "P" if (panel["estimand"] == PRIMARY_ESTIMAND and panel["mode"] == PRIMARY_MODE) else "S"
    slug = _slug(panel["estimand"], panel["endpoint"], panel["deep_model_id"], panel["mode"])
    endpoint_label = str(sem["endpoint_labels"][panel["endpoint"]])
    estimand_label = str(sem["estimand_labels"][panel["estimand"]])
    mode_label = str(sem["mode_labels"][panel["mode"]])
    caption = sem["captions"]["caption"]

    def sx(x):
        return left + (float(x) + 2.0) / 4.0 * plot_w

    def sy(y):
        return top + ((n - 0.5) - float(y)) / span * plot_h

    scope = (
        "%s panel: %d available rows plotted, two planned QC rows in the "
        "table/CSV; 13 station/instrument domains, not independent samples" % (tag, n)
    )
    parts = [
        '<figure id="%s">' % _html(slug),
        "<figcaption><strong>%s</strong></figcaption>" % _html(sem["captions"]["header"]),
        "<h2>%s / %s model-by-preprocessing interaction</h2>"
        % (_html(RESEARCH_QUESTION_ID), _html(FIGURE_ID)),
        (
            '<p class="scope"><strong>Deep strategy:</strong> '
            + _html(MODEL_LABELS[panel["deep_model_id"]])
            + " &middot; <strong>Endpoint:</strong> "
            + _html(endpoint_label)
            + " &middot; <strong>Estimand:</strong> "
            + _html(estimand_label)
            + " &middot; <strong>Mode:</strong> "
            + _html(mode_label)
            + " &middot; <strong>Scope:</strong> "
            + _html(scope)
            + "</p>"
        ),
        '<p class="caption">%s</p>' % _html(caption),
        "<p>Semantic SHA-256: <code>%s</code></p>" % _html(sha),
        ('<p class="scope">%s</p>' % _html(sem["captions"]["qc_note"])),
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
            'role="img" aria-label="%s">' % (width, height, _html(caption))
        ),
    ]
    for tick in XTICKS:
        x = sx(tick)
        parts.append(
            '<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>' % (x, top, x, top + plot_h)
        )
        parts.append(
            '<text class="tick" x="%.1f" y="%d" text-anchor="middle">'
            "%d</text>" % (x, top + plot_h + 20, int(round(tick * 100)))
        )
    parts.append(
        '<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>'
        % (left + plot_w / 2.0, top + plot_h + 46, _html(XLABEL))
    )
    x0 = sx(0.0)
    parts.append(
        '<line class="null" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>' % (x0, top, x0, top + plot_h)
    )
    for row in panel["rows"]:
        py = sy(row["y"])
        parts.append(
            '<text class="ytick" x="%d" y="%.1f" text-anchor="end">%s</text>'
            % (left - 8, py + 5, _html(row["label"]))
        )
        for domain in row["domains"]:
            tip = "%s %s domain %s (%s/%s): %s contexts=%s appearances=%s masters=%s units=%s" % (
                row["comparator_model_id"],
                row["policy_id"],
                domain["domain"],
                domain["station"],
                domain["instrument"],
                domain["point_effect"],
                domain["contexts"],
                domain["unit_appearances"],
                domain["physical_masters"],
                domain["distinct_units"],
            )
            parts.append(
                '<circle class="dp" cx="%.1f" cy="%.1f" r="3" '
                'tabindex="0" data-tip="%s" aria-label="%s"></circle>'
                % (sx(domain["x"]), sy(domain["y"]), _html(tip), _html(tip))
            )
        style = _policy_style(row["policy_id"])
        color = style["color"]
        if row["lower"] is not None:
            low, high = sx(row["lower"]), sx(row["upper"])
            itip = "%s %s %s interval [%s, %s]" % (
                row["comparator_model_id"],
                row["policy_id"],
                panel["mode"],
                row["lower"],
                row["upper"],
            )
            parts.append(
                '<line class="iv" x1="%.1f" y1="%.1f" x2="%.1f" '
                'y2="%.1f" stroke="%s" tabindex="0" data-tip="%s" '
                'aria-label="%s"></line>'
                % (low, py, high, py, _html(color), _html(itip), _html(itip))
            )
            for cap in (low, high):
                parts.append(
                    '<line class="cap" x1="%.1f" y1="%.1f" x2="%.1f" '
                    'y2="%.1f" stroke="%s"></line>' % (cap, py - 5, cap, py + 5, _html(color))
                )
        else:
            rtip = "%s %s %s interval NA: %s" % (
                row["comparator_model_id"],
                row["policy_id"],
                panel["mode"],
                row["reason"] or "",
            )
            parts.append(
                '<text class="natick" x="%.1f" y="%.1f" '
                'text-anchor="end" tabindex="0" data-tip="%s" '
                'aria-label="%s">interval NA</text>'
                % (sx(1.96), sy(float(row["y"]) + 0.30), _html(rtip), _html(rtip))
            )
        otip = "%s %s: point %s (%s/%s/%s)" % (
            row["comparator_model_id"],
            row["policy_id"],
            row["point"],
            panel["estimand"],
            panel["endpoint"],
            panel["mode"],
        )
        point_x = sx(row["point"])
        if _is_square(style):
            parts.append(
                '<rect class="op" x="%.1f" y="%.1f" width="8" '
                'height="8" fill="%s" tabindex="0" data-tip="%s" '
                'aria-label="%s"></rect>'
                % (point_x - 4, py - 4, _html(color), _html(otip), _html(otip))
            )
        else:
            parts.append(
                '<circle class="op" cx="%.1f" cy="%.1f" r="4.5" '
                'fill="%s" tabindex="0" data-tip="%s" '
                'aria-label="%s"></circle>' % (point_x, py, _html(color), _html(otip), _html(otip))
            )
    parts.append("</svg>")
    parts.append(
        '<div class="tools"><label><input type="checkbox" id="dpoints" '
        'checked> domain points</label> <label><input type="checkbox" '
        'id="intervals" checked> intervals</label> '
        '<a class="btn" download="%s" href="data:text/csv;charset=utf-8,%s">'
        "CSV</a></div>" % (_html(slug + ".csv"), quote(csv_text))
    )
    parts.append(_table("Overall rows", OVERALL_KEYS, panel["overall"]))
    parts.append(
        _table("Domain rows", DOMAIN_KEYS, [d for row in panel["rows"] for d in row["domains"]])
    )
    parts.append('<div class="tip" role="status" aria-live="polite"></div>')
    parts.append(
        _json_block(
            "panel-data-" + slug,
            {
                "figure_id": FIGURE_ID,
                "research_question_id": RESEARCH_QUESTION_ID,
                "schema_version": SCHEMA_VERSION,
                "source_semantic_sha256": sem["source_semantic_sha256"],
                "semantic_sha256": sha,
                "slug": slug,
                "estimand": panel["estimand"],
                "endpoint": panel["endpoint"],
                "deep_model_id": panel["deep_model_id"],
                "mode": panel["mode"],
                "scope": tag,
                "metadata": sem["metadata"],
                "csv": csv_text,
                "rows": panel["rows"],
                "overall": panel["overall"],
            },
        )
    )
    parts.append(_JS)
    parts.append("</figure>")
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>%s</title>'
        "<style>%s</style></head><body>%s</body></html>" % (_html(slug), _CSS, "".join(parts))
    )


def _panel(sem, estimand, endpoint, deep, mode, sha):
    data = _panel_data(sem, estimand, endpoint, deep, mode)
    csv_text = _panel_csv(data, sha)
    return {
        "slug": _slug(estimand, endpoint, deep, mode),
        "semantic_sha256": sha,
        "tex": _panel_tex(sem, data, sha),
        "html": _panel_html(sem, data, sha, csv_text),
        "csv": csv_text,
    }


def prepare_f04(prepared):
    """Authenticate and project a reviewed interaction adapter for rendering."""
    if not isinstance(prepared, dict):
        _fail("prepared: not a mapping")
    source = prepared.get("semantic")
    if not isinstance(source, dict):
        _fail("semantic: not a mapping")
    source_sha = _canonical_sha(source)
    if source_sha != prepared.get("semantic_sha256"):
        _fail("semantic SHA mismatch")
    if not isinstance(prepared.get("manifest"), dict):
        _fail("manifest: not a mapping")
    overall = [
        _copy_row(row, OVERALL_KEYS, "f04_interactions")
        for row in source.get("f04_interactions", [])
    ]
    domains = [_copy_row(row, DOMAIN_KEYS, "f04_domains") for row in source.get("f04_domains", [])]
    order = _validate(overall, domains)
    meta = _clean_metadata(source.get("metadata"))
    semantic = _build_semantic(overall, domains, order, meta, source_sha)
    sha = _canonical_sha(semantic)
    panels = [
        _panel(semantic, e, en, deep, mode, sha)
        for e in ESTIMANDS
        for en in ENDPOINTS
        for deep in DEEP_MODELS
        for mode in MODES
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
        "slugs": [panel["slug"] for panel in panels],
    }
    return {"semantic": semantic, "semantic_sha256": sha, "panels": panels, "manifest": manifest}
