"""P08-F02 bounded descriptive renderer (source-only, prepared/unreviewed/unpublished).

Pure string rendering: no statistics are recomputed, nothing is written or compiled,
and no remote assets are referenced.  The returned native TeX / HTML / CSV source keeps
the accepted incoming values unchanged.
"""

# Embedded TeX/HTML/JS uses readable literal lines and %-formatting to avoid
# doubling native TeX/CSS braces (same convention as existing figure renderers).
# ruff: noqa: E501, UP031

from __future__ import annotations

import copy
import csv
import hashlib
import html as _html_module
import io
import json
import math

ESTIMANDS = ("equal_context", "pooled_four_fold")
ENDPOINTS = ("M01", "M06")
MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M", "P05-SELECTED")
POLICIES = ("PP-U-SG", "PP-U-ARPLS")
X_REFERENCE = "PP-U-MIN"

ESTIMAND_SLUGS = {"equal_context": "equal-context", "pooled_four_fold": "pooled-four-fold"}
ENDPOINT_SLUGS = {"M01": "m01", "M06": "m06"}
MODEL_SLUGS = {
    "C-RBF-SVM": "c-rbf-svm",
    "C-RANDOM-FOREST": "c-random-forest",
    "C-EXTRA-TREES": "c-extra-trees",
    "D0-M": "d0-m",
    "P05-SELECTED": "p05-selected",
}
POLICY_STYLE = {
    "PP-U-SG": {
        "color": "#0072B2",
        "tex": "f02sg",
        "marker": "circle",
        "tex_marker": "*",
        "legend": "SG",
    },
    "PP-U-ARPLS": {
        "color": "#D55E00",
        "tex": "f02arpls",
        "marker": "square",
        "tex_marker": "square*",
        "legend": "arPLS",
    },
}
MODEL_LABELS = {
    "C-RBF-SVM": "RBF-SVM",
    "C-RANDOM-FOREST": "Random Forest",
    "C-EXTRA-TREES": "Extra Trees",
    "D0-M": "Ordinary CNN",
    "P05-SELECTED": "Source-selected CNN",
}
ESTIMAND_LABELS = {
    "equal_context": "primary: equally weighted contexts within domain then equal domains",
    "pooled_four_fold": "sensitivity: reconstruct each four-fold pooled prediction set before scoring",
}
ENDPOINT_LABELS = {
    "M01": "M01 individual-spectrum predictions",
    "M06": "M06 combined predictions per physical sample",
}
POLICY_LABELS = {
    "PP-U-SG": "PP-U-SG: impulse replacement + smoothing + minmax",
    "PP-U-ARPLS": "PP-U-ARPLS: impulse replacement + baseline correction + minmax",
}
PANEL_TITLES = {"PP-U-SG": "A  Smoothing (SG)", "PP-U-ARPLS": "B  Baseline correction (arPLS)"}

PAIR_KEYS = (
    "estimand",
    "contrast_id",
    "family_id",
    "endpoint",
    "model_id",
    "policy_id",
    "domain",
    "station",
    "instrument",
    "contexts",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
    "x_balanced_accuracy",
    "y_balanced_accuracy",
    "effect",
)
SUPPORT_KEYS = ("contexts", "unit_appearances", "physical_masters", "distinct_units")
APPROVED_METADATA = (
    "population",
    "independent_unit",
    "selection",
    "interval_caption",
    "interaction_caption",
    "counts_reference",
    "display",
)
AXES = {
    "x_label": "MIN balanced accuracy",
    "y_label": "Policy balanced accuracy",
    "xlim": [0.0, 1.0],
    "ylim": [0.0, 1.0],
    "ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
}
TOLERANCE = 1e-12

_TEX_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_SVG_W, _SVG_H, _SVG_GAP = 380, 320, 24
_PAD_L, _PAD_B, _PAD_T, _PAD_R = 60, 52, 40, 16

TABLE_HEADERS = (
    "D#",
    "Domain",
    "Station",
    "Instrument",
    "Policy",
    "x (MIN)",
    "y (policy)",
    "effect",
    "contexts",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
)


def _canonical_json(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _canonical_sha(obj):
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _tex(value):
    return "".join(_TEX_MAP.get(char, char) for char in str(value))


def _html(value):
    return _html_module.escape(str(value), quote=True)


def _json_script(obj):
    return (
        _canonical_json(obj).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )


def _tick(value):
    return "%g" % value


def _px(value, low, high, out_low, out_high):
    return out_low + (value - low) / (high - low) * (out_high - out_low)


def _caption_text():
    return (
        "RQ-S01: preprocessing and chemical identification. Source-only selection/calibration; "
        "selected CNN identity fixed across policies. SG = impulse replacement, smoothing, minmax; "
        "arPLS = impulse replacement, baseline correction, minmax. MIN = minimal minmax. "
        "Primary: equal contexts within domain, then equal domains. Pooled-fold sensitivity "
        "reconstructs each four-fold prediction set before scoring. "
        "M01 individual-spectrum predictions; M06 combined predictions per physical sample. "
        "M06 averages model probabilities within sample/instrument then equally across instruments, "
        "never input spectra or hard labels. "
        "598 original spectra / 69 physical samples / 10 instruments; 557 spectra in 13 held "
        "station/instrument domains, 260 contexts. Per-domain independent support in the CSV/table. "
        "D1..D13 are lexicographic station/instrument domains, not samples. Diagonal y=x; the 520 points across all panels "
        "are paired domain measurements, not 520 independent observations. Claim limit: cannot establish "
        "causal background removal or new-instrument performance; no error bars, no significance stars."
    )


def _check_row(row):
    if not isinstance(row, dict) or set(row) != set(PAIR_KEYS):
        raise ValueError("f02 pair has unexpected keys")
    if row["estimand"] not in ESTIMANDS:
        raise ValueError("unknown estimand")
    if row["endpoint"] not in ENDPOINTS:
        raise ValueError("unknown endpoint")
    if row["model_id"] not in MODELS:
        raise ValueError("unknown model_id")
    if row["policy_id"] not in POLICIES:
        raise ValueError("unknown policy_id")
    for key in ("x_balanced_accuracy", "y_balanced_accuracy", "effect"):
        value = row[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("non-finite accuracy or effect")
    for key in ("x_balanced_accuracy", "y_balanced_accuracy"):
        if not 0.0 <= row[key] <= 1.0:
            raise ValueError("accuracy outside [0,1]")
    if abs(row["effect"] - (row["y_balanced_accuracy"] - row["x_balanced_accuracy"])) > TOLERANCE:
        raise ValueError("effect does not equal y - x")
    for key in SUPPORT_KEYS:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("support counts must be positive non-bool integers")
    for key in ("contrast_id", "family_id", "domain", "station", "instrument"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError("aggregate labels must be nonempty strings")


def _validate_pairs(pairs):
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("f02_pairs must be a non-empty list")
    seen = set()
    for row in pairs:
        _check_row(row)
        key = (row["estimand"], row["endpoint"], row["model_id"], row["policy_id"], row["domain"])
        if key in seen:
            raise ValueError("duplicate f02 row")
        seen.add(key)
    domains = {row["domain"] for row in pairs}
    if len(domains) != 13:
        raise ValueError("expected exactly 13 domains with no hidden filtering")
    expected = len(ESTIMANDS) * len(ENDPOINTS) * len(MODELS) * len(POLICIES) * len(domains)
    if len(seen) != expected or len(pairs) != expected:
        raise ValueError("incomplete f02 grid")
    wanted = {
        (e, ep, m, p, d)
        for e in ESTIMANDS
        for ep in ENDPOINTS
        for m in MODELS
        for p in POLICIES
        for d in domains
    }
    if seen != wanted:
        raise ValueError("incomplete f02 grid")
    identities = {}
    for row in pairs:
        identity = (row["station"], row["instrument"])
        if identities.setdefault(row["domain"], identity) != identity:
            raise ValueError("inconsistent domain identity")
    if len(set(identities.values())) != 13:
        raise ValueError("domain aliases are not unique station/instrument pairs")
    return sorted(domains, key=lambda d: identities[d])


def _labels(semantic_in):
    incoming = semantic_in.get("labels")
    incoming = incoming if isinstance(incoming, dict) else {}
    endpoints = dict(ENDPOINT_LABELS)
    policies = dict(POLICY_LABELS)
    if isinstance(incoming.get("endpoints"), dict):
        endpoints.update({k: str(v) for k, v in incoming["endpoints"].items() if k in ENDPOINTS})
    if isinstance(incoming.get("policies"), dict):
        policies.update({k: str(v) for k, v in incoming["policies"].items() if k in POLICIES})
    return {
        "models": dict(MODEL_LABELS),
        "estimands": dict(ESTIMAND_LABELS),
        "endpoints": endpoints,
        "policies": policies,
    }


def prepare_f02(prepared):
    if not isinstance(prepared, dict):
        raise ValueError("prepared payload must be a mapping")
    semantic_in = prepared.get("semantic")
    if not isinstance(semantic_in, dict):
        raise ValueError("prepared payload missing semantic mapping")
    if _canonical_sha(semantic_in) != prepared.get("semantic_sha256"):
        raise ValueError("incoming semantic hash does not match")
    pairs = semantic_in.get("f02_pairs")
    domains = _validate_pairs(pairs)
    domain_index = {name: "D%d" % (index + 1) for index, name in enumerate(domains)}
    labels = _labels(semantic_in)
    source_meta = semantic_in.get("metadata", {})
    if set(source_meta) != set(APPROVED_METADATA):
        raise ValueError("missing approved model metadata")
    expected_population = dict(
        primary_spectra=598,
        held_spectra=557,
        masters=69,
        instruments=10,
        held_domains=13,
        contexts=260,
    )
    if (
        source_meta["population"] != expected_population
        or source_meta["independent_unit"] != "physical_master"
    ):
        raise ValueError("population or independent unit changed")
    if any(not isinstance(source_meta[k], str) for k in APPROVED_METADATA if k != "population"):
        raise ValueError("metadata text must be scalar strings")
    metadata = copy.deepcopy(source_meta)
    semantic = {
        "figure_id": "P08-F02",
        "research_question_id": "RQ-S01",
        "schema_version": "nato-sers-p08-f02-render-v1",
        "source_model_semantic_sha256": prepared["semantic_sha256"],
        "f02_pairs": copy.deepcopy(pairs),
        "estimands": list(ESTIMANDS),
        "endpoints": list(ENDPOINTS),
        "models": list(MODELS),
        "policies": list(POLICIES),
        "x_reference": X_REFERENCE,
        "order": {
            "estimands": list(ESTIMANDS),
            "endpoints": list(ENDPOINTS),
            "models": list(MODELS),
            "policies": list(POLICIES),
        },
        "domain_index": domain_index,
        "axes": dict(AXES),
        "style": {policy: dict(POLICY_STYLE[policy]) for policy in POLICIES},
        "labels": labels,
        "captions": {
            "figure": _caption_text(),
            "endpoints": labels["endpoints"],
            "policies": labels["policies"],
        },
        "metadata": metadata,
    }
    sha = _canonical_sha(semantic)
    panels = [
        _panel(
            "f02-%s-%s-%s" % (ESTIMAND_SLUGS[e], ENDPOINT_SLUGS[ep], MODEL_SLUGS[m]),
            sha,
            e,
            ep,
            m,
            domain_index,
            pairs,
            labels,
            metadata,
        )
        for e in ESTIMANDS
        for ep in ENDPOINTS
        for m in MODELS
    ]
    manifest = {
        "figure_id": "P08-F02",
        "status": "prepared",
        "review": "unreviewed",
        "publication": "unpublished",
        "reviewed": False,
        "published": False,
        "semantic_sha256": sha,
        "panels": [panel["slug"] for panel in panels],
        "notes": "Source strings only; nothing was written, compiled or fetched.",
    }
    return {"semantic": semantic, "semantic_sha256": sha, "panels": panels, "manifest": manifest}


def _panel(slug, sha, estimand, endpoint, model, domain_index, pairs, labels, metadata):
    by_policy = {}
    for policy in POLICIES:
        rows = [
            row
            for row in pairs
            if row["estimand"] == estimand
            and row["endpoint"] == endpoint
            and row["model_id"] == model
            and row["policy_id"] == policy
        ]
        rows.sort(key=lambda row: int(domain_index[row["domain"]][1:]))
        by_policy[policy] = rows
    ctx = {
        "slug": slug,
        "sha": sha,
        "estimand": estimand,
        "endpoint": endpoint,
        "model": model,
        "domain_index": domain_index,
        "by_policy": by_policy,
        "labels": labels,
        "metadata": metadata,
    }
    return {
        "slug": slug,
        "semantic_sha256": sha,
        "tex": _tex_panel(ctx),
        "html": _html_panel(ctx),
        "csv": _csv_panel(ctx),
    }


def _tex_panel(ctx):
    labels = ctx["labels"]
    lines = [
        r"\documentclass[border=0pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{f02sg}{HTML}{0072B2}",
        r"\definecolor{f02arpls}{HTML}{D55E00}",
        r"\begin{document}",
        r"\begin{minipage}{181.86mm}",
        r"\centering",
        r"{\bfseries\fontsize{10}{12}\selectfont P08-F02 "
        + _tex(labels["models"][ctx["model"]])
        + " / "
        + _tex(ENDPOINT_LABELS[ctx["endpoint"]])
        + " / "
        + ("Primary (P)" if ctx["estimand"] == "equal_context" else "Sensitivity (S)")
        + r"}\par\vspace{6mm}",
        "% semantic SHA-256: " + ctx["sha"],
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1, horizontal sep=10mm, vertical sep=6mm},",
        r"  width=75mm, height=65mm,",
        r"  xmin=0, xmax=1, ymin=0, ymax=1,",
        r"  xtick={0,0.25,0.5,0.75,1}, ytick={0,0.25,0.5,0.75,1},",
        r"  tick label style={font=\fontsize{8}{9}\selectfont},",
        r"  label style={font=\fontsize{8}{9}\selectfont},",
        r"  title style={font=\bfseries\fontsize{10}{12}\selectfont},",
        r"  xlabel={MIN balanced accuracy}, ylabel={Policy balanced accuracy},",
        r"]",
    ]
    for policy in POLICIES:
        style = POLICY_STYLE[policy]
        coords = " ".join(
            "(%r,%r)" % (row["x_balanced_accuracy"], row["y_balanced_accuracy"])
            for row in ctx["by_policy"][policy]
        )
        lines.append(r"\nextgroupplot[title={" + _tex(PANEL_TITLES[policy]) + r"}]")
        lines.append(
            r"\addplot[black, dash dot, line width=0.6pt, samples=2, domain=0:1, forget plot] {x};"
        )
        lines.append(
            r"\addplot[only marks, mark="
            + style["tex_marker"]
            + r", mark size=1.7pt, color="
            + style["tex"]
            + r", mark options={draw=black,line width=0.8pt}] coordinates {"
            + coords
            + r"};"
        )
    lines.extend(
        [
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            r"\par\vspace{4pt}",
            r"{\fontsize{8}{10}\selectfont " + _tex(_caption_text()) + r"}\par",
            r"\end{minipage}",
            r"\end{document}",
        ]
    )
    return "\n".join(lines)


def _csv_panel(ctx):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["semantic_sha256"] + list(TABLE_HEADERS))
    for policy in POLICIES:
        for row in ctx["by_policy"][policy]:
            writer.writerow(
                [
                    ctx["sha"],
                    ctx["domain_index"][row["domain"]],
                    row["domain"],
                    row["station"],
                    row["instrument"],
                    row["policy_id"],
                    row["x_balanced_accuracy"],
                    row["y_balanced_accuracy"],
                    row["effect"],
                    row["contexts"],
                    row["unit_appearances"],
                    row["physical_masters"],
                    row["distinct_units"],
                ]
            )
    return buffer.getvalue()


def _svg_points(rows, domain_index, policy, color, shape):
    out = []
    for row in rows:
        cx = _px(row["x_balanced_accuracy"], 0.0, 1.0, _PAD_L, _SVG_W - _PAD_R)
        cy = _px(row["y_balanced_accuracy"], 0.0, 1.0, _SVG_H - _PAD_B, _PAD_T)
        tip = (
            "%s | domain %s | station %s | instrument %s | support contexts=%d unit_appearances=%d "
            "physical_masters=%d distinct_units=%d | x=%.6g y=%.6g effect=%.6g"
        ) % (
            row["policy_id"],
            domain_index[row["domain"]],
            row["station"],
            row["instrument"],
            row["contexts"],
            row["unit_appearances"],
            row["physical_masters"],
            row["distinct_units"],
            row["x_balanced_accuracy"],
            row["y_balanced_accuracy"],
            row["effect"],
        )
        if shape == "circle":
            mark = (
                '<circle cx="%.2f" cy="%.2f" r="3.6" fill="%s" stroke="#000" stroke-width="0.8"/>'
                % (cx, cy, color)
            )
        else:
            mark = (
                '<rect x="%.2f" y="%.2f" width="7.2" height="7.2" fill="%s" stroke="#000" stroke-width="0.8"/>'
                % (cx - 3.6, cy - 3.6, color)
            )
        out.append(
            '<g class="pt" tabindex="0" role="img" data-policy="%s" aria-label="%s">%s<title>%s</title></g>'
            % (policy, _html(tip), mark, _html(tip))
        )
    return "".join(out)


def _svg_subplot(offset, title, policy, rows, domain_index, color, shape):
    x0, x1, y0, y1 = _PAD_L, _SVG_W - _PAD_R, _SVG_H - _PAD_B, _PAD_T
    out = [
        '<g class="subplot" transform="translate(%d,0)">' % offset,
        '<rect x="%d" y="%d" width="%d" height="%d" fill="#fff" stroke="#000" stroke-width="1"/>'
        % (x0, y1, x1 - x0, y0 - y1),
    ]
    for tick in AXES["ticks"]:
        tx, ty = _px(tick, 0.0, 1.0, x0, x1), _px(tick, 0.0, 1.0, y0, y1)
        out.append(
            '<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#000" stroke-width="0.5"/>'
            % (tx, y0, tx, y1)
        )
        out.append(
            '<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#000" stroke-width="0.5"/>'
            % (x0, ty, x1, ty)
        )
        out.append(
            '<text x="%.2f" y="%d" text-anchor="middle" font-size="16">%s</text>'
            % (tx, y0 + 20, _tick(tick))
        )
        out.append(
            '<text x="%d" y="%.2f" text-anchor="end" font-size="16">%s</text>'
            % (x0 - 6, ty + 5, _tick(tick))
        )
    out.append(
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#000" stroke-width="0.8" stroke-dasharray="7 3 1 3"/>'
        % (x0, y0, x1, y1)
    )
    out.append(
        '<text x="%.2f" y="%d" text-anchor="middle" font-size="16">MIN balanced accuracy</text>'
        % ((x0 + x1) / 2.0, y0 + 44)
    )
    mid = (y0 + y1) / 2.0
    out.append(
        '<text x="16" y="%.2f" text-anchor="middle" font-size="16" transform="rotate(-90 16 %.2f)">'
        "Policy balanced accuracy</text>" % (mid, mid)
    )
    out.append(
        '<text x="%.2f" y="%d" text-anchor="middle" font-size="16" font-weight="bold">%s</text>'
        % ((x0 + x1) / 2.0, y1 - 12, _html(title))
    )
    out.append(_svg_points(rows, domain_index, policy, color, shape))
    out.append("</g>")
    return "".join(out)


def _svg_panel(ctx):
    width = 2 * _SVG_W + _SVG_GAP
    parts = [
        '<svg width="%d" height="%d" viewBox="0 0 %d %d" role="group" aria-label="P08-F02 scatter panels">'
        % (width, _SVG_H, width, _SVG_H)
    ]
    for index, policy in enumerate(POLICIES):
        style = POLICY_STYLE[policy]
        parts.append(
            _svg_subplot(
                index * (_SVG_W + _SVG_GAP),
                PANEL_TITLES[policy],
                policy,
                ctx["by_policy"][policy],
                ctx["domain_index"],
                style["color"],
                style["marker"],
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _fmt(value):
    return ("%.6g" % value) if isinstance(value, float) else str(value)


_NOTES = (
    "<ul>"
    "<li>RQ-S01 / P08-F02 descriptive paired-domain scatter; no automatic winner selection.</li>"
    "<li>Model source-only selection and calibration; no held-domain tuning. "
    "Selected CNN identity is fixed across policies.</li>"
    "<li>Scope: primary estimand equal_context (equally weighted contexts within domain then equal domains); "
    "sensitivity estimand pooled_four_fold (reconstruct each four-fold pooled prediction set before scoring).</li>"
    "<li>Endpoints: M01 individual-spectrum predictions; M06 combined predictions per physical sample, averaging "
    "model probabilities within physical sample/instrument then equal instrument mean, never averaging input spectra "
    "or hard labels.</li>"
    "<li>Population: 598 original spectra, 69 physical samples, 10 instruments; 557 spectra in 13 held "
    "station/instrument domains and 260 contexts. The 520 panel points are paired domain measurements, not 520 "
    "independent observations.</li>"
    "<li>Aggregation: each point pairs the common PP-U-MIN reference score (x) with one policy score (y) for one "
    "domain; per-domain independent support is listed in the table and CSV.</li>"
    "<li>D1..D13 are lexicographic station/instrument domain indices, not sample indices.</li>"
    "<li>Diagonal meaning: y=x; above favours the policy, on it is a tie, below is unfavorable.</li>"
    "<li>Claim limit: this descriptive scatter cannot establish causal background removal or new-instrument "
    "performance; no error bars or significance stars (intervals elsewhere).</li>"
    "</ul>"
)

_JS_TEMPLATE = """<script>
(function(){var slug="__SLUG__",svg=document.getElementById("svg-"+slug),tip=document.getElementById("tip-"+slug);
function show(ev){var el=ev.target&&ev.target.closest?ev.target.closest(".pt"):null;if(!el)return;
tip.textContent=el.getAttribute("aria-label")||"";tip.style.display="block";
var r=el.getBoundingClientRect();tip.style.left=(r.left+r.width/2)+"px";tip.style.top=(r.top-8)+"px";}
function hide(){tip.style.display="none";}
svg.addEventListener("mouseover",show);svg.addEventListener("mouseout",hide);
svg.addEventListener("focusin",show);svg.addEventListener("focusout",hide);
document.querySelectorAll('.toggle[data-panel="'+slug+'"]').forEach(function(box){
box.addEventListener("change",function(){var on=box.checked,p=box.getAttribute("data-policy");
svg.querySelectorAll('.pt[data-policy="'+p+'"]').forEach(function(el){el.style.display=on?"":"none";});});});
document.getElementById("dl-"+slug).addEventListener("click",function(){
var payload=JSON.parse(document.getElementById("payload-"+slug).textContent);
var blob=new Blob([payload.csv],{type:"text/csv;charset=utf-8"}),url=URL.createObjectURL(blob),a=document.createElement("a");
a.href=url;a.download=slug+".csv";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);});})();
</script>"""


def _html_panel(ctx):
    slug, sha = ctx["slug"], ctx["sha"]
    labels = ctx["labels"]
    svg = _svg_panel(ctx).replace("<svg ", '<svg id="svg-%s" ' % slug, 1)
    payload = _json_script({"semantic_sha256": sha, "slug": slug, "csv": _csv_panel(ctx)})
    rows = []
    for policy in POLICIES:
        for row in ctx["by_policy"][policy]:
            cells = [
                ctx["domain_index"][row["domain"]],
                row["domain"],
                row["station"],
                row["instrument"],
                row["policy_id"],
                _fmt(row["x_balanced_accuracy"]),
                _fmt(row["y_balanced_accuracy"]),
                _fmt(row["effect"]),
                row["contexts"],
                row["unit_appearances"],
                row["physical_masters"],
                row["distinct_units"],
            ]
            rows.append(
                "<tr>" + "".join("<td>" + _html(cell) + "</td>" for cell in cells) + "</tr>"
            )
    toggles = "".join(
        '<label><input type="checkbox" class="toggle" data-panel="%s" data-policy="%s" checked> %s</label> '
        % (slug, policy, _html(POLICY_STYLE[policy]["legend"]))
        for policy in POLICIES
    )
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>P08-F02 " + _html(labels["models"][ctx["model"]]) + "</title>",
        "<style>",
        'body{font-family:"Times New Roman",Times,serif;color:#000;font-size:16px;margin:16px;}',
        "h1{font-size:20px;} h2{font-size:18px;}",
        ".tooltip{position:fixed;display:none;background:#fff;border:1px solid #000;padding:6px;font-size:14px;max-width:460px;z-index:9;}",
        ".pt{cursor:pointer;outline:none;} .pt:focus circle,.pt:focus rect{stroke:#000;stroke-width:2.5;}",
        "table{border-collapse:collapse;font-size:14px;} td,th{border:1px solid #000;padding:3px 5px;}",
        "</style></head><body>",
        "<h1>RQ-S01 / P08-F02 &mdash; Preprocessing: paired domain accuracy</h1>",
        "<p>"
        + _html(labels["models"][ctx["model"]])
        + " &middot; "
        + _html(labels["endpoints"][ctx["endpoint"]])
        + " &middot; "
        + _html(labels["estimands"][ctx["estimand"]])
        + "</p>",
        "<p>semantic SHA-256: <code>"
        + _html(sha)
        + "</code> &middot; x-reference "
        + _html(X_REFERENCE)
        + "</p>",
        "<div>"
        + toggles
        + '<button type="button" id="dl-'
        + slug
        + '">Download CSV</button></div>',
        svg,
        '<div class="tooltip" id="tip-' + slug + '" role="status" aria-live="polite"></div>',
        "<p>" + _html(_caption_text()) + "</p>",
        "<h2>Notes and claim limits</h2>",
        _NOTES,
        "<h2>Full data table (all rows retained)</h2>",
        "<table><thead><tr>"
        + "".join("<th>" + _html(head) + "</th>" for head in TABLE_HEADERS)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>",
        '<script type="application/json" id="payload-' + slug + '">' + payload + "</script>",
        _JS_TEMPLATE.replace("__SLUG__", slug),
        "</body></html>",
    ]
    return "".join(parts)
