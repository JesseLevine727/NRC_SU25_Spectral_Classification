"""P08 U1 compact training-diagnostic renderer (approved U1 reporting).

Pure deterministic string rendering of already-authenticated training
diagnostic curves onto 24 policy/recipe/stage panels as standalone
PGFPlots/TikZ and offline HTML/SVG. No new statistics, resampling, fits,
selection or file IO; the pure writer owns files, compile and review.
"""

from __future__ import annotations

# Literal TeX/HTML/JavaScript templates retain explicit percent interpolation.
# ruff: noqa: E501, UP031
import math

from atlas_sers.visualization.p08_f02_render import _canonical_json, _canonical_sha, _json_script
from atlas_sers.visualization.p08_f02_render import _html as _escape_html

__all__ = ["prepare_training_render"]

SCHEMA_VERSION = "nato-sers-p08-training-render-v1"
SOURCE_SCHEMA = "nato-sers-p08-training-figure-data-v1"
FIGURE_ID = "P08-U1-training-diagnostics"
SCOPE = "E (descriptive training diagnostics; no performance claim)"
RQ = "RQ-S01"

POLICY_ORDER = ("PP-U-SG", "PP-U-ARPLS")
RECIPE_ORDER = ("D0-M", "D1", "D2", "D3")
STAGE_ORDER = ("source_fit", "calibration_model_fit", "final_refit")

POPULATION = {"spectra": 598, "masters": 69, "instruments": 10}
OKABE_ITO_HEX = {
    "blue": "0072B2",
    "vermillion": "D55E00",
    "green": "009E73",
    "purple": "CC79A7",
    "black": "000000",
}

OBJECTIVE_METRICS = ("chemical_ce", "total_loss", "supcon_loss", "paired_loss")
NLL_METRICS = ("train_nll", "validation_nll")
BA_METRICS = ("train_balanced_accuracy", "validation_balanced_accuracy")
METRIC_ORDER = OBJECTIVE_METRICS + NLL_METRICS + BA_METRICS

GROUP_FIELDS = frozenset(
    {
        "policy_id",
        "recipe",
        "stage",
        "planned_jobs",
        "monitored_jobs",
        "missing_history_jobs",
        "max_recorded_epoch",
        "status",
        "coverage",
    }
)
CURVE_FIELDS = frozenset(
    {
        "policy_id",
        "recipe",
        "stage",
        "epoch",
        "metric",
        "n_runs_at_epoch",
        "finite_count",
        "undefined_count",
        "median",
        "q10",
        "q90",
        "reason",
    }
)
CSV_FIELDS = (
    "policy_id",
    "recipe",
    "stage",
    "planned_jobs",
    "monitored_jobs",
    "missing_history_jobs",
    "status",
    "coverage",
    "max_recorded_epoch",
    "epoch",
    "metric",
    "n_runs_at_epoch",
    "finite_count",
    "undefined_count",
    "median",
    "q10",
    "q90",
    "reason",
)

METRIC_STYLE = {
    "chemical_ce": ("blue", "circle", "solid"),
    "total_loss": ("vermillion", "square", "dashed"),
    "supcon_loss": ("green", "triangle", "dotted"),
    "paired_loss": ("purple", "diamond", "dashdotdotted"),
    "train_nll": ("blue", "circle", "solid"),
    "validation_nll": ("vermillion", "square", "dashed"),
    "train_balanced_accuracy": ("blue", "circle", "solid"),
    "validation_balanced_accuracy": ("vermillion", "square", "dashed"),
    "n_runs_at_epoch": ("black", "none", "solid"),
}
LEGEND = {
    "chemical_ce": "CE",
    "total_loss": "Total",
    "supcon_loss": "SupCon",
    "paired_loss": "Paired",
    "train_nll": "Train NLL",
    "validation_nll": "Val NLL",
    "train_balanced_accuracy": "Train BA",
    "validation_balanced_accuracy": "Val BA",
    "n_runs_at_epoch": "Runs",
}
AXIS_TITLES = {
    "A": "A sampled minibatch objective",
    "B": "B post-epoch NLL",
    "C": "C balanced accuracy",
    "D": "D contributing fitting runs",
}
AXIS_LABELS = {
    "A": "objective (loss)",
    "B": "NLL (loss)",
    "C": "balanced accuracy (BA)",
    "D": "runs (count)",
}

_TEX_COLOR = {
    "blue": "oi-blue",
    "vermillion": "oi-vermillion",
    "green": "oi-green",
    "purple": "oi-purple",
    "black": "oi-black",
}
_TIKZ_DASH = {
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dashdotdotted": "dash dot dot",
}
_TIKZ_MARK = {"circle": "*", "square": "square*", "triangle": "triangle*", "diamond": "diamond*"}
_SVG_COLOR = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "black": "#000000",
}
_SVG_DASH = {"solid": "none", "dashed": "5,3", "dotted": "1,3", "dashdotdotted": "6,2,1,2,1,2"}

_TEX_REPL = {
    "\\": "\\textbackslash{}",
    "{": "\\{",
    "}": "\\}",
    "_": "\\_",
    "%": "\\%",
    "&": "\\&",
    "#": "\\#",
    "$": "\\$",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}


def _tex_escape(value):
    return "".join(_TEX_REPL.get(ch, ch) for ch in str(value))


def _g17(value):
    if value is None:
        return ""
    return format(float(value), ".17g")


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _g17(value)
    text = str(value)
    if any(ch in text for ch in ',"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _slug(policy, recipe, stage):
    return "p08-u1-%s-%s-%s" % (policy.lower(), recipe.lower(), stage.replace("_", "-"))


def _validate_semantic(semantic):
    if not isinstance(semantic, dict):
        raise ValueError("semantic must be a dict")
    required = {
        "schema_version",
        "figure_id",
        "caption",
        "groups",
        "curves",
        "metric_order",
        "policy_order",
        "recipe_order",
        "stage_order",
        "population",
    }
    if set(semantic) != required:
        raise ValueError("semantic has an unexpected field set")
    if semantic["schema_version"] != SOURCE_SCHEMA:
        raise ValueError("unknown source schema version")
    if semantic["figure_id"] != FIGURE_ID:
        raise ValueError("unknown figure_id")
    if not isinstance(semantic["caption"], str) or not semantic["caption"]:
        raise ValueError("caption must be a non-empty string")
    if list(semantic["metric_order"]) != list(METRIC_ORDER):
        raise ValueError("metric_order mismatch")
    if list(semantic["policy_order"]) != list(POLICY_ORDER):
        raise ValueError("policy_order mismatch")
    if list(semantic["recipe_order"]) != list(RECIPE_ORDER):
        raise ValueError("recipe_order mismatch")
    if list(semantic["stage_order"]) != list(STAGE_ORDER):
        raise ValueError("stage_order mismatch")
    population = semantic["population"]
    if (
        not isinstance(population, dict)
        or population != POPULATION
        or not all(_is_int(v) for v in population.values())
    ):
        raise ValueError("population must be exactly 598 spectra / 69 masters / 10 instruments")

    groups = semantic["groups"]
    if not isinstance(groups, list) or len(groups) != 24:
        raise ValueError("exactly 24 groups are required")
    expected_keys = {(p, r, s) for p in POLICY_ORDER for r in RECIPE_ORDER for s in STAGE_ORDER}
    gmap = {}
    for group in groups:
        if not isinstance(group, dict) or set(group) != GROUP_FIELDS:
            raise ValueError("group has an unexpected field set")
        key = (group["policy_id"], group["recipe"], group["stage"])
        if key in gmap:
            raise ValueError("duplicate group key")
        if (
            group["policy_id"] not in POLICY_ORDER
            or group["recipe"] not in RECIPE_ORDER
            or group["stage"] not in STAGE_ORDER
        ):
            raise ValueError("unknown group identity")
        for field in ("planned_jobs", "monitored_jobs", "missing_history_jobs"):
            if not _is_int(group[field]) or group[field] < 0:
                raise ValueError("group counts must be non-negative integers")
        if group["planned_jobs"] != group["monitored_jobs"] + group["missing_history_jobs"]:
            raise ValueError("planned must equal monitored plus missing")
        monitored = group["monitored_jobs"]
        if group["planned_jobs"] == 0:
            if (
                monitored != 0
                or group["status"] != "no_registered_jobs"
                or group["coverage"] != "NA"
            ):
                raise ValueError("inconsistent no_registered_jobs group")
        elif monitored == 0:
            if group["status"] != "missing_histories" or group["coverage"] != "NA":
                raise ValueError("inconsistent missing_histories group")
        elif group["missing_history_jobs"] > 0:
            if group["status"] != "partial_history_coverage" or group["coverage"] != "partial":
                raise ValueError("inconsistent partial coverage group")
        else:
            if group["status"] != "complete_history_coverage" or group["coverage"] != "complete":
                raise ValueError("inconsistent complete coverage group")
        max_epoch = group["max_recorded_epoch"]
        if max_epoch is None:
            if monitored != 0:
                raise ValueError("groups with monitored jobs need a max_recorded_epoch")
        elif not _is_int(max_epoch) or max_epoch < 1 or max_epoch > 200 or monitored == 0:
            raise ValueError("invalid max_recorded_epoch")
        gmap[key] = group
    if set(gmap) != expected_keys:
        raise ValueError("group key set mismatch")

    curves = semantic["curves"]
    if not isinstance(curves, list):
        raise ValueError("curves must be a list")
    lookup = {}
    for curve in curves:
        if not isinstance(curve, dict) or set(curve) != CURVE_FIELDS:
            raise ValueError("curve has an unexpected field set")
        key = (curve["policy_id"], curve["recipe"], curve["stage"], curve["epoch"], curve["metric"])
        if key in lookup:
            raise ValueError("duplicate curve row")
        lookup[key] = curve
        group = gmap.get((curve["policy_id"], curve["recipe"], curve["stage"]))
        if group is None:
            raise ValueError("curve references unknown group")
        if curve["metric"] not in METRIC_ORDER:
            raise ValueError("unknown metric")
        if not _is_int(curve["epoch"]) or curve["epoch"] < 1:
            raise ValueError("epoch must be a positive integer")
        if group["max_recorded_epoch"] is None or curve["epoch"] > group["max_recorded_epoch"]:
            raise ValueError("curve epoch out of range")
        n_runs = curve["n_runs_at_epoch"]
        finite = curve["finite_count"]
        undefined = curve["undefined_count"]
        for value in (n_runs, finite, undefined):
            if not _is_int(value) or value < 0:
                raise ValueError("curve counts must be non-negative integers")
        if not 1 <= n_runs <= group["monitored_jobs"]:
            raise ValueError("n_runs_at_epoch out of range")
        if finite + undefined != n_runs:
            raise ValueError("finite plus undefined must equal n_runs")
        quantiles = (curve["q10"], curve["median"], curve["q90"])
        if finite > 0:
            if not all(_is_num(value) for value in quantiles):
                raise ValueError("finite quantiles must be finite")
            if not quantiles[0] <= quantiles[1] <= quantiles[2]:
                raise ValueError("quantiles must be ordered")
            if curve["reason"] is not None:
                raise ValueError("finite rows must not carry a reason")
        else:
            if any(value is not None for value in quantiles):
                raise ValueError("undefined rows must carry None quantiles")
            if curve["reason"] != "metric_not_recorded":
                raise ValueError("undefined rows must record metric_not_recorded")
        if curve["metric"] in ("train_balanced_accuracy", "validation_balanced_accuracy"):
            for value in quantiles:
                if value is not None and not 0.0 <= value <= 1.0:
                    raise ValueError("balanced accuracy out of range")

    for key, group in gmap.items():
        max_epoch = group["max_recorded_epoch"]
        if max_epoch is None:
            continue
        rows = {}
        for epoch in range(1, max_epoch + 1):
            run_counts = set()
            for metric in METRIC_ORDER:
                curve = lookup.get((key[0], key[1], key[2], epoch, metric))
                if curve is None:
                    raise ValueError("curve grid is incomplete")
                rows[(epoch, metric)] = curve
                run_counts.add(curve["n_runs_at_epoch"])
            if len(run_counts) != 1:
                raise ValueError("n_runs_at_epoch must match across metrics")
        for metric in METRIC_ORDER:
            series = [rows[(e, metric)] for e in range(1, max_epoch + 1)]
            counts = [row["n_runs_at_epoch"] for row in series]
            if counts[0] != group["monitored_jobs"]:
                raise ValueError("initial n_runs_at_epoch must equal monitored_jobs")
            if any(counts[i] < counts[i + 1] for i in range(len(counts) - 1)):
                raise ValueError("n_runs_at_epoch must be nonincreasing")
            pattern = [row["median"] is not None for row in series]
            if any(pattern) and not all(pattern):
                raise ValueError("metric must be all-None or fully finite")
            if (
                group["stage"] != "source_fit"
                and metric in NLL_METRICS + BA_METRICS
                and any(pattern)
            ):
                raise ValueError("non-source NLL/BA must be structurally None")
            if (metric == "supcon_loss" and group["recipe"] not in ("D1", "D3")) or (
                metric == "paired_loss" and group["recipe"] not in ("D2", "D3")
            ):
                if any(pattern):
                    raise ValueError("disabled auxiliary metric must be structurally None")
    return gmap


def _authenticate(prepared):
    if not isinstance(prepared, dict):
        raise ValueError("prepared input must be a dict")
    if set(prepared) != {"semantic", "semantic_sha256", "manifest"}:
        raise ValueError("prepared input has an unexpected field set")
    semantic = prepared["semantic"]
    source_sha = prepared["semantic_sha256"]
    if not isinstance(source_sha, str) or source_sha != _canonical_sha(semantic):
        raise ValueError("semantic canonical hash mismatch")
    gmap = _validate_semantic(semantic)
    return semantic, source_sha, gmap


def _bounds(curves, metrics):
    lows = [0.0]
    highs = [1.0]
    for curve in curves:
        if curve["metric"] in metrics and curve["q10"] is not None:
            lows.append(float(curve["q10"]))
            highs.append(float(curve["q90"]))
    low = min(lows)
    high = max(highs)
    if high <= low:
        high = low + 1.0
    pad = 0.05 * (high - low)
    return [low - pad, high + pad]


def _build_axis(semantic, gmap):
    by_rs = {}
    for curve in semantic["curves"]:
        by_rs.setdefault((curve["recipe"], curve["stage"]), []).append(curve)
    axis = {}
    for recipe in RECIPE_ORDER:
        for stage in STAGE_ORDER:
            curves = by_rs.get((recipe, stage), [])
            maxes = []
            monitored = []
            for policy in POLICY_ORDER:
                group = gmap[(policy, recipe, stage)]
                monitored.append(group["monitored_jobs"])
                if group["max_recorded_epoch"] is not None:
                    maxes.append(group["max_recorded_epoch"])
            xmax = float(max([30] + maxes))
            ticks = [1.0]
            tick_step = int(math.ceil(xmax / 50.0)) * 10
            for tick in range(tick_step, int(xmax) + 1, tick_step):
                if tick <= xmax:
                    ticks.append(float(tick))
            dedup = []
            seen = set()
            for tick in ticks:
                if tick not in seen:
                    seen.add(tick)
                    dedup.append(tick)
            dmax = float(max([1] + monitored))
            a_lo, a_hi = _bounds(curves, OBJECTIVE_METRICS)
            b_lo, b_hi = _bounds(curves, NLL_METRICS)
            axis[(recipe, stage)] = {
                "xlim": [1.0, xmax],
                "xticks": dedup,
                "A": {"ylim": [a_lo, a_hi], "yticks": _five_ticks(a_lo, a_hi)},
                "B": {"ylim": [b_lo, b_hi], "yticks": _five_ticks(b_lo, b_hi)},
                "C": {"ylim": [0.0, 1.0], "yticks": [0.0, 0.25, 0.5, 0.75, 1.0]},
                "D": {"ylim": [0.0, 1.1 * dmax], "yticks": _count_ticks(dmax)},
            }
    return axis


def _no_data_message(group):
    if group["planned_jobs"] == 0:
        return (
            "No registered fitting jobs for %s/%s/%s (planned 0, monitored 0, missing 0, status %s)."
            % (
                group["policy_id"],
                group["recipe"],
                group["stage"],
                group["status"],
            )
        )
    return (
        "No complete monitor history for %s/%s/%s (planned %d, monitored %d, missing %d, status %s)."
        % (
            group["policy_id"],
            group["recipe"],
            group["stage"],
            group["planned_jobs"],
            group["monitored_jobs"],
            group["missing_history_jobs"],
            group["status"],
        )
    )


def _metrics_map(curve_rows):
    mm = {}
    for curve in curve_rows:
        mm.setdefault(curve["metric"], []).append(curve)
    runs = []
    for curve in curve_rows:
        if curve["metric"] == "chemical_ce":
            runs.append(
                {
                    "epoch": curve["epoch"],
                    "metric": "n_runs_at_epoch",
                    "median": float(curve["n_runs_at_epoch"]),
                    "q10": None,
                    "q90": None,
                    "finite_count": curve["finite_count"],
                    "n_runs_at_epoch": curve["n_runs_at_epoch"],
                }
            )
    mm["n_runs_at_epoch"] = sorted(runs, key=lambda c: c["epoch"])
    for key in mm:
        mm[key].sort(key=lambda c: c["epoch"])
    return mm


def _csv(group, curve_rows):
    lines = [",".join(CSV_FIELDS)]
    base = [
        group["policy_id"],
        group["recipe"],
        group["stage"],
        group["planned_jobs"],
        group["monitored_jobs"],
        group["missing_history_jobs"],
        group["status"],
        group["coverage"],
        group["max_recorded_epoch"],
    ]
    if not curve_rows:
        lines.append(",".join(_csv_cell(v) for v in base + [None] * 9))
    else:
        for curve in curve_rows:
            row = base + [
                curve["epoch"],
                curve["metric"],
                curve["n_runs_at_epoch"],
                curve["finite_count"],
                curve["undefined_count"],
                curve["median"],
                curve["q10"],
                curve["q90"],
                curve["reason"],
            ]
            lines.append(",".join(_csv_cell(v) for v in row))
    return "\n".join(lines)


def _tex_axis(axis_id, metric_list, message, axis, metrics):
    xlim = axis["xlim"]
    sub = axis[axis_id]
    ylo, yhi = sub["ylim"]
    opts = ["width=75mm", "height=58mm", "title={%s}" % _tex_escape(AXIS_TITLES[axis_id])]
    opts.append("xlabel={epoch}")
    opts.append("ylabel={%s}" % _tex_escape(AXIS_LABELS[axis_id]))
    opts.append("xmin=%s, xmax=%s" % (_g17(xlim[0]), _g17(xlim[1])))
    opts.append("xtick={%s}" % ",".join(_g17(t) for t in axis["xticks"]))
    opts.append("ymin=%s, ymax=%s" % (_g17(ylo), _g17(yhi)))
    yticks = sub.get("yticks")
    if yticks is not None:
        opts.append("ytick={%s}" % ",".join(_g17(t) for t in yticks))
        opts.append("yticklabels={%s}" % ",".join(format(t, ".3g") for t in yticks))
    opts.append("tick label style={font=\\fontsize{8}{9}\\selectfont, text=black}")
    opts.append("label style={font=\\fontsize{8}{9}\\selectfont, text=black}")
    opts.append("title style={font=\\fontsize{10}{11}\\selectfont, text=black}")
    if not message and axis_id in ("A", "B", "C"):
        opts.append(
            "legend style={at={(0.5,-0.24)},anchor=north,draw=none,fill=none,"
            "font=\\fontsize{8}{9}\\selectfont,row sep=1pt,column sep=4pt,legend columns=2}"
        )
    out = ["\\nextgroupplot[" + ", ".join(opts) + "]"]
    if message:
        out.append(
            "\\node[align=center, text width=55mm, font=\\fontsize{8}{9}\\selectfont] "
            "at (axis cs:%s,%s) {%s};"
            % (_g17((xlim[0] + xlim[1]) / 2.0), _g17((ylo + yhi) / 2.0), _tex_escape(message))
        )
        return "\n".join(out)
    for metric in metric_list:
        rows = metrics.get(metric, [])
        finite = [c for c in rows if c["median"] is not None]
        if not finite:
            continue
        color, marker, dash = METRIC_STYLE[metric]
        tex_color = _TEX_COLOR[color]
        if axis_id != "D":
            band = (
                " ".join("(%s,%s)" % (_g17(c["epoch"]), _g17(c["q10"])) for c in finite)
                + " "
                + " ".join("(%s,%s)" % (_g17(c["epoch"]), _g17(c["q90"])) for c in reversed(finite))
            )
            out.append(
                "\\addplot[fill=%s,opacity=0.15,draw=none,forget plot] coordinates {%s};"
                % (tex_color, band)
            )
        mark = ""
        if marker != "none":
            mark = (
                "mark=%s, mark options={draw=black}, mark repeat=10, mark size=1.2pt, "
                % _TIKZ_MARK[marker]
            )
        coords = " ".join("(%s,%s)" % (_g17(c["epoch"]), _g17(c["median"])) for c in finite)
        out.append(
            "\\addplot[color=%s, %s%s, line width=0.8pt] coordinates {%s};"
            % (tex_color, mark, _TIKZ_DASH[dash], coords)
        )
        if axis_id != "D":
            out.append("\\addlegendentry{%s}" % _tex_escape(LEGEND[metric]))
    return "\n".join(out)


def _tex_for(view):
    group = view["group"]
    stage = group["stage"]
    header = "%s | %s | %s" % (group["policy_id"], group["recipe"], stage)
    coverage = "planned=%d, monitored=%d, missing-history=%d, status=%s" % (
        group["planned_jobs"],
        group["monitored_jobs"],
        group["missing_history_jobs"],
        group["status"],
    )
    lines = [
        "\\documentclass[border=0pt,10pt]{standalone}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{lmodern}",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.17}",
        "\\usepgfplotslibrary{groupplots}",
        "\\usepackage{xcolor}",
        "\\definecolor{oi-blue}{HTML}{0072B2}",
        "\\definecolor{oi-vermillion}{HTML}{D55E00}",
        "\\definecolor{oi-green}{HTML}{009E73}",
        "\\definecolor{oi-purple}{HTML}{CC79A7}",
        "\\definecolor{oi-black}{HTML}{000000}",
        "% semantic_sha256: " + view["sha"],
        "\\begin{document}",
        "\\begin{minipage}{181.86mm}",
        "\\centering",
        "{\\bfseries\\fontsize{10}{12}\\selectfont " + _tex_escape(header) + "}\\\\[2mm]",
        "{\\fontsize{8}{10}\\selectfont " + _tex_escape(coverage) + "}\\\\[3mm]",
    ]
    if view["no_data_message"]:
        lines.append("\\begin{tikzpicture}")
        lines.append(
            "\\begin{groupplot}[group style={group size=1 by 1}, width=75mm, height=58mm, "
            "axis lines=left, xmin=0, xmax=1, ymin=0, ymax=1, xtick=\\empty, ytick=\\empty]"
        )
        lines.append("\\nextgroupplot")
        lines.append(
            "\\node[align=center, text width=60mm, font=\\fontsize{8}{9}\\selectfont] "
            "at (axis cs:0.5,0.5) {%s};" % _tex_escape(view["no_data_message"])
        )
        lines.append("\\end{groupplot}")
        lines.append("\\end{tikzpicture}")
    else:
        lines.append("\\begin{tikzpicture}")
        lines.append(
            "\\begin{groupplot}[group style={group size=2 by 2, horizontal sep=14mm, vertical sep=28mm}]"
        )
        specs = [
            ("A", OBJECTIVE_METRICS, None),
            (
                "B",
                NLL_METRICS,
                None if stage == "source_fit" else "Not recorded: training-only fit",
            ),
            ("C", BA_METRICS, None if stage == "source_fit" else "Not recorded: training-only fit"),
            ("D", ("n_runs_at_epoch",), None),
        ]
        for axis_id, metric_list, message in specs:
            lines.append(_tex_axis(axis_id, metric_list, message, view["axis"], view["metrics"]))
        lines.append("\\end{groupplot}")
        lines.append("\\end{tikzpicture}")
    lines.append("\\par\\vspace{6mm}")
    lines.append("\\begin{minipage}{175mm}")
    lines.append("{\\fontsize{8}{10}\\selectfont " + _tex_escape(view["caption"]) + "}")
    lines.append("\\end{minipage}")
    lines.append("\\end{minipage}")
    lines.append("\\end{document}")
    return "\n".join(lines)


def _five_ticks(low, high):
    if high <= low:
        high = low + 1.0
    return [low + (high - low) * i / 4.0 for i in range(5)]


def _count_ticks(dmax):
    top = int(math.ceil(dmax))
    if top <= 5:
        return [float(i) for i in range(top + 1)]
    step = int(math.ceil(top / 5.0))
    ticks = list(range(0, top + 1, step))
    if ticks[-1] != top:
        ticks.append(top)
    return [float(t) for t in ticks]


def _wrap_text(text, limit):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _svg_marker(marker, px, py, radius, stroke, metric):
    if marker == "square":
        return (
            '<rect data-series="%s" x="%s" y="%s" width="%s" height="%s" fill="%s" stroke="#000" stroke-width="0.6" pointer-events="none"/>'
            % (
                metric,
                format(px - radius, ".2f"),
                format(py - radius, ".2f"),
                format(radius * 2.0, ".2f"),
                format(radius * 2.0, ".2f"),
                stroke,
            )
        )
    if marker == "triangle":
        points = "%s,%s %s,%s %s,%s" % (
            format(px, ".2f"),
            format(py - radius, ".2f"),
            format(px - radius, ".2f"),
            format(py + radius, ".2f"),
            format(px + radius, ".2f"),
            format(py + radius, ".2f"),
        )
        return (
            '<polygon data-series="%s" points="%s" fill="%s" stroke="#000" stroke-width="0.6" pointer-events="none"/>'
            % (metric, points, stroke)
        )
    if marker == "diamond":
        points = "%s,%s %s,%s %s,%s %s,%s" % (
            format(px, ".2f"),
            format(py - radius * 1.3, ".2f"),
            format(px + radius * 1.3, ".2f"),
            format(py, ".2f"),
            format(px, ".2f"),
            format(py + radius * 1.3, ".2f"),
            format(px - radius * 1.3, ".2f"),
            format(py, ".2f"),
        )
        return (
            '<polygon data-series="%s" points="%s" fill="%s" stroke="#000" stroke-width="0.6" pointer-events="none"/>'
            % (metric, points, stroke)
        )
    return (
        '<circle data-series="%s" cx="%s" cy="%s" r="%s" fill="%s" stroke="#000" stroke-width="0.6" pointer-events="none"/>'
        % (
            metric,
            format(px, ".2f"),
            format(py, ".2f"),
            format(radius, ".2f"),
            stroke,
        )
    )


def _svg_subplot(axis_id, metrics, metric_list, axis, message, group):
    width, height = 500, 400
    left, right, top, bottom = 78, 20, 44, 62
    pw = width - left - right
    ph = height - top - bottom
    x0, x1 = axis["xlim"]
    y0, y1 = axis[axis_id]["ylim"]
    if y1 <= y0:
        y1 = y0 + 1.0

    def sx(value):
        return left + (float(value) - x0) / (x1 - x0) * pw

    def sy(value):
        return top + (1.0 - (float(value) - y0) / (y1 - y0)) * ph

    def f(value):
        return format(float(value), ".6f")

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="500" height="400" role="img">'
    ]
    out.append(
        '<rect x="%s" y="%s" width="%s" height="%s" fill="#fff" stroke="#000"/>'
        % (f(left), f(top), f(pw), f(ph))
    )
    for tick in axis["xticks"]:
        px = sx(tick)
        out.append(
            '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="#999"/>'
            % (f(px), f(top + ph), f(px), f(top + ph + 4))
        )
        out.append(
            '<text x="%s" y="%s" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">%s</text>'
            % (f(px), f(top + ph + 22), _escape_html(_g17(tick)))
        )
    y_ticks = axis[axis_id].get("yticks") or _five_ticks(y0, y1)
    for tick in y_ticks:
        py = sy(tick)
        out.append(
            '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="#999"/>'
            % (f(left - 4), f(py), f(left), f(py))
        )
        out.append(
            '<text x="%s" y="%s" text-anchor="end" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">%s</text>'
            % (f(left - 6), f(py + 5), _escape_html(format(float(tick), ".3g")))
        )
    out.append(
        '<text x="%s" y="%s" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">epoch</text>'
        % (f(left + pw / 2.0), f(height - 6))
    )
    out.append(
        '<text transform="rotate(-90 %s %s)" x="%s" y="%s" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">%s</text>'
        % (f(20), f(top + ph / 2.0), f(20), f(top + ph / 2.0), _escape_html(AXIS_LABELS[axis_id]))
    )
    out.append(
        '<text x="%s" y="%s" text-anchor="start" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">%s</text>'
        % (f(left), f(20), _escape_html(AXIS_TITLES[axis_id]))
    )
    if message:
        lines = _wrap_text(message, 40)
        for index, line in enumerate(lines):
            out.append(
                '<text x="%s" y="%s" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">%s</text>'
                % (
                    f(left + pw / 2.0),
                    f(top + ph / 2.0 + (index - (len(lines) - 1) / 2.0) * 22),
                    _escape_html(line),
                )
            )
        out.append("</svg>")
        return "".join(out)
    for metric in metric_list:
        rows = metrics.get(metric, [])
        finite = [c for c in rows if c["median"] is not None]
        if not finite:
            continue
        color, marker, line = METRIC_STYLE[metric]
        stroke = _SVG_COLOR[color]
        if axis_id != "D":
            points = [(sx(c["epoch"]), sy(c["q10"])) for c in finite]
            points += [(sx(c["epoch"]), sy(c["q90"])) for c in reversed(finite)]
            poly = " ".join("%s,%s" % (f(px), f(py)) for px, py in points)
            out.append(
                '<polygon data-series="%s" points="%s" fill="%s" opacity="0.15" stroke="none" pointer-events="none"/>'
                % (metric, poly, stroke)
            )
        poly = " ".join("%s,%s" % (f(sx(c["epoch"])), f(sy(c["median"]))) for c in finite)
        out.append(
            '<polyline data-series="%s" points="%s" fill="none" stroke="%s" stroke-width="1.0" stroke-dasharray="%s" pointer-events="none"/>'
            % (metric, poly, stroke, _SVG_DASH[line])
        )
        for index, curve in enumerate(finite):
            px = sx(curve["epoch"])
            py = sy(curve["median"])
            payload = {
                "metric": curve["metric"],
                "epoch": curve["epoch"],
                "median": curve["median"],
                "q10": curve["q10"],
                "q90": curve["q90"],
                "finite_count": curve["finite_count"],
                "n_runs_at_epoch": curve["n_runs_at_epoch"],
                "planned_jobs": group["planned_jobs"],
                "monitored_jobs": group["monitored_jobs"],
                "missing_history_jobs": group["missing_history_jobs"],
            }
            data = _escape_html(_canonical_json(payload))
            out.append(
                '<circle class="p08-hover" data-hover="%s" data-series="%s" cx="%s" cy="%s" r="6" fill="transparent" pointer-events="all" tabindex="0" focusable="true"/>'
                % (data, metric, f(px), f(py))
            )
            if marker != "none" and index % 10 == 0:
                out.append(_svg_marker(marker, px, py, 3.2, stroke, metric))
    out.append("</svg>")
    return "".join(out)


def _svg_message(group, message):
    lines = _wrap_text(message, 40)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="500" height="400" role="img">',
        '<rect x="12" y="12" width="476" height="376" fill="#fff" stroke="#000"/>',
    ]
    start_y = 180 - (len(lines) - 1) * 12
    for index, line in enumerate(lines):
        parts.append(
            '<text x="250" y="%d" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">%s</text>'
            % (start_y + index * 24, _escape_html(line))
        )
    summary = "planned=%d monitored=%d missing=%d" % (
        group["planned_jobs"],
        group["monitored_jobs"],
        group["missing_history_jobs"],
    )
    parts.append(
        '<text x="250" y="%d" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16" fill="#000">%s</text></svg>'
        % (start_y + len(lines) * 24 + 12, _escape_html(summary))
    )
    return "".join(parts)


def _legend_swatch(metric):
    color, marker, line = METRIC_STYLE[metric]
    stroke = _SVG_COLOR[color]
    return (
        '<svg width="45" height="18" viewBox="0 0 45 18" aria-hidden="true">'
        '<line x1="2" x2="43" y1="9" y2="9" stroke="%s" '
        'stroke-width="1.2" stroke-dasharray="%s"/>'
        % (stroke, _SVG_DASH[line])
        + (_svg_marker(marker, 22, 9, 3.2, stroke, "legend-" + metric) if marker != "none" else "")
        + "</svg>"
    )


def _table(group, curve_rows):
    head = "<tr><th>metric</th><th>epoch</th><th>median</th><th>q10</th><th>q90</th><th>finite</th><th>n_runs</th></tr>"
    body = []
    if not curve_rows:
        body.append(
            "<tr><td>%s</td><td>NA</td><td></td><td></td><td></td><td></td><td>%d</td></tr>"
            % (_escape_html(group["status"]), group["monitored_jobs"])
        )
    else:
        for curve in curve_rows:
            body.append(
                "<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td></tr>"
                % (
                    _escape_html(curve["metric"]),
                    curve["epoch"],
                    _g17(curve["median"]),
                    _g17(curve["q10"]),
                    _g17(curve["q90"]),
                    curve["finite_count"],
                    curve["n_runs_at_epoch"],
                )
            )
    return '<div class="p08-table-wrap"><table>' + head + "".join(body) + "</table></div>"


_CSS = (
    'body{font-family:"Times New Roman",Times,serif;color:#000;background:#fff;font-size:16px;margin:12px;}'
    "h1{font-size:18px;}h3{font-size:16px;margin:6px 0;}"
    ".p08-grid{display:grid;grid-template-columns:repeat(2,500px);gap:10px;overflow:auto;}"
    ".p08-checks label{margin-right:8px;font-size:16px;display:inline-flex;align-items:center;}"
    ".p08-key{font-size:16px;margin:6px 0;}"
    ".p08-table-wrap{overflow:auto;max-width:100%;max-height:360px;margin-top:10px;}"
    "table{border-collapse:collapse;font-size:14px;}th,td{border:1px solid #bbb;padding:2px 5px;text-align:right;}"
    ".p08-tip{position:fixed;display:none;background:#fff;color:#000;border:1px solid #000;padding:4px 6px;"
    "font-family:'Times New Roman',Times,serif;font-size:16px;white-space:pre-line;pointer-events:none;z-index:10;}"
    ".p08-hover{cursor:pointer;}.p08-hover:focus{outline:3px solid #000;outline-offset:2px;}"
    ".p08-note{font-size:16px;}.p08-caption{font-size:16px;margin-top:10px;}"
    ".p08-download{font-family:'Times New Roman',Times,serif;font-size:16px;margin:6px 0;}"
)

_JS = (
    "(function(){var root=document.querySelector('[id^=\"p08-panel-\"]');if(!root){return;}"
    "var dataEl=root.querySelector('script[type=\"application/json\"]');"
    "var data=dataEl?JSON.parse(dataEl.textContent):null;"
    'var boxes=root.querySelectorAll("input.p08-metric");'
    'function apply(){hideTip();Array.prototype.forEach.call(boxes,function(b){var m=b.getAttribute("data-metric");'
    'Array.prototype.forEach.call(root.querySelectorAll(\'[data-series="\'+m+\'"]\'),function(el){el.style.display=b.checked?"":"none";});});}'
    'Array.prototype.forEach.call(boxes,function(b){b.addEventListener("change",apply);});apply();'
    'var tip=root.querySelector(".p08-tip");var active=null;'
    'function place(el,ev){if(!tip){return;}tip.style.visibility="hidden";tip.style.display="block";'
    "var pad=8;var vw=window.innerWidth||document.documentElement.clientWidth;var vh=window.innerHeight||document.documentElement.clientHeight;"
    'var left=0;var top=0;if(ev&&typeof ev.clientX==="number"){left=ev.clientX;top=ev.clientY;}else if(el&&el.getBoundingClientRect){var r=el.getBoundingClientRect();left=r.left+r.width;top=r.bottom;}'
    "var tw=tip.offsetWidth;var th=tip.offsetHeight;var x=left+12;var y=top+12;"
    "if(x+tw+pad>vw){x=vw-tw-pad;}if(y+th+pad>vh){y=top-th-12;}if(x<pad){x=pad;}if(y<pad){y=pad;}"
    'tip.style.left=x+"px";tip.style.top=y+"px";tip.style.visibility="visible";}'
    'function showTip(el,ev){if(!tip||!el){return;}var d=JSON.parse(el.getAttribute("data-hover"));'
    'tip.textContent="metric="+d.metric+"\\nepoch="+d.epoch+"\\nmedian="+d.median+"\\nq10="+d.q10+"\\nq90="+d.q90'
    '+"\\nfinite="+d.finite_count+"\\nn_runs="+d.n_runs_at_epoch+"\\nplanned="+d.planned_jobs'
    '+"\\nmonitored="+d.monitored_jobs+"\\nmissing="+d.missing_history_jobs;'
    "active=el;place(el,ev);}"
    'function hideTip(){if(tip){tip.style.display="none";}active=null;}'
    'Array.prototype.forEach.call(root.querySelectorAll("[data-hover]"),function(el){'
    'el.addEventListener("mouseenter",function(ev){showTip(el,ev);});'
    'el.addEventListener("mousemove",function(ev){if(active===el){showTip(el,ev);}});'
    'el.addEventListener("focus",function(ev){showTip(el,ev);});'
    'el.addEventListener("mouseleave",hideTip);'
    'el.addEventListener("blur",hideTip);'
    'el.addEventListener("click",function(ev){if(active===el){hideTip();}else{showTip(el,ev);}});'
    'el.addEventListener("keydown",function(ev){if(ev.key==="Escape"){hideTip();}});});'
    'var dl=root.querySelector(".p08-download");'
    'if(dl&&data&&data.csv){dl.addEventListener("click",function(){var blob=new Blob([data.csv],{type:"text/csv;charset=utf-8"});'
    'var url=URL.createObjectURL(blob);var a=document.createElement("a");a.href=url;a.download=(data.slug||"panel")+".csv";'
    "document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);});}"
    "})();"
)


def _html_for(view):
    group = view["group"]
    stage = group["stage"]
    if view["no_data_message"]:
        sub_html = _svg_message(group, view["no_data_message"])
    else:
        chunks = []
        for axis_id, metric_list in (
            ("A", OBJECTIVE_METRICS),
            ("B", NLL_METRICS),
            ("C", BA_METRICS),
            ("D", ("n_runs_at_epoch",)),
        ):
            message = None
            if stage != "source_fit" and axis_id in ("B", "C"):
                message = "Not recorded: training-only fit"
            chunks.append(
                _svg_subplot(axis_id, view["metrics"], metric_list, view["axis"], message, group)
            )
        sub_html = "".join(chunks)
    checks = []
    for metric in list(METRIC_ORDER) + ["n_runs_at_epoch"]:
        rows = view["metrics"].get(metric, [])
        if any(c["median"] is not None for c in rows):
            checks.append(
                '<label><input class="p08-metric" type="checkbox" data-metric="%s" checked> %s %s</label>'
                % (metric, _legend_swatch(metric), _escape_html(LEGEND[metric]))
            )
    checks_html = '<div class="p08-checks">' + "".join(checks) + "</div>"
    header = "%s | %s | %s" % (group["policy_id"], group["recipe"], stage)
    key_html = (
        '<div class="p08-key"><strong>Key:</strong> '
        "CE=chemical cross entropy; Total=total loss; "
        "SupCon=supervised contrastive; Paired=paired consistency; "
        "NLL=negative loglikelihood; BA=balanced accuracy.</div>"
    )
    download_html = (
        '<button type="button" class="p08-download" aria-label="Download CSV for %s">Download CSV</button>'
        % _escape_html(header)
    )
    coverage = "planned=%d monitored=%d missing-history=%d status=%s coverage=%s max_epoch=%s" % (
        group["planned_jobs"],
        group["monitored_jobs"],
        group["missing_history_jobs"],
        group["status"],
        group["coverage"],
        group["max_recorded_epoch"],
    )
    payload = {
        "slug": view["slug"],
        "semantic_sha256": view["sha"],
        "source_sha256": view["source_sha"],
        "policy_id": group["policy_id"],
        "recipe": group["recipe"],
        "stage": stage,
        "group": group,
        "curves": view["curve_rows"],
        "axis": view["axis"],
        "styles": {
            m: {"color": c, "marker": mk, "line": ln} for m, (c, mk, ln) in METRIC_STYLE.items()
        },
        "legend": LEGEND,
        "caption": view["caption"],
        "scope": SCOPE,
        "rq": RQ,
        "csv": view["csv"],
    }
    embedded = '<script type="application/json">' + _json_script(payload) + "</script>"
    doc = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>%s</title>" % _escape_html(header),
        "<style>%s</style>" % _CSS,
        "</head><body>",
        '<div id="p08-panel-%s">' % view["slug"],
        "<h1>%s</h1>" % _escape_html(header),
        "<p class='p08-note'>figure=%s; RQ=%s; scope=%s; semantic_sha256=%s; source_sha256=%s</p>"
        % (
            _escape_html(FIGURE_ID),
            _escape_html(RQ),
            _escape_html(SCOPE),
            _escape_html(view["sha"]),
            _escape_html(view["source_sha"]),
        ),
        "<p class='p08-note'>%s</p>" % _escape_html(coverage),
        checks_html,
        key_html,
        download_html,
        '<div class="p08-grid">%s</div>' % sub_html,
        _table(group, view["curve_rows"]),
        "<p class='p08-caption'>%s</p>" % _escape_html(view["caption"]),
        '<div class="p08-tip" role="tooltip" aria-live="polite"></div>',
        embedded,
        "<script>%s</script>" % _JS,
        "</div></body></html>",
    ]
    return "".join(doc)


def _render_panel(semantic, group, axis, new_sha, caption, source_sha):
    policy = group["policy_id"]
    recipe = group["recipe"]
    stage = group["stage"]
    slug = _slug(policy, recipe, stage)
    curve_rows = [
        c
        for c in semantic["curves"]
        if c["policy_id"] == policy and c["recipe"] == recipe and c["stage"] == stage
    ]
    order = {m: i for i, m in enumerate(METRIC_ORDER)}
    curve_rows.sort(key=lambda c: (order.get(c["metric"], 99), c["epoch"]))
    csv_text = _csv(group, curve_rows)
    if group["max_recorded_epoch"] is None:
        message = _no_data_message(group)
        metrics_map = {}
    else:
        message = None
        metrics_map = _metrics_map(curve_rows)
    view = {
        "group": group,
        "metrics": metrics_map,
        "axis": axis,
        "slug": slug,
        "sha": new_sha,
        "source_sha": source_sha,
        "caption": caption,
        "csv": csv_text,
        "curve_rows": curve_rows,
        "no_data_message": message,
    }
    return {
        "slug": slug,
        "policy_id": policy,
        "recipe": recipe,
        "stage": stage,
        "semantic_sha256": new_sha,
        "tex": _tex_for(view),
        "html": _html_for(view),
        "csv": csv_text,
    }


def prepare_training_render(prepared):
    """Render the 24 approved training-diagnostic panels (pure strings)."""
    semantic, source_sha, gmap = _authenticate(prepared)
    caption = (
        "RQ-S01 / Scope E: descriptive training diagnostics. "
        "400-1800 cm-1 inputs: SG uses impulse replacement and Savitzky-Golay smoothing; "
        "arPLS uses impulse replacement and baseline correction; both finish with [0,1] "
        "min-max scaling. CE = chemical cross-entropy; SupCon = supervised contrastive; "
        "Paired = paired consistency; NLL = negative log-likelihood; BA = balanced accuracy. "
        + semantic["caption"]
    )
    axis = _build_axis(semantic, gmap)
    definitions = {
        "scope": SCOPE,
        "rq": RQ,
        "caption": caption,
        "axis_titles": dict(AXIS_TITLES),
        "axis_labels": dict(AXIS_LABELS),
        "axis": {("%s|%s" % (r, s)): axis[(r, s)] for r in RECIPE_ORDER for s in STAGE_ORDER},
        "styles": {
            m: {"color": c, "marker": mk, "line": ln, "legend": LEGEND[m]}
            for m, (c, mk, ln) in METRIC_STYLE.items()
        },
        "okabe_ito_hex": dict(OKABE_ITO_HEX),
        "csv_fields": list(CSV_FIELDS),
        "mark_repeat_epochs": 10,
        "band_opacity": 0.15,
        "median_line_width": 0.8,
    }
    new_semantic = {
        "schema_version": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "source_schema_version": semantic["schema_version"],
        "source_sha256": source_sha,
        "caption": caption,
        "policy_order": list(semantic["policy_order"]),
        "recipe_order": list(semantic["recipe_order"]),
        "stage_order": list(semantic["stage_order"]),
        "metric_order": list(semantic["metric_order"]),
        "population": dict(semantic["population"]),
        "groups": [dict(g) for g in semantic["groups"]],
        "curves": [dict(c) for c in semantic["curves"]],
        "definitions": definitions,
    }
    new_sha = _canonical_sha(new_semantic)
    panels = []
    slugs = []
    for policy in POLICY_ORDER:
        for recipe in RECIPE_ORDER:
            for stage in STAGE_ORDER:
                panel = _render_panel(
                    semantic,
                    gmap[(policy, recipe, stage)],
                    axis[(recipe, stage)],
                    new_sha,
                    caption,
                    source_sha,
                )
                panels.append(panel)
                slugs.append(panel["slug"])
    manifest = {
        "status": "prepared",
        "reviewed": False,
        "published": False,
        "figure_id": FIGURE_ID,
        "semantic_sha256": new_sha,
        "source_sha256": source_sha,
        "slugs": slugs,
        "counts": {
            "panels": len(panels),
            "groups": len(new_semantic["groups"]),
            "curve_rows": len(new_semantic["curves"]),
        },
    }
    return {
        "semantic": new_semantic,
        "semantic_sha256": new_sha,
        "panels": panels,
        "manifest": manifest,
    }
