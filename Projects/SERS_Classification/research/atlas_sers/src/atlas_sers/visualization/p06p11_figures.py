"""Pure renderers for the four T035 P06/P11 figures.

No metric computation and no I/O: the validated semantic tables are produced
upstream by ``p06p11_figure_data.build_semantics`` and merely serialised here
into self-contained TeX and inline-Plotly HTML.
"""

from __future__ import annotations

# Percent formatting keeps native TeX braces literal and readily auditable.
# ruff: noqa: UP031
import hashlib
import html
import math

from atlas_sers.visualization.p06p11_deletion_data import build_deletion_semantics
from atlas_sers.visualization.p06p11_figure_data import build_semantics
from atlas_sers.visualization.p06p11_interval_data import build_interval_semantics

__all__ = ["FIGURE_IDS", "SEMANTIC_COLUMNS", "build_figures"]

FIGURE_IDS = (
    "F_P06_primary_scatter",
    "F_P06_effect_intervals",
    "F_P06_weight_sensitivity",
    "F_P06_deletion_stability",
)

SEMANTIC_COLUMNS = (
    "figure_id",
    "panel",
    "label",
    "series",
    "x",
    "y",
    "lower",
    "upper",
    "domain",
    "instrument",
    "complete_contexts",
    "remaining_domains",
    "color",
    "marker",
)

_TEX_ESCAPES = {
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

_PGF_MARKERS = {
    "circle": "*",
    "square": "square*",
    "triangle-up": "triangle*",
    "diamond": "diamond*",
    "cross": "x",
}
_PLOTLY_MARKERS = {
    "circle": "circle",
    "square": "square",
    "triangle-up": "triangle-up",
    "diamond": "diamond",
    "cross": "x",
}

_AXIS_LABELS = {
    "F_P06_primary_scatter": (
        r"Selected classical balanced accuracy (\%)",
        r"Selected CNN balanced accuracy (\%)",
    ),
    "F_P06_deletion_stability": (
        r"M01 balanced accuracy difference (percentage points)",
        r"M06 balanced accuracy difference (percentage points)",
    ),
}
_PANEL_TITLES = {
    "F_P06_primary_scatter": {
        "M01": "A Individual spectra",
        "M06": "B Combined sample predictions",
    },
    "F_P06_effect_intervals": {
        "M01": "A Individual spectra",
        "M06": "B Combined sample predictions",
    },
    "F_P06_weight_sensitivity": {
        "M01": "A Individual spectra",
        "M06": "B Combined sample predictions",
    },
    "F_P06_deletion_stability": {"joint": ""},
}
_HOVER_FIELDS = {
    "F_P06_primary_scatter": ("label", "domain", "instrument", "complete_contexts"),
    "F_P06_deletion_stability": ("label", "domain", "instrument", "remaining_domains"),
}
_INTERVAL_XLABEL = r"Balanced accuracy difference (percentage points)"
_HTML_FONT = "Times New Roman, Times, serif"


def _tex_escape(value):
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in str(value))


def _num(value):
    return format(float(value), ".15g")


def _is_missing(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return value is None


def _ordered_unique(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _panels(semantic):
    order = {"M01": 0, "M06": 1, "joint": 2}
    return sorted(
        _ordered_unique(semantic["panel"].tolist()), key=lambda p: (order.get(str(p), 9), str(p))
    )


def _series_label(series):
    return str(series).replace("_", " ")


def _color_names(semantic):
    names = {}
    for value in semantic["color"].tolist():
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in names:
            names[text] = "p06color%d" % len(names)
    return names


def _equal_limits(semantic, columns):
    values = []
    for column in columns:
        values.extend(float(v) for v in semantic[column].tolist() if not _is_missing(v))
    low, high = min(values + [0.0]), max(values + [0.0])
    pad = 0.1 * (high - low) if high > low else 1.0
    return low - pad, high + pad


def _semantic_sha(semantic):
    return hashlib.sha256(
        semantic.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _wrap_tex(colors, sha, body):
    lines = [
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\usepackage{tikz}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"% sha256: " + sha,
    ]
    lines.extend(
        r"\definecolor{%s}{HTML}{%s}" % (name, hexv.lstrip("#").upper())
        for hexv, name in colors.items()
    )
    lines += [r"\begin{document}", r"\begin{tikzpicture}"]
    lines += body
    lines += [r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(lines) + "\n"


def _tex_nodes(title, caption):
    return [
        r"\node[anchor=south, font=\bfseries\fontsize{10}{12}\selectfont]"
        r" at (current bounding box.north) {%s};" % _tex_escape(title),
        r"\node[anchor=north, text width=17cm, align=center,"
        r" font=\fontsize{8}{10}\selectfont]"
        r" at (current bounding box.south) {%s};" % _tex_escape(caption),
    ]


def _tex_scatter(fid, semantic, title, caption, sha):
    colors = _color_names(semantic)
    panels = _panels(semantic)
    xlabel, ylabel = _AXIS_LABELS[fid]
    is_primary = fid == "F_P06_primary_scatter"
    if is_primary:
        xlabel = r"Selected classical\\balanced accuracy (\%)"
    size = "10cm" if fid == "F_P06_deletion_stability" else "6.6cm"
    if fid == "F_P06_primary_scatter":
        xmin, xmax, ymin, ymax = 0.0, 100.0, 0.0, 100.0
        reference = ("(0,0)", "(100,100)")
    else:
        xmin, xmax = _equal_limits(semantic, ("x", "y"))
        ymin, ymax = xmin, xmax
        reference = None
    body = []
    for index, panel in enumerate(panels):
        sub = semantic[semantic["panel"] == panel]
        panel_title = _PANEL_TITLES[fid].get(str(panel), str(panel))
        opts = "title={%s}" % _tex_escape(panel_title)
        if index == 0:
            opts += (
                r", legend style={at={(0.98,0.02)},anchor=south east,"
                r"draw=black,fill=white,font=\fontsize{8}{10}\selectfont}"
            )
        elif is_primary:
            opts += ", ylabel={}"
        body.append(r"\nextgroupplot[%s]" % opts)
        if reference is not None:
            body.append(
                r"\addplot[forget plot, black, thin, dashed]"
                r" coordinates {%s %s};" % reference
            )
        else:
            body.append(
                r"\addplot[forget plot, black, thin] coordinates {(%s,%s) (%s,%s)};"
                % (_num(0), _num(ymin), _num(0), _num(ymax))
            )
            body.append(
                r"\addplot[forget plot, black, thin] coordinates {(%s,%s) (%s,%s)};"
                % (_num(xmin), _num(0), _num(xmax), _num(0))
            )
        for series in _ordered_unique(sub["series"].tolist()):
            rows = sub[sub["series"] == series]
            coords = " ".join("(%s,%s)" % (_num(r.x), _num(r.y)) for r in rows.itertuples())
            first = rows.iloc[0]
            body.append(
                r"\addplot[only marks, mark=%s, mark size=1.7pt, color=%s]"
                r" coordinates {%s};"
                % (_PGF_MARKERS.get(str(first["marker"]), "*"), colors[str(first["color"])], coords)
            )
            if index == 0:
                body.append(r"\addlegendentry{%s}" % _tex_escape(_series_label(series)))
    options = [
        r"group style={group size=%d by 1, horizontal sep=1.8cm}" % len(panels),
        r"width=%s" % size,
        r"height=%s" % size,
        r"xmin=%s, xmax=%s, ymin=%s, ymax=%s" % (_num(xmin), _num(xmax), _num(ymin), _num(ymax)),
        r"xlabel={%s}" % xlabel,
        r"xlabel style={font=\fontsize{9}{11}\selectfont, color=black%s}"
        % (", align=center" if is_primary else ""),
        r"ylabel={%s}" % ylabel,
        r"ylabel style={font=\fontsize{9}{11}\selectfont, color=black}",
        r"tick label style={font=\fontsize{8}{10}\selectfont, color=black}",
        r"title style={font=\bfseries\fontsize{10}{12}\selectfont, color=black}",
        r"grid=both",
        r"axis equal",
    ]
    group = (
        [r"\begin{groupplot}[" + ",\n".join(options) + "]"]
        + body
        + [r"\end{groupplot}"]
        + _tex_nodes(title, caption)
    )
    return _wrap_tex(colors, sha, group)


def _tex_interval(fid, semantic, title, caption, sha):
    colors = _color_names(semantic)
    panels = _panels(semantic)
    human = _PANEL_TITLES.get(fid, {})
    xmin, xmax = _equal_limits(semantic, ("lower", "upper", "x"))
    counts = [len(semantic[semantic["panel"] == p]) for p in panels]
    n_rows = max(counts) if counts else 1
    body = []
    for index, panel in enumerate(panels):
        sub = semantic[semantic["panel"] == panel]
        rows = sorted(sub.itertuples(), key=lambda r: float(r.y))
        labels = [str(r.label) for r in rows]
        ys = [float(r.y) for r in rows]
        ymin = min(ys) - 0.6
        ymax = max(ys) + 0.6
        panel_title = human.get(str(panel), str(panel))
        opts = "title={%s}" % _tex_escape(panel_title)
        if index == len(panels) - 1:
            opts += ", xlabel={%s}" % _INTERVAL_XLABEL
        opts += ", ytick={%s}" % ",".join(_num(y) for y in ys)
        opts += ", yticklabels={%s}" % ",".join(_tex_escape(label) for label in labels)
        opts += ", ymin=%s, ymax=%s" % (_num(ymin), _num(ymax))
        body.append(r"\nextgroupplot[%s]" % opts)
        body.append(
            r"\addplot[forget plot, black, thin] coordinates {(%s,%s) (%s,%s)};"
            % (_num(0), _num(ymin), _num(0), _num(ymax))
        )
        for row in rows:
            y = float(row.y)
            color = colors[str(row.color)]
            marker = _PGF_MARKERS.get(str(row.marker), "*")
            if not _is_missing(row.lower) and not _is_missing(row.upper):
                body.append(
                    r"\addplot[forget plot, color=%s, thick]"
                    r" coordinates {(%s,%s) (%s,%s)};"
                    % (color, _num(row.lower), _num(y), _num(row.upper), _num(y))
                )
                for edge in (_num(row.lower), _num(row.upper)):
                    body.append(
                        r"\addplot[forget plot, color=%s, thick] coordinates {"
                        r"(%s,%s) (%s,%s)};" % (color, edge, _num(y - 0.14), edge, _num(y + 0.14))
                    )
            if not _is_missing(row.x):
                body.append(
                    r"\addplot[only marks, mark=%s, mark size=1.8pt, color=%s]"
                    r" coordinates {(%s,%s)};" % (marker, color, _num(row.x), _num(y))
                )
    options = [
        r"group style={group size=1 by %d, vertical sep=1.5cm}" % len(panels),
        r"width=11.5cm",
        r"height=%scm" % _num(0.95 * n_rows + 1.5),
        r"xmin=%s, xmax=%s" % (_num(xmin), _num(xmax)),
        r"tick label style={font=\fontsize{8}{10}\selectfont, color=black}",
        r"yticklabel style={font=\fontsize{8}{10}\selectfont, color=black}",
        r"title style={font=\bfseries\fontsize{10}{12}\selectfont, color=black}",
        r"xlabel style={font=\fontsize{9}{11}\selectfont, color=black}",
        r"xmajorgrids=true",
        r"ymajorgrids=false",
    ]
    group = (
        [r"\begin{groupplot}[" + ",\n".join(options) + "]"]
        + body
        + [r"\end{groupplot}"]
        + _tex_nodes(title, caption)
    )
    return _wrap_tex(colors, sha, group)


def _render_tex(fid, semantic, title, caption, kind, sha):
    if kind == "scatter":
        return _tex_scatter(fid, semantic, title, caption, sha)
    return _tex_interval(fid, semantic, title, caption, sha)


def _hover_text(rows, fields):
    texts = []
    for row in rows.itertuples():
        parts = []
        if not _is_missing(row.x):
            parts.append("x: %s" % _num(row.x))
        if not _is_missing(row.y):
            parts.append("y: %s" % _num(row.y))
        for field in fields:
            value = getattr(row, field)
            if not _is_missing(value):
                parts.append("%s: %s" % (field, value))
        texts.append("<br>".join(html.escape(part) for part in parts))
    return texts


def _base_layout(fig, title):
    fig.update_layout(
        title=dict(
            text=html.escape(str(title)), font=dict(family=_HTML_FONT, size=16, color="#000000")
        ),
        font=dict(family=_HTML_FONT, size=12, color="#000000"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="closest",
        dragmode="pan",
        legend=dict(font=dict(family=_HTML_FONT, size=11, color="#000000")),
    )
    # Preserve declared axis limits when equal-aspect plots resize in the browser.
    fig.update_xaxes(fixedrange=False, constrain="domain")
    fig.update_yaxes(fixedrange=False, constrain="domain")
    return fig


def _html_scatter(fid, semantic, title):
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    panels = _panels(semantic)
    titles = [_PANEL_TITLES[fid].get(str(p), str(p)) for p in panels]
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=titles, horizontal_spacing=0.12)
    xlabel, ylabel = _AXIS_LABELS[fid]
    fields = _HOVER_FIELDS[fid]
    xlabel, ylabel = xlabel.replace(r"\%", "%"), ylabel.replace(r"\%", "%")
    for col, panel in enumerate(panels, start=1):
        sub = semantic[semantic["panel"] == panel]
        for series in _ordered_unique(sub["series"].tolist()):
            rows = sub[sub["series"] == series]
            fig.add_trace(
                go.Scatter(
                    x=rows["x"].tolist(),
                    y=rows["y"].tolist(),
                    mode="markers",
                    name=_series_label(series),
                    legendgroup=str(series),
                    showlegend=(col == 1),
                    marker=dict(
                        color=rows["color"].tolist(),
                        symbol=[
                            _PLOTLY_MARKERS.get(str(m), "circle") for m in rows["marker"].tolist()
                        ],
                        size=10,
                        line=dict(width=1, color="#000000"),
                    ),
                    text=_hover_text(rows, fields),
                    hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            showline=True,
            linecolor="#000000",
            gridcolor="#d9d9d9",
            zeroline=False,
            title_text=xlabel,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            showline=True,
            linecolor="#000000",
            gridcolor="#d9d9d9",
            zeroline=False,
            title_text=ylabel,
            row=1,
            col=col,
        )
    if fid == "F_P06_primary_scatter":
        for col in range(1, len(panels) + 1):
            xref = "x" if col == 1 else "x%d" % col
            yref = "y" if col == 1 else "y%d" % col
            fig.update_xaxes(range=[0, 100], row=1, col=col)
            fig.update_yaxes(range=[0, 100], scaleanchor=xref, scaleratio=1, row=1, col=col)
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=100,
                y1=100,
                xref=xref,
                yref=yref,
                line=dict(color="#000000", width=1, dash="dash"),
            )
    else:
        low, high = _equal_limits(semantic, ("x", "y"))
        fig.update_xaxes(range=[low, high])
        fig.update_yaxes(range=[low, high], scaleanchor="x", scaleratio=1)
        fig.add_shape(
            type="line",
            x0=0,
            y0=low,
            x1=0,
            y1=high,
            line=dict(color="#000000", width=1, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=low,
            y0=0,
            x1=high,
            y1=0,
            line=dict(color="#000000", width=1, dash="dash"),
        )
    fig = _base_layout(fig, title)
    fig.update_layout(width=1100, height=650)
    return fig


def _html_interval(fid, semantic, title):
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    panels = _panels(semantic)
    human = _PANEL_TITLES.get(fid, {})
    low, high = _equal_limits(semantic, ("lower", "upper", "x"))
    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[human.get(str(p), str(p)) for p in panels],
        vertical_spacing=0.14,
    )
    for index, panel in enumerate(panels, start=1):
        sub = semantic[semantic["panel"] == panel]
        rows = sorted(sub.itertuples(), key=lambda r: float(r.y))
        ys = [float(r.y) for r in rows]
        ymin = min(ys) - 0.6
        ymax = max(ys) + 0.6
        xref = "x" if index == 1 else "x%d" % index
        yref = "y" if index == 1 else "y%d" % index
        fig.add_shape(
            type="line",
            x0=0,
            y0=ymin,
            x1=0,
            y1=ymax,
            xref=xref,
            yref=yref,
            line=dict(color="#000000", width=1, dash="dash"),
        )
        for row in rows:
            y = float(row.y)
            color = str(row.color)
            if not _is_missing(row.lower) and not _is_missing(row.upper):
                fig.add_shape(
                    type="line",
                    x0=float(row.lower),
                    y0=y,
                    x1=float(row.upper),
                    y1=y,
                    xref=xref,
                    yref=yref,
                    line=dict(color=color, width=2),
                )
                for edge in (float(row.lower), float(row.upper)):
                    fig.add_shape(
                        type="line",
                        x0=edge,
                        y0=y - 0.14,
                        x1=edge,
                        y1=y + 0.14,
                        xref=xref,
                        yref=yref,
                        line=dict(color=color, width=2),
                    )
            if not _is_missing(row.x):
                point = _num(row.x)
                lower = "-" if _is_missing(row.lower) else _num(row.lower)
                upper = "-" if _is_missing(row.upper) else _num(row.upper)
                hover = "%s<br>x: %s<br>lower: %s<br>upper: %s" % (
                    html.escape(str(row.label)),
                    point,
                    lower,
                    upper,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[float(row.x)],
                        y=[y],
                        mode="markers",
                        showlegend=False,
                        marker=dict(
                            color=color,
                            size=10,
                            symbol=_PLOTLY_MARKERS.get(str(row.marker), "circle"),
                        ),
                        text=[hover],
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=index,
                    col=1,
                )
        fig.update_yaxes(
            tickvals=ys,
            ticktext=[html.escape(str(r.label)) for r in rows],
            range=[ymin, ymax],
            row=index,
            col=1,
        )
    fig.update_xaxes(
        showline=True, linecolor="#000000", gridcolor="#d9d9d9", zeroline=False, range=[low, high]
    )
    fig.update_xaxes(title_text=_INTERVAL_XLABEL, row=len(panels), col=1)
    fig.update_yaxes(showline=True, linecolor="#000000", gridcolor="#d9d9d9", zeroline=False)
    height = 1000 if fid == "F_P06_effect_intervals" else 800
    fig = _base_layout(fig, title)
    fig.update_layout(width=1100, height=height, margin=dict(l=260))
    return fig


def _render_html(fid, semantic, title, caption, kind, sha):
    if kind == "scatter":
        fig = _html_scatter(fid, semantic, title)
    else:
        fig = _html_interval(fid, semantic, title)
    document = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id=fid,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    caption_html = (
        '<div id="%s-caption" style="font-family:%s;color:#000000;'
        "background:#ffffff;max-width:17cm;margin:12px auto;text-align:center;"
        'font-size:12px;">'
        '<div style="font-weight:bold;font-size:14px;">%s</div>'
        "<div>%s</div>"
        '<div style="font-size:10px;">SHA-256: %s</div></div>'
        % (fid, _HTML_FONT, html.escape(str(title)), html.escape(str(caption)), sha)
    )
    if "</body>" in document:
        return document.replace("</body>", caption_html + "\n</body>", 1)
    return document + caption_html


def build_figures(tables):
    semantics = build_semantics(tables)
    if set(semantics) != set(FIGURE_IDS):
        semantics.update(build_interval_semantics(tables))
        semantics.update(build_deletion_semantics(tables))
    figures = {}
    for fid in FIGURE_IDS:
        spec = semantics[fid]
        semantic = spec["semantic"]
        title = spec["title"]
        caption = spec["caption"]
        kind = spec["kind"]
        sha = _semantic_sha(semantic)
        figures[fid] = {
            "semantic": semantic,
            "sha256": sha,
            "tex": _render_tex(fid, semantic, title, caption, kind, sha),
            "html": _render_html(fid, semantic, title, caption, kind, sha),
            "title": title,
            "caption": caption,
        }
    return figures
