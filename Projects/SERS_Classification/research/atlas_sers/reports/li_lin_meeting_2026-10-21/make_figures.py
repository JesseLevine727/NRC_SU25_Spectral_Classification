"""Write native TikZ/PGFPlots figures for the 2026-10-21 deck from data/ and private_data/.

Each figure is a standalone tikzpicture in figures/, \\input by deck.tex. Colours are
defined in the deck preamble (validated light-mode palette; see the dataviz checks).
Heatmaps use one blue sequential ramp; scatter plots use at most three hues plus shapes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PRIVATE = HERE / "private_data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["C-EXTRA-TREES", "C-RANDOM-FOREST", "C-RBF-SVM", "D0-M", "P05-SELECTED"]
MODEL_SHORT = {"C-EXTRA-TREES": "Extra Trees", "C-RANDOM-FOREST": "Random Forest", "C-RBF-SVM": "SVM",
               "D0-M": "CNN", "P05-SELECTED": "CNN (selected)"}
STATIONS = ["cwa", "pills", "surfaces"]
STATION_LABEL = {"cwa": "CWA", "pills": "Pills", "surfaces": "Surfaces"}
CLASSES = {"cwa": ["4_nitrophenol", "ethanol", "ethyl_paraoxon"],
           "pills": ["4_ANPP", "benzyl_fentanyl", "blank"],
           "surfaces": ["4_ANPP", "acetaminophen", "benzyl_fentanyl"]}
CHEM_SHORT = {"4_nitrophenol": "4-NP", "ethanol": "EtOH", "ethyl_paraoxon": "paraoxon", "4_ANPP": "4-ANPP",
              "benzyl_fentanyl": "BnF", "blank": "blank", "acetaminophen": "APAP"}
CHEM_LONG = {"4_nitrophenol": "4-nitrophenol", "ethanol": "ethanol", "ethyl_paraoxon": "ethyl paraoxon",
             "4_ANPP": "4-ANPP", "benzyl_fentanyl": "benzyl fentanyl", "blank": "blank",
             "acetaminophen": "acetaminophen"}

# Blue sequential ramp (reference palette steps), anchored at the light surface for zero.
RAMP = ["FCFCFB", "CDE2FB", "9EC5F4", "6DA7EC", "3987E5", "256ABF", "184F95", "0D366B"]


def heat(t):
    """Return (tikz colour spec, text colour) for t in [0, 1]."""
    t = float(np.clip(t, 0, 1))
    pos = t * (len(RAMP) - 1)
    i = min(int(pos), len(RAMP) - 2)
    f = pos - i
    a = np.array([int(RAMP[i][k:k + 2], 16) for k in (0, 2, 4)])
    b = np.array([int(RAMP[i + 1][k:k + 2], 16) for k in (0, 2, 4)])
    r, g, bl = np.rint(a + f * (b - a)).astype(int)
    return f"{{rgb,255:red,{r};green,{g};blue,{bl}}}", ("white" if t > 0.55 else "cInk")


def write(name, body):
    (FIG / f"{name}.tex").write_text(body.strip() + "\n")


# --------------------------------------------------------------------------- coverage

def fig_coverage():
    d = pd.read_csv(DATA / "coverage_substrate_instrument.csv")
    inst = ["Agilent-1", "Agilent-3", "Mira-1", "Mira-2", "Mira-3", "Pendar-1", "Pendar-2", "Pendar-3", "RMX-1", "RMX-2"]
    subs = [("pSERS silver", 324), ("H-Kit", 153), ("NRC", 98), ("GaN/polymer", 23)]
    vmax = d.n.max()
    out = [r"\begin{tikzpicture}[x=0.6cm,y=0.5cm,font=\tiny]"]
    for _, r in d.iterrows():
        x, y, n = int(r.x), int(r.y), int(r.n)
        if n == 0:
            out.append(rf"\fill[cNA] ({x},{y}) rectangle +(0.94,0.9);")
        else:
            col, txt = heat(0.15 + 0.85 * n / vmax)
            out.append(rf"\fill[fill={col}] ({x},{y}) rectangle +(0.94,0.9);")
            out.append(rf"\node[text={txt}] at ({x + 0.47},{y + 0.45}) {{{n}}};")
    for i, name in enumerate(inst):
        out.append(rf"\node[rotate=45,anchor=south west,inner sep=1pt] at ({i + 0.25},4.0) {{{name}}};")
    for j, (s, tot) in enumerate(subs):
        out.append(rf"\node[anchor=east] at (-0.05,{3 - j + 0.45}) {{{s} \textcolor{{cInk2}}{{({tot})}}}};")
    out.append(r"\fill[cNA] (0,-0.9) rectangle +(0.6,0.6); \node[anchor=west] at (0.65,-0.6) {not measured};")
    out.append(r"\node[anchor=west] at (4.2,-0.6) {numbers = spectra per substrate $\times$ instrument};")
    out.append(r"\end{tikzpicture}")
    write("coverage_substrate_instrument", "\n".join(out))


def fig_chem_substrate():
    d = pd.read_csv(DATA / "coverage_chemical_substrate.csv")
    chems = ["4-nitrophenol", "ethanol", "ethyl paraoxon", "4-ANPP", "benzyl fentanyl", "acetaminophen", "blank"]
    subs = ["pSERS", "H-Kit", "NRC", "GaN"]
    out = [r"\begin{tikzpicture}[x=0.78cm,y=0.31cm,font=\tiny]"]
    for _, r in d.iterrows():
        x, y, n = int(r.x), int(r.y), int(r.n)
        if n == 0:
            out.append(rf"\fill[cNA] ({x},{y}) rectangle +(0.95,0.9);")
        else:
            col, txt = heat(0.15 + 0.85 * n / 15)
            out.append(rf"\fill[fill={col}] ({x},{y}) rectangle +(0.95,0.9);")
            out.append(rf"\node[text={txt}] at ({x + 0.475},{y + 0.45}) {{{n}}};")
    for i, s in enumerate(subs):
        out.append(rf"\node[anchor=south] at ({i + 0.475},7.0) {{{s}}};")
    for j, c in enumerate(chems):
        out.append(rf"\node[anchor=east] at (-0.05,{6 - j + 0.45}) {{{c}}};")
    out.append(r"\fill[cNA] (0,-1.15) rectangle +(0.4,0.75); \node[anchor=west] at (0.45,-0.78) {never measured together};")
    out.append(r"\end{tikzpicture}")
    write("coverage_chemical_substrate", "\n".join(out))


# --------------------------------------------------------------------------- PCA

def fig_pca(station="pills"):
    brands = [("Agilent", "cBlue", "*"), ("Mira", "cOrange", "square*"), ("Pendar", "cAqua", "triangle*")]
    chems = [("4_ANPP", "cViolet", "diamond*"), ("benzyl_fentanyl", "cGreen", "pentagon*"), ("blank", "cMagenta", "*")]
    stats = {"minimal": (95, 43), "baseline": (59, 52)}
    var = {}
    for tag in ("minimal", "baseline"):
        d = pd.read_csv(PRIVATE / f"pca_{tag}_{station}.csv")
        var[tag] = d
    out = [r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=2 by 2,horizontal sep=0.35cm,vertical sep=0.55cm},",
           r"  width=4.3cm,height=3.3cm,ticks=none,scale only axis=false,",
           r"  axis line style={cGrid},label style={font=\tiny,text=cInk2},title style={font=\scriptsize,yshift=-3pt},",
           r"  every axis plot/.append style={only marks,mark size=1.15pt,mark options={draw=white,line width=0.15pt}}]"]
    for tag, rowname in (("minimal", "Minimal preprocessing"), ("baseline", "Baseline-corrected")):
        d = var[tag]
        for panel in ("brand", "chem"):
            title = ("coloured by instrument brand" if panel == "brand" else "coloured by chemical") if tag == "minimal" else ""
            ylab = rowname
            opts = [f"title={{{title}}}"] if title else []
            opts.append(f"ylabel={{{ylab}}}" if panel == "brand" else "")
            out.append(r"\nextgroupplot[" + ",".join(o for o in opts if o) + "]")
            groups = brands if panel == "brand" else chems
            key = "brand" if panel == "brand" else "target_analyte"
            for val, col, mark in groups:
                g = d[d[key] == val]
                coords = " ".join(f"({a:.3f},{b:.3f})" for a, b in zip(g.pc1, g.pc2))
                out.append(rf"\addplot[mark={mark},color={col},mark options={{fill={col}}}] coordinates {{{coords}}};")
    out.append(r"\end{groupplot}")
    # Row annotations: nearest-neighbour agreement, excluding the same physical sample.
    for row, tag in ((1, "minimal"), (2, "baseline")):
        b, c = stats[tag]
        out.append(rf"\node[anchor=north,font=\tiny,text=cInk2] at ($(group c1r{row}.south east)!0.5!(group c2r{row}.south west)$) "
                   rf"{{nearest other sample: same brand {b}\% \,$\cdot$\, same chemical {c}\%}};")
    # Legends
    leg = r"\matrix[anchor=west,font=\tiny,column sep=1pt,row sep=0pt] at ($(group c2r1.east)+(0.15,0)$) {"
    for val, col, mark in brands:
        leg += rf"\draw[fill={col},draw=white] plot[mark={mark},mark size=1.6pt] coordinates {{(0,0)}}; & \node[anchor=west] {{{val}}}; \\"
    leg += "};"
    out.append(leg)
    leg = r"\matrix[anchor=west,font=\tiny,column sep=1pt,row sep=0pt] at ($(group c2r2.east)+(0.15,0)$) {"
    for val, col, mark in chems:
        leg += rf"\draw[fill={col},draw=white] plot[mark={mark},mark size=1.6pt] coordinates {{(0,0)}}; & \node[anchor=west] {{{CHEM_LONG[val]}}}; \\"
    leg += "};"
    out.append(leg)
    out.append(r"\end{tikzpicture}")
    write(f"pca_{station}", "\n".join(out))


# --------------------------------------------------------------------------- results

def fig_model_prep():
    d = pd.read_csv(DATA / "model_by_preprocessing.csv")
    names = [n.replace("CNN (selected recipe)", "CNN (selected)") for n in d.model]
    yt = ",".join(str(v) for v in d.y)
    ytl = ",".join("{" + n + "}" for n in names)
    out = [r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=2 by 1,horizontal sep=0.25cm,yticklabels at=edge left},",
           r"  width=2.55cm,height=3.4cm,scale only axis,ymin=0.4,ymax=5.6,",
           rf"  ytick={{{yt}}},yticklabels={{{ytl}}},",
           r"  tick label style={font=\tiny},title style={font=\scriptsize},xlabel style={font=\tiny},",
           r"  xmajorgrids,grid style={cGrid,very thin},axis line style={cInk2},tick style={cInk2},",
           r"  y axis line style={draw=none},ytick style={draw=none}]"]
    for ep, title, lo, hi in (("M01", "single spectrum", 63, 79), ("M06", "combined per sample", 68, 89)):
        out.append(rf"\nextgroupplot[title={{{title}}},xmin={lo},xmax={hi},xlabel={{balanced accuracy (\%)}}]")
        for _, r in d.iterrows():
            out.append(rf"\draw[cGrid,line width=1.2pt] (axis cs:{r[f'{ep}_minimal']},{r.y}) -- (axis cs:{r[f'{ep}_baseline']},{r.y});")
        for key, col, mark in (("minimal", "cBlue", "*"), ("smoothed", "cAqua", "triangle*"), ("baseline", "cOrange", "diamond*")):
            coords = " ".join(f"({r[f'{ep}_{key}']},{r.y})" for _, r in d.iterrows())
            out.append(rf"\addplot[only marks,mark={mark},mark size=2.1pt,color={col},mark options={{fill={col},draw=white,line width=0.3pt}}] coordinates {{{coords}}};")
        for _, r in d.iterrows():
            out.append(rf"\node[font=\tiny,anchor=west,text=cInk] at (axis cs:{r[f'{ep}_baseline']},{r.y}) {{\,{r[f'{ep}_baseline']:.1f}}};")
    out.append(r"\end{groupplot}")
    out.append(r"\matrix[anchor=north,font=\tiny,column sep=2pt] at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-0.55)$) {"
               r"\draw[fill=cBlue,draw=white] plot[mark=*,mark size=2pt] coordinates {(0,0)}; & \node[anchor=west]{minimal}; &"
               r"\draw[fill=cAqua,draw=white] plot[mark=triangle*,mark size=2pt] coordinates {(0,0)}; & \node[anchor=west]{smoothed (SG)}; &"
               r"\draw[fill=cOrange,draw=white] plot[mark=diamond*,mark size=2pt] coordinates {(0,0)}; & \node[anchor=west]{baseline (arPLS)}; \\};")
    out.append(r"\end{tikzpicture}")
    write("model_by_preprocessing", "\n".join(out))


def confusion_block(long, policy, station, model, x0, y0, cell, title=None, labels=True, col_labels=True):
    out = []
    g = long[(long.policy == policy) & (long.station == station) & (long.model_id == model)]
    for _, r in g.iterrows():
        x = x0 + r.x * cell
        y = y0 + r.y * cell
        col, txt = heat(r.pct / 100)
        out.append(rf"\fill[fill={col}] ({x:.3f},{y:.3f}) rectangle +({cell * 0.95:.3f},{cell * 0.95:.3f});")
        out.append(rf"\node[text={txt}] at ({x + cell * 0.475:.3f},{y + cell * 0.475:.3f}) {{{r.pct:.0f}}};")
    cls = CLASSES[station]
    if labels:
        for i, c in enumerate(cls):
            out.append(rf"\node[anchor=east,inner sep=1pt] at ({x0 - 0.02:.3f},{y0 + (2 - i) * cell + cell * 0.475:.3f}) {{{CHEM_SHORT[c]}}};")
            if col_labels:
                out.append(rf"\node[anchor=north east,rotate=45,inner sep=1pt] at ({x0 + i * cell + cell * 0.6:.3f},{y0 - 0.02:.3f}) {{{CHEM_SHORT[c]}}};")
    if title:
        out.append(rf"\node[anchor=south,font=\scriptsize] at ({x0 + 1.5 * cell:.3f},{y0 + 3 * cell:.3f}) {{{title}}};")
    return out


def fig_confusion_main():
    long = pd.read_csv(DATA / "confusion_long.csv")
    cell = 0.38
    out = [r"\begin{tikzpicture}[font=\tiny]"]
    for row, model in enumerate(["C-EXTRA-TREES", "P05-SELECTED"]):
        for col, st in enumerate(STATIONS):
            x0 = 0.95 + col * 2.2
            y0 = (1 - row) * 2.15
            out += confusion_block(long, "baseline", st, model, x0, y0, cell,
                                   title=STATION_LABEL[st] if row == 0 else None)
        label = "Extra Trees" if model == "C-EXTRA-TREES" else "CNN (selected)"
        out.append(rf"\node[rotate=90,anchor=south,font=\scriptsize] at (-0.15,{(1 - row) * 2.15 + 1.5 * cell:.3f}) {{{label}}};")
    out.append(r"\node[anchor=north west,text=cInk2,align=left] at (-0.4,-0.7) {rows = true chemical, columns = predicted;\\\% of each true chemical; baseline-corrected};")
    out.append(r"\end{tikzpicture}")
    write("confusion_main", "\n".join(out))


def fig_confusion_all(policy):
    long = pd.read_csv(DATA / "confusion_long.csv")
    cell = 0.3
    step = 1.12
    out = [r"\begin{tikzpicture}[font=\tiny]"]
    for row, model in enumerate(MODELS):
        for col, st in enumerate(STATIONS):
            x0 = 1.6 + col * 2.6
            y0 = (4 - row) * step
            out += confusion_block(long, policy, st, model, x0, y0, cell,
                                   title=STATION_LABEL[st] if row == 0 else None, labels=True,
                                   col_labels=(row == len(MODELS) - 1))
        out.append(rf"\node[anchor=east,font=\scriptsize] at (0.75,{(4 - row) * step + 1.5 * cell:.3f}) {{{MODEL_SHORT[model]}}};")
    out.append(r"\end{tikzpicture}")
    write(f"confusion_all_{policy}", "\n".join(out))


def fig_per_class():
    long = pd.read_csv(DATA / "confusion_long.csv")
    diag = long[(long.true == long.pred) & (long.policy == "baseline")]
    cw, ch = 0.5, 0.36
    short = {"C-EXTRA-TREES": "Extra Trees", "C-RANDOM-FOREST": "Rand. Forest", "C-RBF-SVM": "SVM",
             "D0-M": "CNN", "P05-SELECTED": "CNN (sel.)"}
    out = [r"\begin{tikzpicture}[font=\tiny]"]
    row = 0
    rows = [(st, c) for st in STATIONS for c in CLASSES[st]]
    n = len(rows)
    for st, c in rows:
        y = (n - 1 - row) * ch - (STATIONS.index(st)) * 0.12
        for j, m in enumerate(MODELS):
            v = diag[(diag.station == st) & (diag.true == c) & (diag.model_id == m)].pct.iloc[0]
            col, txt = heat(v / 100)
            out.append(rf"\fill[fill={col}] ({j * cw:.3f},{y:.3f}) rectangle +({cw * 0.95:.3f},{ch * 0.9:.3f});")
            out.append(rf"\node[text={txt}] at ({j * cw + cw * 0.475:.3f},{y + ch * 0.45:.3f}) {{{v:.0f}}};")
        lab = CHEM_LONG[c]
        out.append(rf"\node[anchor=east,inner sep=1pt] at (-0.03,{y + ch * 0.45:.3f}) {{{lab}}};")
        if CLASSES[st].index(c) == 1:
            out.append(rf"\node[rotate=90,anchor=south,font=\tiny\bfseries,inner sep=1pt] at (-1.62,{y + ch * 0.45:.3f}) {{{STATION_LABEL[st]}}};")
        row += 1
    for j, m in enumerate(MODELS):
        out.append(rf"\node[anchor=south west,rotate=60,inner sep=1pt] at ({j * cw + 0.12:.3f},{n * ch - 0.02:.3f}) {{{short[m]}}};")
    out.append(r"\end{tikzpicture}")
    write("per_class", "\n".join(out))


def fig_scatter():
    d = pd.read_csv(DATA / "per_instrument.csv")
    d = d[d.model_id == "C-EXTRA-TREES"]
    marks = {"cwa": ("o", "cInk"), "pills": ("square*", "cInk"), "surfaces": ("triangle*", "cInk")}
    # Anchor = which side of the label touches the point.
    callouts = {("pills", "Mira-3"): "west", ("surfaces", "RMX-2"): "south", ("surfaces", "Mira-1"): "east",
                ("cwa", "Pendar-2"): "west", ("surfaces", "Agilent-3"): "north west"}
    out = [r"\begin{tikzpicture}",
           r"\begin{axis}[width=3.4cm,height=3.4cm,scale only axis,xmin=30,xmax=102,ymin=30,ymax=102,",
           r"  xtick={40,60,80,100},ytick={40,60,80,100},tick label style={font=\tiny},label style={font=\tiny},",
           r"  xlabel={minimal (\%)},ylabel={baseline-corrected (\%)},xmajorgrids,ymajorgrids,grid style={cGrid,very thin},",
           r"  axis line style={cInk2},tick style={cInk2},clip=false]",
           r"\addplot[cInk2,dashed,thin,domain=30:102] {x};"]
    for st in STATIONS:
        g = d[d.station == st]
        mk, col = marks[st]
        coords = " ".join(f"({r.minimal},{r.baseline})" for _, r in g.iterrows())
        fill = "white" if mk == "o" else col
        out.append(rf"\addplot[only marks,mark={'*' if mk == 'o' else mk},mark size=2.0pt,color={col},mark options={{fill={fill},draw={col},line width=0.5pt}}] coordinates {{{coords}}};")
    for (st, inst), pos in callouts.items():
        r = d[(d.station == st) & (d.instrument == inst)].iloc[0]
        sign = "+" if r.change >= 0 else "$-$"
        out.append(rf"\node[font=\tiny,anchor={pos},inner sep=2.5pt,text=cInk] "
                   rf"at (axis cs:{r.minimal},{r.baseline}) {{{inst} {sign}{abs(r.change):.0f}}};")
    out.append(r"\end{axis}")
    out.append(r"\matrix[anchor=north,font=\tiny,column sep=1pt] at (1.7,-0.72) {"
               r"\draw[fill=white,draw=cInk,line width=0.5pt] plot[mark=*,mark size=1.8pt] coordinates {(0,0)}; & \node[anchor=west]{CWA}; &"
               r"\draw[fill=cInk,draw=cInk] plot[mark=square*,mark size=1.8pt] coordinates {(0,0)}; & \node[anchor=west]{Pills}; &"
               r"\draw[fill=cInk,draw=cInk] plot[mark=triangle*,mark size=1.8pt] coordinates {(0,0)}; & \node[anchor=west]{Surfaces}; \\"
               r"\draw[cInk2,dashed] (-0.12,0) -- (0.12,0); & \node[anchor=west]{no change}; & & \node[anchor=west]{above = helped}; & & \\};")
    out.append(r"\end{tikzpicture}")
    write("scatter_instruments", "\n".join(out))


def fig_substrate():
    d = pd.read_csv(DATA / "substrate_domains.csv")
    # Order: H-Kit (CWA), pSERS Pills, pSERS Surfaces.
    order = [("H-Kit", "CWA"), ("pSERS silver", "Pills"), ("pSERS silver", "Surfaces")]
    W = 3.0            # axis width (cm) for 30..100 %
    xs = lambda v: (v - 30) / 70 * W
    rh = 0.255
    state = {"supports_portability": ("sGood", r"\ding{51}"), "inconclusive": ("sWarn", "?"),
             "inferior_portability": ("sCrit", r"\ding{55}")}
    out = [r"\begin{tikzpicture}[font=\tiny]"]
    y = 0
    rows = []
    for sub, st in order:
        g = d[(d.substrate == sub) & (d.station == st)]
        rows.append(("header", f"{sub} $\\cdot$ {st}", None))
        for _, r in g.iterrows():
            rows.append(("row", r.instrument, r))
    n = len(rows)
    top = n * rh
    for v in (40, 60, 80, 100):
        out.append(rf"\draw[cGrid,very thin] ({xs(v):.3f},0) -- ({xs(v):.3f},{top:.3f});")
        out.append(rf"\node[anchor=north,text=cInk2] at ({xs(v):.3f},0) {{{v}}};")
    out.append(rf"\draw[cInk2,dashed] ({xs(60):.3f},0) -- ({xs(60):.3f},{top:.3f});")
    out.append(rf"\draw[cInk2,densely dotted] ({xs(100/3):.3f},0) -- ({xs(100/3):.3f},{top:.3f});")
    out.append(rf"\node[anchor=south,text=cInk2,align=center,font=\tiny] at ({xs(60):.3f},{top:.3f}) {{pass bar}};")
    out.append(rf"\node[anchor=south,text=cInk2,font=\tiny] at ({xs(33.3):.3f},{top:.3f}) {{chance}};")
    for i, (kind, label, r) in enumerate(rows):
        yy = top - (i + 0.5) * rh
        if kind == "header":
            out.append(rf"\node[anchor=west,font=\tiny\bfseries] at (-1.55,{yy:.3f}) {{{label}}};")
            continue
        out.append(rf"\node[anchor=east,inner sep=1pt] at (-0.05,{yy:.3f}) {{{label}}};")
        if not np.isnan(r.et_arpls_ba):
            col, sym = state[r.et_arpls_state]
            out.append(rf"\draw[{col},line width=1.1pt] ({xs(r.et_arpls_lo):.3f},{yy:.3f}) -- ({xs(r.et_arpls_hi):.3f},{yy:.3f});")
            out.append(rf"\draw[cInk2,fill=white,line width=0.4pt] ({xs(r.et_min_ba):.3f},{yy:.3f}) circle (1.3pt);")
            out.append(rf"\fill[{col},draw=white,line width=0.3pt] ({xs(r.et_arpls_ba):.3f},{yy:.3f}) circle (1.7pt);")
            out.append(rf"\node[anchor=west,text={col}] at ({W + 0.05:.3f},{yy:.3f}) {{{sym}}};")
    out.append(rf"\node[anchor=north,text=cInk2] at ({W / 2:.3f},-0.28) {{held-out-instrument balanced accuracy (\%)}};")
    out.append(r"\end{tikzpicture}")
    write("substrate_domains", "\n".join(out))


def fig_capacity():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.3, 1.8, 3.3, 3.7, 5.2])
    line = np.polyfit(x, y, 1)
    poly = np.polyfit(x, y, 4)
    xn, yn = 5.8, 5.9
    xx = np.linspace(0.7, 6.0, 120)
    line_pts = " ".join(f"({a:.3f},{np.polyval(line, a):.3f})" for a in xx)
    poly_pts = " ".join(f"({a:.3f},{np.clip(np.polyval(poly, a), -2, 9):.3f})" for a in xx)
    out = [r"\begin{tikzpicture}",
           r"\begin{axis}[width=3.0cm,height=1.35cm,scale only axis,xmin=0.5,xmax=6.2,ymin=0,ymax=8.5,ticks=none,",
           r"  axis lines=left,axis line style={cInk2},clip=true]",
           rf"\addplot[cBlue,line width=1pt] coordinates {{{line_pts}}};",
           rf"\addplot[cOrange,line width=1pt] coordinates {{{poly_pts}}};",
           "\\addplot[only marks,mark=*,mark size=1.8pt,color=cInk,mark options={fill=cInk,draw=white}] coordinates {" +
           " ".join(f"({a},{b})" for a, b in zip(x, y)) + "};",
           rf"\addplot[only marks,mark=star,mark size=3pt,color=cInk,line width=0.8pt] coordinates {{({xn},{yn})}};",
           rf"\node[font=\tiny,anchor=north west,inner sep=1pt] at (axis cs:{xn - 0.6},{yn - 0.6}) {{new}};",
           r"\end{axis}",
           r"\matrix[anchor=north west,font=\tiny,row sep=0pt,column sep=1pt] at (-0.1,-0.12) {"
           r"\draw[cBlue,line width=1pt] (0,0) -- (0.35,0); & \node[anchor=west]{line, $W$=2: 5 points $>$ 2, one best fit}; \\"
           r"\draw[cOrange,line width=1pt] (0,0) -- (0.35,0); & \node[anchor=west]{curve, $W$=5: 5 = 5, hits every point, misses the new one}; \\};",
           r"\end{tikzpicture}"]
    write("capacity_schematic", "\n".join(out))


def table_per_instrument():
    d = pd.read_csv(DATA / "per_instrument.csv")
    rows = []
    keys = d[d.model_id == "C-EXTRA-TREES"][["station", "instrument"]].values.tolist()
    for st, inst in keys:
        cells = []
        for m in MODELS:
            r = d[(d.model_id == m) & (d.station == st) & (d.instrument == inst)].iloc[0]
            cells.append(f"{r.minimal:.0f} $\\to$ {r.baseline:.0f}")
        rows.append(f"{STATION_LABEL[st]} & {inst} & " + " & ".join(cells) + r" \\")
    head = " & ".join(MODEL_SHORT[m] for m in MODELS)
    body = "\n".join(rows)
    tex = (r"\begin{tabular}{ll" + "c" * len(MODELS) + "}\n\\toprule\nStation & Instrument & " + head +
           " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}")
    write("table_per_instrument", tex)


def fig_spectra():
    """Same 4-ANPP and blank pill samples on Mira-3 and Agilent-3 at each preprocessing stage."""
    stages = [("raw", "raw"), ("minimal", "minimal"), ("smoothed", "smoothed"), ("baseline", "baseline")]
    insts = [("Mira-3", "Mira-3: mostly background"), ("Agilent-3", "Agilent-3: mostly peaks")]
    series = [("4_ANPP", "cViolet"), ("blank", "cMagenta")]
    out = [r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=2 by 4,horizontal sep=0.25cm,vertical sep=0.12cm,",
           r"    xticklabels at=edge bottom,xlabels at=edge bottom},",
           r"  width=3.75cm,height=0.70cm,scale only axis,xmin=400,xmax=1800,xtick={600,1000,1400,1800},",
           r"  ylabel style={rotate=-90,anchor=east,font=\tiny,xshift=2pt},",
           r"  tick label style={font=\tiny},label style={font=\tiny},title style={font=\scriptsize,yshift=-2pt},",
           r"  axis line style={cGrid},tick style={cGrid},ytick=\empty,",
           r"  every axis plot/.append style={line width=0.45pt,line join=round}]"]
    for row, (tag, label) in enumerate(stages):
        for col, (inst, title) in enumerate(insts):
            opts = []
            if row == 0:
                opts.append(f"title={{{title}}}")
            if col == 0:
                opts.append(f"ylabel={{{label}}}")
            if row == len(stages) - 1:
                opts.append(r"xlabel={Raman shift (cm$^{-1}$)}")
            if tag == "raw":
                # Shared raw axis per instrument so the background level is visible.
                ys = [pd.read_csv(PRIVATE / f"spectrum_{inst}_{c}_raw.csv").y for c, _ in series]
                hi = max(float(y.max()) for y in ys)
                opts.append(f"ymin=0,ymax={hi * 1.05:.1f}")
            else:
                opts.append("ymin=-0.05,ymax=1.08")
            out.append(r"\nextgroupplot[" + ",".join(opts) + "]")
            for chem, col_name in series:
                d = pd.read_csv(PRIVATE / f"spectrum_{inst}_{chem}_{tag}.csv")
                coords = " ".join(f"({x:.0f},{y:.4g})" for x, y in zip(d.x, d.y))
                out.append(rf"\addplot[{col_name}] coordinates {{{coords}}};")
    out.append(r"\end{groupplot}")
    out.append(r"\matrix[anchor=north,font=\tiny,column sep=2pt] at ($(group c1r4.south)!0.5!(group c2r4.south)+(0,-0.62)$) {"
               r"\draw[cViolet,line width=0.9pt] (0,0) -- (0.35,0); & \node[anchor=west]{4-ANPP sample}; &"
               r"\draw[cMagenta,line width=0.9pt] (0,0) -- (0.35,0); & \node[anchor=west]{blank sample}; \\};")
    out.append(r"\end{tikzpicture}")
    write("spectra_preprocessing", "\n".join(out))


def fig_background_gain():
    d = pd.read_csv(DATA / "background_vs_gain.csv")
    d = d[d.model_id == "C-EXTRA-TREES"]
    marks = {"cwa": ("*", "white"), "pills": ("square*", "cInk"), "surfaces": ("triangle*", "cInk")}
    callouts = {("surfaces", "RMX-2"): "south east", ("pills", "Mira-3"): "east", ("surfaces", "Mira-1"): "east",
                ("cwa", "Pendar-2"): "east", ("surfaces", "Pendar-2"): "east"}
    out = [r"\begin{tikzpicture}",
           r"\begin{axis}[width=3.6cm,height=2.55cm,scale only axis,xmin=15,xmax=105,ymin=-19,ymax=44,",
           r"  xtick={20,40,60,80,100},ytick={-10,0,10,20,30},tick label style={font=\tiny},label style={font=\tiny},",
           r"  xlabel={arPLS background (\% of raw signal)},ylabel={gain from correction (points)},",
           r"  xmajorgrids,ymajorgrids,grid style={cGrid,very thin},axis line style={cInk2},tick style={cInk2},clip=false]",
           r"\addplot[cInk2,dashed,thin,domain=15:105] {0};",
           r"\node[font=\tiny,anchor=south,inner sep=2.5pt] at (axis cs:30,2.5) {Agilent};"]
    for st, (mk, fill) in marks.items():
        g = d[d.station == st]
        coords = " ".join(f"({r.background_pct},{r.change})" for _, r in g.iterrows())
        out.append(rf"\addplot[only marks,mark={mk},mark size=2.0pt,color=cInk,mark options={{fill={fill},draw=cInk,line width=0.5pt}}] coordinates {{{coords}}};")
    for (st, inst), anchor in callouts.items():
        r = d[(d.station == st) & (d.instrument == inst)].iloc[0]
        sign = "+" if r.change >= 0 else "$-$"
        out.append(rf"\node[font=\tiny,anchor={anchor},inner sep=2.5pt] at (axis cs:{r.background_pct},{r.change}) {{{inst} {sign}{abs(r.change):.0f}}};")
    out.append(r"\end{axis}")
    out.append(r"\matrix[anchor=north,font=\tiny,column sep=1pt] at (1.8,-0.72) {"
               r"\draw[fill=white,draw=cInk,line width=0.5pt] plot[mark=*,mark size=1.8pt] coordinates {(0,0)}; & \node[anchor=west]{CWA}; &"
               r"\draw[fill=cInk,draw=cInk] plot[mark=square*,mark size=1.8pt] coordinates {(0,0)}; & \node[anchor=west]{Pills}; &"
               r"\draw[fill=cInk,draw=cInk] plot[mark=triangle*,mark size=1.8pt] coordinates {(0,0)}; & \node[anchor=west]{Surfaces}; \\};")
    out.append(r"\end{tikzpicture}")
    write("background_vs_gain", "\n".join(out))


if __name__ == "__main__":
    fig_coverage()
    fig_chem_substrate()
    fig_pca("pills")
    fig_model_prep()
    fig_confusion_main()
    fig_confusion_all("baseline")
    fig_confusion_all("minimal")
    fig_per_class()
    fig_scatter()
    fig_substrate()
    fig_capacity()
    table_per_instrument()
    fig_spectra()
    fig_background_gain()
    print("figures:", sorted(p.name for p in FIG.glob("*.tex")))
