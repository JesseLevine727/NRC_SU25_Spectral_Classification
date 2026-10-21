#!/usr/bin/env python
"""Phase 1: inventory and common-axis loader for the Paper 2 pilot.

Reads raw spectra in place (never writes to the data tree), builds
outputs/inventory.csv and common 1 cm-1-axis archives for Block A, Block B,
neat references and substrate-only backgrounds, and plots cell means.

Run from this folder:
    /home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python load.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "Data"))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")

BLOCK_A_ROOT = os.path.join(DATA, "Jesse_Dataset_Update", "SERs")
BLOCK_B_ROOT = os.path.join(DATA, "Jesse_Dataset_v2", "SERS")
PURE_RAMAN_ROOT = os.path.join(DATA, "Jesse_Dataset_v2", "Pure_Raman_txt")
UPDATE_RAMAN_ROOT = os.path.join(DATA, "Jesse_Dataset_Update", "Raman")
BACKGROUND_ROOT = os.path.join(DATA, "Jesse_Dataset_v2", "SERS", "Backgrounds", "AuNP")

BLOCK_A_ANALYTES = {"4-nitrophenol", "benzenethiol", "pyridine"}
BLOCK_B_ANALYTE_DIRS = {
    "4-mercaptobenzoic acid": "4-mercaptobenzoic acid",
    "2,6-pyridinecarboxilic acid": "2,6-pyridinecarboxylic acid",
    "4-mercaptobenzonitrile": "4-mercaptobenzonitrile",
}
BLOCK_B_SERS_DIRS = re.compile(r"^(AgNP|AuNP)\s*-\s*(632|785)\s*nm$", re.IGNORECASE)

ANALYTE_CANON = {
    "4-np": "4-nitrophenol",
    "4np": "4-nitrophenol",
    "4-nitrophenol": "4-nitrophenol",
    "bt": "benzenethiol",
    "benzenethiol": "benzenethiol",
    "pyridine": "pyridine",
    "4-mercaptobenzoic acid": "4-mercaptobenzoic acid",
    "2,6-pyridinecarboxilic acid": "2,6-pyridinecarboxylic acid",
    "2,6-pyridinecarboxylic acid": "2,6-pyridinecarboxylic acid",
    "4-mercaptobenzonitrile": "4-mercaptobenzonitrile",
}
CODE = {
    "4-nitrophenol": "4NP",
    "benzenethiol": "BT",
    "pyridine": "Py",
    "4-mercaptobenzoic acid": "MBA",
    "2,6-pyridinecarboxylic acid": "PDCA",
    "4-mercaptobenzonitrile": "MBN",
}
SUBSTRATE_CANON = {
    "ag": "Ag",
    "agnp": "Ag",
    "au": "Au",
    "aunp": "Au",
    "pico": "PICO",
    "psers": "pSERS",
    "pserss": "pSERS",
}
SUBSTRATE_ORDER = ["Ag", "Au", "PICO", "pSERS"]
POS_RE = re.compile(
    r"(?:map|amp)\s*(\d+)[^/]*?[Yy]\s*0*(\d+)\s*_\s*[Xx]\s*0*(\d+)", re.IGNORECASE
)
LASER_RE = re.compile(r"(632|633|785)")


def canon_analyte(name: str) -> str | None:
    key = re.sub(r"\s+", " ", name.strip().lower())
    return ANALYTE_CANON.get(key)


def canon_substrate(name: str) -> str | None:
    key = re.sub(r"[\s_\-]", "", name.strip().lower())
    return SUBSTRATE_CANON.get(key)


def canon_laser(name: str) -> str:
    m = LASER_RE.search(name)
    if not m:
        return ""
    return "632" if m.group(1) in ("632", "633") else "785"


def read_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter="\t", dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected two columns, got {arr.shape}")
    x, y = arr[:, 0], arr[:, 1]
    if len(x) < 10:
        raise ValueError(f"only {len(x)} points")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("shift axis is not strictly increasing")
    return x, y


def parse_position(basename: str) -> tuple[str, int, int]:
    m = POS_RE.search(basename)
    if m:
        return f"map{m.group(1)}", int(m.group(2)), int(m.group(3))
    return "single", -1, -1


def common_axis(records: list[dict]) -> np.ndarray:
    """1 cm-1 grid strictly inside the range measured by every spectrum."""
    lo = max(r["shift_min"] for r in records)
    hi = min(r["shift_max"] for r in records)
    start, stop = int(np.ceil(lo - 1e-9)), int(np.floor(hi + 1e-9))
    if stop - start < 100:
        raise ValueError(f"common axis too short: {start}-{stop}")
    axis = np.arange(start, stop + 1, 1.0)
    for r in records:
        if axis[0] < r["shift_min"] - 1e-9 or axis[-1] > r["shift_max"] + 1e-9:
            raise AssertionError("interpolation would extrapolate")
    return axis


def to_matrix(records: list[dict], axis: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(axis, r["x_raw"], r["y_raw"]) for r in records])


def make_record(path: str, block: str, analyte: str, substrate: str, laser: str,
                map_id: str) -> dict:
    x, y = read_txt(path)
    map_label, yy, xx = parse_position(os.path.basename(path))
    if map_label == "single":
        # Keep single spectra distinct (e.g. reference files "analyte - 1.txt").
        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
        map_label = f"single_{stem}"
        if map_id.endswith("_single"):
            map_id = map_id[: -len("_single")] + "_" + map_label
    return {
        "block": block,
        "analyte": analyte,
        "substrate": substrate,
        "laser": laser,
        "map_id": map_id,
        "map_label": map_label,
        "y": yy,
        "x": xx,
        "path": os.path.relpath(path, DATA),
        "n_points": int(len(x)),
        "shift_min": float(x.min()),
        "shift_max": float(x.max()),
        "x_raw": x,
        "y_raw": y,
    }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_block_a() -> list[dict]:
    records = []
    for adir in sorted(os.listdir(BLOCK_A_ROOT)):
        apath = os.path.join(BLOCK_A_ROOT, adir)
        if not os.path.isdir(apath):
            continue
        analyte = canon_analyte(adir.split(" - ")[0])
        if analyte not in BLOCK_A_ANALYTES:
            continue
        for sdir in sorted(os.listdir(apath)):
            spath = os.path.join(apath, sdir)
            substrate = canon_substrate(sdir)
            if not os.path.isdir(spath) or substrate not in SUBSTRATE_ORDER:
                continue
            for f in sorted(os.listdir(spath)):
                if not f.lower().endswith(".txt") or "avg" in f.lower():
                    continue
                map_label, yy, xx = parse_position(f)
                map_id = f"A_{CODE[analyte]}_{substrate}_632_{map_label}"
                records.append(make_record(os.path.join(spath, f), "A", analyte,
                                           substrate, "632", map_id))
    return records


def discover_block_b() -> list[dict]:
    records = []
    for adir, analyte in BLOCK_B_ANALYTE_DIRS.items():
        apath = os.path.join(BLOCK_B_ROOT, adir)
        for sdir in sorted(os.listdir(apath)):
            m = BLOCK_B_SERS_DIRS.match(sdir)
            if not m:
                continue
            substrate = "Ag" if m.group(1).lower().startswith("ag") else "Au"
            laser = canon_laser(m.group(2))
            spath = os.path.join(apath, sdir)
            for f in sorted(os.listdir(spath)):
                if not f.lower().endswith(".txt") or "avg" in f.lower():
                    continue
                map_label, yy, xx = parse_position(f)
                map_id = f"B_{CODE[analyte]}_{substrate}_{laser}_{map_label}"
                records.append(make_record(os.path.join(spath, f), "B", analyte,
                                           substrate, laser, map_id))
    return records


def discover_references() -> list[dict]:
    records = []
    # Pure Raman txt: laser not encoded in names/folders -> unknown.
    for f in sorted(os.listdir(PURE_RAMAN_ROOT)):
        if not f.lower().endswith(".txt"):
            continue
        stem = os.path.splitext(f)[0]
        analyte = canon_analyte(stem.split(" - ")[0])
        if analyte not in {"benzenethiol", "pyridine"}:
            continue
        map_label, _, _ = parse_position(f)
        map_id = f"REF_{CODE[analyte]}_unknown_pure_{map_label}"
        records.append(make_record(os.path.join(PURE_RAMAN_ROOT, f), "reference",
                                   analyte, "none", "unknown", map_id))

    # Update Raman references, laser encoded in the subfolder.
    for adir, analyte in [("4-Nitrophenol", "4-nitrophenol"),
                          ("4-mercaptobenzonitrile", "4-mercaptobenzonitrile")]:
        apath = os.path.join(UPDATE_RAMAN_ROOT, adir)
        for sdir in sorted(os.listdir(apath)):
            spath = os.path.join(apath, sdir)
            if not os.path.isdir(spath):
                continue
            laser = canon_laser(sdir)
            for f in sorted(os.listdir(spath)):
                if not f.lower().endswith(".txt") or "avg" in f.lower():
                    continue
                map_label, _, _ = parse_position(f)
                map_id = f"REF_{CODE[analyte]}_{laser}_update_{map_label}"
                records.append(make_record(os.path.join(spath, f), "reference",
                                           analyte, "none", laser, map_id))

    # Block B neat powder references, laser encoded in the subfolder.
    for adir, analyte in [("4-mercaptobenzoic acid", "4-mercaptobenzoic acid"),
                          ("2,6-pyridinecarboxilic acid", "2,6-pyridinecarboxylic acid")]:
        ppath = os.path.join(BLOCK_B_ROOT, adir, "powder - non-SERs")
        for sdir in sorted(os.listdir(ppath)):
            spath = os.path.join(ppath, sdir)
            if not os.path.isdir(spath):
                continue
            laser = canon_laser(sdir)
            for f in sorted(os.listdir(spath)):
                if not f.lower().endswith(".txt") or "avg" in f.lower():
                    continue
                map_label, _, _ = parse_position(f)
                map_id = f"REF_{CODE[analyte]}_{laser}_powder_{map_label}"
                records.append(make_record(os.path.join(spath, f), "reference",
                                           analyte, "none", laser, map_id))
    return records


def discover_backgrounds() -> list[dict]:
    records = []
    for sdir in sorted(os.listdir(BACKGROUND_ROOT)):
        spath = os.path.join(BACKGROUND_ROOT, sdir)
        if not os.path.isdir(spath):
            continue
        laser = canon_laser(sdir)
        for f in sorted(os.listdir(spath)):
            if not f.lower().endswith(".txt") or "avg" in f.lower():
                continue
            map_label, _, _ = parse_position(f)
            map_id = f"BG_Au_{laser}_{map_label}"
            records.append(make_record(os.path.join(spath, f), "background", "background",
                                       "Au", laser, map_id))
    return records


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def inventory_frame(records: list[dict]) -> pd.DataFrame:
    cols = ["block", "analyte", "substrate", "laser", "map_id", "y", "x", "path",
            "n_points", "shift_min", "shift_max"]
    rows = []
    for r in records:
        row = {k: r[k] for k in cols}
        row["y"] = r["y"] if r["y"] >= 0 else np.nan
        row["x"] = r["x"] if r["x"] >= 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def save_npz(path: str, records: list[dict], axis: np.ndarray) -> None:
    X = to_matrix(records, axis)
    np.savez_compressed(
        path,
        X=X,
        axis=axis,
        block=np.array([r["block"] for r in records]),
        analyte=np.array([r["analyte"] for r in records]),
        substrate=np.array([r["substrate"] for r in records]),
        laser=np.array([r["laser"] for r in records]),
        map_id=np.array([r["map_id"] for r in records]),
        map_label=np.array([r["map_label"] for r in records]),
        y=np.array([r["y"] for r in records], dtype=int),
        x=np.array([r["x"] for r in records], dtype=int),
        path=np.array([r["path"] for r in records]),
    )


def minmax_rows(X: np.ndarray) -> np.ndarray:
    lo = X.min(axis=1, keepdims=True)
    span = X.max(axis=1, keepdims=True) - lo
    return (X - lo) / np.maximum(span, np.finfo(float).eps)


def reference_mean(records: list[dict], analyte: str, laser: str,
                   axis: np.ndarray) -> np.ndarray | None:
    """Mean of min-max-scaled reference spectra, preferring the exact laser."""
    subset = [r for r in records if r["analyte"] == analyte and r["laser"] == laser]
    if not subset and laser:
        subset = [r for r in records if r["analyte"] == analyte
                  and r["laser"] in ("", "unknown")]
    if not subset:
        subset = [r for r in records if r["analyte"] == analyte]
    if not subset:
        return None
    X = minmax_rows(to_matrix(subset, axis))
    return X.mean(axis=0)


def plot_block_cell_means(records: list[dict], axis: np.ndarray, refs: list[dict],
                          block: str, outpath: str) -> None:
    if block == "A":
        analytes = ["4-nitrophenol", "benzenethiol", "pyridine"]
        cells = [("Ag", "632"), ("Au", "632"), ("PICO", "632"), ("pSERS", "632")]
    else:
        analytes = ["4-mercaptobenzoic acid", "2,6-pyridinecarboxylic acid",
                    "4-mercaptobenzonitrile"]
        cells = [("Ag", "632"), ("Ag", "785"), ("Au", "632"), ("Au", "785")]

    grouped = defaultdict(list)
    for r in records:
        grouped[(r["analyte"], r["substrate"], r["laser"])].append(r)

    fig, axes = plt.subplots(len(analytes), len(cells), figsize=(16, 10.5),
                             sharex=True, sharey=True)
    for i, analyte in enumerate(analytes):
        for j, (substrate, laser) in enumerate(cells):
            ax = axes[i, j]
            rs = grouped.get((analyte, substrate, laser), [])
            if not rs:
                ax.text(0.5, 0.5, "no data\n(4-NP / Au not measured)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=9, color="0.4")
                ax.set_axis_off()
                continue
            X = to_matrix(rs, axis)
            mean = minmax_rows(X.mean(axis=0, keepdims=True))[0]
            n_maps = len({r["map_id"] for r in rs})
            ax.plot(axis, mean, color="#1f5fa8", lw=1.1,
                    label=f"cell mean (n={len(rs)} spots, {n_maps} map)")
            ref = reference_mean(refs, analyte, laser, axis)
            if ref is not None:
                ref = (ref - ref.min()) / max(ref.max() - ref.min(), 1e-12)
                ax.plot(axis, ref, color="0.35", lw=0.9, ls="--", label="neat reference")
            if i == 0:
                ax.set_title(f"{substrate}  |  {laser} nm", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{CODE[analyte]}\nmin-max intensity", fontsize=9)
            if i == len(analytes) - 1:
                ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.tick_params(labelsize=8)
            if i == 0 and j == len(cells) - 1:
                ax.legend(fontsize=7, loc="upper right", frameon=False)
    fig.suptitle(f"Block {block}: mean SERS spectrum per cell (min-max scaled for display)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    for p in [BLOCK_A_ROOT, BLOCK_B_ROOT, PURE_RAMAN_ROOT, UPDATE_RAMAN_ROOT,
              BACKGROUND_ROOT]:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)

    a = discover_block_a()
    b = discover_block_b()
    refs = discover_references()
    bgs = discover_backgrounds()
    print(f"Block A spectra: {len(a)}")
    print(f"Block B spectra: {len(b)}")
    print(f"Reference spectra: {len(refs)}")
    print(f"Background spectra: {len(bgs)}")

    # Hard count checks against the raw folders.
    assert len(a) == 275, len(a)
    assert len(b) == 780, len(b)
    assert len(refs) == 265, len(refs)
    assert len(bgs) == 98, len(bgs)

    axis_a = common_axis(a)
    axis_b = common_axis(b)
    axis_ref = common_axis(refs)
    axis_bg = common_axis(bgs)
    print(f"Block A axis: {axis_a[0]:.0f}-{axis_a[-1]:.0f} cm-1 ({len(axis_a)} points)")
    print(f"Block B axis: {axis_b[0]:.0f}-{axis_b[-1]:.0f} cm-1 ({len(axis_b)} points)")
    print(f"Reference axis: {axis_ref[0]:.0f}-{axis_ref[-1]:.0f} cm-1 ({len(axis_ref)} points)")
    print(f"Background axis: {axis_bg[0]:.0f}-{axis_bg[-1]:.0f} cm-1 ({len(axis_bg)} points)")

    inv = inventory_frame(a + b + refs + bgs)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    save_npz(os.path.join(OUT, "spectra_blockA.npz"), a, axis_a)
    save_npz(os.path.join(OUT, "spectra_blockB.npz"), b, axis_b)
    save_npz(os.path.join(OUT, "references.npz"), refs, axis_ref)
    save_npz(os.path.join(OUT, "backgrounds.npz"), bgs, axis_bg)

    plot_block_cell_means(a, axis_a, refs, "A", os.path.join(FIG, "cell_means_blockA.png"))
    plot_block_cell_means(b, axis_b, refs, "B", os.path.join(FIG, "cell_means_blockB.png"))
    print("Wrote outputs/inventory.csv and four .npz archives")
    print("Wrote figures/cell_means_blockA.png and figures/cell_means_blockB.png")


if __name__ == "__main__":
    main()
