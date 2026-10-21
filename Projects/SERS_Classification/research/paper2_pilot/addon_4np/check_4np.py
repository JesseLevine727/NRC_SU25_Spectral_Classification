#!/usr/bin/env python
"""Paper 2 pilot add-on: chip-to-chip versus substrate variation for 4-nitrophenol.

Reads the repeat 4-NP-on-silver Galactic SPC files, validates the binary layout,
puts them on the same 1 cm-1 axis as the block A maps (intersection only), and
compares within-preparation, between-preparation, between-substrate and
between-chemical distances.

Primary preprocessing is per-spectrum min-max.  Sensitivity is despike + arPLS +
min-max with the same parameters as phase 2.

Writes only inside this folder: figure_4np.png and RESULTS_4NP.md.
"""

from __future__ import annotations

import os
import struct
import sys
from itertools import combinations
from pathlib import Path

# Keep imports from writing .pyc files outside this folder (atlas_sers is read-only).
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SPC_DIR = Path(
    "/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/"
    "Data/Jesse_Dataset_v2/SERS/4-Nitrophenol"
)
BLOCK_A_NPZ = HERE.parent / "outputs" / "spectra_blockA.npz"
INVENTORY_CSV = HERE.parent / "outputs" / "inventory.csv"
ATLAS_SRC = HERE.parents[1] / "atlas_sers" / "src"

FIGURE = HERE / "figure_4np.png"
RESULTS = HERE / "RESULTS_4NP.md"

SPC_SAMPLES = {
    # name used in the analysis: five R1-R5 files
    "P1 23pp-SC (Ag-23pp-SC-4NP)": [
        SPC_DIR / f"Ag-23pp-SC-4NP-R{i}.spc" for i in range(1, 6)
    ],
    "P2 23pp-SC (S03_23PP_SC_4NP)": [
        SPC_DIR / f"S03_23PP_SC_4NP_R{i:02d}.spc" for i in range(1, 6)
    ],
    "P3 pSERS (S01-Ag-psers-4NP)": [
        SPC_DIR / f"S01-Ag-psers-4NP-R{i}.spc" for i in range(1, 6)
    ],
}

BLOCK_MAPS = {
    "block A 4NP/Ag": "A_4NP_Ag_632_map1",
    "block A 4NP/PICO": "A_4NP_PICO_632_map1",
    "block A 4NP/pSERS": "A_4NP_pSERS_632_map1",
    "block A BT/Ag": "A_BT_Ag_632_map1",
    "block A Py/Ag": "A_Py_Ag_632_map1",
}

SUBSTRATE_PAIRS = [
    ("block A 4NP/Ag", "block A 4NP/PICO"),
    ("block A 4NP/Ag", "block A 4NP/pSERS"),
    ("block A 4NP/PICO", "block A 4NP/pSERS"),
]
CHEMICAL_PAIRS = [
    ("block A 4NP/Ag", "block A BT/Ag"),
    ("block A 4NP/Ag", "block A Py/Ag"),
    ("block A BT/Ag", "block A Py/Ag"),
]

EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- #
# Minimal Galactic SPC new-format reader
# --------------------------------------------------------------------------- #
class SpcFormatError(RuntimeError):
    """Raised when a file does not match the expected little-endian SPC layout."""


def read_spc(path: Path) -> dict:
    """Read one Galactic SPC new-format file, little-endian.

    The layout used here is the one described in ADDON_BRIEF.md:
      512-byte main header: flags, version, exponent, int32 point count,
      float64 first/last x, int32 subfile count;
      optional float32 x array when bit 0 of the flags byte is set;
      then per subfile a 32-byte header and point-count y values
      (float32 when the subfile exponent byte is 0x80, otherwise scaled int32).

    The function is deliberately strict: any size mismatch, non-finite value or
    inconsistent header aborts with SpcFormatError instead of guessing.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 512:
        raise SpcFormatError(f"{path.name}: shorter than the 512-byte main header")

    flags = raw[0]
    version = raw[1]
    exponent = raw[2]
    n_points = struct.unpack_from("<i", raw, 4)[0]
    x_first, x_last = struct.unpack_from("<dd", raw, 8)
    n_sub = struct.unpack_from("<i", raw, 24)[0]

    if version != 0x4B:
        raise SpcFormatError(f"{path.name}: version 0x{version:02x}, expected 0x4b")
    if n_points < 2:
        raise SpcFormatError(f"{path.name}: implausible point count {n_points}")
    if n_sub < 1:
        raise SpcFormatError(f"{path.name}: implausible subfile count {n_sub}")
    if not np.isfinite([x_first, x_last]).all() or not x_first < x_last:
        raise SpcFormatError(
            f"{path.name}: bad x range [{x_first}, {x_last}]"
        )

    offset = 512
    explicit_x = bool(flags & 0x01)
    if explicit_x:
        need = 4 * n_points
        if len(raw) < offset + need:
            raise SpcFormatError(f"{path.name}: truncated explicit x array")
        x = np.frombuffer(raw, dtype="<f4", count=n_points, offset=offset).astype(float)
        offset += need
        if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise SpcFormatError(f"{path.name}: explicit x array is not increasing")
    else:
        x = np.linspace(float(x_first), float(x_last), n_points)
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-6, atol=1e-6):
        raise SpcFormatError(f"{path.name}: x grid is not uniform")

    ys, sub_exponents = [], []
    for sub in range(n_sub):
        if len(raw) < offset + 32:
            raise SpcFormatError(f"{path.name}: truncated subfile {sub} header")
        sub_exp = raw[offset + 1]
        # The point count sits at +16 in this layout; some writers leave it 0.
        sub_npts = struct.unpack_from("<i", raw, offset + 16)[0]
        if sub_npts not in (0, n_points):
            raise SpcFormatError(
                f"{path.name}: subfile {sub} says {sub_npts} points, "
                f"main header says {n_points}"
            )
        offset += 32

        nbytes = 4 * n_points
        if len(raw) < offset + nbytes:
            raise SpcFormatError(f"{path.name}: truncated y data in subfile {sub}")
        if sub_exp == 0x80:
            y = np.frombuffer(
                raw, dtype="<f4", count=n_points, offset=offset
            ).astype(float)
        else:
            y = np.frombuffer(
                raw, dtype="<i4", count=n_points, offset=offset
            ).astype(float)
            # Galactic integer scaling: stored integer / 2**(exponent - 64).
            y = y / (2.0 ** (int(sub_exp) - 64))
        offset += nbytes
        if not np.all(np.isfinite(y)):
            raise SpcFormatError(f"{path.name}: non-finite y values in subfile {sub}")
        ys.append(y)
        sub_exponents.append(int(sub_exp))

    trailing = len(raw) - offset
    if trailing != 0:
        raise SpcFormatError(
            f"{path.name}: {trailing} unexpected trailing bytes; "
            "layout is ambiguous"
        )

    return {
        "path": path,
        "flags": int(flags),
        "version": int(version),
        "exponent": int(exponent),
        "n_points": int(n_points),
        "x_first": float(x_first),
        "x_last": float(x_last),
        "n_sub": int(n_sub),
        "x": x,
        "y": np.vstack(ys),
        "y0": ys[0],
        "sub_exponents": sub_exponents,
        "explicit_x": explicit_x,
    }


def validate_spc() -> dict:
    """Parse every R scan, report headers and check the five scans look alike."""
    print("=" * 78)
    print("SPC VALIDATION")
    print("=" * 78)

    records = {}
    for sample, files in SPC_SAMPLES.items():
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError("missing SPC files: " + ", ".join(missing))
        records[sample] = [read_spc(p) for p in files]

        print(f"\n{sample}")
        for rec in records[sample]:
            exp = ",".join(f"0x{e:02x}" for e in rec["sub_exponents"])
            print(
                f"  {rec['path'].name:32s} {rec['n_points']:4d} pts  "
                f"x=[{rec['x'][0]:.1f}, {rec['x'][-1]:.1f}] cm-1  "
                f"n_sub={rec['n_sub']}  sub_exp={exp}  "
                f"y=[{rec['y0'].min():.4g}, {rec['y0'].max():.4g}]"
            )

        # All five scans must share the same axis and length.
        first = records[sample][0]
        for rec in records[sample][1:]:
            if rec["n_points"] != first["n_points"] or not np.allclose(
                rec["x"], first["x"]
            ):
                raise SpcFormatError(
                    f"{sample}: scans do not share the same x axis"
                )
        Y = np.vstack([r["y0"] for r in records[sample]])
        corr = [
            float(np.corrcoef(a, b)[0, 1])
            for a, b in combinations(Y, 2)
        ]
        print(
            f"  five-scan pairwise Pearson r: min={min(corr):.6f}  "
            f"median={np.median(corr):.6f}  max={max(corr):.6f}"
        )
        records[sample] = {
            "files": [r["path"] for r in records[sample]],
            "x": first["x"],
            "Y": Y,
            "n_points": first["n_points"],
            "x_first": first["x"][0],
            "x_last": first["x"][-1],
            "sub_exponents": first["sub_exponents"],
            "scan_corr": corr,
        }

    lengths = {r["n_points"] for r in records.values()}
    ranges = {(round(r["x_first"], 6), round(r["x_last"], 6)) for r in records.values()}
    if len(lengths) != 1 or len(ranges) != 1:
        raise SpcFormatError(
            f"SPC samples disagree on point count/range: {lengths}, {ranges}"
        )

    print(
        f"\nAll 15 scans parsed unambiguously: {lengths.pop()} points, "
        f"x=[{records['P1 23pp-SC (Ag-23pp-SC-4NP)']['x_first']:.1f}, "
        f"{records['P1 23pp-SC (Ag-23pp-SC-4NP)']['x_last']:.1f}] cm-1, "
        "uniform 1 cm-1 grid, float32 y (subfile exponent 0x80)."
    )
    return records


# --------------------------------------------------------------------------- #
# Block A, common axis, preprocessing
# --------------------------------------------------------------------------- #
def load_block_a() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = np.load(BLOCK_A_NPZ, allow_pickle=False)
    X = data["X"].astype(float)
    axis = data["axis"].astype(float)
    map_id = data["map_id"]
    analyte = data["analyte"]
    substrate = data["substrate"]

    print("\n" + "=" * 78)
    print("BLOCK A")
    print("=" * 78)
    print(f"X={X.shape}, axis=[{axis[0]:.1f}, {axis[-1]:.1f}] cm-1, "
          f"step={np.median(np.diff(axis)):.3f} cm-1")

    groups = {}
    for label, mid in BLOCK_MAPS.items():
        mask = map_id == mid
        if not mask.any():
            raise ValueError(f"map {mid} not found in {BLOCK_A_NPZ}")
        groups[label] = X[mask]
        print(
            f"  {label:22s} {mid:24s} n={int(mask.sum()):2d}  "
            f"{analyte[mask][0]} / {substrate[mask][0]}"
        )

    # Light cross-check against inventory.csv (read-only).
    if INVENTORY_CSV.exists():
        import csv

        wanted = set(BLOCK_MAPS.values())
        counts = {}
        with INVENTORY_CSV.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["map_id"] in wanted:
                    counts[row["map_id"]] = counts.get(row["map_id"], 0) + 1
        for mid in sorted(wanted):
            n_npz = int((map_id == mid).sum())
            if counts.get(mid) != n_npz:
                raise ValueError(
                    f"inventory/npz count mismatch for {mid}: "
                    f"{counts.get(mid)} vs {n_npz}"
                )
        print("  inventory.csv spot counts agree with the npz metadata.")

    return X, axis, groups


def resample_rows(x: np.ndarray, Y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Linear interpolation onto target, refusing to extrapolate."""
    if target[0] < x[0] - 1e-9 or target[-1] > x[-1] + 1e-9:
        raise ValueError(
            f"target [{target[0]}, {target[-1]}] leaves source "
            f"[{x[0]}, {x[-1]}]; refusing to extrapolate"
        )
    return np.vstack([np.interp(target, x, row) for row in Y])


def minmax_rows(Y: np.ndarray) -> np.ndarray:
    lo = Y.min(axis=1, keepdims=True)
    span = Y.max(axis=1, keepdims=True) - lo
    return (Y - lo) / np.maximum(span, EPS)


def preprocess(Y: np.ndarray, version: str) -> np.ndarray:
    """Per-spectrum preprocessing, matching phase 2."""
    if version == "minmax":
        out = minmax_rows(Y)
    elif version == "arpls":
        from atlas_sers.preprocessing.representations import (
            arpls_correct,
            despike,
        )

        clean = despike(Y, window=5, threshold=10.0)
        corrected = arpls_correct(
            clean,
            lam=100000.0,
            iterations=12,
            relative_tolerance=0.001,
            logistic_clip=60.0,
        )
        out = minmax_rows(corrected)
    else:
        raise ValueError(version)
    return np.clip(out, 0.0, None)


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #
def distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (1 - Pearson correlation, RMSE) between two mean spectra."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return 1.0 - r, rmse


def summarize(pairs: list[tuple[str, str, float, float]]) -> dict:
    d = np.array([p[2] for p in pairs], dtype=float)
    e = np.array([p[3] for p in pairs], dtype=float)
    return {
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "mean_rmse": float(np.mean(e)),
        "pairs": pairs,
    }


def analyze(
    spc_records: dict,
    block_groups_raw: dict[str, np.ndarray],
    common: np.ndarray,
    version: str,
) -> dict:
    """Distances for one preprocessing version."""
    # Build the four 4-NP-on-silver preparations and the supporting maps.
    raw_groups = {}
    for label, rec in spc_records.items():
        short = label.split(" (")[0]  # P1 23pp-SC, P2 23pp-SC, P3 pSERS
        raw_groups[short] = resample_rows(rec["x"], rec["Y"], common)
    for label, Y in block_groups_raw.items():
        raw_groups[label] = resample_rows(
            np.load(BLOCK_A_NPZ, allow_pickle=False)["axis"].astype(float),
            Y,
            common,
        )

    prepped = {k: preprocess(Y, version) for k, Y in raw_groups.items()}
    means = {k: Y.mean(axis=0) for k, Y in prepped.items()}

    within = {}
    for k, Y in prepped.items():
        ds, es = [], []
        for row in Y:
            d, e = distance(row, means[k])
            ds.append(d)
            es.append(e)
        within[k] = {
            "n": int(Y.shape[0]),
            "mean": float(np.mean(ds)),
            "min": float(np.min(ds)),
            "max": float(np.max(ds)),
            "mean_rmse": float(np.mean(es)),
        }

    prep_labels = ["P1 23pp-SC", "P2 23pp-SC", "P3 pSERS", "block A 4NP/Ag"]
    prep_pairs = []
    for a, b in combinations(prep_labels, 2):
        d, e = distance(means[a], means[b])
        prep_pairs.append((a, b, d, e))

    substrate_pairs = []
    for a, b in SUBSTRATE_PAIRS:
        d, e = distance(means[a], means[b])
        substrate_pairs.append((a, b, d, e))

    chemical_pairs = []
    for a, b in CHEMICAL_PAIRS:
        d, e = distance(means[a], means[b])
        chemical_pairs.append((a, b, d, e))

    prep10 = next(p for p in prep_pairs if p[0] == "P1 23pp-SC" and p[1] == "P2 23pp-SC")
    spc_only = [p for p in prep_pairs if p[0] != "block A 4NP/Ag"
                and p[1] != "block A 4NP/Ag"]

    prep_all = summarize(prep_pairs)
    sub = summarize(substrate_pairs)
    chem = summarize(chemical_pairs)
    return {
        "version": version,
        "prepped": prepped,
        "means": means,
        "within": within,
        "prep_pairs": prep_pairs,
        "prep_all": prep_all,
        "prep_23pp": {"d": prep10[2], "rmse": prep10[3]},
        "prep_spc_only": summarize(spc_only),
        "substrate": sub,
        "chemical": chem,
        "ratio_substrate": sub["mean"] / prep_all["mean"],
        "ratio_chemical": chem["mean"] / prep_all["mean"],
        "ratio_substrate_23pp": sub["mean"] / prep10[2],
        "ratio_chemical_23pp": chem["mean"] / prep10[2],
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(
    spc_records: dict,
    common: np.ndarray,
    res: dict,
    alt: dict,
    block_axis: np.ndarray,
) -> None:
    raw_p1 = spc_records["P1 23pp-SC (Ag-23pp-SC-4NP)"]
    p1_prepped = res["prepped"]["P1 23pp-SC"]

    colors = {
        "P1 23pp-SC": "#1f77b4",
        "P2 23pp-SC": "#ff7f0e",
        "P3 pSERS": "#2ca02c",
        "block A 4NP/Ag": "#d62728",
        "block A 4NP/PICO": "#ff7f0e",
        "block A 4NP/pSERS": "#2ca02c",
        "block A BT/Ag": "#9467bd",
        "block A Py/Ag": "#8c564b",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8))

    # (a) one parsed raw spectrum, full native SPC range.
    ax = axes[0, 0]
    ax.plot(raw_p1["x"], raw_p1["Y"][0], color="0.2", lw=0.8)
    ax.set_title("(a) Parsed SPC validation: Ag-23pp-SC-4NP-R1")
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("intensity (counts)")
    ax.text(
        0.03,
        0.95,
        f"{raw_p1['n_points']} points\n"
        f"{raw_p1['x_first']:.0f}–{raw_p1['x_last']:.0f} cm$^{{-1}}$\n"
        "float32 y (0x80)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(fc="white", ec="0.7", alpha=0.85),
    )

    # (b) five scans of the same sample, min-max.
    ax = axes[0, 1]
    for i in range(p1_prepped.shape[0]):
        ax.plot(common, p1_prepped[i], lw=0.8, alpha=0.85, label=f"R{i + 1}")
    ax.set_title("(b) Five scans of P1 after per-spectrum min–max")
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("min–max intensity")
    ax.legend(ncol=5, fontsize=7, frameon=False)
    corr = raw_p1["scan_corr"]
    ax.text(
        0.03,
        0.95,
        f"pairwise r\n{min(corr):.5f}–{max(corr):.5f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(fc="white", ec="0.7", alpha=0.85),
    )

    # (c) four preparations of 4-NP on silver.
    ax = axes[0, 2]
    for label in ["P1 23pp-SC", "P2 23pp-SC", "P3 pSERS", "block A 4NP/Ag"]:
        ax.plot(common, res["means"][label], lw=1.0, label=label,
                color=colors[label])
    ax.set_title("(c) 4-NP on silver: four preparation means")
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("min–max intensity")
    ax.legend(fontsize=7, frameon=False)

    # (d) three substrates, 4-NP.
    ax = axes[1, 0]
    for label in ["block A 4NP/Ag", "block A 4NP/PICO", "block A 4NP/pSERS"]:
        ax.plot(common, res["means"][label], lw=1.0, label=label,
                color=colors[label])
    ax.set_title("(d) Block A 4-NP mean spectra by substrate")
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("min–max intensity")
    ax.legend(fontsize=8, frameon=False)

    # (e) three chemicals on Ag.
    ax = axes[1, 1]
    for label in ["block A 4NP/Ag", "block A BT/Ag", "block A Py/Ag"]:
        ax.plot(common, res["means"][label], lw=1.0, label=label,
                color=colors[label])
    ax.set_title("(e) Block A mean spectra on Ag by chemical")
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("min–max intensity")
    ax.legend(fontsize=8, frameon=False)

    # (f) distance summary, min-max and arPLS side by side, log scale.
    ax = axes[1, 2]
    labels = [
        "within\nSPC prep",
        "within\nblkA 4NP/Ag",
        "between\nprep (all 6)",
        "23pp-SC\npair",
        "between\nsubstrate",
        "between\nchemical",
    ]
    spc_within = np.mean(
        [res["within"][k]["mean"] for k in ["P1 23pp-SC", "P2 23pp-SC", "P3 pSERS"]]
    )
    vals_mm = [
        spc_within,
        res["within"]["block A 4NP/Ag"]["mean"],
        res["prep_all"]["mean"],
        res["prep_23pp"]["d"],
        res["substrate"]["mean"],
        res["chemical"]["mean"],
    ]
    spc_within_alt = np.mean(
        [alt["within"][k]["mean"] for k in ["P1 23pp-SC", "P2 23pp-SC", "P3 pSERS"]]
    )
    vals_alt = [
        spc_within_alt,
        alt["within"]["block A 4NP/Ag"]["mean"],
        alt["prep_all"]["mean"],
        alt["prep_23pp"]["d"],
        alt["substrate"]["mean"],
        alt["chemical"]["mean"],
    ]
    xpos = np.arange(len(labels))
    w = 0.38
    ax.bar(xpos - w / 2, vals_mm, w, label="min–max", color="#1f77b4")
    ax.bar(xpos + w / 2, vals_alt, w, label="arPLS + min–max", color="#ff7f0e")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("1 − Pearson (log scale)")
    ax.set_title("(f) Distance summary")
    ax.legend(fontsize=8, frameon=False)
    for i, (a, b) in enumerate(zip(vals_mm, vals_alt)):
        ax.text(i - w / 2, a * 1.15, f"{a:.3g}", ha="center", fontsize=6.5,
                rotation=90, va="bottom")
        ax.text(i + w / 2, b * 1.15, f"{b:.3g}", ha="center", fontsize=6.5,
                rotation=90, va="bottom")
    ax.set_ylim(1e-4, 3.0)

    fig.suptitle(
        "4-nitrophenol add-on: repeat preparations, substrates and chemicals"
        " (block A common axis 300–2447 cm$^{-1}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def write_report(
    spc_records: dict,
    common: np.ndarray,
    block_axis: np.ndarray,
    res: dict,
    alt: dict,
) -> None:
    spc_lab = "P1 23pp-SC (Ag-23pp-SC-4NP)"
    rec = spc_records[spc_lab]
    corr_all = np.concatenate([r["scan_corr"] for r in spc_records.values()])

    def row(label: str, d: float, rmse: float, bold: bool = False,
            nd: int = 4) -> str:
        if bold:
            return f"| **{label}** | **{fmt(d, nd)} ({fmt(rmse)})** |"
        return f"| {label} | {fmt(d, nd)} ({fmt(rmse)}) |"

    pp = {(a, b): (d, e) for a, b, d, e in res["prep_pairs"]}
    sp = {(a, b): (d, e) for a, b, d, e in res["substrate"]["pairs"]}
    ch = {(a, b): (d, e) for a, b, d, e in res["chemical"]["pairs"]}

    other_keys = ["block A 4NP/PICO", "block A 4NP/pSERS",
                  "block A BT/Ag", "block A Py/Ag"]
    other_d = [res["within"][k]["mean"] for k in other_keys]
    other_e = [res["within"][k]["mean_rmse"] for k in other_keys]
    other_text = (
        f"{min(other_d):.4f}–{max(other_d):.4f} "
        f"({min(other_e):.3f}–{max(other_e):.3f})"
    )

    md = f"""# 4-NP add-on: chip-to-chip vs substrate and chemical variation

**Data and axis.** 15 SPC scans (3 samples × R1–R5) parsed unambiguously:
2201 points, **300–2500 cm⁻¹**, 1 cm⁻¹, float32 y; the five scans of each sample
agree (overall pairwise r = {corr_all.min():.6f}–{corr_all.max():.6f}); `avg` files were
ignored. Block A is {block_axis[0]:.0f}–{block_axis[-1]:.0f} cm⁻¹, so the `.spc` range **does not
match** (SPC {rec['x_first']:.0f}–{rec['x_last']:.0f} vs {block_axis[0]:.0f}–{block_axis[-1]:.0f}); a different
instrument/session is likely. The common axis is the intersection,
**{common[0]:.0f}–{common[-1]:.0f} cm⁻¹ ({len(common)} points), no extrapolation**. Primary
preprocessing: per-spectrum min–max; distance is 1 − Pearson between mean
spectra (RMSE on the same min–max scale). 632 nm only.

## Distance table (primary, min–max)

| Comparison | d (RMSE) |
|---|---:|
| **Within preparation** | |
{row("P1 23pp-SC, 5 scans", res['within']['P1 23pp-SC']['mean'], res['within']['P1 23pp-SC']['mean_rmse'], nd=5)}
{row("P2 23pp-SC, 5 scans", res['within']['P2 23pp-SC']['mean'], res['within']['P2 23pp-SC']['mean_rmse'], nd=5)}
{row("P3 pSERS, 5 scans", res['within']['P3 pSERS']['mean'], res['within']['P3 pSERS']['mean_rmse'], nd=5)}
{row("block A 4NP/Ag, 25 spots", res['within']['block A 4NP/Ag']['mean'], res['within']['block A 4NP/Ag']['mean_rmse'])}
| other block A maps, 25 spots each (4NP/PICO, 4NP/pSERS, BT/Ag, Py/Ag) | {other_text} |
| **Between preparations, 4-NP on silver** | |
{row("P1–P2 (both 23pp-SC)", *pp[("P1 23pp-SC", "P2 23pp-SC")])}
{row("P1–P3", *pp[("P1 23pp-SC", "P3 pSERS")])}
{row("P2–P3", *pp[("P2 23pp-SC", "P3 pSERS")])}
{row("P1–P4 (block A 4NP/Ag)", *pp[("P1 23pp-SC", "block A 4NP/Ag")])}
{row("P2–P4", *pp[("P2 23pp-SC", "block A 4NP/Ag")])}
{row("P3–P4", *pp[("P3 pSERS", "block A 4NP/Ag")])}
{row("mean, all six", res['prep_all']['mean'], res['prep_all']['mean_rmse'], bold=True)}
| **Between substrates, 4-NP** | |
{row("Ag–PICO", *sp[("block A 4NP/Ag", "block A 4NP/PICO")])}
{row("Ag–pSERS", *sp[("block A 4NP/Ag", "block A 4NP/pSERS")])}
{row("PICO–pSERS", *sp[("block A 4NP/PICO", "block A 4NP/pSERS")])}
{row("mean", res['substrate']['mean'], res['substrate']['mean_rmse'], bold=True)}
| **Between chemicals, Ag** | |
{row("4-NP–benzenethiol", *ch[("block A 4NP/Ag", "block A BT/Ag")])}
{row("4-NP–pyridine", *ch[("block A 4NP/Ag", "block A Py/Ag")])}
{row("benzenethiol–pyridine", *ch[("block A BT/Ag", "block A Py/Ag")])}
{row("mean", res['chemical']['mean'], res['chemical']['mean_rmse'], bold=True)}

## Ratios

- **between-substrate / between-preparation = {res['substrate']['mean']:.4f} / {res['prep_all']['mean']:.4f} = {res['ratio_substrate']:.2f}**
- **between-chemical / between-preparation = {res['chemical']['mean']:.4f} / {res['prep_all']['mean']:.4f} = {res['ratio_chemical']:.2f}**

The denominator matters: the inclusive mean ({res['prep_all']['mean']:.4f}) gives
{res['ratio_substrate']:.2f} and {res['ratio_chemical']:.2f}; the largest preparation pair
({res['prep_all']['max']:.4f}) gives {res['substrate']['mean'] / res['prep_all']['max']:.2f} and
{res['chemical']['mean'] / res['prep_all']['max']:.2f}; the same-protocol 23pp-SC pair
(d = {res['prep_23pp']['d']:.5f}) gives {res['ratio_substrate_23pp']:.0f} and {res['ratio_chemical_23pp']:.0f}.

**Sensitivity.** arPLS + min–max (phase 2 settings) gives between-preparation
{alt['prep_all']['mean']:.3f}, 23pp pair {alt['prep_23pp']['d']:.4f}, between-substrate
{alt['substrate']['mean']:.3f} and between-chemical {alt['chemical']['mean']:.3f}; the ratios
become {alt['ratio_substrate']:.2f} and {alt['ratio_chemical']:.2f}.

## What this means

Substrate differences are larger than the mean between-preparation distance
({res['ratio_substrate']:.1f}× under min–max, {alt['ratio_substrate']:.1f}× after arPLS), but the mean
substrate distance is only {res['substrate']['mean'] / res['prep_all']['max']:.1f}× the largest preparation
pair because one substrate pair (Ag–pSERS) is small; the other two (Ag–PICO,
PICO–pSERS) exceed every preparation pair. Against the same-protocol 23pp-SC
repeat pair, substrate and chemical distances are far larger
({res['ratio_substrate_23pp']:.0f}× and {res['ratio_chemical_23pp']:.0f}×), so reproducibility within one protocol
is much tighter than the differences being classified. Chemical differences on
Ag are smaller than the inclusive preparation mean under min–max
({res['ratio_chemical']:.2f}×) because a shared broad background dominates, but arPLS raises them
to {alt['ratio_chemical']:.2f}×.

## Caveats

- Preparation methods are unknown: P1/P2 are `23pp-SC`, P3 is `psers`, P4 is the
  block A Ag map; the between-preparation number is an upper bound, not a clean
  replicate estimate.
- The `.spc` range differs from block A and the SPC spectra have a much stronger
  broad background, so an instrument/session difference is likely; the 300 cm⁻¹
  cutoff also excludes the dominant low-wavenumber background of the block A
  pSERS map, making min–max distances window-sensitive.
- Only one chemical (4-NP) has repeats; each substrate/chemical comparison uses
  one block A map. The five SPC scans may be repeated acquisitions, not
  independent spots, so within-preparation spread may be underestimated.
  Min–max retains background; the arPLS ratios differ substantially
  ({res['ratio_substrate']:.2f}→{alt['ratio_substrate']:.2f}, {res['ratio_chemical']:.2f}→{alt['ratio_chemical']:.2f}).

*Generated by `check_4np.py`; figure in `figure_4np.png`.*
"""
    RESULTS.write_text(md)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    try:
        spc_records = validate_spc()
    except (SpcFormatError, FileNotFoundError) as exc:
        print("\nSTOP: SPC file could not be parsed unambiguously.")
        print(f"  {exc}")
        print("No results were written.")
        return 1

    # Import atlas_sers only after validation, and without writing bytecode.
    if not (ATLAS_SRC / "atlas_sers").exists():
        print(f"STOP: atlas_sers source not found at {ATLAS_SRC}")
        return 1
    sys.path.insert(0, str(ATLAS_SRC))

    X, block_axis, block_groups_raw = load_block_a()

    # Common 1 cm-1 axis: intersection only.
    spc_x = spc_records["P1 23pp-SC (Ag-23pp-SC-4NP)"]["x"]
    lo = max(float(spc_x[0]), float(block_axis[0]))
    hi = min(float(spc_x[-1]), float(block_axis[-1]))
    common = np.arange(np.ceil(lo - 1e-9), np.floor(hi + 1e-9) + 1.0, 1.0)

    print("\n" + "=" * 78)
    print("COMMON AXIS")
    print("=" * 78)
    print(f"SPC range      : {spc_x[0]:.1f}–{spc_x[-1]:.1f} cm-1 ({len(spc_x)} pts)")
    print(f"Block A range  : {block_axis[0]:.1f}–{block_axis[-1]:.1f} cm-1 "
          f"({len(block_axis)} pts)")
    print(f"Intersection   : {common[0]:.1f}–{common[-1]:.1f} cm-1 ({len(common)} pts)")
    print("Ranges match   :", bool(
        np.isclose(spc_x[0], block_axis[0]) and np.isclose(spc_x[-1], block_axis[-1])
    ))

    res = analyze(spc_records, block_groups_raw, common, "minmax")
    alt = analyze(spc_records, block_groups_raw, common, "arpls")

    def show(name: str, r: dict) -> None:
        print("\n" + "=" * 78)
        print(f"DISTANCES: {name}")
        print("=" * 78)
        for k, v in r["within"].items():
            print(f"  within {k:24s} d={v['mean']:.5f} "
                  f"[{v['min']:.5f}, {v['max']:.5f}]  rmse={v['mean_rmse']:.4f}")
        print("  between preparations (4-NP on silver):")
        for a, b, d, e in r["prep_pairs"]:
            print(f"    {a:22s} – {b:22s} d={d:.5f}  rmse={e:.4f}")
        print(f"    mean all six d={r['prep_all']['mean']:.5f}; "
              f"23pp-SC pair d={r['prep_23pp']['d']:.5f}")
        print("  between substrates (4-NP):")
        for a, b, d, e in r["substrate"]["pairs"]:
            print(f"    {a:22s} – {b:22s} d={d:.5f}  rmse={e:.4f}")
        print(f"    mean d={r['substrate']['mean']:.5f}")
        print("  between chemicals (Ag):")
        for a, b, d, e in r["chemical"]["pairs"]:
            print(f"    {a:22s} – {b:22s} d={d:.5f}  rmse={e:.4f}")
        print(f"    mean d={r['chemical']['mean']:.5f}")
        print(f"  ratios: substrate/prep={r['ratio_substrate']:.2f}  "
              f"chemical/prep={r['ratio_chemical']:.2f}  "
              f"(vs 23pp: {r['ratio_substrate_23pp']:.0f}, "
              f"{r['ratio_chemical_23pp']:.0f})")

    show("min–max (primary)", res)
    show("arPLS + min–max (sensitivity)", alt)

    make_figure(spc_records, common, res, alt, block_axis)
    write_report(spc_records, common, block_axis, res, alt)

    print("\n" + "=" * 78)
    print("WROTE")
    print("=" * 78)
    print(f"  {FIGURE}")
    print(f"  {RESULTS}")
    print(f"\nRESULTS_4NP.md: {RESULTS}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        # Do not leave a matplotlib cache inside the deliverable folder.
        import shutil

        shutil.rmtree(HERE / ".mplconfig", ignore_errors=True)
