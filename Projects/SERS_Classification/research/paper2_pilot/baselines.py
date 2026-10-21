#!/usr/bin/env python
"""Phase 2: chemometric baselines for the Paper 2 pilot.

No neural networks, CPU only, fixed seeds.  All scaling / PCA / models are fit
inside training folds; each map is kept entirely in train or test.

Experiments
    E1  reference recovery with NMF and MCR-ALS (k, k-1, k+1)
    E2  variance decomposition of map-mean spectra
    E3  held-out cell prediction (additive, other substrate, other analyte)
    E4  leave-one-substrate-out / leave-one-laser-out analyte classification

Run from this folder with the project venv:
    .../.venv/bin/python baselines.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.signal import find_peaks
from sklearn.decomposition import NMF, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "atlas_sers", "src")))
from atlas_sers.preprocessing.representations import arpls_correct, despike  # noqa: E402

OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 0
VERSIONS = ("minmax", "arpls")
VERSION_LABEL = {"minmax": "min–max", "arpls": "arPLS + min–max"}
BLOCKS = ("A", "B")
K_GRID = {"A": [6, 7, 8], "B": [4, 5, 6]}
K_NOMINAL = {"A": 7, "B": 5}          # analytes + substrates
SUPP_K = {"A": [], "B": [7]}          # B design has 4 substrate x laser cells
PREF_LASER = {"A": {"4-nitrophenol": ["632"],
                    "benzenethiol": ["unknown"],
                    "pyridine": ["unknown"]},
              "B": {"4-mercaptobenzoic acid": ["632", "785"],
                    "2,6-pyridinecarboxylic acid": ["632", "785"],
                    "4-mercaptobenzonitrile": ["632", "785"]}}
CODE = {"4-nitrophenol": "4NP", "benzenethiol": "BT", "pyridine": "Py",
        "4-mercaptobenzoic acid": "MBA", "2,6-pyridinecarboxylic acid": "PDCA",
        "4-mercaptobenzonitrile": "MBN"}
ANALYTES = {"A": ["4-nitrophenol", "benzenethiol", "pyridine"],
            "B": ["4-mercaptobenzoic acid", "2,6-pyridinecarboxylic acid",
                  "4-mercaptobenzonitrile"]}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# Loading and preprocessing
# --------------------------------------------------------------------------- #
def load_block(block: str):
    d = np.load(os.path.join(OUT, f"spectra_block{block}.npz"), allow_pickle=False)
    meta = pd.DataFrame({k: d[k] for k in
                         ["analyte", "substrate", "laser", "map_id", "map_label",
                          "y", "x", "path"]})
    return d["X"].astype(float), d["axis"].astype(float), meta


def minmax_rows(X: np.ndarray) -> np.ndarray:
    lo = X.min(axis=1, keepdims=True)
    span = X.max(axis=1, keepdims=True) - lo
    return (X - lo) / np.maximum(span, np.finfo(float).eps)


def preprocess(X: np.ndarray, version: str) -> np.ndarray:
    if version == "minmax":
        out = minmax_rows(X)
    elif version == "arpls":
        clean = despike(X, window=5, threshold=10.0)
        corrected = arpls_correct(clean, lam=100000.0, iterations=12,
                                  relative_tolerance=0.001, logistic_clip=60.0)
        out = minmax_rows(corrected)
    else:
        raise ValueError(version)
    return np.clip(out, 0.0, None)


def map_means(X: np.ndarray, meta: pd.DataFrame):
    rows, mats = [], []
    for map_id, idx in meta.groupby("map_id", sort=False).groups.items():
        idx = np.asarray(idx)
        first = meta.loc[idx[0]]
        rows.append({"map_id": map_id, "analyte": first["analyte"],
                     "substrate": first["substrate"], "laser": first["laser"],
                     "n_spots": len(idx)})
        mats.append(X[idx].mean(axis=0))
    return pd.DataFrame(rows), np.vstack(mats)


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def minmax_vec(y: np.ndarray) -> np.ndarray:
    lo, hi = float(y.min()), float(y.max())
    return (y - lo) / max(hi - lo, 1e-12)


def group_r2(Y: np.ndarray, labels: np.ndarray) -> float:
    grand = Y.mean(axis=0)
    ss_tot = float(((Y - grand) ** 2).sum())
    ss_res = 0.0
    for level in np.unique(labels):
        m = labels == level
        ss_res += float(((Y[m] - Y[m].mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def design_matrix(rows: pd.DataFrame, factors: list[str]) -> np.ndarray:
    cols = [np.ones(len(rows))]
    for factor in factors:
        levels = sorted(rows[factor].unique().tolist())
        for level in levels[1:]:
            cols.append((rows[factor].to_numpy() == level).astype(float))
    return np.column_stack(cols)


def additive_r2(Y: np.ndarray, rows: pd.DataFrame, factors: list[str]) -> float:
    D = design_matrix(rows, factors)
    coef = np.linalg.lstsq(D, Y, rcond=None)[0]
    resid = Y - D @ coef
    grand = Y.mean(axis=0)
    return 1.0 - float((resid ** 2).sum()) / float(((Y - grand) ** 2).sum())


# --------------------------------------------------------------------------- #
# NMF and MCR-ALS
# --------------------------------------------------------------------------- #
def fit_nmf(X: np.ndarray, k: int):
    model = NMF(n_components=k, init="nndsvda", random_state=SEED,
                max_iter=2000, tol=1e-4)
    W = model.fit_transform(X)
    return W, model.components_.copy()


def mcr_als(X: np.ndarray, S0: np.ndarray, max_iter: int = 50,
            tol: float = 1e-6):
    """Alternating non-negative least squares, initialized from NMF spectra."""
    n, p = X.shape
    k = S0.shape[0]
    S = np.maximum(S0, 0.0).copy()
    S /= np.maximum(S.max(axis=1, keepdims=True), np.finfo(float).eps)
    C = np.zeros((n, k))
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        for i in range(n):
            C[i] = nnls(S.T, X[i])[0]
        S_new = np.empty_like(S)
        for j in range(p):
            S_new[:, j] = nnls(C, X[:, j])[0]
        scale = S_new.max(axis=1, keepdims=True)
        scale[scale <= np.finfo(float).eps] = 1.0
        S_new /= scale
        C *= scale[:, 0]
        delta = np.linalg.norm(S_new - S) / max(np.linalg.norm(S), 1e-12)
        S = S_new
        if delta < tol:
            break
    return C, S, n_iter


# --------------------------------------------------------------------------- #
# References and peaks
# --------------------------------------------------------------------------- #
def load_references():
    d = np.load(os.path.join(OUT, "references.npz"), allow_pickle=False)
    meta = pd.DataFrame({k: d[k] for k in ["analyte", "laser", "map_id", "path"]})
    return d["X"].astype(float), d["axis"].astype(float), meta


def reference_mean(ref_X, ref_axis, ref_meta, analyte, laser):
    m = (ref_meta["analyte"].to_numpy() == analyte) & \
        (ref_meta["laser"].to_numpy() == laser)
    if not np.any(m):
        return None
    return minmax_rows(ref_X[m]).mean(axis=0)


def reference_mean_version(ref_X, ref_axis, ref_meta, analyte, laser, version):
    """Reference mean after the same per-spectrum preprocessing as the SERS data."""
    m = (ref_meta["analyte"].to_numpy() == analyte) & \
        (ref_meta["laser"].to_numpy() == laser)
    if not np.any(m):
        return None
    return preprocess(ref_X[m], version).mean(axis=0)


def align_reference(block_axis, ref_axis, y):
    lo = max(block_axis[0], ref_axis[0])
    hi = min(block_axis[-1], ref_axis[-1])
    mask = (block_axis >= lo) & (block_axis <= hi)
    return np.interp(block_axis[mask], ref_axis, y), mask


def top_reference_peaks(y, axis, n=8):
    idx, _ = find_peaks(y, distance=5)
    if len(idx) == 0:
        return np.array([])
    idx = idx[np.argsort(y[idx])[::-1][:n]]
    return axis[idx]


def component_peaks(y, axis):
    span = float(y.max() - y.min())
    if span <= 0:
        return np.array([])
    idx, _ = find_peaks(y, prominence=0.02 * span, distance=5)
    return axis[idx]


def peak_match_fraction(ref_peaks, comp_peaks, tol=10.0):
    if len(ref_peaks) == 0:
        return 0, 0, float("nan")
    matched = int(sum(np.any(np.abs(comp_peaks - p) <= tol)
                      for p in ref_peaks)) if len(comp_peaks) else 0
    return matched, len(ref_peaks), matched / len(ref_peaks)


# --------------------------------------------------------------------------- #
# E1
# --------------------------------------------------------------------------- #
def run_e1(data, ref_X, ref_axis, ref_meta):
    rows = []
    for block in BLOCKS:
        axis = data[block]["axis"]
        for version in VERSIONS:
            X = data[block]["X"][version]
            meta = data[block]["meta"]

            # Do-nothing baseline and reference caches, computed once per analyte/ref.
            cache = {}
            for analyte in ANALYTES[block]:
                sers_mean = X[meta["analyte"].to_numpy() == analyte].mean(axis=0)
                for ref_laser in PREF_LASER[block][analyte]:
                    ref_y = reference_mean_version(ref_X, ref_axis, ref_meta,
                                                   analyte, ref_laser, version)
                    if ref_y is None:
                        continue
                    rv, mask = align_reference(axis, ref_axis, ref_y)
                    ref_peaks = top_reference_peaks(rv, axis[mask])
                    base_r = corr(rv, sers_mean[mask])
                    cache[(analyte, ref_laser)] = (rv, mask, ref_peaks)
                    rows.append({
                        "block": block, "preprocessing": version,
                        "experiment": "E1", "method": "do-nothing", "k": -1,
                        "analyte": analyte, "ref_laser": ref_laser,
                        "corr": base_r, "abs_corr": abs(base_r),
                        "best_component": -1, "peaks_matched": 0,
                        "peaks_total": len(ref_peaks), "peak_fraction": float("nan"),
                        "baseline_corr": base_r,
                    })

            for k in K_GRID[block] + SUPP_K[block]:
                _, H_nmf = fit_nmf(X, k)
                _, H_mcr, _ = mcr_als(X, H_nmf)
                for (analyte, ref_laser), (rv, mask, ref_peaks) in cache.items():
                    for method, H in [("NMF", H_nmf), ("MCR-ALS", H_mcr)]:
                        comps = [h[mask] for h in H]
                        rs = np.array([corr(rv, h) for h in comps])
                        # NMF/MCR components are non-negative, so a negative
                        # correlation is anti-correlation, not recovery: take
                        # the best positive correlation.
                        best = int(np.nanargmax(rs)) if np.any(np.isfinite(rs)) else -1
                        if best >= 0:
                            matched, total, frac = peak_match_fraction(
                                ref_peaks, component_peaks(comps[best], axis[mask]))
                        else:
                            matched, total, frac = 0, len(ref_peaks), float("nan")
                        rows.append({
                            "block": block, "preprocessing": version,
                            "experiment": "E1", "method": method, "k": k,
                            "analyte": analyte, "ref_laser": ref_laser,
                            "corr": float(rs[best]) if best >= 0 else float("nan"),
                            "abs_corr": float(abs(rs[best])) if best >= 0 else float("nan"),
                            "best_component": best,
                            "peaks_matched": matched, "peaks_total": total,
                            "peak_fraction": frac,
                            "baseline_corr": float("nan"),
                        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# E2
# --------------------------------------------------------------------------- #
def run_e2(data):
    rows = []
    for block in BLOCKS:
        for version in VERSIONS:
            mm, Y = map_means(data[block]["X"][version], data[block]["meta"])
            factors = ["analyte", "substrate"] + (["laser"] if block == "B" else [])
            r_analyte = group_r2(Y, mm["analyte"].to_numpy())
            r_substrate = group_r2(Y, mm["substrate"].to_numpy())
            r_laser = group_r2(Y, mm["laser"].to_numpy()) if block == "B" else float("nan")
            r_add = additive_r2(Y, mm, factors)
            r_full = group_r2(Y, mm["map_id"].to_numpy())
            rows.append({
                "block": block, "preprocessing": version, "experiment": "E2",
                "n_maps": len(mm), "r2_analyte": r_analyte,
                "r2_substrate": r_substrate, "r2_laser": r_laser,
                "r2_additive": r_add, "r2_full_cell": r_full,
                "interaction": r_full - r_add,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# E3
# --------------------------------------------------------------------------- #
def run_e3(data):
    rows = []
    for block in BLOCKS:
        for version in VERSIONS:
            mm, Y = map_means(data[block]["X"][version], data[block]["meta"])
            factors = ["analyte", "substrate"] + (["laser"] if block == "B" else [])
            D = design_matrix(mm, factors)
            for i in range(len(mm)):
                train = np.arange(len(mm)) != i
                # All factor levels must remain represented after the hold-out.
                for factor in factors:
                    assert set(mm.loc[train, factor]) == set(mm[factor]), \
                        f"level lost when holding out {mm.loc[i, 'map_id']}"
                coef = np.linalg.lstsq(D[train], Y[train], rcond=None)[0]
                pred_add = D[i] @ coef

                test = mm.loc[i]
                if block == "B":
                    m_b = ((mm["analyte"] == test["analyte"]) &
                           (mm["substrate"] != test["substrate"]) &
                           (mm["laser"] == test["laser"])).to_numpy()
                    m_c = ((mm["substrate"] == test["substrate"]) &
                           (mm["analyte"] != test["analyte"]) &
                           (mm["laser"] == test["laser"])).to_numpy()
                else:
                    m_b = ((mm["analyte"] == test["analyte"]) &
                           (mm["substrate"] != test["substrate"])).to_numpy()
                    m_c = ((mm["substrate"] == test["substrate"]) &
                           (mm["analyte"] != test["analyte"])).to_numpy()
                pred_b = Y[m_b].mean(axis=0) if m_b.any() else np.full(Y.shape[1], np.nan)
                pred_c = Y[m_c].mean(axis=0) if m_c.any() else np.full(Y.shape[1], np.nan)
                actual = Y[i]
                for method, pred, n_donors in [
                        ("additive", pred_add, int(train.sum())),
                        ("same analyte / other substrate", pred_b, int(m_b.sum())),
                        ("same substrate / other analyte", pred_c, int(m_c.sum()))]:
                    rows.append({
                        "block": block, "preprocessing": version,
                        "experiment": "E3", "method": method,
                        "held_out": test["map_id"], "analyte": test["analyte"],
                        "substrate": test["substrate"], "laser": test["laser"],
                        "n_donors": n_donors,
                        "corr": corr(actual, pred) if np.all(np.isfinite(pred)) else float("nan"),
                        "rmse": rmse(actual, pred) if np.all(np.isfinite(pred)) else float("nan"),
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# E4
# --------------------------------------------------------------------------- #
def make_folds(meta: pd.DataFrame, scheme: str):
    if scheme == "LOSO":
        groups = meta["substrate"].to_numpy()
    elif scheme == "LOLO":
        groups = meta["laser"].to_numpy()
    else:
        raise ValueError(scheme)
    folds = []
    for g in sorted(np.unique(groups)):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        folds.append((str(g), train, test))
    return folds


def run_e4(data):
    rows = []
    for block in BLOCKS:
        schemes = ["LOSO"] if block == "A" else ["LOSO", "LOLO"]
        for version in VERSIONS:
            X = data[block]["X"][version]
            meta = data[block]["meta"]
            y = meta["analyte"].to_numpy()
            for scheme in schemes:
                for fold_name, train, test in make_folds(meta, scheme):
                    models = {
                        "Random Forest": RandomForestClassifier(
                            n_estimators=500, random_state=SEED, n_jobs=-1),
                        "Extra Trees": ExtraTreesClassifier(
                            n_estimators=500, random_state=SEED, n_jobs=-1),
                        "PCA + LDA": Pipeline([
                            ("pca", PCA(n_components=20, random_state=SEED)),
                            ("lda", LinearDiscriminantAnalysis())]),
                    }
                    for model_name, model in models.items():
                        model.fit(X[train], y[train])
                        proba = model.predict_proba(X[test])
                        classes = model.classes_
                        pred_spot = classes[np.argmax(proba, axis=1)]
                        ba_spot = balanced_accuracy_score(y[test], pred_spot)

                        test_meta = meta.iloc[test].reset_index(drop=True)
                        map_true, map_pred = [], []
                        for map_id in test_meta["map_id"].unique():
                            m = (test_meta["map_id"] == map_id).to_numpy()
                            map_true.append(test_meta.loc[m, "analyte"].iloc[0])
                            map_pred.append(classes[np.argmax(proba[m].mean(axis=0))])
                        ba_map = balanced_accuracy_score(map_true, map_pred)

                        rows.append({
                            "block": block, "preprocessing": version,
                            "experiment": "E4", "scheme": scheme, "fold": fold_name,
                            "model": model_name,
                            "n_train_spots": len(train), "n_test_spots": len(test),
                            "n_train_maps": meta.iloc[train]["map_id"].nunique(),
                            "n_test_maps": test_meta["map_id"].nunique(),
                            "ba_spectrum": ba_spot, "ba_map": ba_map,
                        })
                        print(f"  E4 {block} {version:6s} {scheme} fold={fold_name:4s} "
                              f"{model_name:14s} BA spot={ba_spot:.3f} map={ba_map:.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_e1_corr(e1: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for i, block in enumerate(BLOCKS):
        for j, version in enumerate(VERSIONS):
            ax = axes[i, j]
            d = e1[(e1.block == block) & (e1.preprocessing == version) &
                   (e1.k == K_NOMINAL[block])]
            d_base = e1[(e1.block == block) & (e1.preprocessing == version) &
                        (e1.method == "do-nothing")]
            keys = e1[(e1.block == block) & (e1.preprocessing == version)][
                ["analyte", "ref_laser"]].drop_duplicates().values.tolist()
            labels = [f"{CODE[a]}\n{r}" for a, r in keys]
            x = np.arange(len(keys))
            width = 0.26
            for off, method in zip([-width, 0, width],
                                   ["do-nothing", "NMF", "MCR-ALS"]):
                source = d_base if method == "do-nothing" else d
                vals = [float(source[(source.analyte == a) & (source.ref_laser == r) &
                                     (source.method == method)]["abs_corr"].iloc[0])
                        for a, r in keys]
                ax.bar(x + off, vals, width, label=method)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f"Block {block} — {VERSION_LABEL[version]} (k={K_NOMINAL[block]})")
            ax.set_ylabel("|Pearson r| reference vs component")
            ax.grid(axis="y", alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_e1_components(data, e1, ref_X, ref_axis, ref_meta, outpath: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for i, block in enumerate(BLOCKS):
        axis = data[block]["axis"]
        X = data[block]["X"]["minmax"]
        meta = data[block]["meta"]
        k = K_NOMINAL[block]
        _, H = fit_nmf(X, k)
        for j, analyte in enumerate(ANALYTES[block]):
            ax = axes[i, j]
            d = e1[(e1.block == block) & (e1.preprocessing == "minmax") &
                   (e1.k == k) & (e1.method == "NMF") & (e1.analyte == analyte)]
            d = d.sort_values("abs_corr", ascending=False)
            best_row = d.iloc[0]
            ref_y = reference_mean(ref_X, ref_axis, ref_meta, analyte,
                                   best_row["ref_laser"])
            rv, mask = align_reference(axis, ref_axis, ref_y)
            h = H[int(best_row["best_component"])][mask]
            ax.plot(axis[mask], minmax_vec(h), lw=1.1,
                    label=f"NMF comp {int(best_row['best_component'])}")
            ax.plot(axis[mask], minmax_vec(rv), color="0.3", ls="--", lw=0.9,
                    label=f"ref ({best_row['ref_laser']})")
            ax.set_title(f"{CODE[analyte]}  r={best_row['corr']:.2f}, "
                         f"peaks {int(best_row['peaks_matched'])}/"
                         f"{int(best_row['peaks_total'])}", fontsize=9)
            ax.tick_params(labelsize=8)
            if j == 0:
                ax.set_ylabel("min–max intensity")
            if i == 1:
                ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("E1: best NMF component vs neat reference (min–max preprocessing)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_e2(e2: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, block in zip(axes, BLOCKS):
        d = e2[e2.block == block]
        factors = ["analyte", "substrate"] + (["laser"] if block == "B" else []) + \
                  ["additive", "full_cell"]
        labels = ["analyte", "substrate"] + (["laser"] if block == "B" else []) + \
                 ["additive", "full cells"]
        x = np.arange(len(factors))
        width = 0.38
        for off, version in zip([-width / 2, width / 2], VERSIONS):
            dd = d[d.preprocessing == version].iloc[0]
            vals = [dd[f"r2_{f}"] for f in factors]
            ax.bar(x + off, vals, width, label=VERSION_LABEL[version])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Block {block} ({int(d.n_maps.iloc[0])} map means)")
        ax.set_ylabel("fraction of variance explained")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_e3(e3: pd.DataFrame, outpath: str):
    methods = ["additive", "same analyte / other substrate",
               "same substrate / other analyte"]
    short = ["additive", "other substrate", "other analyte"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, block in zip(axes, BLOCKS):
        x = np.arange(len(methods))
        width = 0.38
        for off, version in zip([-width / 2, width / 2], VERSIONS):
            d = e3[(e3.block == block) & (e3.preprocessing == version)]
            vals = [d[d.method == m]["corr"].mean() for m in methods]
            ax.bar(x + off, vals, width, label=VERSION_LABEL[version])
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Block {block}: held-out cell mean prediction")
        ax.set_ylabel("mean Pearson r")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_e4(e4: pd.DataFrame, outpath: str):
    schemes = [("A", "LOSO", "Block A: leave-one-substrate-out"),
               ("B", "LOSO", "Block B: leave-one-substrate-out"),
               ("B", "LOLO", "Block B: leave-one-laser-out")]
    models = ["Random Forest", "Extra Trees", "PCA + LDA"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (block, scheme, title) in zip(axes, schemes):
        x = np.arange(len(models))
        width = 0.38
        for off, version in zip([-width / 2, width / 2], VERSIONS):
            d = e4[(e4.block == block) & (e4.scheme == scheme) &
                   (e4.preprocessing == version)]
            vals = [d[d.model == m]["ba_map"].mean() for m in models]
            ax.bar(x + off, vals, width, label=VERSION_LABEL[version])
        ax.axhline(1 / 3, color="k", ls=":", lw=1, label="chance (33%)")
        ax.set_xticks(x)
        ax.set_xticklabels(["RF", "ET", "PCA+LDA"], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("balanced accuracy (per map)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Loading blocks and fitting preprocessing …")
    data = {}
    ref_X, ref_axis, ref_meta = load_references()
    for block in BLOCKS:
        X, axis, meta = load_block(block)
        data[block] = {"axis": axis, "meta": meta, "X": {}}
        for version in VERSIONS:
            data[block]["X"][version] = preprocess(X, version)
        print(f"  Block {block}: {X.shape[0]} spectra x {X.shape[1]} shifts, "
              f"{meta['map_id'].nunique()} maps")

    print("E1: NMF and MCR-ALS reference recovery …")
    e1 = run_e1(data, ref_X, ref_axis, ref_meta)
    e1.to_csv(os.path.join(OUT, "e1_reference_recovery.csv"), index=False)

    print("E2: variance decomposition …")
    e2 = run_e2(data)
    e2.to_csv(os.path.join(OUT, "e2_variance.csv"), index=False)

    print("E3: held-out cell prediction …")
    e3 = run_e3(data)
    e3.to_csv(os.path.join(OUT, "e3_unseen.csv"), index=False)

    print("E4: classification …")
    e4 = run_e4(data)
    e4.to_csv(os.path.join(OUT, "e4_classification.csv"), index=False)

    e4_summary = (e4.groupby(["block", "preprocessing", "scheme", "model"])[
        ["ba_spectrum", "ba_map"]].agg(["mean", "std"]).reset_index())
    e4_summary.columns = ["_".join(c).strip("_") for c in e4_summary.columns]
    e4_summary.to_csv(os.path.join(OUT, "e4_summary.csv"), index=False)

    print("Writing figures …")
    plot_e1_corr(e1, os.path.join(FIG, "e1_reference_corr.png"))
    plot_e1_components(data, e1, ref_X, ref_axis, ref_meta,
                       os.path.join(FIG, "e1_best_components.png"))
    plot_e2(e2, os.path.join(FIG, "e2_variance.png"))
    plot_e3(e3, os.path.join(FIG, "e3_unseen.png"))
    plot_e4(e4, os.path.join(FIG, "e4_classification.png"))

    meta = {
        "seed": SEED,
        "k_grid": K_GRID,
        "k_nominal": K_NOMINAL,
        "supp_k": SUPP_K,
        "versions": list(VERSIONS),
        "nmf": {"init": "nndsvda", "max_iter": 2000, "tol": 1e-4},
        "mcr_als": {"max_iter": 50, "tol": 1e-6, "init": "NMF"},
        "despike": {"window": 5, "threshold": 10.0},
        "arpls": {"lam": 100000, "iterations": 12,
                  "relative_tolerance": 0.001, "logistic_clip": 60.0},
        "pca_lda_components": 20,
        "rf_et_trees": 500,
        "e1_reference_rule": "Block A: laser given per analyte (BT/Py unknown); "
                             "Block B: both 632 and 785 references evaluated.",
        "e1_reference_preprocessing": "same per-spectrum preprocessing as the SERS data",
        "e3_donor_rule": "Block B donor means matched the held-out laser.",
    }
    with open(os.path.join(OUT, "phase2_run.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
