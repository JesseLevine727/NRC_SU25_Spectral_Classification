#!/usr/bin/env python
"""Phase 3: VAE model ladder for the Paper 2 pilot.

Experiments
    E1  chemical part vs neat Raman reference and the phase-2 additive SERS signature
    E2  map-grouped logistic-regression probes, MIG (VAE / beta-VAE) and a swap test
    E3  held-out whole cells (CVAE and physics-structured VAE only)
    E4  leave-one-substrate-out / leave-one-laser-out analyte classification

Hard rules implemented here
    * only files under research/paper2_pilot/ are read or written;
    * the independent unit is one map: every map is wholly in train or test;
    * early stopping uses a random 10% of spots drawn from training maps only;
    * fixed configuration, seeds 0/1/2, no hyperparameter search;
    * test maps never enter training, validation or early stopping.

Run from this folder with the project venv:
    .../.venv/bin/python run_phase3.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mutual_info_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "atlas_sers", "src")))
from atlas_sers.preprocessing.representations import arpls_correct, despike  # noqa: E402

from vae_models import (  # noqa: E402
    BetaVAE, CVAE, ConvDecoder, PhysicsVAE, TrainConfig, VAE, count_parameters,
    physics_diagnostics, train_model, vae_diagnostics,
)

OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEEDS = [0, 1, 2]
BLOCKS = ["A", "B"]
PREPS = ["minmax", "arpls"]
VAE_MODELS = ["VAE", "beta-VAE", "CVAE", "PhysicsVAE"]
CHEM_MODELS = ["CVAE", "PhysicsVAE"]
BASELINE_MODELS = ["VAE", "beta-VAE", "CVAE", "PhysicsVAE"]

K_NOMINAL = {"A": 7, "B": 5}
PREF_LASER = {"A": {"4-nitrophenol": ["632"],
                    "benzenethiol": ["unknown"],
                    "pyridine": ["unknown"]},
              "B": {"4-mercaptobenzoic acid": ["632", "785"],
                    "2,6-pyridinecarboxylic acid": ["632", "785"],
                    "4-mercaptobenzonitrile": ["632", "785"]}}
CODE = {"4-nitrophenol": "4NP", "benzenethiol": "BT", "pyridine": "Py",
        "4-mercaptobenzoic acid": "MBA", "2,6-pyridinecarboxylic acid": "PDCA",
        "4-mercaptobenzonitrile": "MBN"}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Pandas4Warning.*")
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Data loading and preprocessing
# --------------------------------------------------------------------------- #
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
    return np.clip(out, 0.0, None).astype(np.float32)


@dataclass
class BlockData:
    block: str
    axis: np.ndarray
    meta: pd.DataFrame
    X_raw: np.ndarray
    X: dict
    analytes: list
    substrates: list
    lasers: list
    ya: np.ndarray
    ys: np.ndarray
    yl: np.ndarray
    cells: pd.DataFrame
    use_laser: bool

    @property
    def n_a(self):
        return len(self.analytes)

    @property
    def n_s(self):
        return len(self.substrates)

    @property
    def n_l(self):
        return len(self.lasers)


def load_block(block: str, preps=PREPS) -> BlockData:
    d = np.load(os.path.join(OUT, f"spectra_block{block}.npz"), allow_pickle=False)
    X_raw = d["X"].astype(float)
    meta = pd.DataFrame({k: d[k] for k in
                         ["analyte", "substrate", "laser", "map_id", "map_label",
                          "y", "x", "path"]})
    analytes = sorted(meta["analyte"].unique().tolist())
    substrates = sorted(meta["substrate"].unique().tolist())
    lasers = sorted(meta["laser"].unique().tolist())
    ya = np.array([analytes.index(v) for v in meta["analyte"]], dtype=np.int64)
    ys = np.array([substrates.index(v) for v in meta["substrate"]], dtype=np.int64)
    yl = np.array([lasers.index(v) for v in meta["laser"]], dtype=np.int64)
    cells = (meta.groupby("map_id", sort=True)
             .agg(analyte=("analyte", "first"), substrate=("substrate", "first"),
                  laser=("laser", "first"), n_spots=("analyte", "size"))
             .reset_index())
    X = {p: preprocess(X_raw, p) for p in preps}
    return BlockData(block=block, axis=d["axis"].astype(float), meta=meta,
                     X_raw=X_raw, X=X, analytes=analytes, substrates=substrates,
                     lasers=lasers, ya=ya, ys=ys, yl=yl, cells=cells,
                     use_laser=(block == "B"))


def load_references():
    d = np.load(os.path.join(OUT, "references.npz"), allow_pickle=False)
    meta = pd.DataFrame({k: d[k] for k in ["analyte", "laser", "map_id", "path"]})
    X = d["X"].astype(float)
    return X, d["axis"].astype(float), meta


def map_means(X: np.ndarray, meta: pd.DataFrame):
    rows, mats = [], []
    for map_id, idx in meta.groupby("map_id", sort=True).groups.items():
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


def sers_signature(X: np.ndarray, meta: pd.DataFrame, block: str) -> dict:
    """Phase-2 additive analyte effect: mean over cells of (cell mean - cell's
    mean over analytes).  Each cell is (substrate) in A and (substrate, laser)
    in B.  This is a baseline-derived target, not ground truth."""
    mm, Y = map_means(X, meta)
    groups = ["substrate"] + (["laser"] if block == "B" else [])
    sig = {}
    for analyte in sorted(mm["analyte"].unique()):
        effects = []
        group_cols = groups
        for key, idx in mm.groupby(group_cols, sort=True).groups.items():
            idx = np.asarray(idx)
            sub = mm.loc[idx]
            m_a = (sub["analyte"] == analyte).to_numpy()
            if not np.any(m_a):
                continue
            cell_mean = Y[idx[m_a]].mean(axis=0)
            other_mean = Y[idx[~m_a]].mean(axis=0) if np.any(~m_a) else cell_mean
            effects.append(cell_mean - other_mean)
        sig[analyte] = np.mean(np.vstack(effects), axis=0) if effects else None
    return sig


# --------------------------------------------------------------------------- #
# Fold construction
# --------------------------------------------------------------------------- #
def _levels_present(sub: pd.DataFrame, full: pd.DataFrame, block: str) -> bool:
    if set(sub["analyte"]) != set(full["analyte"]):
        return False
    if set(sub["substrate"]) != set(full["substrate"]):
        return False
    if block == "B" and set(sub["laser"]) != set(full["laser"]):
        return False
    return True


def make_grouped_folds(cells: pd.DataFrame, block: str, n_splits: int = 5,
                       seed: int = 0):
    """Deterministic map-grouped folds with every factor level in every train set."""
    maps = np.array(sorted(cells["map_id"].unique()))
    rng = np.random.RandomState(seed)
    for attempt in range(5000):
        order = rng.permutation(maps)
        folds = [list(order[k::n_splits]) for k in range(n_splits)]
        folds = [f for f in folds if f]
        ok = True
        for f in folds:
            train = cells[~cells["map_id"].isin(f)]
            if not _levels_present(train, cells, block):
                ok = False
                break
        if ok:
            return [(f"fold{k}", f) for k, f in enumerate(folds)]
    raise RuntimeError("could not build valid grouped folds")


def e4_folds(cells: pd.DataFrame, scheme: str):
    key = "substrate" if scheme == "LOSO" else "laser"
    out = []
    for level in sorted(cells[key].unique()):
        test = cells.loc[cells[key] == level, "map_id"].tolist()
        train = cells.loc[cells[key] != level, "map_id"].tolist()
        out.append((str(level), train, test))
    return out


def split_val(train_idx: np.ndarray, seed: int, frac: float = 0.10):
    """Random 10% of training spots; comes from training maps by construction."""
    rng = np.random.RandomState(seed + 7919)
    perm = rng.permutation(len(train_idx))
    n_val = max(1, int(round(frac * len(train_idx))))
    val = train_idx[perm[:n_val]]
    fit = train_idx[perm[n_val:]]
    return np.sort(fit), np.sort(val)


# --------------------------------------------------------------------------- #
# Global bookkeeping
# --------------------------------------------------------------------------- #
FIT_TIMES = []
FAILURES = []
DIAGNOSTICS = []
CHEM_PARTS = {}
SWAP_EXAMPLES = {}
RF_CACHE = {}


def add_failure(experiment, block, prep, model, seed, stage, reason, tag=""):
    FAILURES.append(dict(experiment=experiment, block=block, preprocessing=prep,
                         model=model, seed=seed, stage=stage, reason=str(reason),
                         tag=tag))
    print(f"  !! failure {experiment} {block} {prep} {model} seed={seed} "
          f"{stage}: {reason}")


def add_time(experiment, block, prep, model, seed, seconds, n_fits=1, tag=""):
    FIT_TIMES.append(dict(experiment=experiment, block=block, preprocessing=prep,
                          model=model, seed=seed, seconds=float(seconds),
                          n_fits=int(n_fits), tag=tag))


def indices_for_maps(data: BlockData, maps) -> np.ndarray:
    maps = set(maps)
    return np.flatnonzero(data.meta["map_id"].isin(maps).to_numpy())


def build_model(data: BlockData, model_name: str, prep: str,
                fit_idx: np.ndarray, seed: int):
    in_len = data.X[prep].shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model_name == "VAE":
        return VAE(in_len, latent_dim=8, beta=1.0)
    if model_name == "beta-VAE":
        return BetaVAE(in_len, latent_dim=8, beta=4.0)
    if model_name == "CVAE":
        if data.use_laser:
            return CVAE(in_len, dims=(4, 4, 2),
                        n_classes=(data.n_a, data.n_s, data.n_l),
                        use_laser=True, beta=1.0)
        return CVAE(in_len, dims=(4, 4), n_classes=(data.n_a, data.n_s),
                    use_laser=False, beta=1.0)
    if model_name == "PhysicsVAE":
        Xtr = data.X[prep][fit_idx]
        A = np.zeros((data.n_a, in_len), dtype=np.float32)
        B = np.zeros((data.n_s, in_len), dtype=np.float32)
        overall = Xtr.mean(axis=0)
        for k in range(data.n_a):
            m = data.ya[fit_idx] == k
            A[k] = Xtr[m].mean(axis=0) if np.any(m) else overall
        for k in range(data.n_s):
            m = data.ys[fit_idx] == k
            B[k] = Xtr[m].mean(axis=0) if np.any(m) else overall
        A = np.clip(A, 1e-3, None)
        B = np.clip(B, 1e-3, None)
        return PhysicsVAE(in_len, data.n_a, data.n_s, n_lasers=data.n_l,
                          use_laser=data.use_laser, init_A=A, init_B=B)
    raise ValueError(model_name)


def fit_model(data: BlockData, prep: str, model_name: str, train_maps,
              seed: int, experiment: str, tag: str = ""):
    """Fit one model on the given training maps; returns (model, fit info)."""
    t0 = time.time()
    train_idx = indices_for_maps(data, train_maps)
    fit_idx, val_idx = split_val(train_idx, seed)
    model = build_model(data, model_name, prep, fit_idx, seed).to(DEVICE)
    X_t = torch.tensor(data.X[prep], dtype=torch.float32, device=DEVICE)
    ya = torch.tensor(data.ya, dtype=torch.long, device=DEVICE)
    ys = torch.tensor(data.ys, dtype=torch.long, device=DEVICE)
    yl = torch.tensor(data.yl, dtype=torch.long, device=DEVICE)
    cfg = TrainConfig(seed=seed)
    try:
        res = train_model(model, X_t, ya, ys, yl,
                          torch.tensor(fit_idx, dtype=torch.long, device=DEVICE),
                          cfg, DEVICE,
                          val_idx=torch.tensor(val_idx, dtype=torch.long, device=DEVICE))
    except Exception as exc:  # noqa: BLE001
        add_failure(experiment, data.block, prep, model_name, seed,
                    "training", repr(exc), tag)
        add_time(experiment, data.block, prep, model_name, seed,
                 time.time() - t0, 1, tag)
        return None, dict(failure=repr(exc), seconds=time.time() - t0,
                          params=count_parameters(model))
    seconds = time.time() - t0
    add_time(experiment, data.block, prep, model_name, seed, seconds, 1, tag)
    info = dict(failure=res.failure, best_epoch=res.best_epoch,
                best_val=res.best_val, epochs_run=res.epochs_run,
                seconds=seconds, params=count_parameters(model),
                fit_idx=fit_idx, val_idx=val_idx, train_idx=train_idx)
    if res.failure:
        add_failure(experiment, data.block, prep, model_name, seed,
                    "training", res.failure, tag)
    # Collapse diagnostics on training maps only (descriptive, never on test maps).
    tr_t = torch.tensor(fit_idx, dtype=torch.long, device=DEVICE)
    try:
        if model.kind in ("vae", "cvae"):
            info["diagnostics"] = vae_diagnostics(model, X_t[tr_t])
        else:
            info["diagnostics"] = physics_diagnostics(model, X_t[tr_t], yl[tr_t])
    except Exception as exc:  # noqa: BLE001
        info["diagnostics"] = {"error": repr(exc)}
    collapsed = False
    diag = info["diagnostics"]
    if model.kind in ("vae", "cvae"):
        blocks = diag if model.kind == "vae" else {k: v for k, v in diag.items()}
        for name, val in blocks.items():
            if isinstance(val, dict) and "active_units" in val:
                if val["active_units"] == 0 or val.get("kl_total", 1.0) < 0.05:
                    collapsed = True
    elif isinstance(diag, dict) and "error" not in diag:
        # Confident p/q is fine; a single predicted class is collapse.
        if diag.get("n_p_classes", 99) <= 1 or diag.get("n_q_classes", 99) <= 1:
            collapsed = True
    info["collapsed"] = bool(collapsed)
    DIAGNOSTICS.append(dict(
        experiment=experiment, block=data.block, preprocessing=prep,
        model=model_name, seed=seed, tag=tag,
        best_epoch=info.get("best_epoch"), epochs_run=info.get("epochs_run"),
        best_val=info.get("best_val"), collapsed=bool(collapsed),
        diagnostics=json.dumps(diag, default=str)))
    return model, info


def tensors_for(data: BlockData, prep: str):
    X = torch.tensor(data.X[prep], dtype=torch.float32, device=DEVICE)
    ya = torch.tensor(data.ya, dtype=torch.long, device=DEVICE)
    ys = torch.tensor(data.ys, dtype=torch.long, device=DEVICE)
    yl = torch.tensor(data.yl, dtype=torch.long, device=DEVICE)
    return X, ya, ys, yl


# --------------------------------------------------------------------------- #
# E1: chemical part vs references and SERS signature
# --------------------------------------------------------------------------- #
def run_e1(data: dict, ref_X, ref_axis, ref_meta, e1_p2):
    rows = []
    for block in BLOCKS:
        d = data[block]
        for prep in PREPS:
            X = d.X[prep]
            sig = sers_signature(X, d.meta, block)
            # Phase-2 NMF baseline for the SERS-signature target.
            nmf = NMF(n_components=K_NOMINAL[block], init="nndsvda",
                      random_state=0, max_iter=2000, tol=1e-4)
            W = nmf.fit_transform(X)
            H = nmf.components_
            analyte_mean = {a: X[d.ya == k].mean(axis=0)
                            for k, a in enumerate(d.analytes)}
            for analyte in d.analytes:
                target = sig[analyte]
                rs = np.array([corr(target, h) for h in H])
                best = int(np.nanargmax(rs)) if np.any(np.isfinite(rs)) else -1
                rows.append(dict(
                    block=block, preprocessing=prep, model="NMF (phase 2)",
                    seed=-1, analyte=analyte, target="SERS signature",
                    ref_laser="", corr=float(rs[best]) if best >= 0 else np.nan,
                    abs_corr=float(abs(rs[best])) if best >= 0 else np.nan,
                    peaks_matched=np.nan, peaks_total=np.nan,
                    peak_fraction=np.nan,
                    note=f"best of k={K_NOMINAL[block]} NMF components vs signature"))
                rows.append(dict(
                    block=block, preprocessing=prep, model="analyte mean (phase 2)",
                    seed=-1, analyte=analyte, target="SERS signature",
                    ref_laser="", corr=corr(target, analyte_mean[analyte]),
                    abs_corr=abs(corr(target, analyte_mean[analyte])),
                    peaks_matched=np.nan, peaks_total=np.nan,
                    peak_fraction=np.nan,
                    note="analyte mean spectrum vs signature"))

            for model_name in CHEM_MODELS:
                for seed in SEEDS:
                    model, info = fit_model(d, prep, model_name, d.cells["map_id"],
                                            seed, "E1")
                    if model is None:
                        continue
                    X_t, _, _, yl_t = tensors_for(d, prep)
                    with torch.no_grad():
                        if model_name == "CVAE":
                            blocks = model.latent_blocks(X_t)
                            chem = {}
                            sub_mean = blocks["sub"].mean(dim=0, keepdim=True)
                            las_mean = (blocks.get("laser").mean(dim=0, keepdim=True)
                                        if d.use_laser else None)
                            for k, analyte in enumerate(d.analytes):
                                m = torch.tensor(d.ya == k, device=DEVICE)
                                zc = blocks["chem"][m].mean(dim=0, keepdim=True)
                                zd = {"chem": zc, "sub": sub_mean}
                                if d.use_laser:
                                    zd["laser"] = las_mean
                                chem[analyte] = model.decode_blocks(zd)[0].cpu().numpy()
                        else:
                            A = model.analyte_spectra().cpu().numpy()
                            chem = {a: A[k] for k, a in enumerate(d.analytes)}
                    for analyte in d.analytes:
                        CHEM_PARTS[(block, model_name, seed, analyte)] = chem[analyte]
                        # (a) neat Raman reference, phase-2 laser rules
                        for ref_laser in PREF_LASER[block][analyte]:
                            m = ((ref_meta["analyte"] == analyte) &
                                 (ref_meta["laser"] == ref_laser)).to_numpy()
                            if not np.any(m):
                                continue
                            ref_y = preprocess(ref_X[m], prep).mean(axis=0)
                            rv, mask = align_reference(d.axis, ref_axis, ref_y)
                            ref_peaks = top_reference_peaks(rv, d.axis[mask])
                            comp = chem[analyte][mask]
                            r = corr(rv, comp)
                            matched, total, frac = peak_match_fraction(
                                ref_peaks, component_peaks(comp, d.axis[mask]))
                            rows.append(dict(
                                block=block, preprocessing=prep, model=model_name,
                                seed=seed, analyte=analyte,
                                target="Raman reference", ref_laser=ref_laser,
                                corr=r, abs_corr=abs(r), peaks_matched=matched,
                                peaks_total=total, peak_fraction=frac,
                                note=f"early stop epoch {info.get('best_epoch')}"))
                        # (b) phase-2 additive SERS signature (baseline-derived)
                        r = corr(sig[analyte], chem[analyte])
                        rows.append(dict(
                            block=block, preprocessing=prep, model=model_name,
                            seed=seed, analyte=analyte, target="SERS signature",
                            ref_laser="", corr=r, abs_corr=abs(r),
                            peaks_matched=np.nan, peaks_total=np.nan,
                            peak_fraction=np.nan,
                            note=f"early stop epoch {info.get('best_epoch')}"))
                    del model
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

            # Phase-2 NMF / MCR / do-nothing reference rows, same target.
            for method, label in [("do-nothing", "do-nothing (phase 2)"),
                                  ("NMF", "NMF (phase 2)"),
                                  ("MCR-ALS", "MCR-ALS (phase 2)")]:
                sub = e1_p2[(e1_p2.block == block) &
                            (e1_p2.preprocessing == prep) &
                            (e1_p2.method == method)]
                if method != "do-nothing":
                    sub = sub[sub.k == K_NOMINAL[block]]
                for _, r0 in sub.iterrows():
                    rows.append(dict(
                        block=block, preprocessing=prep, model=label, seed=-1,
                        analyte=r0["analyte"], target="Raman reference",
                        ref_laser=r0["ref_laser"], corr=r0["corr"],
                        abs_corr=r0["abs_corr"],
                        peaks_matched=r0["peaks_matched"],
                        peaks_total=r0["peaks_total"],
                        peak_fraction=r0["peak_fraction"],
                        note="phase 2 value, same reference rule"))
    e1 = pd.DataFrame(rows)
    e1.to_csv(os.path.join(OUT, "phase3_e1.csv"), index=False)
    return e1


# --------------------------------------------------------------------------- #
# E2 helpers
# --------------------------------------------------------------------------- #
def fit_probe(Ztr, ytr, Zte, n_classes):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Ztr)
    Xte = scaler.transform(Zte)
    lr = LogisticRegression(max_iter=3000, C=1.0, random_state=0)
    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(Xte)
    full = np.zeros((len(Zte), n_classes), dtype=float)
    full[:, lr.classes_] = proba
    return full


def pooled_probe_scores(records):
    """Pool held-out folds, then compute BA per spectrum and per map."""
    out = {}
    for key, recs in records.items():
        P = np.vstack([r["proba"] for r in recs])
        y = np.concatenate([r["y"] for r in recs])
        maps = np.concatenate([r["maps"] for r in recs])
        pred = P.argmax(axis=1)
        ba_spot = float(balanced_accuracy_score(y, pred))
        map_true, map_pred = [], []
        for m in np.unique(maps):
            mm = maps == m
            map_true.append(int(y[mm][0]))
            map_pred.append(int(P[mm].mean(axis=0).argmax()))
        ba_map = float(balanced_accuracy_score(map_true, map_pred))
        fold_spot = [r["ba_spectrum"] for r in recs
                     if np.isfinite(r["ba_spectrum"])]
        fold_map = [r["ba_map"] for r in recs if np.isfinite(r["ba_map"])]
        out[key] = dict(ba_spectrum_pooled=ba_spot, ba_map_pooled=ba_map,
                        ba_spectrum_foldmean=float(np.mean(fold_spot))
                        if fold_spot else np.nan,
                        ba_map_foldmean=float(np.mean(fold_map))
                        if fold_map else np.nan,
                        n_folds=len(recs), n_spots=len(y),
                        n_maps=len(np.unique(maps)),
                        n_classes=len(np.unique(y)))
    return out


def mig_scores(Z: np.ndarray, labels: np.ndarray, n_bins: int = 8):
    """MIG for one categorical factor using quantile-binned latent dims."""
    if len(np.unique(labels)) < 2 or Z.shape[1] < 2:
        return dict(mig=np.nan, mi_top1=np.nan, mi_top2=np.nan, entropy=np.nan)
    Zb = np.zeros_like(Z, dtype=int)
    for j in range(Z.shape[1]):
        qs = np.quantile(Z[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        Zb[:, j] = np.digitize(Z[:, j], qs)
    mis = np.array([mutual_info_score(labels, Zb[:, j]) for j in range(Z.shape[1])])
    order = np.argsort(mis)[::-1]
    counts = np.bincount(labels)
    p = counts[counts > 0] / counts.sum()
    H = float(-(p * np.log(p)).sum())
    if H <= 0:
        return dict(mig=np.nan, mi_top1=np.nan, mi_top2=np.nan, entropy=H)
    return dict(mig=float((mis[order[0]] - mis[order[1]]) / H),
                mi_top1=float(mis[order[0]]), mi_top2=float(mis[order[1]]),
                entropy=H)


def _rf_for_fold(data: BlockData, prep: str, fold_key, train_idx: np.ndarray):
    key = (data.block, prep, fold_key)
    if key not in RF_CACHE:
        X = data.X[prep]
        rf_a = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        rf_s = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        rf_a.fit(X[train_idx], data.ya[train_idx])
        rf_s.fit(X[train_idx], data.ys[train_idx])
        RF_CACHE[key] = (rf_a, rf_s)
    return RF_CACHE[key]


def swap_test(data: BlockData, prep: str, model_name: str, model, seed: int,
              fold_id: int, fold_key, train_idx, test_idx,
              blocks_tr=None, blocks_te=None, phys_tr=None, phys_te=None):
    """Decode chemical block i with substrate block j; RF reads the result."""
    X = data.X[prep]
    X_t, ya_t, ys_t, yl_t = tensors_for(data, prep)
    rf_a, rf_s = _rf_for_fold(data, prep, fold_key, train_idx)
    rng = np.random.RandomState(seed * 1009 + fold_id * 37 + 11)
    pairs = []
    for ii in range(len(test_idx)):
        s_i = int(data.ys[test_idx[ii]])
        l_i = int(data.yl[test_idx[ii]])
        cand = np.flatnonzero((data.ys[train_idx] != s_i) &
                              ((data.yl[train_idx] == l_i) if data.use_laser else True))
        if len(cand) == 0:
            continue
        js = rng.choice(cand, size=min(5, len(cand)), replace=False)
        for jj in js:
            pairs.append((ii, int(jj)))
    if not pairs:
        return None
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    ii_t = torch.tensor(ii, dtype=torch.long, device=DEVICE)
    jj_t = torch.tensor(jj, dtype=torch.long, device=DEVICE)
    test_t = torch.tensor(test_idx, dtype=torch.long, device=DEVICE)
    train_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        if model_name == "CVAE":
            chem_i = blocks_te["chem"][ii_t]
            sub_j = blocks_tr["sub"][jj_t]
            zd = {"chem": chem_i, "sub": sub_j}
            if data.use_laser:
                zd["laser"] = blocks_te["laser"][ii_t]
            decoded = model.decode_blocks(zd)
            # chemical-only reference: substrate block at its overall mean
            sub_mean = blocks_tr["sub"].mean(dim=0, keepdim=True).expand(len(ii), -1)
            zd2 = {"chem": chem_i, "sub": sub_mean}
            if data.use_laser:
                zd2["laser"] = blocks_te["laser"][ii_t]
            chem_only = model.decode_blocks(zd2)
        else:
            p_i = phys_te["p"][ii_t]
            q_j = phys_tr["q"][jj_t]
            gain_i = phys_te["gain"][ii_t]
            offset_i = phys_te["offset"][ii_t]
            las_i = yl_t[test_t][ii_t]
            decoded = model.decode_mixture(p_i, q_j, las_i, gain_i, offset_i)
            q_mean = phys_tr["q"].mean(dim=0, keepdim=True).expand(len(ii), -1)
            chem_only = model.decode_mixture(p_i, q_mean, las_i, gain_i, offset_i)
    dec = decoded.cpu().numpy()
    pred_a = rf_a.predict(dec)
    pred_s = rf_s.predict(dec)
    a_i = data.ya[test_idx][ii]
    s_j = data.ys[train_idx][jj]
    s_i = data.ys[test_idx][ii]
    p_a = float(np.mean(pred_a == a_i))
    p_s = float(np.mean(pred_s == s_j))
    p_joint = float(np.mean((pred_a == a_i) & (pred_s == s_j)))
    p_own_s = float(np.mean(pred_s == s_i))
    example_key = (data.block, model_name)
    if example_key not in SWAP_EXAMPLES:
        # first pair: real i, real j, swapped decode, chemical-only decode
        SWAP_EXAMPLES[example_key] = dict(
            axis=data.axis.copy(),
            real_i=X[test_idx[ii[0]]].copy(),
            real_j=X[train_idx[jj[0]]].copy(),
            decoded=dec[0].copy(), chem_only=chem_only[0].cpu().numpy().copy(),
            analyte_i=data.analytes[a_i[0]], substrate_i=data.substrates[s_i[0]],
            substrate_j=data.substrates[s_j[0]],
            laser_i=data.lasers[int(data.yl[test_idx[ii[0]]])])
    return dict(n_pairs=len(pairs), p_analyte_i=p_a, p_substrate_j=p_s,
                p_joint=p_joint, p_own_substrate=p_own_s,
                chance_analyte=1.0 / data.n_a,
                chance_substrate=1.0 / data.n_s)


# --------------------------------------------------------------------------- #
# E2: probes, MIG and swap test
# --------------------------------------------------------------------------- #
def run_e2(data: dict):
    probe_rows, summary_rows, mig_rows, swap_rows = [], [], [], []
    for block in BLOCKS:
        d = data[block]
        prep = "minmax"
        X_t, ya_t, ys_t, yl_t = tensors_for(d, prep)
        folds = make_grouped_folds(d.cells, block, n_splits=5, seed=0)
        for model_name in VAE_MODELS:
            for seed in SEEDS:
                print(f"  E2 {block} {prep} {model_name} seed={seed}", flush=True)
                records = {}       # (block_name,target) -> list of dicts
                mig_fold = []
                swap_fold = []
                for fold_id, (fold_key, test_maps) in enumerate(folds):
                    train_maps = [m for m in d.cells["map_id"] if m not in set(test_maps)]
                    model, info = fit_model(d, prep, model_name, train_maps, seed,
                                            "E2", f"{fold_key}")
                    if model is None:
                        continue
                    train_idx = info["train_idx"]
                    test_idx = indices_for_maps(d, test_maps)
                    tr = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
                    te = torch.tensor(test_idx, dtype=torch.long, device=DEVICE)
                    with torch.no_grad():
                        if model_name in ("VAE", "beta-VAE"):
                            Ztr = model.latent_mean(X_t[tr]).cpu().numpy()
                            Zte = model.latent_mean(X_t[te]).cpu().numpy()
                            probes = [("full", "analyte", Ztr, Zte, ya_t[te]),
                                      ("full", "substrate", Ztr, Zte, ys_t[te])]
                            blocks_tr = blocks_te = phys_tr = phys_te = None
                        elif model_name == "CVAE":
                            blocks_tr = model.latent_blocks(X_t[tr])
                            blocks_te = model.latent_blocks(X_t[te])
                            probes = [
                                ("chem", "analyte", blocks_tr["chem"].cpu().numpy(),
                                 blocks_te["chem"].cpu().numpy(), ya_t[te]),
                                ("sub", "substrate", blocks_tr["sub"].cpu().numpy(),
                                 blocks_te["sub"].cpu().numpy(), ys_t[te]),
                                ("chem", "substrate", blocks_tr["chem"].cpu().numpy(),
                                 blocks_te["chem"].cpu().numpy(), ys_t[te]),
                                ("sub", "analyte", blocks_tr["sub"].cpu().numpy(),
                                 blocks_te["sub"].cpu().numpy(), ya_t[te]),
                            ]
                            if d.use_laser:
                                probes.append(("laser", "laser",
                                               blocks_tr["laser"].cpu().numpy(),
                                               blocks_te["laser"].cpu().numpy(),
                                               yl_t[te]))
                            phys_tr = phys_te = None
                        else:
                            p_tr, q_tr, g_tr, o_tr = model.probabilities(X_t[tr], yl_t[tr])
                            p_te, q_te, g_te, o_te = model.probabilities(X_t[te], yl_t[te])
                            phys_tr = dict(p=p_tr, q=q_tr, gain=g_tr, offset=o_tr)
                            phys_te = dict(p=p_te, q=q_te, gain=g_te, offset=o_te)
                            probes = [
                                ("p", "analyte", p_tr.cpu().numpy(),
                                 p_te.cpu().numpy(), ya_t[te]),
                                ("q", "substrate", q_tr.cpu().numpy(),
                                 q_te.cpu().numpy(), ys_t[te]),
                                ("p", "substrate", p_tr.cpu().numpy(),
                                 p_te.cpu().numpy(), ys_t[te]),
                                ("q", "analyte", q_tr.cpu().numpy(),
                                 q_te.cpu().numpy(), ya_t[te]),
                            ]
                            blocks_tr = blocks_te = None
                        for block_name, target, Ztr, Zte, yte in probes:
                            n_classes = {"analyte": d.n_a, "substrate": d.n_s,
                                         "laser": d.n_l}[target]
                            try:
                                proba = fit_probe(Ztr, (ya_t if target == "analyte"
                                                        else ys_t if target == "substrate"
                                                        else yl_t)[tr].cpu().numpy(),
                                                  Zte, n_classes)
                            except Exception as exc:  # noqa: BLE001
                                add_failure("E2", block, prep, model_name, seed,
                                            "probe", repr(exc), f"{fold_key}/{block_name}->{target}")
                                continue
                            y_true = yte.cpu().numpy()
                            maps = d.meta["map_id"].to_numpy()[test_idx]
                            pred = proba.argmax(axis=1)
                            if len(np.unique(y_true)) > 1:
                                ba_spot = float(balanced_accuracy_score(y_true, pred))
                                map_true, map_pred = [], []
                                for m in np.unique(maps):
                                    mm = maps == m
                                    map_true.append(int(y_true[mm][0]))
                                    map_pred.append(int(proba[mm].mean(axis=0).argmax()))
                                ba_map = float(balanced_accuracy_score(map_true, map_pred))
                            else:
                                ba_spot = ba_map = np.nan
                            key = (block_name, target)
                            records.setdefault(key, []).append(
                                dict(proba=proba, y=y_true, maps=maps,
                                     ba_spectrum=ba_spot, ba_map=ba_map,
                                     fold=fold_key))
                            probe_rows.append(dict(
                                block=block, preprocessing=prep, model=model_name,
                                seed=seed, fold=fold_key, block_name=block_name,
                                target=target, ba_spectrum=ba_spot, ba_map=ba_map,
                                n_test_spots=len(y_true),
                                n_test_maps=len(np.unique(maps))))
                        # MIG for the unsupervised models on held-out latents
                        if model_name in ("VAE", "beta-VAE"):
                            Zte_t = torch.tensor(Zte, device=DEVICE)
                            for factor, lab in [("analyte", ya_t[te]),
                                                ("substrate", ys_t[te])]:
                                mig = mig_scores(Zte, lab.cpu().numpy())
                                mig_rows.append(dict(
                                    block=block, preprocessing=prep,
                                    model=model_name, seed=seed, fold=fold_key,
                                    factor=factor, n_test_spots=len(lab),
                                    **mig))
                        # swap test
                        if model_name in ("CVAE", "PhysicsVAE"):
                            sw = swap_test(d, prep, model_name, model, seed,
                                           fold_id, fold_key, train_idx, test_idx,
                                           blocks_tr, blocks_te, phys_tr, phys_te)
                            if sw is not None:
                                sw.update(dict(block=block, preprocessing=prep,
                                               model=model_name, seed=seed,
                                               fold=fold_key))
                                swap_fold.append(sw)
                    del model
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

                pooled = pooled_probe_scores(records)
                for (block_name, target), vals in pooled.items():
                    summary_rows.append(dict(
                        block=block, preprocessing=prep, model=model_name,
                        seed=seed, block_name=block_name, target=target,
                        **vals))
                if swap_fold:
                    n = sum(s["n_pairs"] for s in swap_fold)
                    swap_rows.append(dict(
                        block=block, preprocessing=prep, model=model_name,
                        seed=seed, n_pairs=n,
                        p_analyte_i=sum(s["p_analyte_i"] * s["n_pairs"] for s in swap_fold) / n,
                        p_substrate_j=sum(s["p_substrate_j"] * s["n_pairs"] for s in swap_fold) / n,
                        p_joint=sum(s["p_joint"] * s["n_pairs"] for s in swap_fold) / n,
                        p_own_substrate=sum(s["p_own_substrate"] * s["n_pairs"] for s in swap_fold) / n,
                        chance_analyte=1.0 / d.n_a,
                        chance_substrate=float(np.mean([s["chance_substrate"] for s in swap_fold])),
                        n_folds=len(swap_fold)))

        # Raw-spectrum baseline under exactly the same map-grouped folds.
        raw_records = {}
        for fold_id, (fold_key, test_maps) in enumerate(folds):
            train_maps = [m for m in d.cells["map_id"] if m not in set(test_maps)]
            train_idx = indices_for_maps(d, train_maps)
            test_idx = indices_for_maps(d, test_maps)
            pca = PCA(n_components=20, random_state=0)
            Ztr = pca.fit_transform(d.X[prep][train_idx])
            Zte = pca.transform(d.X[prep][test_idx])
            for target, y_all, ytr, n_classes in [
                    ("analyte", d.ya, d.ya[train_idx], d.n_a),
                    ("substrate", d.ys, d.ys[train_idx], d.n_s)]:
                proba = fit_probe(Ztr, ytr, Zte, n_classes)
                y_true = y_all[test_idx]
                maps = d.meta["map_id"].to_numpy()[test_idx]
                pred = proba.argmax(axis=1)
                ba_spot = float(balanced_accuracy_score(y_true, pred)) \
                    if len(np.unique(y_true)) > 1 else np.nan
                map_true, map_pred = [], []
                for m in np.unique(maps):
                    mm = maps == m
                    map_true.append(int(y_true[mm][0]))
                    map_pred.append(int(proba[mm].mean(axis=0).argmax()))
                ba_map = float(balanced_accuracy_score(map_true, map_pred)) \
                    if len(np.unique(map_true)) > 1 else np.nan
                key = ("raw_pca20", target)
                raw_records.setdefault(key, []).append(
                    dict(proba=proba, y=y_true, maps=maps,
                         ba_spectrum=ba_spot, ba_map=ba_map, fold=fold_key))
        pooled = pooled_probe_scores(raw_records)
        for (block_name, target), vals in pooled.items():
            summary_rows.append(dict(
                block=block, preprocessing=prep,
                model="PCA20+LR (raw, phase-2-style)", seed=-1,
                block_name=block_name, target=target, **vals))

    pd.DataFrame(probe_rows).to_csv(os.path.join(OUT, "phase3_e2_probes.csv"),
                                    index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "phase3_e2_summary.csv"),
                                      index=False)
    pd.DataFrame(mig_rows).to_csv(os.path.join(OUT, "phase3_e2_mig.csv"), index=False)
    swap = pd.DataFrame(swap_rows)
    swap.to_csv(os.path.join(OUT, "phase3_e2_swap.csv"), index=False)
    if SWAP_EXAMPLES:
        arrays = {}
        for (block, model_name), ex in SWAP_EXAMPLES.items():
            for k, v in ex.items():
                if k == "axis":
                    arrays[f"{block}_{model_name}_axis"] = v
                elif isinstance(v, str):
                    arrays[f"{block}_{model_name}_{k}"] = np.array(v)
                else:
                    arrays[f"{block}_{model_name}_{k}"] = v
        np.savez_compressed(os.path.join(OUT, "phase3_swap_example.npz"), **arrays)
    return (pd.DataFrame(summary_rows), pd.DataFrame(mig_rows), swap)


# --------------------------------------------------------------------------- #
# E3: unseen combinations
# --------------------------------------------------------------------------- #
def physics_cell_prediction(model: PhysicsVAE, data: BlockData, prep: str,
                            train_idx: np.ndarray, analyte: str, substrate: str,
                            laser: str):
    a_idx = data.analytes.index(analyte)
    s_idx = data.substrates.index(substrate)
    l_idx = data.lasers.index(laser)
    X_t, _, ys_t, yl_t = tensors_for(data, prep)
    tr = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        A = model.analyte_spectra()
        B = model.substrate_spectra()
        G = model.gain_matrix()
        seen_s = np.unique(data.ys[train_idx][data.ya[train_idx] == a_idx])
        g_mean = G[a_idx, torch.tensor(seen_s, device=DEVICE)].mean()
        p, q, gain, offset = model.probabilities(X_t[tr], yl_t[tr])
        m_a = torch.tensor(data.ya[train_idx] == a_idx, device=DEVICE)
        gain_a = gain[m_a].mean()
        offset_a = offset[m_a].mean()
        las = torch.tensor([l_idx], dtype=torch.long, device=DEVICE)
        r = model.laser_response(las)[0]
        pred = gain_a * r * (g_mean * A[a_idx] + B[s_idx]) + offset_a
    return pred.cpu().numpy()


def run_e3(data: dict, e3_p2):
    rows = []
    for block in BLOCKS:
        d = data[block]
        prep = "minmax"
        X_t, _, _, yl_t = tensors_for(d, prep)
        for model_name in CHEM_MODELS:
            for seed in SEEDS:
                print(f"  E3 {block} {prep} {model_name} seed={seed}", flush=True)
                for _, cell in d.cells.iterrows():
                    held_map = cell["map_id"]
                    train_maps = [m for m in d.cells["map_id"] if m != held_map]
                    test_idx = indices_for_maps(d, [held_map])
                    actual = d.X[prep][test_idx].mean(axis=0)
                    model, info = fit_model(d, prep, model_name, train_maps, seed,
                                            "E3", held_map)
                    if model is None:
                        continue
                    try:
                        if model_name == "CVAE":
                            train_idx = info["train_idx"]
                            tr = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
                            with torch.no_grad():
                                blocks_tr = model.latent_blocks(X_t[tr])
                                m_a = torch.tensor(d.ya[train_idx] ==
                                                   d.analytes.index(cell["analyte"]),
                                                   device=DEVICE)
                                m_s = torch.tensor(d.ys[train_idx] ==
                                                   d.substrates.index(cell["substrate"]),
                                                   device=DEVICE)
                                zd = {"chem": blocks_tr["chem"][m_a].mean(dim=0, keepdim=True),
                                      "sub": blocks_tr["sub"][m_s].mean(dim=0, keepdim=True)}
                                if d.use_laser:
                                    m_l = torch.tensor(d.yl[train_idx] ==
                                                       d.lasers.index(cell["laser"]),
                                                       device=DEVICE)
                                    zd["laser"] = blocks_tr["laser"][m_l].mean(dim=0,
                                                                               keepdim=True)
                                pred = model.decode_blocks(zd)[0].cpu().numpy()
                        else:
                            pred = physics_cell_prediction(
                                model, d, prep, info["train_idx"], cell["analyte"],
                                cell["substrate"], cell["laser"])
                        if not np.all(np.isfinite(pred)):
                            add_failure("E3", block, prep, model_name, seed,
                                        "prediction", "non-finite decoded spectrum",
                                        held_map)
                            continue
                        rows.append(dict(
                            block=block, preprocessing=prep, model=model_name,
                            seed=seed, held_out=held_map, analyte=cell["analyte"],
                            substrate=cell["substrate"], laser=cell["laser"],
                            n_train_maps=len(train_maps), n_test_spots=len(test_idx),
                            corr=corr(actual, pred), rmse=rmse(actual, pred),
                            note=f"early stop epoch {info.get('best_epoch')}"))
                    except Exception as exc:  # noqa: BLE001
                        add_failure("E3", block, prep, model_name, seed,
                                    "prediction", repr(exc), held_map)
                    del model
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
        # phase-2 baseline rows for the same held-out cells (min-max), once per block
        sub = e3_p2[(e3_p2.block == block) & (e3_p2.preprocessing == prep)]
        for method, label in [("additive", "phase2: additive"),
                              ("same analyte / other substrate",
                               "phase2: other substrate"),
                              ("same substrate / other analyte",
                               "phase2: other analyte")]:
            for _, r0 in sub[sub.method == method].iterrows():
                rows.append(dict(
                    block=block, preprocessing=prep, model=label, seed=-1,
                    held_out=r0["held_out"], analyte=r0["analyte"],
                    substrate=r0["substrate"], laser=r0["laser"],
                    n_train_maps=np.nan, n_test_spots=np.nan,
                    corr=r0["corr"], rmse=r0["rmse"], note="phase 2 value"))
    e3 = pd.DataFrame(rows)
    e3.to_csv(os.path.join(OUT, "phase3_e3.csv"), index=False)
    return e3


# --------------------------------------------------------------------------- #
# E4: unseen substrate / laser
# --------------------------------------------------------------------------- #
def run_e4(data: dict, e4_p2):
    rows = []
    for block in BLOCKS:
        d = data[block]
        schemes = ["LOSO"] if block == "A" else ["LOSO", "LOLO"]
        for prep in PREPS:
            X_t, ya_t, ys_t, yl_t = tensors_for(d, prep)
            for scheme in schemes:
                for fold_key, train_maps, test_maps in e4_folds(d.cells, scheme):
                    train_idx = indices_for_maps(d, train_maps)
                    test_idx = indices_for_maps(d, test_maps)
                    tr = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
                    te = torch.tensor(test_idx, dtype=torch.long, device=DEVICE)
                    y_true = d.ya[test_idx]
                    maps = d.meta["map_id"].to_numpy()[test_idx]
                    for model_name in VAE_MODELS:
                        for seed in SEEDS:
                            print(f"  E4 {block} {prep} {scheme}/{fold_key} "
                                  f"{model_name} seed={seed}", flush=True)
                            model, info = fit_model(d, prep, model_name, train_maps,
                                                    seed, "E4", f"{scheme}/{fold_key}")
                            if model is None:
                                continue
                            try:
                                with torch.no_grad():
                                    if model_name in ("VAE", "beta-VAE"):
                                        Ztr = model.latent_mean(X_t[tr]).cpu().numpy()
                                        Zte = model.latent_mean(X_t[te]).cpu().numpy()
                                        proba = fit_probe(Ztr, d.ya[train_idx],
                                                          Zte, d.n_a)
                                    elif model_name == "CVAE":
                                        blocks_te = model.latent_blocks(X_t[te])
                                        logits = model.cls_heads[0](blocks_te["chem"])
                                        proba = F.softmax(logits, dim=1).cpu().numpy()
                                    else:
                                        p, q, gain, offset = model.probabilities(X_t[te], yl_t[te])
                                        proba = p.cpu().numpy()
                                pred = proba.argmax(axis=1)
                                ba_spot = float(balanced_accuracy_score(y_true, pred)) \
                                    if len(np.unique(y_true)) > 1 else np.nan
                                map_true, map_pred = [], []
                                for m in np.unique(maps):
                                    mm = maps == m
                                    map_true.append(int(y_true[mm][0]))
                                    map_pred.append(int(proba[mm].mean(axis=0).argmax()))
                                ba_map = float(balanced_accuracy_score(map_true, map_pred)) \
                                    if len(np.unique(map_true)) > 1 else np.nan
                                rows.append(dict(
                                    block=block, preprocessing=prep, scheme=scheme,
                                    fold=fold_key, model=model_name, seed=seed,
                                    n_train_spots=len(train_idx),
                                    n_test_spots=len(test_idx),
                                    n_train_maps=len(train_maps),
                                    n_test_maps=len(test_maps),
                                    ba_spectrum=ba_spot, ba_map=ba_map,
                                    note=f"early stop epoch {info.get('best_epoch')}"))
                            except Exception as exc:  # noqa: BLE001
                                add_failure("E4", block, prep, model_name, seed,
                                            "prediction", repr(exc),
                                            f"{scheme}/{fold_key}")
                            del model
                            if DEVICE.type == "cuda":
                                torch.cuda.empty_cache()
                    # phase-2 baseline rows, same folds
                    sub = e4_p2[(e4_p2.block == block) &
                                (e4_p2.preprocessing == prep) &
                                (e4_p2.scheme == scheme) &
                                (e4_p2.fold == fold_key)]
                    for p2model, label in [("Random Forest", "phase2: Random Forest"),
                                           ("Extra Trees", "phase2: Extra Trees"),
                                           ("PCA + LDA", "phase2: PCA + LDA")]:
                        for _, r0 in sub[sub.model == p2model].iterrows():
                            rows.append(dict(
                                block=block, preprocessing=prep, scheme=scheme,
                                fold=fold_key, model=label, seed=-1,
                                n_train_spots=r0["n_train_spots"],
                                n_test_spots=r0["n_test_spots"],
                                n_train_maps=r0["n_train_maps"],
                                n_test_maps=r0["n_test_maps"],
                                ba_spectrum=r0["ba_spectrum"],
                                ba_map=r0["ba_map"], note="phase 2 value"))
    e4 = pd.DataFrame(rows)
    e4.to_csv(os.path.join(OUT, "phase3_e4.csv"), index=False)
    return e4


# --------------------------------------------------------------------------- #
# Parameter table and figures
# --------------------------------------------------------------------------- #
def write_parameter_table(data: dict):
    rows = []
    for block in BLOCKS:
        d = data[block]
        for model_name in VAE_MODELS:
            idx = np.arange(len(d.meta))[: max(1, len(d.meta) // 2)]
            model = build_model(d, model_name, "minmax", idx, 0)
            c = count_parameters(model)
            rows.append(dict(block=block, model=model_name, in_len=d.X["minmax"].shape[1],
                             **c))
    p = pd.DataFrame(rows)
    p.to_csv(os.path.join(OUT, "phase3_params.csv"), index=False)
    return p


def plot_e1_physics(data: dict, ref_X, ref_axis, ref_meta, e1: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for i, block in enumerate(BLOCKS):
        d = data[block]
        for j, analyte in enumerate(d.analytes):
            ax = axes[i, j]
            ax.set_title(f"Block {block}: {CODE[analyte]}", fontsize=10)
            parts = [CHEM_PARTS[(block, "PhysicsVAE", s, analyte)] for s in SEEDS
                     if (block, "PhysicsVAE", s, analyte) in CHEM_PARTS]
            if parts:
                ax.plot(d.axis, minmax_vec(np.mean(parts, axis=0)),
                        lw=1.3, color="#1f5fa8", label="A_a (model 4, seed mean)")
            # reference (first preferred laser)
            ref_laser = PREF_LASER[block][analyte][0]
            m = ((ref_meta["analyte"] == analyte) &
                 (ref_meta["laser"] == ref_laser)).to_numpy()
            if np.any(m):
                ref_y = preprocess(ref_X[m], "minmax").mean(axis=0)
                rv, mask = align_reference(d.axis, ref_axis, ref_y)
                ax.plot(d.axis[mask], minmax_vec(rv), color="0.3", ls="--", lw=0.9,
                        label=f"Raman ref ({ref_laser})")
            sig = sers_signature(d.X["minmax"], d.meta, block)[analyte]
            ax.plot(d.axis, minmax_vec(sig), color="#c46a00", ls=":", lw=1.0,
                    label="phase-2 SERS signature")
            ax.tick_params(labelsize=8)
            if j == 0:
                ax.set_ylabel("min–max (display)", fontsize=9)
            if i == 1:
                ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Model 4 learned analyte spectra A_a vs neat reference and phase-2 SERS signature",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "phase3_e1_physics_spectra.png"), dpi=200)
    plt.close(fig)


def plot_swap_example():
    path = os.path.join(OUT, "phase3_swap_example.npz")
    if not os.path.exists(path):
        return
    d = np.load(path, allow_pickle=False)
    keys = sorted({k.split("_axis")[0] for k in d.files if k.endswith("_axis")})
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 3.8),
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        axis = d[f"{key}_axis"]
        ax.plot(axis, minmax_vec(d[f"{key}_real_i"]), color="0.2", lw=1.1,
                label=f"real i ({d[f'{key}_analyte_i']} / {d[f'{key}_substrate_i']})")
        ax.plot(axis, minmax_vec(d[f"{key}_real_j"]), color="0.6", lw=1.0,
                label=f"real j (substrate {d[f'{key}_substrate_j']})")
        ax.plot(axis, minmax_vec(d[f"{key}_chem_only"]), color="#1f5fa8", ls="--",
                lw=1.0, label="chemical block only")
        ax.plot(axis, minmax_vec(d[f"{key}_decoded"]), color="#c43b3b", lw=1.2,
                label="chem i + substrate j (swap)")
        ax.set_title(key.replace("_", " "), fontsize=10)
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("min–max (display)")
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phase3_e2_swap_example.png"), dpi=200)
    plt.close(fig)


def plot_e4(e4: pd.DataFrame):
    schemes = [("A", "LOSO", "Block A: leave-substrate-out"),
               ("B", "LOSO", "Block B: leave-substrate-out"),
               ("B", "LOLO", "Block B: leave-laser-out")]
    models = ["VAE", "beta-VAE", "CVAE", "PhysicsVAE"]
    colors = ["#8ab4d8", "#4f86b8", "#2a5d8f", "#123b5c"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (block, scheme, title) in zip(axes, schemes):
        d = e4[(e4.block == block) & (e4.scheme == scheme) &
               (e4.preprocessing == "minmax")]
        x = np.arange(len(models))
        vals = [d[d.model == m]["ba_map"].mean() for m in models]
        errs = [d[d.model == m]["ba_map"].std() for m in models]
        ax.bar(x, vals, yerr=errs, color=colors, capsize=3)
        # phase-2 best baseline (mean over phase-2 models per fold, then mean)
        p2 = d[d.model.str.startswith("phase2:")]
        if len(p2):
            p2_mean = p2.groupby("fold")["ba_map"].mean().mean()
            ax.axhline(p2_mean, color="#c46a00", ls="--", lw=1.2,
                       label=f"phase-2 mean ({p2_mean:.2f})")
            ax.legend(fontsize=7, frameon=False)
        ax.axhline(1 / 3, color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["VAE", "β-VAE", "CVAE", "Physics"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("balanced accuracy per map (mean ± sd over folds/seeds)")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phase3_e4_comparison.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="all",
                        help="comma-separated: e1,e2,e3,e4,figures or all")
    args = parser.parse_args()
    stages = (["e1", "e2", "e3", "e4", "figures"]
              if args.stages == "all" else [s.strip() for s in args.stages.split(",")])
    t_start = time.time()
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    print("Loading blocks and preprocessing ...")
    data = {b: load_block(b) for b in BLOCKS}
    for b in BLOCKS:
        print(f"  Block {b}: {data[b].X['minmax'].shape[0]} spectra x "
              f"{data[b].X['minmax'].shape[1]} shifts, {len(data[b].cells)} maps, "
              f"{data[b].n_a} analytes, {data[b].n_s} substrates, {data[b].n_l} lasers")
    ref_X, ref_axis, ref_meta = load_references()
    e1_p2 = pd.read_csv(os.path.join(OUT, "e1_reference_recovery.csv"))
    e2_p2 = pd.read_csv(os.path.join(OUT, "e2_variance.csv"))
    e3_p2 = pd.read_csv(os.path.join(OUT, "e3_unseen.csv"))
    e4_p2 = pd.read_csv(os.path.join(OUT, "e4_classification.csv"))

    params = write_parameter_table(data)
    print(params.to_string(index=False))

    if "e1" in stages:
        print("E1: chemical part vs references and SERS signature ...")
        run_e1(data, ref_X, ref_axis, ref_meta, e1_p2)
    if "e2" in stages:
        print("E2: probes, MIG and swap test ...")
        run_e2(data)
    if "e3" in stages:
        print("E3: held-out cells ...")
        run_e3(data, e3_p2)
    if "e4" in stages:
        print("E4: unseen substrate / laser ...")
        run_e4(data, e4_p2)
    if "figures" in stages:
        print("Figures ...")
        # E1 reads the chemical-part cache written by a previous E1 run if needed.
        chem_path = os.path.join(OUT, "phase3_chemical_parts.npz")
        if os.path.exists(chem_path) and not CHEM_PARTS:
            z = np.load(chem_path, allow_pickle=False)
            for k in z.files:
                block, model_name, seed, analyte = k.split("|")
                CHEM_PARTS[(block, model_name, int(seed), analyte)] = z[k]
        e1 = pd.read_csv(os.path.join(OUT, "phase3_e1.csv")) \
            if os.path.exists(os.path.join(OUT, "phase3_e1.csv")) else pd.DataFrame()
        if CHEM_PARTS:
            plot_e1_physics(data, ref_X, ref_axis, ref_meta, e1)
        plot_swap_example()
        e4 = pd.read_csv(os.path.join(OUT, "phase3_e4.csv")) \
            if os.path.exists(os.path.join(OUT, "phase3_e4.csv")) else pd.DataFrame()
        if len(e4):
            plot_e4(e4)

    # Save chemical parts for the figure / reproducibility.
    if CHEM_PARTS:
        arrays = {f"{b}|{m}|{s}|{a}": v for (b, m, s, a), v in CHEM_PARTS.items()}
        np.savez_compressed(os.path.join(OUT, "phase3_chemical_parts.npz"), **arrays)

    pd.DataFrame(FIT_TIMES).to_csv(os.path.join(OUT, "phase3_runtime.csv"), index=False)
    fail_cols = ["experiment", "block", "preprocessing", "model", "seed",
                 "stage", "reason", "tag"]
    pd.DataFrame(FAILURES, columns=fail_cols).to_csv(
        os.path.join(OUT, "phase3_failures.csv"), index=False)
    pd.DataFrame(DIAGNOSTICS).to_csv(os.path.join(OUT, "phase3_diagnostics.csv"),
                                     index=False)
    total = time.time() - t_start
    summary = dict(device=str(DEVICE), seeds=SEEDS, total_seconds=total,
                   total_minutes=total / 60.0, n_failures=len(FAILURES),
                   n_fits=len(FIT_TIMES), parameter_table=params.to_dict(orient="records"))
    with open(os.path.join(OUT, "phase3_run.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"Done in {total / 60:.1f} min; {len(FIT_TIMES)} fits; "
          f"{len(FAILURES)} failures.")


if __name__ == "__main__":
    main()
