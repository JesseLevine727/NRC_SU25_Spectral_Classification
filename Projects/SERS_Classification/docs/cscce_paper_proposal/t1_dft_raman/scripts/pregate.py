"""T1 pre-gate: measured Raman vs a DFT-calculated library, fixed methods only.

Pools (D1 design): P1 closed (the calculated spectra of all query molecules),
P2 same-formula isomers from QM9S (cap 1000), P3 true + 20,000 random QM9S.
Kill rules: best fixed top-1 on P2 >= 70% (no room) or best fixed top-5 on P2 < 10%.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter

STICKS, QUERIES, OUTDIR = sys.argv[1:4]
GRID = np.arange(400.0, 3201.0, 1.0)
WINDOWS = {"fp": [(400, 1800)], "fp_ch": [(400, 1800), (2800, 3100)]}
NU0 = 1e7 / 785.0
RNG = np.random.default_rng(0)
THREE = [(0, 1000, 0.9790), (1000, 2000, 0.9818), (2000, 1e9, 0.9619)]  # Rahimi 2026, B3LYP-D3


def rd_table(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(("", "") if m is None else (Chem.MolToInchiKey(m), CalcMolFormula(m)))
    return out


def library_table(smiles, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    chunks = np.array_split(np.array(smiles), 30)
    with Pool(30) as p:
        res = p.map(rd_table, chunks)
    t = pd.DataFrame([r for part in res for r in part], columns=["inchikey", "formula"])
    t.to_csv(cache, index=False)
    return t


def scale(nu, scheme):
    if scheme == "none":
        return nu
    if scheme == "global":
        return nu * 0.965
    out = nu.copy()
    for lo, hi, f in THREE:
        m = (nu >= lo) & (nu < hi)
        out[m] = nu[m] * f
    return out


def render(idx, d, scheme="three", fwhm=10.0, inten="xsec", factor=None):
    """Broadened calculated spectra (len(idx) x len(GRID)), Lorentzian line shape."""
    off, nu_all, act_all = d["offsets"], d["nu"], d["act"]
    out = np.zeros((len(idx), len(GRID)), np.float32)
    g = (fwhm / 2.0) ** 2
    for r, i in enumerate(idx):
        nu = nu_all[off[i]:off[i + 1]].astype(float)
        a = act_all[off[i]:off[i + 1]].astype(float)
        keep = nu > 50
        nu, a = nu[keep], a[keep]
        if inten == "xsec":
            a = a * (NU0 - nu) ** 4 / nu / (1 - np.exp(-1.4388 * nu / 298.0)) / 1e12
        nu = nu * factor if factor is not None else scale(nu, scheme)
        inw = (nu > GRID[0] - 60) & (nu < GRID[-1] + 60)
        nu, a = nu[inw], a[inw]
        out[r] = (a[None, :] * g / ((GRID[:, None] - nu[None, :]) ** 2 + g)).sum(1)
    return out


def mask_for(win):
    m = np.zeros(len(GRID), bool)
    for lo, hi in WINDOWS[win]:
        m |= (GRID >= lo) & (GRID <= hi)
    return m


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def feats(X, m, kind):
    X = np.clip(np.nan_to_num(X), 0, None)
    if kind == "raw":
        return unit(X[:, m])
    if kind == "sqrt":
        return unit(np.sqrt(X[:, m]))
    if kind == "pearson":
        Y = X[:, m] - X[:, m].mean(1, keepdims=True)
        return unit(Y)
    if kind == "deriv":
        Y = savgol_filter(np.sqrt(X), 11, 2, deriv=1, axis=1)
        Y = np.where(m[None, :], Y, 0)
        return unit(Y)
    raise ValueError(kind)


def lag_cos(Q, C, maxlag=13):
    """max over integer shifts of cos(Q, shift(C)); Q, C unit rows on the full GRID."""
    best = np.full((Q.shape[0], C.shape[0]), -np.inf, np.float32)
    for s in range(-maxlag, maxlag + 1):
        best = np.maximum(best, Q @ np.roll(C, s, axis=1).T)
    return best


def peaks(y, m, prom=0.03):
    y = np.clip(np.nan_to_num(y), 0, None) * m
    if y.max() <= 0:
        return np.zeros(0), np.zeros(0)
    ys = savgol_filter(y, 7, 2) / y.max()
    p, _ = find_peaks(ys, prominence=prom)
    return GRID[p], np.sqrt(np.clip(ys[p], 1e-6, None))


def peak_score(pe, we, pc, wc, tol=15.0):
    if len(pe) == 0 or len(pc) == 0:
        return 0.0, 0.0
    cost = np.abs(pe[:, None] - pc[None, :])
    big = cost > tol
    c = np.where(big, 1e6, cost)
    r, k = linear_sum_assignment(c)
    ok = ~big[r, k]
    r, k = r[ok], k[ok]
    w = 0.5 * (we[r].sum() / we.sum() + wc[k].sum() / wc.sum())
    f1 = 2 * len(r) / (len(pe) + len(pc))
    return w, f1


def ranks(S, true_col):
    """S: n_query x n_cand similarity; true_col: index of the true candidate per query."""
    t = S[np.arange(len(S)), true_col]
    return 1 + (S > t[:, None]).sum(1)  # optimistic ties are rare with float scores


def summarise(r):
    r = np.asarray(r)
    return dict(top1=float((r == 1).mean()), top5=float((r <= 5).mean()), mrr=float((1 / r).mean()),
                median_rank=float(np.median(r)))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = dict(np.load(STICKS))
    lib = library_table(list(d["smiles"]), os.path.join(OUTDIR, "qm9s_rdkit.csv"))
    q = pd.read_csv(QUERIES + ".csv")
    Qspec = np.load(QUERIES + ".npy")
    ik2i = {}
    for i, k in enumerate(lib["inchikey"]):
        ik2i.setdefault(k, i)
    blk = {}
    for i, k in enumerate(lib["inchikey"]):
        if isinstance(k, str) and k:
            blk.setdefault(k[:14], i)
    q["lib_idx"] = [ik2i.get(k, blk.get(k[:14], -1)) for k in q["inchikey"]]
    q["match"] = ["full" if k in ik2i else ("skeleton" if k[:14] in blk else "missing") for k in q["inchikey"]]
    print(q[["source", "name", "formula", "lib_idx", "match"]].to_string())
    found = (q.lib_idx >= 0).to_numpy()
    q, Qspec = q[found].reset_index(drop=True), Qspec[found]

    # pools
    formula = lib["formula"].to_numpy()
    true = q["lib_idx"].to_numpy()
    p1 = np.unique(true)
    p3_rand = RNG.choice(len(lib), 20000, replace=False)
    pools = {}
    for j, t in enumerate(true):
        iso = np.flatnonzero(formula == formula[t])
        if len(iso) > 1000:
            iso = np.union1d(RNG.choice(iso, 1000, replace=False), [t])
        pools[("P2", j)] = iso
        pools[("P3", j)] = np.union1d(p3_rand, [t])
        pools[("P1", j)] = p1
    q["p2_size"] = [len(pools[("P2", j)]) for j in range(len(q))]
    cand = np.unique(np.concatenate([v for v in pools.values()]))
    pos = {c: i for i, c in enumerate(cand)}
    pcols = {k: (np.searchsorted(cand, v), int(np.flatnonzero(v == true[k[1]])[0])) for k, v in pools.items()}
    print("candidates to render", len(cand))

    configs = []
    for scheme in ["none", "global", "three"]:
        for fwhm in [6.0, 10.0, 16.0]:
            configs.append(dict(scheme=scheme, fwhm=fwhm))
    results, per_query = [], []
    for cfg in configs:
        C = render(cand, d, scheme=cfg["scheme"], fwhm=cfg["fwhm"])
        for win in ["fp", "fp_ch"]:
            m = mask_for(win)
            use = q["source"].str.startswith("flanagan").to_numpy() if win == "fp_ch" else np.ones(len(q), bool)
            sims = {}
            for kind in ["raw", "sqrt", "pearson", "deriv"]:
                sims[kind] = feats(Qspec[use], m, kind) @ feats(C, m, kind).T
            sims["deriv_lag13"] = lag_cos(feats(Qspec[use], m, "deriv"), feats(C, m, "deriv"))
            for name, S in sims.items():
                for pool in ["P1", "P2", "P3"]:
                    rr = []
                    for jj, j in enumerate(np.flatnonzero(use)):
                        cols, tcol = pcols[(pool, j)]
                        s = S[jj, cols]
                        rr.append(int(1 + (s > s[tcol]).sum()))
                        per_query.append(dict(win=win, **cfg, metric=name, pool=pool, query=j, rank=rr[-1]))
                    results.append(dict(win=win, **cfg, metric=name, pool=pool, n=len(rr), **summarise(rr)))
            print(win, cfg, "done", flush=True)

    # per-candidate global scale search (unscaled sticks, sqrt cosine), fingerprint window
    m = mask_for("fp")
    factors = np.arange(0.95, 1.0001, 0.0025)
    Qf = feats(Qspec, m, "sqrt")
    best = np.full((len(q), len(cand)), -np.inf, np.float32)
    bestd = np.full((len(q), len(cand)), -np.inf, np.float32)
    Qd = feats(Qspec, m, "deriv")
    for f in factors:
        C = render(cand, d, fwhm=10.0, factor=f)
        best = np.maximum(best, Qf @ feats(C, m, "sqrt").T)
        bestd = np.maximum(bestd, lag_cos(Qd, feats(C, m, "deriv"), maxlag=4))
    for name, S in [("scale_search_sqrt", best), ("scale_search_deriv_lag4", bestd)]:
        for pool in ["P1", "P2", "P3"]:
            rr = []
            for j in range(len(q)):
                cols, tcol = pcols[(pool, j)]
                s = S[j, cols]
                rr.append(int(1 + (s > s[tcol]).sum()))
                per_query.append(dict(win="fp", scheme="search", fwhm=10.0, metric=name, pool=pool, query=j, rank=rr[-1]))
            results.append(dict(win="fp", scheme="search", fwhm=10.0, metric=name, pool=pool, n=len(rr), **summarise(rr)))

    # peak assignment (Hungarian, +-15 cm^-1), three-range scaling, P1 and P2 only
    C = render(cand, d, scheme="three", fwhm=10.0)
    cpk = {}
    for j in range(len(q)):
        pe, we = peaks(Qspec[j], m)
        for pool in ["P1", "P2"]:
            sw, sf = [], []
            for c in pools[(pool, j)]:
                if c not in cpk:
                    cpk[c] = peaks(C[pos[c]], m)
                w, f1 = peak_score(pe, we, *cpk[c])
                sw.append(w)
                sf.append(f1)
            tcol = int(np.flatnonzero(pools[(pool, j)] == true[j])[0])
            for name, s in [("peaks_weighted", np.array(sw)), ("peaks_count", np.array(sf))]:
                per_query.append(dict(win="fp", scheme="three", fwhm=10.0, metric=name, pool=pool, query=j,
                                      rank=int(1 + (s > s[tcol]).sum())))
    pq = pd.DataFrame(per_query)
    for (name, pool), g in pq[pq.metric.str.startswith("peaks")].groupby(["metric", "pool"]):
        results.append(dict(win="fp", scheme="three", fwhm=10.0, metric=name, pool=pool, n=len(g), **summarise(g["rank"])))

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUTDIR, "pregate_results.csv"), index=False)
    pq.to_csv(os.path.join(OUTDIR, "pregate_per_query.csv"), index=False)
    q.to_csv(os.path.join(OUTDIR, "queries_matched.csv"), index=False)
    print(res.sort_values(["win", "pool", "top1"], ascending=[True, True, False]).groupby(["win", "pool"]).head(8).to_string())


if __name__ == "__main__":
    main()
