"""Fitted-physics control: no network, 8 parameters fitted per fold on training compounds.

calc spectrum -> rescale band positions per region (1 + d_r, r in <1000, 1000-2000, >2000 cm^-1)
              -> render (FWHM 16) -> multiply intensity by a smooth response exp(sum_k a_k P_k(nu))
              -> sqrt -> unit norm -> cosine with the measured spectrum.
Fitted with the same contrastive objective as the Siamese fine-tuning, on the same splits.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siamese as s  # noqa: E402

DATA, OUT = sys.argv[1:3]
DEV = "cuda"
ORDER = 4
XN = (s.GRID - 0.5 * (s.LO + 1800.0)) / (0.5 * (1800.0 - s.LO))  # [-1, 1] over the window
LEG = torch.stack([torch.special.legendre_polynomial_p(XN, k) for k in range(1, ORDER + 1)])  # ORDER x G


def calc_inputs(D, idx, d, a):
    nu = D.nu[idx]
    reg = torch.where(nu < 1000, d[0], torch.where(nu < 2000, d[1], d[2]))
    sp = s.render(nu * (1 + reg), D.inten[idx], torch.full((len(idx), 1), 16.0, device=DEV))
    resp = torch.exp(a @ LEG)
    x = (sp.clamp_min(0) * resp).sqrt()
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


def full_scores(D, qx, d, a):
    out = torch.empty((qx.shape[0], D.n), device=DEV)
    with torch.no_grad():
        for lo in range(0, D.n, 8192):
            idx = torch.arange(lo, min(D.n, lo + 8192), device=DEV)
            out[:, lo:lo + len(idx)] = qx @ calc_inputs(D, idx, d, a).T
    return out


def main():
    D = s.Data(DATA)
    rng = np.random.default_rng(0)
    results = {}
    splits = os.environ.get("FIT_SPLITS", "compound,class").split(",")
    for split in splits:
        if split == "transfer":  # fit on FIT_TRAIN_SPLITS (default split 0), evaluate every other query once
            tr = np.isin(D.qsplit, [int(v) for v in os.environ.get("FIT_TRAIN_SPLITS", "0").split(",")])
            folds = [(np.flatnonzero(~tr), np.flatnonzero(tr))]
        else:
            hg = np.unique(D.qclass, return_inverse=True)[1] if split == "class" else D.qgroup
            folds = [(np.flatnonzero(hg == h), np.flatnonzero(hg != h)) for h in np.unique(hg)]
        rows = []
        for hq, tq in folds:
            d = torch.zeros(3, device=DEV, requires_grad=True)
            a = torch.zeros(ORDER, device=DEV, requires_grad=True)
            opt = torch.optim.Adam([d, a], lr=0.02)
            cand_true = np.unique(D.qtrue[tq])
            tgt = torch.tensor(np.searchsorted(cand_true, D.qtrue[tq]), device=DEV)
            allowed = np.setdiff1d(np.arange(D.n), np.concatenate([D.qtrue[hq], cand_true]))
            for step in range(300):
                neg = rng.choice(allowed, 4096, replace=False)
                cand = torch.tensor(np.concatenate([cand_true, neg]), device=DEV)
                X = calc_inputs(D, cand, d * 0.02, a)  # d scaled: +-2% per unit
                loss = F.cross_entropy(D.qx[tq] @ X.T / 0.02, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            S = full_scores(D, D.qx[hq], d.detach() * 0.02, a.detach())
            for j, qi in enumerate(hq):
                t = int(D.qtrue[qi])
                sc = S[j]
                rk = int(1 + (sc > sc[t]).sum())
                p3 = torch.tensor(np.union1d(D.p3, [t]), device=DEV)
                p2 = torch.tensor(np.flatnonzero(D.formula == D.formula[t]), device=DEV)
                rows.append(dict(query=int(qi), name=str(D.qname[qi]), rank_full=rk,
                                 rank_P3=int(1 + (sc[p3] > sc[t]).sum()), rank_P2=int(1 + (sc[p2] > sc[t]).sum()),
                                 d=(d.detach() * 0.02).tolist(), a=a.detach().tolist()))
        results[split] = rows
        for p in ["full", "P3", "P2"]:
            v = np.array([r[f"rank_{p}"] for r in rows])
            print(f"fitted physics, leave-one-{split}-out, {p}: top1 {(v == 1).mean():.3f} top5 {(v <= 5).mean():.3f}")
        ex = rows[0]
        print("  example fitted params: d", np.round(ex["d"], 4), "a", np.round(ex["a"], 3))
    json.dump(results, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
