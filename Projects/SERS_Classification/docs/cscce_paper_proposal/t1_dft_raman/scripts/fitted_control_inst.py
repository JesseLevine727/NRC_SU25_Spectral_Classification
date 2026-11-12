"""Post-hoc control for the pooled step 2 runs: the fitted calibration with one parameter set per
instrument (VibraCLIP FT-Raman vs our 785 nm spectra), so it can match an instrument-aware network.

Same model, objective, steps and folds as fitted_control.py; only the parameters are doubled.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siamese as s  # noqa: E402
from fitted_control import ORDER, calc_inputs, full_scores  # noqa: E402

DATA, OUT = sys.argv[1:3]
DEV = "cuda"


def main():
    D = s.Data(DATA)
    inst = (D.qsource == "vibraclip").astype(int)
    rng = np.random.default_rng(0)
    results = {}
    for split in os.environ.get("INST_SPLITS", "compound,class").split(","):
        if split == "transfer":  # step 3: fit on splits 0-2, evaluate split 3
            folds = [(np.flatnonzero(D.qsplit == 3), np.flatnonzero(D.qsplit < 3))]
        else:
            hg = np.unique(D.qclass, return_inverse=True)[1] if split == "class" else D.qgroup
            folds = [(np.flatnonzero(hg == h), np.flatnonzero(hg != h)) for h in np.unique(hg)]
        rows = []
        for hq, tq in folds:
            d = torch.zeros(2, 3, device=DEV, requires_grad=True)
            a = torch.zeros(2, ORDER, device=DEV, requires_grad=True)
            opt = torch.optim.Adam([d, a], lr=0.02)
            cand_true = np.unique(D.qtrue[tq])
            tgt = torch.tensor(np.searchsorted(cand_true, D.qtrue[tq]), device=DEV)
            it = torch.tensor(inst[tq], device=DEV)[:, None]
            allowed = np.setdiff1d(np.arange(D.n), np.concatenate([D.qtrue[hq], cand_true]))
            for step in range(300):
                cand = torch.tensor(np.concatenate([cand_true, rng.choice(allowed, 4096, replace=False)]), device=DEV)
                S0 = D.qx[tq] @ calc_inputs(D, cand, d[0] * 0.02, a[0]).T
                S1 = D.qx[tq] @ calc_inputs(D, cand, d[1] * 0.02, a[1]).T
                loss = F.cross_entropy(torch.where(it == 1, S1, S0) / 0.02, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            for k in (0, 1):
                hk = hq[inst[hq] == k]
                if not len(hk):
                    continue
                S = full_scores(D, D.qx[hk], d[k].detach() * 0.02, a[k].detach())
                for j, qi in enumerate(hk):
                    t = int(D.qtrue[qi])
                    rows.append(dict(query=int(qi), name=str(D.qname[qi]), rank_full=int(1 + (S[j] > S[j, t]).sum()),
                                     inst=k, d=(d[k].detach() * 0.02).tolist(), a=a[k].detach().tolist()))
        results[split] = rows
        v = np.array([r["rank_full"] for r in rows])
        print(f"per-instrument fitted, leave-one-{split}-out: top1 {(v == 1).mean():.3f} top5 {(v <= 5).mean():.3f}", flush=True)
    json.dump(results, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
