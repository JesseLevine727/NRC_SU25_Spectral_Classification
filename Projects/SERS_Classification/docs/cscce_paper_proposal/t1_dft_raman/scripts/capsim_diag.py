"""Diagnostics for capsim_t1.py on the step 2 data: plumbing check, what outranks the true entry, small-library behaviour."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import capsim_t1 as c  # noqa: E402

DATA, FIX = sys.argv[1:3]
P = c.Prep(DATA)
D = P.D
split, group = D.qsplit, D.qgroup
te = split == 1


def top1(r, mask):
    g = group[mask]
    return np.mean([(r[mask][g == u] == 1).mean() for u in np.unique(g)])


# 1. plumbing: sqrt-cosine through this script's queries and library must equal the step 2 fixed baseline
Q = torch.tensor(np.sqrt(np.clip(P.queries("none"), 0, None)), device=c.DEV, dtype=torch.float32)
L = P.lib(16.0).sqrt()
S = torch.nn.functional.normalize(Q, dim=1) @ torch.nn.functional.normalize(L, dim=1).T
r_cos = c.ranks(S, D.qtrue)
r_fix = np.array(json.load(open(FIX))["fixed_sqrt_cos_fwhm16"]["full"]["ranks"])
print(f"1. sqrt-cosine via capsim pipeline: test top-1 {top1(r_cos, te):.3f}; step 2 fixed baseline {top1(r_fix, te):.3f}; "
      f"ranks identical for {np.mean(r_cos == r_fix) * 100:.0f}% of queries")

# 2. what outranks the true entry (paper settings)
cfg = c.PAPER
S = P.scores(cfg)
r = c.ranks(S, D.qtrue)
Lpk = P.lib_peaks(cfg["fwhm"], cfg["ksm"], cfg["feat"])[:, :cfg["npeak"]]
ncp = (Lpk >= 0).sum(1).cpu().numpy()
print(f"2. paper settings: library entries by number of CPs found (<=10): "
      + ", ".join(f"{k}:{(ncp == k).mean() * 100:.0f}%" for k in range(1, 11)))
for qi in np.flatnonzero(te)[:6]:
    s = S[qi].cpu().numpy()
    top = np.argsort(-s)[:200]
    t = D.qtrue[qi]
    print(f"   {str(D.qname[qi])[:28]:28s} true: score {s[t]:.2f}, CPs {ncp[t]:2d}, rank {r[qi]:6d} | top-200: "
          f"score {s[top].min():.2f}-{s[top].max():.2f}, CPs median {np.median(ncp[top]):.0f}")
sym = c.PAPER | dict(sim="pearson", variant="both", h=5)

# 3. small library: the 110 query compounds' own entries only (paper used 16)
cand = np.unique(D.qtrue)
pos = {t: i for i, t in enumerate(cand)}
tq = np.array([pos[t] for t in D.qtrue])
print(f"3. candidate pool = the {len(cand)} query compounds only:")
for name, cf in [("paper 2023 settings", c.PAPER), ("selected (train-tuned)", sym | dict(npeak=10, h=5, ksm=3)),
                 ("sqrt-cosine", None)]:
    if cf is None:
        Ssub = (torch.nn.functional.normalize(Q, dim=1) @ torch.nn.functional.normalize(L, dim=1).T)[:, cand]
    else:
        Ssub = P.scores(cf)[:, cand]
    rs = c.ranks(Ssub, tq)
    print(f"   {name:24s} test top-1 {top1(rs, te):.3f}, top-5 {np.mean(rs[te] <= 5):.3f}")
for n in (16, 50):
    rng = np.random.default_rng(0)
    acc = {"capsim": [], "cos": []}
    Sc, Sk = P.scores(c.PAPER), torch.nn.functional.normalize(Q, dim=1) @ torch.nn.functional.normalize(L, dim=1).T
    for qi in np.flatnonzero(te):
        for _ in range(20):
            pool = np.concatenate([[D.qtrue[qi]], rng.choice(np.setdiff1d(cand, [D.qtrue[qi]]), n - 1, replace=False)])
            for k, M in (("capsim", Sc), ("cos", Sk)):
                v = M[qi, pool].cpu().numpy()
                acc[k].append(float(v[0] > v[1:].max()))
    print(f"   random pools of {n} query compounds (true + {n - 1}), paper settings: CaPSim top-1 {np.mean(acc['capsim']):.3f}, "
          f"sqrt-cosine {np.mean(acc['cos']):.3f}")
