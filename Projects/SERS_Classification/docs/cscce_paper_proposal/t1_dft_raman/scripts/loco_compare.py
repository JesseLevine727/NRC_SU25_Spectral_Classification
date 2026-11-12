"""Leave-one-compound-out over the 110 step 2 compounds (127 spectra): does fitting the frequency stretch to the
identification loss beat fitting it by least squares on assigned bands? Exploratory; step 3 compounds untouched.

Per fold (one compound held out, all spectra of the other 109 used for fitting):
  published   three-range literature scale factors, sqrt cosine (no fit)
  lsq         extra region-wise scale factors by least squares on assigned band pairs (as calib_compare.py)
  stretch     the 2 stretch parameters fitted with the InfoNCE retrieval loss (as fitted_control.py)
  full        all 6 calibration parameters, same loss (= fitted_control.py leave-one-compound-out)
usage: loco_compare.py <data.npz> <out.json>
"""
import json
import os
import sys
import time

os.environ.setdefault("T1_LO", "500")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siamese as s  # noqa: E402

DEV = "cuda"
DATA, OUT = sys.argv[1:3]
D = s.Data(DATA)
G = s.GRID
grid = G.cpu().numpy()
nq = len(D.qtrue)
ORDER = 4
XN = (G - 0.5 * (s.LO + 1800.0)) / (0.5 * (1800.0 - s.LO))
LEG = torch.stack([torch.special.legendre_polynomial_p(XN, k) for k in range(1, ORDER + 1)])
FW = torch.tensor(16.0, device=DEV)


def calc_inputs(idx, d, a):
    nu = D.nu[idx]
    reg = torch.where(nu < 1000, d[0], torch.where(nu < 2000, d[1], d[2]))
    sp = s.render(nu * (1 + reg), D.inten[idx], FW.expand(len(idx), 1))
    x = (sp.clamp_min(0) * torch.exp(a @ LEG)).sqrt()
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


@torch.no_grad()
def rank_held(hq, d, a):
    S = torch.empty((len(hq), D.n), device=DEV)
    for lo in range(0, D.n, 8192):
        idx = torch.arange(lo, min(D.n, lo + 8192), device=DEV)
        S[:, lo:lo + len(idx)] = D.qx[hq] @ calc_inputs(idx, d, a).T
    t = torch.tensor(D.qtrue[hq], device=DEV)
    return (1 + (S > S.gather(1, t[:, None])).sum(1)).cpu().numpy()


def fit(tq, hq, parts, rng):
    d = torch.zeros(3, device=DEV, requires_grad="d" in parts)
    a = torch.zeros(ORDER, device=DEV, requires_grad="a" in parts)
    opt = torch.optim.Adam([p for p in (d, a) if p.requires_grad], lr=0.02)
    cand_true = np.unique(D.qtrue[tq])
    tgt = torch.tensor(np.searchsorted(cand_true, D.qtrue[tq]), device=DEV)
    allowed = np.setdiff1d(np.arange(D.n), np.concatenate([D.qtrue[hq], cand_true]))
    for _ in range(300):
        neg = rng.choice(allowed, 4096, replace=False)
        cand = torch.tensor(np.concatenate([cand_true, neg]), device=DEV)
        loss = F.cross_entropy(D.qx[tq] @ calc_inputs(cand, d * 0.02, a).T / 0.02, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return d.detach() * 0.02, a.detach()


# band pairs per spectrum (measured peak, its own DFT peak), for the least-squares fit
pairs = []
for qi in range(nq):
    y = D.qx[qi].square().cpu().numpy()
    pe, _ = find_peaks(y / y.max(), prominence=0.03)
    t = int(D.qtrue[qi])
    c = s.render(D.nu[t:t + 1], D.inten[t:t + 1], torch.full((1, 1), 10.0, device=DEV))[0].cpu().numpy()
    pc, _ = find_peaks(c / c.max(), prominence=0.03)
    pr = []
    if len(pe) and len(pc):
        cost = np.abs(grid[pe][:, None] - grid[pc][None, :])
        ia, ib = linear_sum_assignment(np.where(cost > 40, 1e6, cost))
        pr = [(grid[pe[i]], grid[pc[j]]) for i, j in zip(ia, ib) if cost[i, j] <= 40]
    pairs.append(pr)

zero_d, zero_a = torch.zeros(3, device=DEV), torch.zeros(ORDER, device=DEV)
R = {k: np.zeros(nq, int) for k in ("published", "lsq", "stretch", "full")}
P = {k: {} for k in ("lsq", "stretch", "full")}
rng = np.random.default_rng(0)
t0 = time.time()
groups = np.unique(D.qgroup)
for gi, h in enumerate(groups):
    hq = np.flatnonzero(D.qgroup == h)
    tq = np.flatnonzero(D.qgroup != h)
    R["published"][hq] = rank_held(hq, zero_d, zero_a)
    num, den = np.zeros(2), np.zeros(2)
    for qi in tq:
        for ve, vc in pairs[qi]:
            k = int(vc >= 1000)
            num[k] += ve * vc
            den[k] += vc * vc
    sc = num / den
    d_lsq = torch.tensor([sc[0] - 1, sc[1] - 1, 0.0], device=DEV, dtype=torch.float32)  # same parameterisation as d
    R["lsq"][hq] = rank_held(hq, d_lsq, zero_a)
    P["lsq"][int(h)] = d_lsq[:2].tolist()
    for name, parts in (("stretch", ("d",)), ("full", ("d", "a"))):
        d, a = fit(tq, hq, parts, rng)
        R[name][hq] = rank_held(hq, d, a)
        P[name][int(h)] = d[:2].tolist() + a.tolist()
    if gi % 10 == 0:
        print(f"  fold {gi}/{len(groups)}, {time.time() - t0:.0f} s; lsq {np.round(sc - 1, 4)}, "
              f"stretch {np.round(P['stretch'][int(h)][:2], 4)}", flush=True)
json.dump(dict(ranks={k: v.tolist() for k, v in R.items()}, params=P), open(OUT, "w"))
print(f"done {time.time() - t0:.0f} s")
