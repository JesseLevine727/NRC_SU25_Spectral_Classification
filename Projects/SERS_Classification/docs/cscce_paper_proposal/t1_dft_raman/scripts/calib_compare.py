"""Does the fitted calibration add anything over existing ways of tuning a DFT library for identification?
Step 2 data only (exploratory, not predeclared): fit on split 0 (our 37 spectra), score once on split 1 (81 compounds).

Methods
  houthuijs   Houthuijs et al. 2023 Anal. Chem. 95, 8998 (SI "Optimization of spectral similarity scoring"): grid
              search over line width and an intensity transform (I^p or log(I + c)), scale factor fixed, chosen by
              the geometric mean rank of the correct entry on the tuning spectra. Their width range (15-75 cm^-1,
              Gaussian) is widened to Raman band widths; line shape, whether the transform acts on the sticks or the
              rendered spectrum, and whether it acts on the DFT side only or on both sides are left open in the
              text, so they are part of the grid.
  lsq_scale   region-wise scale factors (below / above 1000 cm^-1) fitted by least squares on band pairs of the
              training compounds (Hungarian assignment within 40 cm^-1, as residuals.py), then sqrt cosine.
  calib_*     the 6-parameter calibration (fitted_control.py) and its ablations: stretch only (2), response only (4).
usage: calib_compare.py <data.npz> <out_dir>
"""
import itertools
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
qlin = D.qx.square()  # q_x = sqrt(I)/|.|  ->  I/sum(I)
split = D.qsplit
tr, te = np.flatnonzero(split == 0), np.flatnonzero(split == 1)


def render(nu, I, fwhm, shape):
    if shape == "lorentz":
        return s.render(nu, I, torch.full((nu.shape[0], 1), fwhm, device=DEV))
    return (I[:, :, None] * torch.exp(-4 * np.log(2) * (G[None, None] - nu[:, :, None]) ** 2 / fwhm ** 2)).sum(1)


def transform(x, t):
    x = x.clamp_min(0)
    if t[0] == "pow":
        return x ** t[1]
    return torch.log(x / (x.amax(-1, keepdim=True) + 1e-30) + t[1])  # log(I + c), I scaled to max 1


def unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def ranks_of(S, qidx):
    t = torch.tensor(D.qtrue[qidx], device=DEV)
    return (1 + (S > S.gather(1, t[:, None])).sum(1)).cpu().numpy()


def lib_scores(qidx, fn, chunk=4096):
    """fn(nu, inten) -> (n, G) unit-norm library rows; returns (len(qidx), n) cosine scores."""
    out = torch.empty((len(qidx), D.n), device=DEV)
    for lo in range(0, D.n, chunk):
        sl = slice(lo, min(D.n, lo + chunk))
        out[:, sl] = Q[qidx] @ fn(D.nu[sl], D.inten[sl]).T
    return out


def per_comp(r, idx):
    g = D.qgroup[idx]
    return np.array([(r[g == u] == 1).mean() for u in np.unique(g)])


results = {}

# ------------------------------------------------------------------ Houthuijs-style grid search
t0 = time.time()
FWHMS = [4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 75]
TRANS = [("pow", p) for p in (0.3, 0.4, 0.5, 0.6, 0.7, 1.0)] + [("log", c) for c in (0.01, 0.1, 1.0)]
rows = []
for shape, fw, t, where, sides in itertools.product(["gauss", "lorentz"], FWHMS, TRANS, ["sticks", "spectrum"],
                                                    ["dft", "both"]):
    if where == "sticks" and t[0] == "log":
        continue  # log(I + c) of sticks would give every absent band a nonzero floor; only defined on spectra
    Q = unit(transform(qlin, t)) if sides == "both" else unit(qlin)

    def fn(nu, I, fw=fw, shape=shape, t=t, where=where):
        if where == "sticks":
            return unit(render(nu, transform(I, t), fw, shape).clamp_min(0))
        return unit(transform(render(nu, I, fw, shape), t))
    r = ranks_of(lib_scores(np.arange(len(D.qtrue)), fn), np.arange(len(D.qtrue)))
    rows.append(dict(shape=shape, fwhm=fw, trans=f"{t[0]}{t[1]}", where=where, sides=sides,
                     train_gmr=float(np.exp(np.log(r[tr]).mean())), train_top1=float(per_comp(r[tr], tr).mean()),
                     test_top1=float(per_comp(r[te], te).mean()), ranks=r.tolist()))
best = min(rows, key=lambda w: w["train_gmr"])
results["houthuijs_selected"] = best
results["houthuijs_oracle"] = max(rows, key=lambda w: w["test_top1"])
results["houthuijs_grid"] = [{k: v for k, v in w.items() if k != "ranks"} for w in rows]
print(f"Houthuijs grid: {len(rows)} settings in {time.time() - t0:.0f} s; selected "
      f"{ {k: best[k] for k in ('shape', 'fwhm', 'trans', 'where', 'sides')} } train GMR {best['train_gmr']:.1f}",
      flush=True)

# ------------------------------------------------------------------ least-squares scale factors on assigned bands
Q = unit(qlin.sqrt())
num, den = np.zeros(2), np.zeros(2)
npairs = 0
grid = G.cpu().numpy()
for qi in tr:
    y = qlin[qi].cpu().numpy()
    pe, _ = find_peaks(y / y.max(), prominence=0.03)
    t = int(D.qtrue[qi])
    c = render(D.nu[t:t + 1], D.inten[t:t + 1], 10.0, "lorentz")[0].cpu().numpy()
    pc, _ = find_peaks(c / c.max(), prominence=0.03)
    if len(pe) == 0 or len(pc) == 0:
        continue
    cost = np.abs(grid[pe][:, None] - grid[pc][None, :])
    a, b = linear_sum_assignment(np.where(cost > 40, 1e6, cost))
    ok = cost[a, b] <= 40
    for ia, ib in zip(a[ok], b[ok]):
        ve, vc = grid[pe[ia]], grid[pc[ib]]
        k = int(vc >= 1000)
        num[k] += ve * vc
        den[k] += vc * vc
        npairs += 1
sc = num / den
print(f"least-squares extra scale factors from {npairs} assigned band pairs: below 1000 {sc[0]:.4f}, above {sc[1]:.4f}")


def fn_lsq(nu, I):
    f = torch.where(nu < 1000, torch.tensor(float(sc[0]), device=DEV), torch.tensor(float(sc[1]), device=DEV))
    return unit(render(nu * f, I, 16.0, "lorentz").clamp_min(0).sqrt())
r = ranks_of(lib_scores(np.arange(len(D.qtrue)), fn_lsq), np.arange(len(D.qtrue)))
results["lsq_scale"] = dict(scale=sc.tolist(), npairs=npairs, ranks=r.tolist())
r = ranks_of(lib_scores(np.arange(len(D.qtrue)), lambda nu, I: unit(render(nu, I, 16.0, "lorentz").clamp_min(0).sqrt())),
             np.arange(len(D.qtrue)))
results["published_scale"] = dict(ranks=r.tolist())

# ------------------------------------------------------------------ 6-parameter calibration and ablations (as fitted_control.py)
ORDER = 4
XN = (G - 0.5 * (s.LO + 1800.0)) / (0.5 * (1800.0 - s.LO))
LEG = torch.stack([torch.special.legendre_polynomial_p(XN, k) for k in range(1, ORDER + 1)])


def calc_inputs(idx, d, a):
    nu = D.nu[idx]
    reg = torch.where(nu < 1000, d[0], torch.where(nu < 2000, d[1], d[2]))
    sp = s.render(nu * (1 + reg), D.inten[idx], torch.full((len(idx), 1), 16.0, device=DEV))
    x = (sp.clamp_min(0) * torch.exp(a @ LEG)).sqrt()
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


for name, parts in [("calib_full", ("d", "a")), ("calib_stretch_only", ("d",)), ("calib_response_only", ("a",))]:
    rng = np.random.default_rng(0)
    d = torch.zeros(3, device=DEV, requires_grad="d" in parts)
    a = torch.zeros(ORDER, device=DEV, requires_grad="a" in parts)
    opt = torch.optim.Adam([p for p in (d, a) if p.requires_grad], lr=0.02)
    hq = np.flatnonzero(split != 0)
    cand_true = np.unique(D.qtrue[tr])
    tgt = torch.tensor(np.searchsorted(cand_true, D.qtrue[tr]), device=DEV)
    allowed = np.setdiff1d(np.arange(D.n), np.concatenate([D.qtrue[hq], cand_true]))
    for step in range(300):
        neg = rng.choice(allowed, 4096, replace=False)
        cand = torch.tensor(np.concatenate([cand_true, neg]), device=DEV)
        loss = F.cross_entropy(D.qx[tr] @ calc_inputs(cand, d * 0.02, a).T / 0.02, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    Q = D.qx
    dd, aa = d.detach() * 0.02, a.detach()
    out = torch.empty((len(D.qtrue), D.n), device=DEV)
    with torch.no_grad():
        for lo in range(0, D.n, 8192):
            idx = torch.arange(lo, min(D.n, lo + 8192), device=DEV)
            out[:, lo:lo + len(idx)] = Q @ calc_inputs(idx, dd, aa).T
    results[name] = dict(d=dd.tolist(), a=aa.tolist(), ranks=ranks_of(out, np.arange(len(D.qtrue))).tolist())
    print(f"{name}: d {np.round(dd.cpu().numpy(), 4)}, a {np.round(aa.cpu().numpy(), 3)}", flush=True)

os.makedirs(OUT, exist_ok=True)
json.dump(results, open(f"{OUT}/calib_compare.json", "w"), indent=1, default=float)
print(f"done {time.time() - t0:.0f} s")
