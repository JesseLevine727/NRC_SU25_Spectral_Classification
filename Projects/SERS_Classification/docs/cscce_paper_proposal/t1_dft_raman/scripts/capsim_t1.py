"""CaPSim (Ju et al., ACS Nano 2023, 17, 21251) as used against DFT references (Ju et al., PNAS 2025,
122, e2427069122), scored on the step 2 data. Exploratory comparator; not part of any predeclared rule.

Definition followed (ACS Nano 2023, Computational Methods):
  preprocessing  baseline removal + l2 norm (airPLS, BaselineRemoval ZhangFit defaults, as PNAS 2025)
  CaPE           moving-average smoothing (K_smooth), [0,1] scaling, peaks with prominence >= 0.02
                 (Bajomo 2022 PNAS), keep the N_peak strongest by height (2023) or prominence (2025 option)
  step 1         CP locations K_l from the reference spectrum
  step 2         max-pool the reference over each K_l, normalise the compressed vector
  step 3         max-pool the query over the same K_l, normalise
  step 4         CaPSim = inner product of the two compressed vectors (2023); Pearson's r (2025)
Gaps in the text, covered as tuned options instead of guessed:
  - with a single DFT spectrum per compound a CP location is one point, so K_l = [p - h, p + h] with the
    PNAS 2025 shift tolerance h ("distance threshold ... roughly 10, 15, 20 cm^-1 peak shift"; total vs
    half width unclear, so h in 6-20 cm^-1)
  - variant "ref": CP locations from the reference only (2023 formal definition); variant "both": the
    query's own CaPE peaks are added ("CaPE ... applied to both DFT and experimental spectra", 2025)
  - inner product as written (sum over CPs) or divided by the number of CPs (the reported values,
    e.g. 0.311, are below 1, which a sum of [0,1] terms that includes the matched maximum cannot be)
  - DFT line width is not stated: Lorentzian FWHM 10 or 16 cm^-1 (as the fixed baselines)
  - queries were already baseline-corrected upstream (ALS for ours), so airPLS is an option, not forced
Settings are chosen on split 0 (our 37 spectra, the data the calibration is fitted on) and scored once on
split 1 (the 81 step 2 test compounds). Noise-floor subtraction (PNAS 2025) is a constant offset and is
removed by the min-max step, so it is omitted.

usage: capsim_t1.py check <data.npz>       literal vs vectorised agreement + property tests
       capsim_t1.py run <data.npz> <out_dir>
"""
import itertools
import json
import os
import sys
import time

os.environ.setdefault("T1_LO", "500")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siamese as s  # noqa: E402

DEV = "cuda"
G = len(s.GRID)
NMAX = 20
GRIDSPEC = dict(npeak=[5, 10, 15, 20], h=[3, 4, 5, 8, 10], feat=["height", "prom"], norm=["minmax", "linf"],
                sim=["sum", "mean", "pearson"], fwhm=[10.0, 16.0], ksm=[1, 3], base=["none", "airpls"],
                variant=["ref", "both"])
# ACS Nano 2023 settings mapped to our 2 cm^-1 grid: N_peak 10, K_smooth 5 (~4 cm^-1 -> 3 points),
# w_max 36 (~30 cm^-1 wide -> +-15 cm^-1 -> h = 8 points)
PAPER = dict(npeak=10, h=8, feat="height", norm="minmax", sim="sum", fwhm=16.0, ksm=3, base="airpls", variant="ref")


# ------------------------------------------------------------------ preprocessing and CaPE
def airpls(x, lam=100, porder=1, itermax=15):
    """Zhang 2010 airPLS, as BaselineRemoval.ZhangFit (lambda_=100, porder=1, repitition=15)."""
    m = len(x)
    E = sparse.eye(m, format="csc")
    D = E
    for _ in range(porder):
        D = D[1:] - D[:-1]
    H = lam * (D.T @ D)
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = spsolve(sparse.csc_matrix(sparse.diags(w) + H), w * x)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * np.abs(x).sum() or i == itermax:
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = w[-1] = np.exp(i * d[d < 0].max() / dssn)
    return x - z


def preprocess(x, base):
    y = airpls(x) if base == "airpls" else x.copy()
    return y / (np.linalg.norm(y) + 1e-12)


def cape(x, ksm, nmax=NMAX):
    """Peak positions sorted by height and by prominence (strongest first), padded with -1."""
    y = np.convolve(x, np.ones(ksm) / ksm, mode="same") if ksm > 1 else x
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p, props = find_peaks(y, prominence=0.02)
    out = {}
    for feat, v in (("height", y[p]), ("prom", props["prominences"])):
        o = p[np.argsort(-v, kind="stable")][:nmax]
        out[feat] = np.concatenate([o, -np.ones(nmax - len(o), int)])
    return out


# ------------------------------------------------------------------ literal transcription (slow, per pair)
def capsim_literal(q, r, ref_peaks, q_peaks, h, norm, sim):
    locs = [p for p in ref_peaks if p >= 0] + ([p for p in q_peaks if p >= 0] if q_peaks is not None else [])
    if not locs:
        return -np.inf
    Ks = [range(max(0, p - h), min(G, p + h + 1)) for p in locs]           # step 1
    rc = np.array([max(r[k] for k in K) for K in Ks])                    # step 2
    qc = np.array([max(q[k] for k in K) for K in Ks])                    # step 3

    def nz(c):
        if norm == "linf":
            return c / c.max() if c.max() > 0 else np.zeros_like(c)
        return (c - c.min()) / (c.max() - c.min()) if c.max() > c.min() else np.zeros_like(c)
    rc, qc = nz(rc), nz(qc)
    if sim == "sum":                                                     # step 4
        return float(qc @ rc)
    if sim == "mean":
        return float(qc @ rc) / len(locs)
    if len(locs) < 2 or qc.std() == 0 or rc.std() == 0:
        return -1.0
    return float(np.corrcoef(qc, rc)[0, 1])


# ------------------------------------------------------------------ vectorised (GPU)
def maxfilt(X, h):
    return torch.nn.functional.max_pool1d(X[:, None], 2 * h + 1, 1, h)[:, 0]


def _norm(c, m, norm):
    big = torch.finfo(c.dtype).max
    mx = torch.where(m, c, -big).amax(-1, keepdim=True)
    if norm == "linf":
        out = torch.where(mx > 0, c / mx.clamp_min(1e-30), torch.zeros_like(c))
    else:
        mn = torch.where(m, c, big).amin(-1, keepdim=True)
        out = torch.where(mx > mn, (c - mn) / (mx - mn).clamp_min(1e-30), torch.zeros_like(c))
    return out * m


def _sim(qc, rc, m, sim):
    n = m.sum(-1)
    if sim == "sum":
        out = (qc * rc).sum(-1)
    elif sim == "mean":
        out = (qc * rc).sum(-1) / n.clamp_min(1)
    else:
        nn_ = n.clamp_min(1)[..., None]
        qm = qc - (qc.sum(-1, keepdim=True) / nn_)
        rm = rc - (rc.sum(-1, keepdim=True) / nn_)
        qm, rm = qm * m, rm * m
        den = (qm.square().sum(-1) * rm.square().sum(-1)).sqrt()
        out = torch.where((n >= 2) & (den > 0), (qm * rm).sum(-1) / den.clamp_min(1e-30), torch.full_like(den, -1.0))
    return torch.where(n > 0, out, torch.full_like(out, -float("inf")))


def capsim_scores(Q, Qpk, L, Lpk, h, norm, sim, variant, chunk=16384):
    """Q (nq,G) preprocessed queries, Qpk (nq,N) their CaPE peaks; L (nl,G) references, Lpk (nl,N).
    Returns (nq, nl) CaPSim scores."""
    MQ, nq, N = maxfilt(Q, h), Q.shape[0], Lpk.shape[1]
    out = torch.empty((nq, L.shape[0]), device=DEV)
    for lo in range(0, L.shape[0], chunk):
        sl = slice(lo, min(L.shape[0], lo + chunk))
        ML, pk = maxfilt(L[sl], h), Lpk[sl]
        m = pk >= 0
        pc = pk.clamp_min(0)
        rc = ML.gather(1, pc)[None].expand(nq, -1, -1)                     # (nq, nl, N)
        qc = MQ[:, pc.reshape(-1)].reshape(nq, -1, N)
        mm = m[None].expand(nq, -1, -1)
        if variant == "both":
            qm = Qpk >= 0
            qpc = Qpk.clamp_min(0)
            rc2 = ML[:, qpc.reshape(-1)].reshape(-1, nq, Qpk.shape[1]).transpose(0, 1)   # (nq, nl, Nq)
            qc2 = MQ.gather(1, qpc)[:, None].expand(-1, rc.shape[1], -1)
            rc, qc = torch.cat([rc, rc2], -1), torch.cat([qc, qc2], -1)
            mm = torch.cat([mm, qm[:, None].expand(-1, rc.shape[1], -1)], -1)
        out[:, sl] = _sim(_norm(qc, mm, norm), _norm(rc, mm, norm), mm, sim)
    return out


# ------------------------------------------------------------------ data
class Prep:
    def __init__(self, path):
        self.D = s.Data(path)
        self.qlin = (self.D.qx.square()).cpu().numpy().astype(np.float64)   # q_x = sqrt(I)/|.|  ->  I/sum(I)
        self._L, self._Lpk, self._Q, self._Qpk = {}, {}, {}, {}

    def lib(self, fwhm):
        if fwhm not in self._L:
            D, X = self.D, torch.empty((self.D.n, G), device=DEV)
            for lo in range(0, D.n, 4096):
                sl = slice(lo, min(D.n, lo + 4096))
                x = s.render(D.nu[sl], D.inten[sl], torch.full((D.nu[sl].shape[0], 1), fwhm, device=DEV)).clamp_min(0)
                X[sl] = x / (x.norm(dim=1, keepdim=True) + 1e-12)
            self._L[fwhm] = X
        return self._L[fwhm]

    def lib_peaks(self, fwhm, ksm, feat):
        key = (fwhm, ksm)
        if key not in self._Lpk:
            t = time.time()
            X = self.lib(fwhm).cpu().numpy()
            pk = {"height": np.empty((len(X), NMAX), np.int64), "prom": np.empty((len(X), NMAX), np.int64)}
            for i, x in enumerate(X):
                c = cape(x, ksm)
                pk["height"][i], pk["prom"][i] = c["height"], c["prom"]
            self._Lpk[key] = {k: torch.tensor(v, device=DEV) for k, v in pk.items()}
            print(f"  CaPE on library, FWHM {fwhm:g}, K {ksm}: {time.time() - t:.0f} s", flush=True)
        return self._Lpk[key][feat]

    def queries(self, base):
        if base not in self._Q:
            self._Q[base] = np.stack([preprocess(x, base) for x in self.qlin])
        return self._Q[base]

    def q_peaks(self, base, ksm, feat):
        key = (base, ksm)
        if key not in self._Qpk:
            c = [cape(x, ksm) for x in self.queries(base)]
            self._Qpk[key] = {f: torch.tensor(np.stack([e[f] for e in c]), device=DEV) for f in ("height", "prom")}
        return self._Qpk[key][feat]

    def scores(self, cfg, qidx=None):
        Q = torch.tensor(self.queries(cfg["base"]), device=DEV, dtype=torch.float32)
        Qpk = self.q_peaks(cfg["base"], cfg["ksm"], cfg["feat"])[:, :cfg["npeak"]]
        if qidx is not None:
            Q, Qpk = Q[qidx], Qpk[qidx]
        Lpk = self.lib_peaks(cfg["fwhm"], cfg["ksm"], cfg["feat"])[:, :cfg["npeak"]]
        return capsim_scores(Q, Qpk, self.lib(cfg["fwhm"]), Lpk, cfg["h"], cfg["norm"], cfg["sim"], cfg["variant"])


def ranks(S, true):
    t = torch.tensor(true, device=DEV)
    st = S.gather(1, t[:, None])
    return (1 + (S > st).sum(1)).cpu().numpy()


# ------------------------------------------------------------------ checks
def check(path):
    P = Prep(path)
    rng = np.random.default_rng(0)
    qsel = np.array([0, 5, 20, 40, 60, 100])
    nmis = 0
    for cfg in [PAPER, dict(PAPER, variant="both", sim="pearson", norm="linf", feat="prom", npeak=5, h=3, ksm=1,
                            base="none", fwhm=10.0), dict(PAPER, sim="mean", npeak=20, h=10)]:
        S = P.scores(cfg, qsel).cpu().numpy()
        lib = P.lib(cfg["fwhm"]).cpu().numpy()
        Lpk = P.lib_peaks(cfg["fwhm"], cfg["ksm"], cfg["feat"]).cpu().numpy()[:, :cfg["npeak"]]
        Q = P.queries(cfg["base"])
        Qpk = P.q_peaks(cfg["base"], cfg["ksm"], cfg["feat"]).cpu().numpy()[:, :cfg["npeak"]]
        for a, qi in enumerate(qsel):
            for j in np.concatenate([[P.D.qtrue[qi]], rng.choice(P.D.n, 200, replace=False)]):
                v = capsim_literal(Q[qi], lib[j], Lpk[j], Qpk[qi] if cfg["variant"] == "both" else None,
                                   cfg["h"], cfg["norm"], cfg["sim"])
                if not (np.isclose(v, S[a, j], atol=2e-5) or (np.isinf(v) and np.isinf(S[a, j]))):
                    nmis += 1
                    if nmis < 5:
                        print("  mismatch", cfg, qi, j, v, S[a, j])
    print(f"literal vs vectorised: {nmis} mismatches over {3 * len(qsel) * 201} pairs")

    # property tests on synthetic spectra (one-query, one-reference)
    x = np.arange(G) * 2.0 + s.LO

    def spec(pos, amp, w=8.0):
        return sum(a * (w / 2) ** 2 / ((x - p) ** 2 + (w / 2) ** 2) for p, a in zip(pos, amp)) + 1e-6
    ref = spec([700, 1000, 1300, 1600], [1.0, 0.6, 0.8, 0.4])
    rp = cape(ref / np.linalg.norm(ref), 1)["height"][:4]
    print("CaPE on synthetic reference, peaks (cm^-1) by height:", (s.LO + 2 * rp).tolist(), "(expect 700, 1300, 1000, 1600)")

    def one(q, h=5, norm="minmax", sim="sum"):
        return capsim_literal(q / np.linalg.norm(q), ref / np.linalg.norm(ref), rp, None, h, norm, sim)

    def pooled(q, h=5):
        return [round(max(q[max(0, p - h):p + h + 1]), 3) for p in rp]
    noise = 0.03 * np.random.default_rng(1).random(G)
    amp = [1, .6, .8, .4]
    sh6, sh40 = spec([706, 1006, 1306, 1606], amp), spec([740, 1040, 1340, 1640], amp)
    print(f"  identical: {one(ref):.3f}; x7 scale: {one(7 * ref):.3f}; +0.2 offset (minmax): {one(ref + 0.2):.3f}")
    print(f"  raw pooled query at the reference CPs (h = 10 cm^-1): unshifted {pooled(ref)}, +6 cm^-1 {pooled(sh6)}, "
          f"+40 cm^-1 {pooled(sh40)}")
    print(f"  noise-free: +6 cm^-1 {one(sh6):.3f}; +40 cm^-1 {one(sh40):.3f}  (min-max rescales the tails back up)")
    print(f"  with 3% noise: unshifted {one(ref + noise):.3f}; +6 cm^-1 {one(sh6 + noise):.3f}; "
          f"+40 cm^-1 {one(sh40 + noise):.3f}")
    print(f"  extra query peak away from reference CPs (ignored): {one(ref + spec([850], [1.5])):.3f}")


# ------------------------------------------------------------------ run
def main(path, out):
    os.makedirs(out, exist_ok=True)
    P = Prep(path)
    D = P.D
    split, group, hbd = D.qsplit, D.qgroup, np.load(path)["q_hbd"]
    keys = list(GRIDSPEC)
    cfgs = [dict(zip(keys, v)) for v in itertools.product(*GRIDSPEC.values())]
    # drop configurations that cannot differ: airPLS/none and K only matter through their inputs, so keep all;
    # sum/mean coincide when every reference has N peaks, which is checked by keeping both
    rows = []
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        r = ranks(P.scores(cfg), D.qtrue)
        rows.append(dict(cfg=cfg, ranks=r.tolist()))
        if i % 200 == 0:
            print(f"  {i}/{len(cfgs)} configs, {time.time() - t0:.0f} s", flush=True)

    def per_comp(r, mask):
        g = group[mask]
        return np.array([(r[mask][g == u] == 1).mean() for u in np.unique(g)])

    def score(r, mask):
        return per_comp(r, mask).mean(), np.mean(np.log(r[mask]))
    tr, te = split == 0, split == 1
    for row in rows:
        r = np.array(row["ranks"])
        row["train_top1"], row["train_logrank"] = score(r, tr)
        row["test_top1"], _ = score(r, te)
    best_tr = max(rows, key=lambda w: (w["train_top1"], -w["train_logrank"]))
    paper = next(w for w in rows if w["cfg"] == PAPER)
    best_by_variant = {v: max([w for w in rows if w["cfg"]["variant"] == v],
                              key=lambda w: (w["train_top1"], -w["train_logrank"])) for v in GRIDSPEC["variant"]}
    oracle = max(rows, key=lambda w: w["test_top1"])
    json.dump(dict(selected=best_tr, paper=paper, by_variant=best_by_variant, oracle=oracle,
                   all=[dict(cfg=w["cfg"], train_top1=w["train_top1"], test_top1=w["test_top1"]) for w in rows]),
              open(f"{out}/capsim_step2.json", "w"), indent=1, default=float)
    print(f"done {len(cfgs)} configs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2])
    else:
        main(sys.argv[2], sys.argv[3])
