"""Step 3 summary. Applies the decision rule in step3/PREREG.md, then the secondary analyses."""
import glob
import json
import sys

import numpy as np
import pandas as pd

O, DATA = sys.argv[1:3]
d = np.load(DATA)
nq = len(d["q_true"])
split, group, cls, natoms = d["q_split"], d["q_group"], d["q_class"], d["q_natoms"]
test = split == 3
pd.set_option("display.width", 220)


def ranks_from(path, key="rank_full", sub=None):
    rows = json.load(open(path))
    df = pd.DataFrame(rows[sub] if sub else rows["results"])
    out = np.zeros(nq, int)
    out[df["query"].to_numpy()] = df[key].to_numpy()
    return out


def per_compound(hit, mask):
    g = group[mask]
    ug = np.unique(g)
    return ug, np.array([hit[mask][g == u].mean() for u in ug])


def boot(vals, n=2000, seed=0):
    """same bootstrap as step 2 (analyze_v2.py)"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n, len(vals)))
    return np.percentile(vals[idx].mean(1), [2.5, 97.5])


def table(R, mask, label):
    rows = []
    for k, r in R.items():
        ug, h = per_compound((r == 1).astype(float), mask)
        lo, hi = boot(h)
        rr = r[mask]
        rows.append(dict(method=k, compounds=len(ug), top1=h.mean(), ci95=f"{lo:.2f}-{hi:.2f}", top5=(rr <= 5).mean(),
                         top20=(rr <= 20).mean(), median_rank=np.median(rr)))
    print(f"\n{label}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


R = {}
fx = json.load(open(f"{O}/fixed_baseline.json"))
R["fixed_fwhm16"] = np.array(fx["fixed_sqrt_cos_fwhm16"]["full"]["ranks"])
R["fixed_fwhm10"] = np.array(fx["fixed_sqrt_cos_fwhm10"]["full"]["ranks"])
R["calib_37"] = ranks_from(f"{O}/fitted_transfer.json", sub="transfer")
R["calib_all127"] = ranks_from(f"{O}/fitted_transfer_all.json", sub="transfer")
R["calib_per_instrument"] = ranks_from(f"{O}/fitted_inst.json", sub="transfer")
for p in sorted(glob.glob(f"{O}/transfer_seed?.json")):
    R["siamese_" + p[-6]] = ranks_from(p)
R["zeroshot"] = ranks_from(f"{O}/transfer_seed0.json", key="rank_zeroshot")
for p in sorted(glob.glob(f"{O}/transfer_seed?_shuffle.json")):
    R["shuffle_" + p.split("seed")[1][0]] = ranks_from(p)

# ------------------------------------------------------------------ primary decision
ug, cal = per_compound((R["calib_37"] == 1).astype(float), test)
sia_keys = [k for k in R if k.startswith("siamese_")]
assert len(sia_keys) == 3
sia = np.mean([per_compound((R[k] == 1).astype(float), test)[1] for k in sia_keys], 0)
delta = sia - cal
lo, hi = boot(delta)
D = delta.mean()
verdict = "SIAMESE WINS" if (D >= 0.10 and lo > 0) else "CALIBRATION WINS" if (D <= -0.10 and hi < 0) else "TIE"
rep = pd.read_csv(DATA.replace(".npz", "_dft_report.csv"))
print(f"PRIMARY (predeclared): {len(ug)} fresh compounds without O-H/N-H, {int(test.sum())} spectra, "
      f"library {d['nu3'].shape[0]}")
print(f"  excluded (technical DFT failures): {rep[rep.status != 'ok'][['tag', 'smiles', 'status']].values.tolist()}")
print(f"  Siamese top-1 {sia.mean():.3f} (mean of 3 seeds)   calibration (fit on our 37) top-1 {cal.mean():.3f}")
print(f"  Delta {D:+.3f}, paired bootstrap 95% CI {lo:+.3f} to {hi:+.3f}  ->  {verdict}")
f16 = per_compound((R["fixed_fwhm16"] == 1).astype(float), test)[1]
sh = [per_compound((R[k] == 1).astype(float), test)[1].mean() for k in R if k.startswith("shuffle_")]
print(f"  control: fixed {f16.mean():.3f}; shuffle {np.round(sh, 3)}" +
      ("  FLAG: shuffle exceeds fixed by >5 points" if max(sh, default=0) > f16.mean() + 0.05 else ""))

# ------------------------------------------------------------------ secondary
table(R, test, "fresh test set, all methods")
for lab, m in [("<=20 atoms", natoms <= 20), ("21-25 atoms", (natoms > 20) & (natoms <= 25)), ("26-30 atoms", natoms > 25)]:
    table({k: v for k, v in R.items() if not k.startswith("shuffle")}, test & m, f"size tier {lab}")
t = pd.DataFrame({k: (R[k] == 1)[test] for k in ["fixed_fwhm16", "calib_37"] + sia_keys})
t["class"] = cls[test]
print("\ntop-1 by class")
print(t.groupby("class").mean().round(2).assign(n=t.groupby("class").size()).to_string())
per = pd.DataFrame({"name": d["q_name"][test], "atoms": natoms[test], "class": cls[test],
                    **{k: R[k][test] for k in ["fixed_fwhm16", "calib_37", "calib_all127"] + sia_keys}})
print("\nper-compound ranks")
print(per.sort_values("atoms").to_string(index=False))
