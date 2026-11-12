"""Step 2 summary. Applies the decision rule in step2/PREREG.md, then the secondary analyses."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

O, DATA = sys.argv[1:3]
d = np.load(DATA)
nq = len(d["q_true"])
split, group, cls, hbd, names = d["q_split"], d["q_group"], d["q_class"], d["q_hbd"], d["q_name"]
pd.set_option("display.width", 220)


def ranks_from(path, key="rank_full", sub=None):
    rows = json.load(open(path))
    rows = rows[sub] if sub else rows["results"]
    df = pd.DataFrame(rows)
    if "ntrain" in df:
        df = df[df.ntrain == df.ntrain.max()]
    out = np.zeros(nq, int)
    out[df["query"].to_numpy()] = df[key].to_numpy()
    return out


def per_compound(hit, mask):
    """mean hit per compound over its spectra, for the queries in mask."""
    g = group[mask]
    ug = np.unique(g)
    return ug, np.array([hit[mask][g == u].mean() for u in ug])


def boot(vals, n=2000, seed=0):
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


def load_primary(o):
    R = {}
    fx = json.load(open(f"{o}/fixed_baseline.json"))
    R["fixed_fwhm16"] = np.array(fx["fixed_sqrt_cos_fwhm16"]["full"]["ranks"])
    R["fixed_fwhm10"] = np.array(fx["fixed_sqrt_cos_fwhm10"]["full"]["ranks"])
    R["fitted"] = ranks_from(f"{o}/fitted_transfer.json", sub="transfer")
    for p in sorted(glob.glob(f"{o}/transfer_seed?.json")):
        R["siamese_" + p[-6]] = ranks_from(p)
    R["zeroshot"] = ranks_from(f"{o}/transfer_seed0.json", key="rank_zeroshot")
    for p in sorted(glob.glob(f"{o}/transfer_seed?_shuffle.json")):
        R["shuffle_" + p.split("seed")[1][0]] = ranks_from(p)
    return R


# ------------------------------------------------------------------ primary decision
R = load_primary(O)
test = split == 1
ug, fit = per_compound((R["fitted"] == 1).astype(float), test)
sia_keys = [k for k in R if k.startswith("siamese_")]
sia = np.mean([per_compound((R[k] == 1).astype(float), test)[1] for k in sia_keys], 0)
delta = sia - fit
lo, hi = boot(delta)
D = delta.mean()
verdict = "SIAMESE WINS" if (D >= 0.10 and lo > 0) else "CALIBRATION WINS" if (D <= -0.10 and hi < 0) else "TIE"
print(f"PRIMARY (predeclared): {len(ug)} unseen VibraCLIP compounds, full library {d['nu3'].shape[0]}")
print(f"  Siamese top-1 {sia.mean():.3f} (mean of {len(sia_keys)} seeds)   fitted top-1 {fit.mean():.3f}")
print(f"  Delta {D:+.3f}, paired bootstrap 95% CI {lo:+.3f} to {hi:+.3f}  ->  {verdict}")
fx = per_compound((R["fixed_fwhm16"] == 1).astype(float), test)[1]
sh = [per_compound((R[k] == 1).astype(float), test)[1].mean() for k in R if k.startswith("shuffle_")]
print(f"  control: fixed {fx.mean():.3f}; shuffle {np.round(sh, 3)}" +
      ("  FLAG: shuffle exceeds fixed by >5 points" if max(sh, default=0) > fx.mean() + 0.05 else ""))
table(R, test, "primary test set, all methods")
table(R, test & hbd.astype(bool), "test compounds WITH an O-H/N-H donor")
table(R, test & ~hbd.astype(bool), "test compounds WITHOUT an O-H/N-H donor")
table({k: v for k, v in R.items() if not k.startswith("shuffle")}, split == 2,
      "VibraCLIP spectra of our training compounds (same compound, new instrument)")
t = pd.DataFrame({k: (R[k] == 1)[test] for k in ["fixed_fwhm16", "fitted"] + sia_keys})
t["class"] = cls[test]
print("\nprimary test, top-1 by class")
print(t.groupby("class").mean().round(2).assign(n=t.groupby("class").size()).to_string())

# ------------------------------------------------------------------ sensitivity
if os.path.exists(f"{O}/nocorr/fitted_transfer.json"):
    Rn = load_primary(f"{O}/nocorr")
    table({k: v for k, v in Rn.items() if not k.startswith("zero")}, test, "sensitivity: no 1064 nm correction")

# ------------------------------------------------------------------ pooled 110 compounds
pp = f"{O}/pooled"
if os.path.exists(f"{pp}/fitted_control.json"):
    P = {"fixed_fwhm16": R["fixed_fwhm16"],
         "fitted_compound": ranks_from(f"{pp}/fitted_control.json", sub="compound"),
         "fitted_class": ranks_from(f"{pp}/fitted_control.json", sub="class")}
    for p in sorted(glob.glob(f"{pp}/finetune_d2_w32_s1.json")):
        P["siamese_compound"] = ranks_from(p)
    for p in sorted(glob.glob(f"{pp}/byclass/finetune_*byclass_seed*.json")):
        P["siamese_class_" + p[-6]] = ranks_from(p)
    for p in sorted(glob.glob(f"{pp}/finetune_*shuffle*.json")):
        P["shuffle_compound"] = ranks_from(p)
    allq = np.ones(nq, bool)
    table(P, allq, f"pooled: {len(np.unique(group))} compounds, {nq} spectra")
    table(P, split == 1, "pooled, VibraCLIP compounds only")
    p0 = f"{pp}/finetune_d2_w32_s1.json"
    if os.path.exists(p0):
        df = pd.DataFrame(json.load(open(p0))["results"])
        print("\npooled learning curve, leave-one-compound-out (top-1 / top-5, per spectrum)")
        print(df.groupby("ntrain")["rank_full"].apply(lambda x: f"{(x == 1).mean():.3f} / {(x <= 5).mean():.3f}").to_string())
