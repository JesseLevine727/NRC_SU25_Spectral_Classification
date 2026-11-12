"""Summarise the 29-compound rerun: fixed, fitted calibration and Siamese, on all / original / new queries."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siamese import CLASSES  # noqa: E402

O, DATA = sys.argv[1:3]
d = np.load(DATA)
names = d["q_name"]
nq = len(names)
OLD = np.arange(nq) < 28
rng = np.random.default_rng(0)
groups = d["q_group"]


def stats(r, mask=None):
    r = np.asarray(r)
    if mask is not None:
        r = r[mask]
    return dict(n=len(r), top1=(r == 1).mean(), top5=(r <= 5).mean(), top20=(r <= 20).mean(), median=np.median(r))


def boot_ci(r):
    """95% bootstrap CI of top-1, resampling compounds (not spectra)."""
    r = np.asarray(r)
    ug = np.unique(groups)
    vals = []
    for _ in range(2000):
        gs = rng.choice(ug, len(ug))
        idx = np.concatenate([np.flatnonzero(groups == g) for g in gs])
        vals.append((r[idx] == 1).mean())
    return np.percentile(vals, [2.5, 97.5])


def ranks_from_ft(path, full_only=True):
    r = json.load(open(path))["results"]
    df = pd.DataFrame(r)
    df = df[df.ntrain == df.ntrain.max()]
    out = np.zeros(nq, int)
    out[df["query"].to_numpy()] = df["rank_full"].to_numpy()
    return out, pd.DataFrame(r)


rows, per = [], {}
fx = json.load(open(f"{O}/fixed_baseline.json"))
per["fixed"] = np.array(fx["fixed_sqrt_cos_fwhm16"]["full"]["ranks"])
fc = json.load(open(f"{O}/fitted_control.json"))
for split in ["compound", "class"]:
    v = np.zeros(nq, int)
    for row in fc[split]:
        v[row["query"]] = row["rank_full"]
    per[f"fitted_{split}"] = v
for p in sorted(glob.glob(f"{O}/seed*/finetune_*.json")):
    per[f"siamese_compound_{p.split('/')[-2]}"] = ranks_from_ft(p)[0]
for p in sorted(glob.glob(f"{O}/byclass/finetune_*byclass_seed*.json")):
    per[f"siamese_class_{p.split('_')[-1][:-5]}"] = ranks_from_ft(p)[0]
for p in sorted(glob.glob(f"{O}/shuffle/finetune_*shuffle*.json")):
    per["shuffle_" + ("class" if "byclass" in p else "compound") + "_" + p.split("shuffle")[-1][:1]] = ranks_from_ft(p)[0]

for k, v in per.items():
    lo, hi = boot_ci(v)
    a, o, n = stats(v), stats(v, OLD), stats(v, ~OLD)
    rows.append(dict(method=k, top1=a["top1"], ci95=f"{lo:.2f}-{hi:.2f}", top5=a["top5"], top20=a["top20"],
                     median=a["median"], old28_top1=o["top1"], new9_top1=n["top1"], new9_top5=n["top5"]))
pd.set_option("display.width", 200)
print(f"{nq} queries, {len(np.unique(groups))} compounds; full library {d['nu3'].shape[0]} entries")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# learning curve (seed0 run)
p0 = glob.glob(f"{O}/seed0/finetune_*.json")
if p0:
    _, df = ranks_from_ft(p0[0])
    print("\nlearning curve, leave-one-compound-out (held-out top-1 / top-5, full library)")
    print(df.groupby("ntrain")["rank_full"].apply(lambda x: f"{(x == 1).mean():.3f} / {(x <= 5).mean():.3f}").to_string())

cls = np.array([CLASSES[n] for n in names])
t = pd.DataFrame({k: per[k] for k in per if k in ("fixed", "fitted_class") or k.startswith("siamese_class")})
t["class"] = cls
print("\nper class top-1, leave-one-class-out")
print(t.groupby("class").agg(lambda x: (x == 1).mean()).round(2).assign(n=t.groupby("class").size()).to_string())
new = pd.DataFrame({"name": names[~OLD], **{k: per[k][~OLD] for k in per if not k.startswith("shuffle")}})
print("\nnew compounds, full-library rank")
print(new.to_string(index=False))
