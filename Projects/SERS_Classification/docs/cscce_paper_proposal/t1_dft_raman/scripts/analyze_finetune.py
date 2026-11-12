"""Leave-one-compound-out fine-tuning: learning curve and train vs held-out curves."""
import json
import sys

import numpy as np
import pandas as pd

FT, FIXED = sys.argv[1:3]
r = json.load(open(FT))
fx = json.load(open(FIXED))
res = pd.DataFrame(r["results"])


def stats(x):
    x = np.asarray(x)
    return pd.Series(dict(top1=(x == 1).mean(), top5=(x <= 5).mean(), top20=(x <= 20).mean(), median=np.median(x)))


print("fixed baseline (same 28 queries):")
for k, v in fx.items():
    print(f"  {k}: " + ", ".join(f"{p} top1 {v[p]['top1']:.3f} top5 {v[p]['top5']:.3f}" for p in ["full", "P3", "P2"]))
for k, v in fx.items():
    rk = np.array(v["full"]["ranks"])
    print(f"  {k} full top20 {(rk <= 20).mean():.3f} median {np.median(rk):.0f}")

print("\nheld-out results by number of training compounds (averaged over repeats)")
for pool in ["full", "P3", "P2"]:
    t = res.groupby("ntrain")[f"rank_{pool}"].apply(stats).unstack().round(3)
    print(pool)
    print(t.to_string())

c = pd.DataFrame(r["curves"])
rows = []
for step, g in c.groupby("step"):
    held = np.concatenate(g["held_ranks"].to_list())
    tr = np.concatenate(g["train_ranks"].to_list())
    rows.append(dict(step=step, train_top1=(tr == 1).mean(), train_top5=(tr <= 5).mean(),
                     held_top1=(held == 1).mean(), held_top5=(held <= 5).mean(), held_median=np.median(held)))
print("\ntrain vs held-out over fine-tuning steps (20 training compounds, full 130k library)")
print(pd.DataFrame(rows).round(3).to_string(index=False))

last = res[res.ntrain == res.ntrain.max()]
fixed16 = np.array(fx["fixed_sqrt_cos_fwhm16"]["full"]["ranks"])
per = last[["query", "name", "source", "rank_full", "rank_P3", "rank_P2"]].copy()
per["fixed16_full"] = fixed16[per["query"].to_numpy()]
print("\nper query, 20 training compounds vs fixed (full library rank)")
print(per.sort_values("query").to_string(index=False))
