"""Summarise the width x depth sweep: train vs validation curves and zero-shot real results."""
import glob
import json
import sys

import pandas as pd

OUT = sys.argv[1]
rows, curves = [], []
for f in sorted(glob.glob(f"{OUT}/sweep_d*_w*_s*.json")):
    r = json.load(open(f))
    last = r["curve"][-1]
    rows.append(dict(depth=r["depth"], width=r["width"], skip=r.get("skip", 0), params_M=r["params"] / 1e6,
                     train130k_top1=last["train_full"]["top1"], val130k_top1=last["val_full"]["top1"],
                     gap=last["train_full"]["top1"] - last["val_full"]["top1"], val5k_top1=last["val_5k"]["top1"],
                     real_full_top1=r["final_real"]["full"]["top1"], real_full_top5=r["final_real"]["full"]["top5"],
                     real_P3_top1=r["final_real"]["P3"]["top1"], real_P3_top5=r["final_real"]["P3"]["top5"],
                     real_P2_top1=r["final_real"]["P2"]["top1"], minutes=last["sec"] / 60))
    for c in r["curve"]:
        curves.append(dict(cfg=f"d{r['depth']}w{r['width']}", step=c["step"], loss=c["loss"],
                           train=c["train_full"]["top1"], val=c["val_full"]["top1"], real_full=c["real"]["full"]["top1"],
                           real_P3=c["real"]["P3"]["top1"]))
t = pd.DataFrame(rows).sort_values(["depth", "width"])
pd.set_option("display.width", 220)
print(t.round(3).to_string(index=False))
print()
print("validation top-1 (130k library), depth x width")
print(t.pivot(index="depth", columns="width", values="val130k_top1").round(3).to_string())
print("\ntrain - val gap")
print(t.pivot(index="depth", columns="width", values="gap").round(3).to_string())
print("\nreal zero-shot full-library top-1 (fixed baseline 0.25)")
print(t.pivot(index="depth", columns="width", values="real_full_top1").round(3).to_string())
c = pd.DataFrame(curves)
print("\ncurves (train/val = synthetic 130k retrieval top-1)")
print(c.pivot(index="step", columns="cfg", values="val").round(3).to_string())
best = t.sort_values("val130k_top1", ascending=False)
top = best.iloc[0]
near = best[best.val130k_top1 >= top.val130k_top1 - 0.01].sort_values("params_M").iloc[0]
print(f"\nselection rule (best synthetic val, smallest within 1 point): depth {int(near.depth)} width {int(near.width)}")
