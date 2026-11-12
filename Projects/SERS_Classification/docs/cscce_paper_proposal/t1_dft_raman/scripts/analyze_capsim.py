"""CaPSim on the step 2 test compounds next to the step 2 methods (exploratory, not predeclared)."""
import json
import sys

import numpy as np
import pandas as pd

O2, CAP, DATA = sys.argv[1:4]
d = np.load(DATA)
split, group, hbd = d["q_split"], d["q_group"], d["q_hbd"]
pd.set_option("display.width", 220)
sys.argv = [sys.argv[0], O2, DATA]
exec(open(__file__.replace("analyze_capsim.py", "analyze_v2.py")).read().split("# ----")[0])  # helpers only
R2 = load_primary(O2)  # noqa: F821
C = json.load(open(CAP))


def short(c):
    return " ".join(f"{k}={v}" for k, v in c.items())


R = {"fixed_fwhm16": R2["fixed_fwhm16"], "calibration_37": R2["fitted"]}
for k in ["siamese_0", "siamese_1", "siamese_2"]:
    R[k] = R2[k]
R["capsim_selected"] = np.array(C["selected"]["ranks"])
R["capsim_paper2023"] = np.array(C["paper"]["ranks"])
for v, row in C["by_variant"].items():
    R[f"capsim_best_{v}"] = np.array(row["ranks"])
R["capsim_oracle(post hoc)"] = np.array(C["oracle"]["ranks"])

print("CaPSim settings chosen on our 37 training spectra (split 0):")
print("  selected:", short(C["selected"]["cfg"]), f"train top-1 {C['selected']['train_top1']:.3f}")
for v, row in C["by_variant"].items():
    print(f"  best '{v}':", short(row["cfg"]), f"train top-1 {row['train_top1']:.3f}")
print("  paper 2023 settings:", short(C["paper"]["cfg"]), f"train top-1 {C['paper']['train_top1']:.3f}")
print("  oracle (best on the test set itself, optimistic):", short(C["oracle"]["cfg"]))
alls = pd.DataFrame([dict(**a["cfg"], train=a["train_top1"], test=a["test_top1"]) for a in C["all"]])
print(f"  across all {len(alls)} settings, test top-1: median {alls.test.median():.3f}, "
      f"90th pct {alls.test.quantile(.9):.3f}, max {alls.test.max():.3f}; "
      f"train-test rank correlation {alls[['train', 'test']].corr('spearman').iloc[0, 1]:.2f}")

test = split == 1
table(R, test, "step 2 test set (81 VibraCLIP compounds)")  # noqa: F821
table(R, test & ~hbd, "no O-H/N-H donor (post hoc split)")  # noqa: F821
table(R, test & hbd, "with O-H/N-H donor (post hoc split)")  # noqa: F821

ug, cap = per_compound((R["capsim_selected"] == 1).astype(float), test)  # noqa: F821
for name, other in [("calibration_37", per_compound((R["calibration_37"] == 1).astype(float), test)[1]),  # noqa: F821
                    ("siamese (mean of 3)", np.mean([per_compound((R[k] == 1).astype(float), test)[1]  # noqa: F821
                                                     for k in ["siamese_0", "siamese_1", "siamese_2"]], 0)),
                    ("fixed_fwhm16", per_compound((R["fixed_fwhm16"] == 1).astype(float), test)[1])]:  # noqa: F821
    dl = other - cap
    lo, hi = boot(dl)  # noqa: F821
    print(f"{name} minus CaPSim(selected): {dl.mean():+.3f}  (paired bootstrap 95% CI {lo:+.3f} to {hi:+.3f})")

print("\nmain effects across the grid (mean test top-1 by option):")
for k in ["variant", "sim", "norm", "feat", "npeak", "h", "fwhm", "base", "ksm"]:
    print(f"  {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in alls.groupby(k).test.mean().items()))
