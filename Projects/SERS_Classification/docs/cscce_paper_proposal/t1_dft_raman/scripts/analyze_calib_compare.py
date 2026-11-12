"""Summary of calib_compare.py next to the step 2 network (exploratory)."""
import json
import sys

import numpy as np
import pandas as pd

O2, CC, DATA = sys.argv[1:4]
d = np.load(DATA)
split, group, hbd = d["q_split"], d["q_group"], d["q_hbd"]
sys.argv = [sys.argv[0], O2, DATA]
exec(open(__file__.replace("analyze_calib_compare.py", "analyze_v2.py")).read().split("# ----")[0])  # helpers only
R2 = load_primary(O2)  # noqa: F821
C = json.load(open(CC))
R = {"published scale factors + sqrt cosine (fixed)": np.array(C["published_scale"]["ranks"]),
     "least-squares scale factors (assigned bands)": np.array(C["lsq_scale"]["ranks"]),
     "Houthuijs 2023 grid search (selected on train)": np.array(C["houthuijs_selected"]["ranks"]),
     "Houthuijs 2023 grid search (best on test, optimistic)": np.array(C["houthuijs_oracle"]["ranks"]),
     "calibration: stretch only (2)": np.array(C["calib_stretch_only"]["ranks"]),
     "calibration: response only (4)": np.array(C["calib_response_only"]["ranks"]),
     "calibration: full (6)": np.array(C["calib_full"]["ranks"])}
for k in ["siamese_0", "siamese_1", "siamese_2"]:
    R[f"network ({k})"] = R2[k]
pd.set_option("display.max_colwidth", 60)
sel = C["houthuijs_selected"]
print("Houthuijs-style selection:", {k: sel[k] for k in ("shape", "fwhm", "trans", "where", "sides")},
      f"train geometric mean rank {sel['train_gmr']:.1f}")
orc = C["houthuijs_oracle"]
print("Houthuijs-style best on test:", {k: orc[k] for k in ("shape", "fwhm", "trans", "where", "sides")})
print("least-squares extra scale factors:", np.round(C["lsq_scale"]["scale"], 4), f"from {C['lsq_scale']['npairs']} band pairs")
print("check: calib_full reproduces step 2 calibration ranks for "
      f"{np.mean(np.array(C['calib_full']['ranks'])[split == 1] == R2['fitted'][split == 1]) * 100:.0f}% of test queries")
test = split == 1
table(R, test, "step 2 test set, 81 compounds")  # noqa: F821
table(R, test & ~hbd, "no O-H/N-H donor (post hoc)")  # noqa: F821
full = per_compound((R["calibration: full (6)"] == 1).astype(float), test)[1]  # noqa: F821
print("\nfull calibration minus each method (top-1, paired bootstrap 95% CI over 81 compounds):")
for k, r in R.items():
    if k == "calibration: full (6)":
        continue
    dl = full - per_compound((r == 1).astype(float), test)[1]  # noqa: F821
    lo, hi = boot(dl)  # noqa: F821
    print(f"  vs {k:55s} {dl.mean():+.3f}  ({lo:+.3f} to {hi:+.3f})")
g = pd.DataFrame(C["houthuijs_grid"])
print(f"\nHoutuijs grid ({len(g)} settings): test top-1 median {g.test_top1.median():.3f}, max {g.test_top1.max():.3f}; "
      f"train-GMR vs test top-1 Spearman {g[['train_gmr', 'test_top1']].corr('spearman').iloc[0, 1]:.2f}")
for k in ["shape", "fwhm", "trans", "where", "sides"]:
    print(f"  {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in g.groupby(k).test_top1.max().items()))
