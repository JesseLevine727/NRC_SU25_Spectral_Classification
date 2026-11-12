"""Oracle residuals: each measured spectrum against its own calculated spectrum.

Peaks are assigned by Hungarian matching (+-40 cm^-1) after three-range scaling.
Reports position residuals (measured - calculated) and log intensity ratios, so we
can see whether the remaining mismatch is in band positions or band intensities.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pregate import GRID, mask_for, peaks, render  # noqa: E402

STICKS, MATCHED, QUERIES, OUT = sys.argv[1:5]


def main():
    d = dict(np.load(STICKS))
    q = pd.read_csv(MATCHED)
    allq = pd.read_csv(QUERIES + ".csv")
    spec = np.load(QUERIES + ".npy")
    keep = allq["inchikey"].isin(set(q["inchikey"])).to_numpy()
    spec = spec[keep]
    rows = []
    for j, r in q.iterrows():
        win = "fp_ch" if r.source.startswith("flanagan") else "fp"
        m = mask_for(win)
        pe, we = peaks(spec[j], m, prom=0.03)
        C = render([int(r.lib_idx)], d, scheme="three", fwhm=10.0)[0]
        pc, wc = peaks(C, m, prom=0.03)
        if len(pe) == 0 or len(pc) == 0:
            continue
        cost = np.abs(pe[:, None] - pc[None, :])
        big = cost > 40
        a, b = linear_sum_assignment(np.where(big, 1e6, cost))
        ok = ~big[a, b]
        for ia, ib in zip(a[ok], b[ok]):
            rows.append(dict(query=j, name=r["name"], source=r.source, exp_cm=pe[ia], calc_cm=pc[ib],
                             resid=pe[ia] - pc[ib], log_int_ratio=np.log(we[ia] ** 2 / we.max() ** 2)
                             - np.log(wc[ib] ** 2 / wc.max() ** 2)))
        rows.append(dict(query=j, name=r["name"], source=r.source, n_exp_peaks=len(pe), n_calc_peaks=len(pc),
                         n_matched=int(ok.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pr = df.dropna(subset=["resid"])
    cnt = df.dropna(subset=["n_exp_peaks"])
    print("matched peaks", len(pr), "of", int(cnt.n_exp_peaks.sum()), "measured peaks;",
          int(cnt.n_calc_peaks.sum()), "calculated peaks")
    for lo, hi in [(400, 1000), (1000, 1800), (2800, 3100)]:
        s = pr[(pr.exp_cm >= lo) & (pr.exp_cm < hi)]
        if len(s):
            print(f"{lo}-{hi}: n={len(s)}  median |resid| {s.resid.abs().median():.1f}  mean resid {s.resid.mean():+.1f}"
                  f"  within 13: {(s.resid.abs() <= 13).mean():.0%}  |log I ratio| median {s.log_int_ratio.abs().median():.2f}")
    per = pr.groupby(["name", "source"]).agg(n=("resid", "size"), mean_resid=("resid", "mean"),
                                             med_abs=("resid", lambda x: x.abs().median()),
                                             int_spread=("log_int_ratio", "std"))
    per = per.join(cnt.set_index(["name", "source"])[["n_exp_peaks", "n_calc_peaks"]])
    print(per.round(2).to_string())


if __name__ == "__main__":
    main()
