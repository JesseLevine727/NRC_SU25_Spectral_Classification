"""Full-library search (all QM9S, no formula prior) with the best fixed setting,
plus the top-ranked candidates for each query so we can see what the decoys are."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pregate import feats, mask_for, render  # noqa: E402

STICKS, MATCHED, QUERIES, RDK, OUT = sys.argv[1:6]


def main():
    d = dict(np.load(STICKS))
    lib = pd.read_csv(RDK)
    q = pd.read_csv(MATCHED)
    allq = pd.read_csv(QUERIES + ".csv")
    spec = np.load(QUERIES + ".npy")[allq["inchikey"].isin(set(q["inchikey"])).to_numpy()]
    m = mask_for("fp")
    Qf = feats(spec, m, "sqrt")
    n = len(d["offsets"]) - 1
    S = np.zeros((len(q), n), np.float32)
    for lo in range(0, n, 10000):
        idx = np.arange(lo, min(n, lo + 10000))
        S[:, idx] = Qf @ feats(render(idx, d, scheme="three", fwhm=16.0), m, "sqrt").T
    rows, tops = [], []
    for j, r in q.iterrows():
        s = S[j]
        t = int(r.lib_idx)
        rank = int(1 + (s > s[t]).sum())
        same = lib["formula"].to_numpy() == lib["formula"][t]
        rank_iso = int(1 + (s[same] > s[t]).sum())
        order = np.argsort(-s)[:5]
        rows.append(dict(name=r["name"], source=r.source, rank_full=rank, rank_isomer_all=rank_iso,
                         n_isomer_all=int(same.sum()), true_score=float(s[t]),
                         best_other=float(np.max(np.delete(s, t)))))
        iso_order = np.flatnonzero(same)[np.argsort(-s[same])][:4]
        tops.append(dict(name=r["name"], source=r.source, true=d["smiles"][t],
                         top_full=" | ".join(f"{d['smiles'][i]} {s[i]:.3f}" for i in order),
                         top_isomers=" | ".join(f"{d['smiles'][i]} {s[i]:.3f}" for i in iso_order)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT + "_ranks.csv", index=False)
    pd.DataFrame(tops).to_csv(OUT + "_tops.csv", index=False)
    print(res.round(3).to_string())
    r = res["rank_full"]
    print(f"\nfull library ({n}): top1 {(r == 1).mean():.2f} top5 {(r <= 5).mean():.2f} top20 {(r <= 20).mean():.2f}")
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(pd.DataFrame(tops).to_string())


if __name__ == "__main__":
    main()
