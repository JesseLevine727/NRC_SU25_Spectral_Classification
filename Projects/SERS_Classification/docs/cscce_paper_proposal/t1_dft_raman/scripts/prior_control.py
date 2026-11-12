"""Non-neural control: fixed cosine + a 'looks like a known compound' library prior.

prior_i = max over training compounds c of cos(calc_i, calc_c)  (calculated spectra only;
no measured information). Score = fixed_cos(query, calc_i) + lam * prior_i, with lam chosen
inside each leave-one-compound-out fold on the training compounds' own queries.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import siamese as s  # noqa: E402

DATA, OUT = sys.argv[1:3]
LAMS = np.round(np.arange(0, 2.01, 0.1), 2)


def ranks(S, true):
    t = S[np.arange(len(S)), true]
    return 1 + (S > t[:, None]).sum(1)


def main():
    D = s.Data(DATA)
    L16 = D.render_clean(16.0)
    Lp = D.libx  # FWHM 10 calc spectra for the calc-calc prior
    Sq = (D.qx @ L16.T).cpu().numpy()  # fixed similarity, 28 x 130k
    groups = np.unique(D.qgroup)
    glib = {g: int(D.qtrue[np.flatnonzero(D.qgroup == g)[0]]) for g in groups}
    C = (Lp[[glib[g] for g in groups]] @ Lp.T).cpu().numpy()  # 21 x 130k calc-calc cosine
    out = []
    for held in groups:
        train = groups[groups != held]
        hq = np.flatnonzero(D.qgroup == held)
        tq = np.flatnonzero(D.qgroup != held)
        # prior for evaluating training queries: exclude that query's own compound
        best_lam, best_mrr = 0.0, -1
        for lam in LAMS:
            rr = []
            for j in tq:
                others = [k for k, g in enumerate(groups) if g in train and g != D.qgroup[j]]
                prior = C[others].max(0)
                sc = Sq[j] + lam * prior
                rr.append(1 + (sc > sc[D.qtrue[j]]).sum())
            mrr = np.mean(1 / np.array(rr))
            if mrr > best_mrr:
                best_lam, best_mrr = lam, mrr
        prior = C[[k for k, g in enumerate(groups) if g in train]].max(0)
        for j in hq:
            sc = Sq[j] + best_lam * prior
            p_only = prior
            out.append(dict(query=int(j), name=str(D.qname[j]), lam=float(best_lam),
                            rank_fused=int(1 + (sc > sc[D.qtrue[j]]).sum()),
                            rank_prior_only=int(1 + (p_only > p_only[D.qtrue[j]]).sum()),
                            rank_fixed=int(1 + (Sq[j] > Sq[j, D.qtrue[j]]).sum())))
    json.dump(out, open(OUT, "w"), indent=1)
    r = {k: np.array([o[k] for o in out]) for k in ["rank_fixed", "rank_fused", "rank_prior_only"]}
    for k, v in r.items():
        print(f"{k:16s} full-library top1 {(v == 1).mean():.3f} top5 {(v <= 5).mean():.3f} "
              f"top20 {(v <= 20).mean():.3f} median {np.median(v):.0f}")
    print("chosen lambda per fold:", sorted(set(o["lam"] for o in out)))


if __name__ == "__main__":
    main()
