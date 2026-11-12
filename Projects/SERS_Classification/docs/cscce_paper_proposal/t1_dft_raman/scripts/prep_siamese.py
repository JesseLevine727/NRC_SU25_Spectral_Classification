"""Pack QM9S sticks, real queries and pools into one npz for the Siamese experiments."""
import sys

import numpy as np
import pandas as pd

STICKS, RDK, MATCHED, QUERIES, OUT = sys.argv[1:6]
NU0 = 1e7 / 785.0
THREE = [(0, 1000, 0.9790), (1000, 2000, 0.9818), (2000, 1e9, 0.9619)]


def main():
    d = np.load(STICKS)
    off, nu_all, act_all = d["offsets"], d["nu"], d["act"]
    n = len(off) - 1
    K = int(np.diff(off).max())
    nu3 = np.zeros((n, K), np.float32)
    inten = np.zeros((n, K), np.float32)
    for i in range(n):
        nu = nu_all[off[i]:off[i + 1]].astype(float)
        a = act_all[off[i]:off[i + 1]].astype(float)
        ok = nu > 50
        nu, a = nu[ok], a[ok]
        a = a * (NU0 - nu) ** 4 / nu / (1 - np.exp(-1.4388 * nu / 298.0))
        s = nu.copy()
        for lo, hi, f in THREE:
            m = (nu >= lo) & (nu < hi)
            s[m] = nu[m] * f
        inwin = (s > 340) & (s < 1860)  # fingerprint window plus margin
        s, a = s[inwin], a[inwin]
        if len(a) and a.max() > 0:
            a = a / a.max()
        nu3[i, :len(s)] = s
        inten[i, :len(a)] = a
    lib = pd.read_csv(RDK)
    q = pd.read_csv(MATCHED)
    allq = pd.read_csv(QUERIES + ".csv")
    spec = np.load(QUERIES + ".npy")[allq["inchikey"].isin(set(q["inchikey"])).to_numpy()]
    grid = np.arange(400.0, 3201.0, 1.0)
    w = (grid >= 400) & (grid <= 1800)
    x = np.clip(np.nan_to_num(spec[:, w][:, ::2]), 0, None)
    x = np.sqrt(x)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    groups = pd.factorize(q["inchikey"])[0]
    true = q["lib_idx"].to_numpy()
    p3 = np.random.default_rng(0).choice(n, 20000, replace=False)  # same draw as pregate.py
    formula_codes = pd.factorize(lib["formula"])[0]
    rng = np.random.default_rng(1)
    pool = np.setdiff1d(np.arange(n), true)
    val = rng.choice(pool, 5000, replace=False)
    train = np.setdiff1d(pool, val)
    np.savez(OUT, nu3=nu3, inten=inten, q_x=x.astype(np.float32), q_true=true, q_group=groups,
             q_name=q["name"].to_numpy(str), q_source=q["source"].to_numpy(str), p3=p3,
             formula=formula_codes, train_idx=train, val_idx=val, smiles=d["smiles"])
    print("library", n, "max modes in window", int((inten > 0).sum(1).max()), "queries", len(x),
          "compounds", groups.max() + 1, "train", len(train), "val", len(val))


if __name__ == "__main__":
    main()
