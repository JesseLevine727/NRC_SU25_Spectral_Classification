"""Append our B3LYP/TZVP calculations (compounds missing from QM9S) to the Siamese data file.

Library rows are processed exactly as in prep_siamese.py (785 nm cross-section, three-range
scaling, 340-1860 cm^-1 window, max-normalised). The queries for these compounds are the
previously unmatched rows of queries.csv. Train/val/P3 indices are kept unchanged, so every
earlier library index still means the same molecule.
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qm9s_modes as qm  # noqa: E402

NU0 = 1e7 / 785.0
THREE = [(0, 1000, 0.9790), (1000, 2000, 0.9818), (2000, 1e9, 0.9619)]
NEW = {  # file tag -> query name in queries.csv
    "formic_acid": "formic acid", "acetic_acid": "acetic acid", "acetone": "acetone",
    "ethylene_glycol": "ethylene glycol", "tetrahydrofuran": "tetrahydrofuran", "diethylamine": "diethylamine",
    "isobutylamine": "isobutylamine", "dimethoxypropane": "2,2-dimethoxypropane",
}


def sticks(nu, a, K):
    ok = nu > 50
    nu, a = nu[ok], a[ok]
    a = a * (NU0 - nu) ** 4 / nu / (1 - np.exp(-1.4388 * nu / 298.0))
    s = nu.copy()
    for lo, hi, f in THREE:
        m = (nu >= lo) & (nu < hi)
        s[m] = nu[m] * f
    w = (s > 340) & (s < 1860)
    s, a = s[w], a[w]
    a = a / a.max()
    out_s, out_a = np.zeros(K, np.float32), np.zeros(K, np.float32)
    out_s[:len(s)], out_a[:len(a)] = s, a
    return out_s, out_a


def main():
    DATA, RDK, QUERIES, DFT_DIR, OUT = sys.argv[1:6]
    d = dict(np.load(DATA))
    n0, K = d["nu3"].shape
    lib = pd.read_csv(RDK)
    fcodes, funiq = pd.factorize(lib["formula"])
    assert (fcodes == d["formula"]).all()
    funiq = list(funiq)
    allq = pd.read_csv(QUERIES + ".csv")
    spec = np.load(QUERIES + ".npy")
    grid = np.arange(400.0, 3201.0, 1.0)
    win = (grid >= 400) & (grid <= 1800)

    new_nu, new_in, new_smi, new_f, report = [], [], [], [], []
    q_x, q_true, q_group, q_name, q_source = [], [], [], [], []
    g_next = int(d["q_group"].max()) + 1
    for k, (tag, qname) in enumerate(NEW.items()):
        r = np.load(os.path.join(DFT_DIR, f"new_{tag}.npz"))
        nu, act, _, _ = qm.modes(r["z"], r["xyz"], r["dpol"], r["hess"])
        smi = str(r["source"])
        mol = Chem.MolFromSmiles(smi)
        s, a = sticks(nu, act, K)
        new_nu.append(s)
        new_in.append(a)
        new_smi.append(Chem.MolToSmiles(mol))
        f = CalcMolFormula(mol)
        if f not in funiq:
            funiq.append(f)
        new_f.append(funiq.index(f))
        report.append(dict(tag=tag, smiles=smi, formula=f, natoms=len(r["z"]), lowest_cm=round(float(nu.min()), 1),
                           n_imag=int((nu < 0).sum()), max_grad=float(r["grad_max"]), minutes=round(float(r["seconds"]) / 60, 1),
                           isomers_in_library=int((d["formula"] == funiq.index(f)).sum())))
        rows = np.flatnonzero(allq["name"].to_numpy() == qname)
        assert len(rows) >= 1, qname
        for i in rows:
            x = np.clip(np.nan_to_num(spec[i, win][::2]), 0, None)
            x = np.sqrt(x)
            q_x.append(x / np.linalg.norm(x))
            q_true.append(n0 + k)
            q_group.append(g_next)
            q_name.append(qname)
            q_source.append(allq["source"].iloc[i])
        g_next += 1
    print(pd.DataFrame(report).to_string(index=False))
    d["nu3"] = np.concatenate([d["nu3"], np.array(new_nu)])
    d["inten"] = np.concatenate([d["inten"], np.array(new_in)])
    d["smiles"] = np.concatenate([d["smiles"], np.array(new_smi)])
    d["formula"] = np.concatenate([d["formula"], np.array(new_f)])
    d["q_x"] = np.concatenate([d["q_x"], np.array(q_x, np.float32)])
    d["q_true"] = np.concatenate([d["q_true"], np.array(q_true)])
    d["q_group"] = np.concatenate([d["q_group"], np.array(q_group)])
    d["q_name"] = np.concatenate([d["q_name"], np.array(q_name)])
    d["q_source"] = np.concatenate([d["q_source"], np.array(q_source)])
    np.savez(OUT, **d)
    print("library", d["nu3"].shape[0], "queries", len(d["q_x"]), "compounds", d["q_group"].max() + 1)


if __name__ == "__main__":
    main()
