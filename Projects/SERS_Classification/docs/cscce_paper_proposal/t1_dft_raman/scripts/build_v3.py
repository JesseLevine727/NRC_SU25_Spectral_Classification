"""Step 3 data file (see step3/PREREG.md): step 2 file + one DFT entry per test compound + their
VibraCLIP spectra as split 3.

Library rows are processed as in extend_library.py (785 nm cross-section, three-range scaling);
queries as in build_v2.py (500-1800 window, 1064 -> 785 nm correction, sqrt, unit norm). The band
arrays are widened when a larger molecule has more in-window bands. Train/val/P3 indices and every
earlier library index are unchanged.
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qm9s_modes as qm  # noqa: E402
from build_v2 import GRID, HBD, NU785, NU1064, chem_class, load_vib  # noqa: E402
from extend_library import sticks  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def main():
    DATA, TEST, DFT_DIR, VIB, OUT = sys.argv[1:6]
    d = dict(np.load(DATA))
    n_old, K_old = d["nu3"].shape
    test = pd.read_csv(TEST)

    # formula codes: recover code -> formula from one representative per code
    codes = d["formula"]
    first = {c: i for i, c in reversed(list(enumerate(codes)))}
    code_of = {CalcMolFormula(Chem.MolFromSmiles(str(d["smiles"][i]))): c for c, i in first.items()}
    next_code = int(codes.max()) + 1

    new_nu, new_in, new_smi, new_f, done, natoms, report = [], [], [], [], {}, {}, []
    for row in test.itertuples():
        path = os.path.join(DFT_DIR, f"{row.tag}.npz")
        if not os.path.exists(path):
            report.append(dict(tag=row.tag, smiles=row.smiles, status="DFT failed or missing"))
            continue
        r = np.load(path)
        nu, act, _, _ = qm.modes(r["z"], r["xyz"], r["dpol"], r["hess"])
        if (nu < -5).any():
            report.append(dict(tag=row.tag, smiles=row.smiles, status=f"imaginary {nu.min():.0f} cm^-1 left, excluded"))
            continue
        m = Chem.MolFromSmiles(row.smiles)
        assert Chem.MolToInchiKey(m) == row.ik
        s, a = sticks(nu, act, 400)
        new_nu.append(s)
        new_in.append(a)
        new_smi.append(Chem.MolToSmiles(m))
        f = CalcMolFormula(m)
        if f not in code_of:
            code_of[f] = next_code
            next_code += 1
        new_f.append(code_of[f])
        done[row.ik] = n_old + len(new_nu) - 1
        natoms[row.ik] = len(r["z"])
        report.append(dict(tag=row.tag, smiles=row.smiles, status="ok", natoms=len(r["z"]),
                           lowest_cm=round(float(nu[nu > 0].min()), 1), bands=int((s > 0).sum()),
                           minutes=round(float(r["seconds"]) / 60, 1), route=str(r.get("dpol_route", "displace"))))
    rep = pd.DataFrame(report)
    print(rep.to_string(index=False))
    K = max(K_old, int(max((x > 0).sum() for x in new_nu)))
    pad = lambda x: np.pad(x, ((0, 0), (0, K - x.shape[1])))  # noqa: E731
    d["nu3"] = np.concatenate([pad(d["nu3"]), np.array(new_nu)[:, :K]]).astype(np.float32)
    d["inten"] = np.concatenate([pad(d["inten"]), np.array(new_in)[:, :K]]).astype(np.float32)
    d["smiles"] = np.concatenate([d["smiles"], np.array(new_smi)])
    d["formula"] = np.concatenate([d["formula"], np.array(new_f)])

    q = {k: list(d[k]) for k in ["q_x", "q_true", "q_group", "q_name", "q_source", "q_split", "q_class", "q_hbd"]}
    q["q_natoms"] = [0] * len(q["q_x"])  # filled for test queries only
    g_next = int(d["q_group"].max()) + 1
    groups, nspec = {}, 0
    for e in load_vib(VIB):
        m = Chem.MolFromSmiles(e["smile"])
        k = Chem.MolToInchiKey(m)
        if k not in done:
            continue
        w, y = np.asarray(e["raman_spectra"], float)
        o = np.argsort(w)
        w, y = w[o], np.nan_to_num(y[o])
        y = np.interp(GRID, w, y) * ((NU785 - GRID) / (NU1064 - GRID)) ** 4
        y = np.sqrt(np.clip(y, 0, None))
        q["q_x"].append((y / np.linalg.norm(y)).astype(np.float32))
        q["q_true"].append(done[k])
        q["q_group"].append(groups.setdefault(k, g_next + len(groups)))
        q["q_name"].append(e["nist_name"])
        q["q_source"].append("vibraclip")
        q["q_split"].append(3)
        q["q_class"].append(chem_class(m))
        q["q_hbd"].append(m.HasSubstructMatch(HBD))
        q["q_natoms"].append(natoms[k])
        nspec += 1
    d["q_x"] = np.array(q["q_x"], np.float32)
    for k in ["q_true", "q_group", "q_split", "q_hbd", "q_natoms"]:
        d[k] = np.array(q[k])
    for k in ["q_name", "q_source", "q_class"]:
        d[k] = np.array(q[k], dtype=str)
    assert not d["q_hbd"][d["q_split"] == 3].any()
    assert len(groups) == len(done), "a computed test compound has no VibraCLIP spectrum"
    np.savez(OUT, **d)
    rep.to_csv(OUT.replace(".npz", "_dft_report.csv"), index=False)
    print(f"library {d['nu3'].shape[0]} (+{len(new_nu)}), band width {K_old} -> {K}; "
          f"test compounds {len(groups)}, test spectra {nspec}, queries {len(d['q_x'])}")
    print("test classes", pd.Series(d["q_class"][d["q_split"] == 3]).value_counts().to_dict())


if __name__ == "__main__":
    main()
