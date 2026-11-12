"""Step 2 data file: 500-1800 cm^-1 window, VibraCLIP experimental queries appended.

q_split: 0 = our 37 training spectra, 1 = VibraCLIP compounds in QM9S and new to us (primary test),
         2 = VibraCLIP spectra of compounds we already have.
q_class: chemical class (existing CLASSES for old queries, SMARTS rules for new ones).
The pretraining pool is re-drawn with every query compound removed.
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
# CLASSES is read from siamese.py without importing it (this venv has RDKit but not torch)
_src = open(os.path.join(HERE, "siamese.py")).read()
CLASSES = ast.literal_eval(_src[_src.index("CLASSES = {") + 10:_src.index("}", _src.index("CLASSES = {")) + 1])

RDLogger.DisableLog("rdApp.*")
NOCORR = os.environ.get("NO_LASER_CORR") == "1"
LO = 500.0
GRID = np.arange(LO, 1801.0, 2.0)
NU785, NU1064 = 1e7 / 785.0, 1e7 / 1064.0
RULES = [  # first match wins; aromatic first to agree with the existing hand-assigned classes
    ("aromatic", "[a]"), ("acid", "[CX3](=O)[OX2H1]"), ("amide", "[NX3][CX3]=O"), ("ester", "[CX3](=O)[OX2][#6]"),
    ("nitrile", "C#N"), ("nitro", "[N+](=O)[O-]"), ("other", "[CX3]=N[OX2H]"), ("ketone", "[CX3]=O"),
    ("alcohol", "[OX2H]"), ("amine", "[NX3;!$(N[#6]=O)]"), ("ether", "[OD2]([#6])[#6]"), ("alkene", "[$(C=C),$(C#C)]"),
]
RULES = [(c, Chem.MolFromSmarts(s)) for c, s in RULES]
HBD = Chem.MolFromSmarts("[OX2H,NX3;H1,H2]")


def chem_class(m):
    for c, p in RULES:
        if m.HasSubstructMatch(p):
            return c
    return "alkane" if all(a.GetSymbol() == "C" for a in m.GetAtoms()) else "other"


def load_vib(path):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "step2", "scripts"))
    from load_vibraclip import load  # numpy-only unpickler
    return list(load(path).values())


def main():
    DATA, RDK, VIB, OUT = sys.argv[1:5]
    d = dict(np.load(DATA))
    n0 = 129817
    lib = pd.read_csv(RDK)
    keys = list(lib.inchikey) + [Chem.MolToInchiKey(Chem.MolFromSmiles(str(s))) for s in d["smiles"][n0:]]
    dup = lib.inchikey.duplicated(keep=False).to_numpy()
    key2idx = {k: i for i, k in enumerate(keys) if i >= n0 or not dup[i]}

    # old queries: cut 400-500 and renormalise (q_x = sqrt(I)/|.|, so this equals recomputing on 500-1800)
    x_old = d["q_x"][:, int((LO - 400) / 2):]
    x_old = x_old / np.linalg.norm(x_old, axis=1, keepdims=True)
    old_true = d["q_true"]
    old_keys = {keys[t]: g for t, g in zip(old_true, d["q_group"])}
    q = dict(q_x=list(x_old), q_true=list(old_true), q_group=list(d["q_group"]), q_name=list(d["q_name"]),
             q_source=list(d["q_source"]), q_split=[0] * len(old_true),
             q_class=[CLASSES[str(n)] for n in d["q_name"]],
             q_hbd=[Chem.MolFromSmiles(str(d["smiles"][t])).HasSubstructMatch(HBD) for t in old_true])
    g_next = int(d["q_group"].max()) + 1
    new_group = {}
    skipped = 0
    for e in load_vib(VIB):
        m = Chem.MolFromSmiles(e["smile"])
        k = Chem.MolToInchiKey(m)
        if k in old_keys:
            split, grp = 2, old_keys[k]
        elif k in key2idx and key2idx[k] < n0:
            split = 1
            grp = new_group.setdefault(k, g_next + len(new_group))
        else:
            skipped += 1
            continue
        w, y = np.asarray(e["raman_spectra"], float)
        o = np.argsort(w)
        w, y = w[o], np.nan_to_num(y[o])
        y = np.interp(GRID, w, y)
        if not NOCORR:
            y = y * ((NU785 - GRID) / (NU1064 - GRID)) ** 4
        y = np.sqrt(np.clip(y, 0, None))
        q["q_x"].append(y / np.linalg.norm(y))
        q["q_true"].append(key2idx[k])
        q["q_group"].append(grp)
        q["q_name"].append(e["nist_name"])
        q["q_source"].append("vibraclip")
        q["q_split"].append(split)
        q["q_class"].append(q["q_class"][list(q["q_group"]).index(grp)] if split == 2 else chem_class(m))
        q["q_hbd"].append(m.HasSubstructMatch(HBD))
    d["q_x"] = np.array(q["q_x"], np.float32)
    for k in ["q_true", "q_group", "q_split", "q_hbd"]:
        d[k] = np.array(q[k])
    for k in ["q_name", "q_source", "q_class"]:
        d[k] = np.array(q[k], dtype=str)
    pool = np.setdiff1d(np.arange(n0), d["q_true"])
    rng = np.random.default_rng(1)
    d["val_idx"] = rng.choice(pool, 5000, replace=False)
    d["train_idx"] = np.setdiff1d(pool, d["val_idx"])
    d["grid_lo"] = np.array(LO)
    np.savez(OUT, **d)
    sp = d["q_split"]
    print("queries", len(sp), "by split", np.bincount(sp).tolist(), "compounds", len(np.unique(d["q_group"])),
          "test compounds", len(np.unique(d["q_group"][sp == 1])), "skipped (not in library)", skipped)
    print("test classes", pd.Series(d["q_class"][sp == 1]).value_counts().to_dict())
    print("test with O-H/N-H donor", int(d["q_hbd"][sp == 1].sum()), "of", int((sp == 1).sum()))
    print("pretrain pool", len(d["train_idx"]), "val", len(d["val_idx"]))


if __name__ == "__main__":
    main()
