"""Measured query spectra: Flanagan & Glavin 2025 (785 nm) plus our normal Raman.

Each query is the mean spectrum of one compound from one source, interpolated to
a 1 cm^-1 grid and baseline-corrected (asymmetric least squares).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from scipy import sparse
from scipy.sparse.linalg import spsolve

FLAN, NRC, EXT, OUT = sys.argv[1:5]
GRID = np.arange(400.0, 3201.0, 1.0)

# Flanagan label -> (name, SMILES); compounds outside QM9 (Br, Cl, S, >9 heavy atoms) are skipped
FLANAGAN = {
    "1,3-Dimethyl-2-imidazolidinone": ("1,3-dimethyl-2-imidazolidinone", "CN1CCN(C)C1=O"),
    "2 - Propanol": ("2-propanol", "CC(C)O"),
    "2, 2 - Dimethoxy Propane": ("2,2-dimethoxypropane", "COC(C)(C)OC"),
    "4-Methyl-2-pentanone": ("4-methyl-2-pentanone", "CC(C)CC(C)=O"),
    "Methyl Isobutyl Ketone": ("methyl isobutyl ketone (2nd bottle)", "CC(C)CC(C)=O"),
    "Acetic acid": ("acetic acid", "CC(=O)O"),
    "Acetone": ("acetone", "CC(C)=O"),
    "Acetonitrile": ("acetonitrile", "CC#N"),
    "Benzaldehyde": ("benzaldehyde", "O=Cc1ccccc1"),
    "Butyl Acetate": ("butyl acetate", "CCCCOC(C)=O"),
    "Cyclohexane": ("cyclohexane", "C1CCCCC1"),
    "Diethylamine": ("diethylamine", "CCNCC"),
    "Diethylene Glycol": ("diethylene glycol", "OCCOCCO"),
    "Ethanol": ("ethanol", "CCO"),
    "Ethyl Acetate": ("ethyl acetate", "CCOC(C)=O"),
    "Ethylene Glycol": ("ethylene glycol", "OCCO"),
    "Formic acid": ("formic acid", "OC=O"),
    "Isobutylamine": ("isobutylamine", "CC(C)CN"),
    "Methanol": ("methanol", "CO"),
    "N,N - Dimethylformahide": ("N,N-dimethylformamide", "CN(C)C=O"),
    "n-Heptane": ("n-heptane", "CCCCCCC"),
    "n-Hexane": ("n-hexane", "CCCCCC"),
    "Propyl acetate": ("propyl acetate", "CCCOC(C)=O"),
    "Tert-butanol": ("tert-butanol", "CC(C)(C)O"),
    "Tert-butyl methyl ether": ("tert-butyl methyl ether", "COC(C)(C)C"),
    "Tetrahydrofuran": ("tetrahydrofuran", "C1CCOC1"),
    "Toluene": ("toluene", "Cc1ccccc1"),
}
FLAN_FORMULA_CHECK = {"1,3-Dimethyl-2-imidazolidinone": "C5H10N2O", "2, 2 - Dimethoxy Propane": "C5H12O2",
                      "Diethylene Glycol": "C4H10O3", "Isobutylamine": "C4H11N", "Tert-butyl methyl ether": "C5H12O"}

# our normal Raman: (name, SMILES, glob relative to the NRC Data dir or ExtendingSiamese data dir)
OURS = [
    ("benzene", "c1ccccc1", NRC + "/Jesse_Dataset_v2/Pure_Raman_txt/benzene - *.txt"),
    ("pyridine", "c1ccncc1", NRC + "/Jesse_Dataset_v2/Pure_Raman_txt/pyridine - *.txt"),
    ("N,N-dimethylformamide", "CN(C)C=O", NRC + "/Jesse_Dataset_v2/Pure_Raman_txt/n,n-dimethylformamide - *.txt"),
    ("ethanol", "CCO", NRC + "/Jesse_Dataset_v2/Pure_Raman_txt/etoh*.txt"),
    ("methanol", "CO", NRC + "/Jesse_Dataset_v2/Pure_Raman_txt/meoh*.txt"),
    ("toluene", "Cc1ccccc1", NRC + "/Jesse_Dataset_PT2/Toluene/txt/*.txt"),
    ("acetonitrile", "CC#N", NRC + "/Jesse_Dataset_PT2/Acetonitrile/txt/*.txt"),
    ("diethylamine", "CCNCC", NRC + "/Jesse_Dataset_PT2/diethylamine/txt/*.txt"),
    ("n-hexane", "CCCCCC", NRC + "/Jesse_Dataset_PT2/n-hexane/txt/*.txt"),
    ("aniline", "Nc1ccccc1", EXT + "/Feb26_Spectra/Aniline/*.txt"),
]


def als(y, lam=1e6, p=0.01, niter=10):
    n = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2), dtype=float)
    DD = lam * D.dot(D.T)
    w = np.ones(n)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def finish(x, y):
    """Interpolate to GRID (NaN outside measured range) and remove the baseline."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    lo, hi = max(GRID[0], x[0]), min(GRID[-1], x[-1])
    m = (GRID >= lo) & (GRID <= hi)
    out = np.full(GRID.shape, np.nan)
    yi = np.interp(GRID[m], x, y)
    out[m] = yi - als(yi)
    return out


def main():
    rows, specs = [], []
    df = pd.read_csv(FLAN)
    wn = np.array([float(c) for c in df.columns if c != "label"])
    X = df.drop(columns="label").to_numpy(float)
    for lab, (name, smi) in FLANAGAN.items():
        sel = (df["label"] == lab).to_numpy()
        assert sel.sum() > 0, lab
        mol = Chem.MolFromSmiles(smi)
        if lab in FLAN_FORMULA_CHECK:
            assert CalcMolFormula(mol) == FLAN_FORMULA_CHECK[lab], (lab, CalcMolFormula(mol))
        specs.append(finish(wn, X[sel].mean(0)))
        rows.append(dict(source="flanagan_785", name=name, smiles=smi, n_spectra=int(sel.sum())))
    for name, smi, pat in OURS:
        files = sorted(glob.glob(pat))
        assert files, pat
        arrs = [np.loadtxt(f) for f in files]
        x = arrs[0][:, 0]
        ys = [np.interp(x, a[np.argsort(a[:, 0]), 0], a[np.argsort(a[:, 0]), 1]) for a in arrs]
        specs.append(finish(x, np.mean(ys, 0)))
        rows.append(dict(source="nrc_" + ("feb26" if "Feb26" in pat else "labspec"), name=name, smiles=smi,
                         n_spectra=len(files)))
    meta = pd.DataFrame(rows)
    for i, r in meta.iterrows():
        mol = Chem.MolFromSmiles(r.smiles)
        meta.loc[i, "formula"] = CalcMolFormula(mol)
        meta.loc[i, "inchikey"] = Chem.MolToInchiKey(mol)
        meta.loc[i, "heavy"] = mol.GetNumHeavyAtoms()
    meta.to_csv(OUT + ".csv", index=False)
    np.save(OUT + ".npy", np.array(specs))
    print(meta.to_string())


if __name__ == "__main__":
    main()
