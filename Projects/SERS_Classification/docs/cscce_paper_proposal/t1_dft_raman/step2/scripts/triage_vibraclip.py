"""Triage the VibraCLIP experimental molecule list against QM9S and our DFT budget (no unpickling)."""
import io
import pickletools
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
PKL, LIB, OURS, OUT = sys.argv[1:5]
smis = []
for op, arg, _ in pickletools.genops(io.BytesIO(open(PKL, "rb").read())):
    if op.name.endswith("UNICODE") and isinstance(arg, str):
        smis.append(arg)
keys = set(pd.read_csv(LIB).inchikey)
ours = {Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in OURS.split(",")}
HB = Chem.MolFromSmarts("[OX2H,NX3;H1,H2,H3,$([NX4+])]")  # O-H or N-H donors (carboxylic acids included)
rows = []
for s in dict.fromkeys(smis):
    m = Chem.MolFromSmiles(s)
    if m is None:
        rows.append(dict(smiles=s, ok=False))
        continue
    mh = Chem.AddHs(m)
    el = sorted({a.GetSymbol() for a in m.GetAtoms()})
    ik = Chem.MolToInchiKey(m)
    rows.append(dict(smiles=s, ok=True, heavy=m.GetNumHeavyAtoms(), atoms=mh.GetNumAtoms(), elements="".join(el),
                     chno_f=set(el) <= {"C", "H", "N", "O", "F"}, charged=any(a.GetFormalCharge() for a in m.GetAtoms()) and "[N+](=O)[O-]" not in s,
                     hbond_donor=m.HasSubstructMatch(HB), in_qm9s=ik in keys, in_ours=ik in ours,
                     mw=round(Descriptors.MolWt(m), 1)))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
ok = df[df.ok]
print("unique SMILES", len(df), "parsable", len(ok))
print("elements only C/H/N/O/F:", int(ok.chno_f.sum()))
print("in QM9S:", int(ok.in_qm9s.sum()), "  in our 8 new:", int(ok.in_ours.sum()))
print("in QM9S and no O-H/N-H donor:", int((ok.in_qm9s & ~ok.hbond_donor).sum()))
nq = ok[~ok.in_qm9s & ok.chno_f]
for lim in (20, 25, 30, 40):
    sub = nq[nq.atoms <= lim]
    print(f"not in QM9S, CHNOF, <= {lim} atoms: {len(sub)}  (no donor: {int((~sub.hbond_donor).sum())})")
print("heavy-atom histogram:", ok.heavy.describe()[["min", "25%", "50%", "75%", "max"]].to_dict())
print("\nin QM9S:"); print(ok[ok.in_qm9s][["smiles", "atoms", "hbond_donor"]].to_string(index=False))
