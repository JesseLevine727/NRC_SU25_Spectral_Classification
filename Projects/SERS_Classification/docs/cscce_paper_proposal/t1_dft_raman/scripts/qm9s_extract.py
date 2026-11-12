"""Extract stick spectra (wavenumber, Raman activity) for every QM9S molecule."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zipfile
from multiprocessing import Pool

import numpy as np

from qm9s_modes import modes, parse

ZIP, OUT = sys.argv[1], sys.argv[2]
NPROC = 30


def work(k):
    zp = zipfile.ZipFile(ZIP)
    names = sorted(n for n in zp.namelist() if n.endswith(".csv"))[k::NPROC]
    out = []
    for name in names:
        try:
            qid, q9, smi, z, xyz, dpol, hess = parse(zp.read(name).decode())
            nu, act, _, _ = modes(z, xyz, dpol, hess)
            out.append((qid, q9, smi, nu.astype(np.float32), act.astype(np.float32)))
        except Exception as e:  # keep going, report later
            out.append((-1, -1, f"ERR {name} {e}", np.zeros(0, np.float32), np.zeros(0, np.float32)))
    return out


if __name__ == "__main__":
    with Pool(NPROC) as p:
        parts = p.map(work, range(NPROC))
    recs = sorted((r for part in parts for r in part), key=lambda r: r[0])
    errs = [r[2] for r in recs if r[0] < 0]
    recs = [r for r in recs if r[0] >= 0]
    off = np.cumsum([0] + [len(r[3]) for r in recs])
    np.savez_compressed(
        OUT,
        qm9s_id=np.array([r[0] for r in recs]),
        qm9_id=np.array([r[1] for r in recs]),
        smiles=np.array([r[2] for r in recs]),
        offsets=off,
        nu=np.concatenate([r[3] for r in recs]),
        act=np.concatenate([r[4] for r in recs]),
    )
    print("molecules", len(recs), "errors", len(errs))
    for e in errs[:10]:
        print(e)
