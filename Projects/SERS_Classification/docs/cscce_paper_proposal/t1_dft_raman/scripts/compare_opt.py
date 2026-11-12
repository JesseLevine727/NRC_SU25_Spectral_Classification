"""Compare frequencies/activities of a from-SMILES optimised run with the stored QM9S entry (rotation invariant)."""
import os, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qm9s_modes as qm
ZIP, NPZ, QID = sys.argv[1], sys.argv[2], int(sys.argv[3])
rec = qm.parse(zipfile.ZipFile(ZIP).read(f"qm9s_csv/{QID:06d}.csv").decode())
r = np.load(NPZ)
nu0, a0, _, _ = qm.modes(*rec[3:])
nu1, a1, _, _ = qm.modes(r["z"], r["xyz"], r["dpol"], r["hess"])
print(f"QM9S {QID} {rec[2]} vs our optimised geometry: max|grad| {float(r['grad_max']):.1e}, {float(r['seconds'])/60:.1f} min")
for f0, f1, x0, x1 in zip(nu0, nu1, a0, a1):
    print(f"  {f0:8.1f} {f1:8.1f}  act {x0:8.2f} {x1:8.2f}")
