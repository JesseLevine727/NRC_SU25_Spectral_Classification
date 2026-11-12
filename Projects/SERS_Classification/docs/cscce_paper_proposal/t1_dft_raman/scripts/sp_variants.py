"""Single points at a stored QM9S geometry with several method variants: energy, dipole, polarizability, max gradient."""
import os, sys, zipfile, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qm9s_modes as qm
from pyscf import dft, gto, lib
from pyscf.prop.polarizability.rks import Polarizability
lib.num_threads(32)
ZIP, QID = sys.argv[1], int(sys.argv[2])
text = zipfile.ZipFile(ZIP).read(f"qm9s_csv/{QID:06d}.csv").decode()
rec = qm.parse(text)
import csv, io
rows = [[c for c in r[1:] if c != ""] for r in list(csv.reader(io.StringIO(text)))[1:]]
E0 = float(rows[1][0]) / 27.211386245988
mu0 = np.array(rows[3], float); pol0 = np.array(rows[6], float)
print(f"stored: E {E0:.6f} Ha  mu {np.round(mu0,4)} |mu| {np.linalg.norm(mu0):.4f}  pol {np.round(pol0,3)}")
for basis, xc, grid in [("def2-tzvp", "b3lyp", (99, 590)), ("def2-tzvp", "b3lyp5", (99, 590)), ("def2-tzvp", "b3lyp", (75, 302)),
                        ("6-31g(2df,p)", "b3lyp", (99, 590))]:
    mol = gto.M(atom=[(int(a), tuple(r)) for a, r in zip(rec[3], rec[4])], basis=basis, unit="Angstrom", verbose=0)
    mol.cart = False
    mf = dft.RKS(mol); mf.xc = xc; mf.grids.atom_grid = grid; mf.conv_tol = 1e-11; e = mf.kernel()
    mu = mf.dip_moment(unit="Debye", verbose=0)
    g = mf.nuc_grad_method().kernel()
    p = Polarizability(mf).polarizability()
    pv = [p[0,0], p[0,1], p[1,1], p[0,2], p[1,2], p[2,2]]
    print(f"{basis:13s} {xc:7s} {grid}: E {e:.6f} (dE {e-E0:+.6f})  mu {np.round(mu,4)} |mu| {np.linalg.norm(mu):.4f}  pol {np.round(pv,3)}  max|g| {np.abs(g).max():.1e}")
