import os, sys, zipfile, csv, io, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qm9s_modes as qm
from tzvp_basis import tzvp
from pyscf import dft, gto, lib
from pyscf.prop.polarizability.rks import Polarizability
lib.num_threads(32)
ZIP = sys.argv[1]
for QID in map(int, sys.argv[2:]):
    text = zipfile.ZipFile(ZIP).read(f"qm9s_csv/{QID:06d}.csv").decode()
    rec = qm.parse(text)
    rows = [[c for c in r[1:] if c != ""] for r in list(csv.reader(io.StringIO(text)))[1:]]
    E0 = float(rows[1][0]) / 27.211386245988
    mu0 = np.array(rows[3], float); pol0 = np.array(rows[6], float)
    print(f"QM9S {QID} {rec[2]} stored: E {E0:.6f}  |mu| {np.linalg.norm(mu0):.4f}  pol {np.round(pol0,3)}")
    for cart in (False, True):
        mol = gto.M(atom=[(int(a), tuple(r)) for a, r in zip(rec[3], rec[4])], basis=tzvp(), unit="Angstrom", verbose=0, cart=cart)
        mf = dft.RKS(mol); mf.xc = "b3lyp"; mf.grids.atom_grid = (99, 590); mf.conv_tol = 1e-11; e = mf.kernel()
        mu = mf.dip_moment(unit="Debye", verbose=0)
        g = mf.nuc_grad_method().kernel()
        p = Polarizability(mf).polarizability()
        pv = [p[0,0], p[0,1], p[1,1], p[0,2], p[1,2], p[2,2]]
        print(f"  TZVP cart={cart}: nao {mol.nao}  E {e:.6f} (dE {e-E0:+.6f})  |mu| {np.linalg.norm(mu):.4f}  pol {np.round(pv,3)}  max|g| {np.abs(g).max():.1e}")
