"""B3LYP/TZVP Hessian and polarizability derivatives in the QM9S format, with PySCF.

qm9s <id>:     use the stored geometry of a QM9S molecule (no optimisation), to check that we
               reproduce its stored Hessian and polarizability derivatives.
smiles <smi>:  RDKit conformers -> lowest MMFF energy -> B3LYP/TZVP optimisation -> same outputs.

QM9S is B3LYP/TZVP in Gaussian 16 (Ahlrichs 1994 TZVP, spherical d), not def2-TZVP: with this basis
PySCF reproduces the stored methanol energy (0.05 mHa), dipole and polarizability.

Output npz: z, xyz (A), hess (3n x 3n, eV/A^2), dpol (n x 18, Bohr^3/A, per atom
[x: xx xy yy zx zy zz][y: ...][z: ...]), grad_max (Hartree/Bohr at the final geometry).
Polarizability derivatives: by default from finite-field analytic gradients (dpol_field.py;
Raman activities within 0.25% of the displacement route on acetonitrile, THF and 2,2-DMP, about
19x faster at 19 atoms). DPOL_ROUTE=displace gives the step 1 route: central differences (step
0.005 A) of the analytic (CPKS) static polarizability on a (75, 302) grid, in parallel workers.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
STEP = 0.005  # A
COMP = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]  # xx xy yy zx zy zz


DF = os.environ.get("DFT_DF", "0") == "1"  # density fitting (def2-universal-jkfit)
# "field": finite-field analytic gradients (dpol_field.py, 19 gradients); "displace": 6n CPKS polarizabilities
ROUTE = os.environ.get("DPOL_ROUTE", "field")


def build(z, xyz, threads, grid=(99, 590)):
    from pyscf import dft, gto, lib
    from tzvp_basis import tzvp
    lib.num_threads(threads)
    mol = gto.M(atom=[(int(a), tuple(r)) for a, r in zip(z, xyz)], basis=tzvp(), unit="Angstrom",
                verbose=0, max_memory=6000)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"  # PySCF >= 2.3: VWN-RPA variant, same as Gaussian's B3LYP
    mf.grids.atom_grid = grid  # (99, 590) = Gaussian "ultrafine"
    mf.conv_tol = 1e-11
    if DF:
        mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    return mol, mf


def polar(z, xyz, threads, dm0=None):
    from pyscf.prop.polarizability.rks import Polarizability
    _, mf = build(z, xyz, threads, grid=(75, 302))
    mf.kernel(dm0=dm0)
    assert mf.converged
    p = Polarizability(mf)
    p.conv_tol = 1e-10
    return np.asarray(p.polarizability())


def _worker(args):
    z, xyz, threads, dm0 = args
    return polar(z, xyz, threads, dm0)


def conformer(smi):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    cids = AllChem.EmbedMultipleConfs(m, numConfs=50, randomSeed=0)
    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    best = int(np.argmin([e for _, e in res]))
    pos = m.GetConformer(cids[best]).GetPositions()
    return np.array([a.GetAtomicNum() for a in m.GetAtoms()]), pos


def run(z, xyz, optimise, nproc, threads, log, ckpt=None):
    t0 = time.time()
    from pyscf import lib
    if ckpt and os.path.exists(ckpt):  # optimisation and Hessian already done; redo only the dpol step
        c = np.load(ckpt)
        z, xyz, hess, g = c["z"], c["xyz"], c["hess"], c["grad"]
        log(f"  resumed optimised geometry and Hessian from {os.path.basename(ckpt)}")
        return _dpol(z, xyz, hess, g, nproc, threads, log, t0 - float(c["seconds"]))
    mol, mf = build(z, xyz, nproc * threads)
    mf.kernel()
    if optimise:
        from pyscf.geomopt.geometric_solver import optimize
        mol = optimize(mf, maxsteps=200, convergence_set="GAU_TIGHT")
        xyz = mol.atom_coords(unit="Angstrom")
        mol, mf = build(z, xyz, nproc * threads)
        mf.kernel()
        log(f"  optimised ({time.time() - t0:.0f} s)")
    assert mf.converged
    lib.num_threads(nproc * threads)
    g = mf.nuc_grad_method().kernel()
    h = mf.Hessian().kernel()  # natm, natm, 3, 3 in Hartree/Bohr^2
    n = len(z)
    hess = h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) * HARTREE_EV / BOHR_A ** 2
    log(f"  hessian ({time.time() - t0:.0f} s), max |grad| {np.abs(g).max():.2e} Ha/Bohr")
    if ckpt:
        np.savez(ckpt, z=z, xyz=xyz, hess=hess, grad=g, seconds=time.time() - t0)
    return _dpol(z, xyz, hess, g, nproc, threads, log, t0)


def _dpol(z, xyz, hess, g, nproc, threads, log, t0):
    n = len(z)
    if ROUTE == "field":
        from dpol_field import dpol_ff
        dpol = dpol_ff(z, xyz, log=log)
        return dict(z=z, xyz=xyz, hess=hess, dpol=dpol, grad_max=np.abs(g).max(), seconds=time.time() - t0,
                    dpol_route=ROUTE)
    _, mf = build(z, xyz, nproc * threads)  # displacement route: reference density for the workers
    mf.kernel()
    dm0 = mf.make_rdm1()
    jobs = []
    for a in range(n):
        for k in range(3):
            for s in (+1, -1):
                x = xyz.copy()
                x[a, k] += s * STEP
                jobs.append((z, x, threads, dm0))
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc) as pool:
        al = np.array(pool.map(_worker, jobs, chunksize=1)).reshape(n, 3, 2, 3, 3)
    dal = (al[:, :, 0] - al[:, :, 1]) / (2 * STEP)  # n, 3(coord), 3, 3 in Bohr^3/A
    dpol = np.stack([dal[:, :, i, j] for i, j in COMP], -1).reshape(n, 18)
    log(f"  polarizability derivatives ({time.time() - t0:.0f} s)")
    return dict(z=z, xyz=xyz, hess=hess, dpol=dpol, grad_max=np.abs(g).max(), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["qm9s", "smiles"])
    ap.add_argument("value")
    ap.add_argument("out")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    def log(s):
        print(s, flush=True)

    if a.kind == "qm9s":
        import qm9s_modes as qm
        rec = qm.parse(zipfile.ZipFile(a.zip).read(f"qm9s_csv/{int(a.value):06d}.csv").decode())
        z, xyz = rec[3], rec[4]
        log(f"QM9S {a.value} {rec[2]} natoms {len(z)}")
        r = run(z, xyz, False, a.nproc, a.threads, log)
    else:
        z, xyz = conformer(a.value)
        log(f"SMILES {a.value} natoms {len(z)}")
        r = run(z, xyz, True, a.nproc, a.threads, log, ckpt=a.out.replace(".npz", "_hess.npz"))
    np.savez(a.out, **r, source=a.value)


if __name__ == "__main__":
    main()
