"""Speed test: small-basis pre-optimisation + density-fitted TZVP optimisation and Hessian, against
an exact-route result for the same molecule (step 3 run 1, 32 threads on an otherwise idle machine).

The polarizability step (dpol_field.py) is unchanged. Prints stage times and the differences in
geometry, frequencies and Raman activities.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_raman as dr  # noqa: E402
import qm9s_modes as qm  # noqa: E402
from dpol_field import dpol_ff  # noqa: E402

SMI, REF = sys.argv[1:3]
REF_TIMES = dict(opt=float(sys.argv[3]), hess=float(sys.argv[4]))
NT = 32


def rks(z, xyz, basis, grid, df):
    from pyscf import dft, gto, lib
    lib.num_threads(NT)
    mol = gto.M(atom=[(int(a), tuple(r)) for a, r in zip(z, xyz)], basis=basis, unit="Angstrom", verbose=0,
                max_memory=16000)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.atom_grid = grid
    mf.conv_tol = 1e-11
    if df:
        mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    return mf


def main():
    from pyscf.geomopt.geometric_solver import optimize
    from tzvp_basis import tzvp
    z, xyz = dr.conformer(SMI)
    T = {}
    t = time.time()
    mf = rks(z, xyz, "def2-svp", (75, 302), False)
    mf.kernel()
    xyz = optimize(mf, maxsteps=200).atom_coords(unit="Angstrom")
    T["pre-opt def2-SVP"] = time.time() - t
    t = time.time()
    mf = rks(z, xyz, tzvp(), (99, 590), True)
    mf.kernel()
    xyz = optimize(mf, maxsteps=200, convergence_set="GAU_TIGHT").atom_coords(unit="Angstrom")
    T["opt DF-TZVP"] = time.time() - t
    t = time.time()
    mf = rks(z, xyz, tzvp(), (99, 590), True)
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    h = mf.Hessian().kernel()
    n = len(z)
    hess = h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) * dr.HARTREE_EV / dr.BOHR_A ** 2
    T["hessian DF-TZVP"] = time.time() - t
    t = time.time()
    dpol = dpol_ff(z, xyz, log=lambda s: None)
    T["dpol (unchanged route)"] = time.time() - t

    r = np.load(REF)
    D0 = np.linalg.norm(r["xyz"][:, None] - r["xyz"][None], axis=-1)
    D1 = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    nu0, a0, _, _ = qm.modes(r["z"], r["xyz"], r["dpol"], r["hess"])
    nu1, a1, _, _ = qm.modes(z, xyz, dpol, hess)
    big = a0 > 0.01 * a0.max()
    print("stage times (s):", {k: round(v) for k, v in T.items()})
    opt_new = T["pre-opt def2-SVP"] + T["opt DF-TZVP"]
    print(f"optimisation: {opt_new:.0f} s vs exact {REF_TIMES['opt']:.0f} s  -> x{REF_TIMES['opt'] / opt_new:.2f}")
    print(f"hessian:      {T['hessian DF-TZVP']:.0f} s vs exact {REF_TIMES['hess']:.0f} s  -> x{REF_TIMES['hess'] / T['hessian DF-TZVP']:.2f}")
    tot_new = opt_new + T["hessian DF-TZVP"] + T["dpol (unchanged route)"]
    tot_ref = REF_TIMES["opt"] + REF_TIMES["hess"] + T["dpol (unchanged route)"]
    print(f"whole run:    {tot_new:.0f} s vs {tot_ref:.0f} s  -> x{tot_ref / tot_new:.2f}")
    print(f"max interatomic distance change {np.abs(D1 - D0).max():.4f} A; max|grad| {np.abs(g).max():.1e} Ha/Bohr")
    m = nu0 > 50
    print(f"frequencies (>50 cm^-1): max |diff| {np.abs(nu1 - nu0)[m].max():.2f} cm^-1, mean {np.abs(nu1 - nu0)[m].mean():.2f}")
    print(f"activities (modes > 1% of max): max rel diff {np.max(np.abs(a1 - a0)[big] / a0[big]):.4f}, "
          f"median {np.median(np.abs(a1 - a0)[big] / a0[big]):.4f}")
    print(f"lowest frequency {nu1.min():.1f} cm^-1 (exact route {nu0.min():.1f})")


if __name__ == "__main__":
    main()
