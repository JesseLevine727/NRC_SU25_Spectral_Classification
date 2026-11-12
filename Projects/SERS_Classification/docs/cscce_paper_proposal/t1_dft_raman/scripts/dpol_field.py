"""Polarizability derivatives from finite-field analytic nuclear gradients.

d(alpha_ij)/dR = -d^2 g_R / dF_i dF_j, with g_R the analytic B3LYP/TZVP gradient in a uniform field F.
19 gradient evaluations per molecule (0, +-F_i, +-F_i+-F_j) instead of 6n polarizability calculations,
so the cost no longer grows with the number of displaced atoms. Same basis and functional as
dft_raman.py; the field term enters the core Hamiltonian and its basis-centre derivative enters the
gradient, and the grid response is included so the grid moves with the atoms as it does in the
displacement route.

check <npz>: recompute dpol at the stored geometry and compare with the stored dpol.
"""
import itertools
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BOHR_A = 0.529177210903
FIELD = 0.003  # a.u.
COMP = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]  # xx xy yy zx zy zz


def field_gradient(z, xyz, F, threads, grid, dm0=None):
    from pyscf import dft, gto, lib
    from tzvp_basis import tzvp
    lib.num_threads(threads)
    mol = gto.M(atom=[(int(a), tuple(r)) for a, r in zip(z, xyz)], basis=tzvp(), unit="Angstrom",
                verbose=0, max_memory=4000)
    F = np.asarray(F, float)
    r = mol.intor("int1e_r")  # <mu|r_k|nu>, origin at 0
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.atom_grid = grid
    # the orbital gradient sets the density (and so the nuclear gradient) precision; the energy test is
    # loose because total energies of the larger molecules carry ~1e-12 Ha of numerical noise
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 150
    h0 = mf.get_hcore()
    hf = h0 + np.einsum("k,kij->ij", F, r)  # electron in field: +F.r
    mf.get_hcore = lambda *a, **k: hf
    mf.kernel(dm0=dm0)
    if not mf.converged:  # fall back once to a slightly looser orbital gradient, from the last density
        mf.conv_tol_grad = 5e-8
        mf.kernel(dm0=mf.make_rdm1())
    assert mf.converged
    g = mf.nuc_grad_method()
    g.grid_response = True
    irp = mol.intor("int1e_irp", comp=9).reshape(3, 3, mol.nao, mol.nao)  # <mu| r_k d_d |nu>
    dfield = -np.einsum("k,kdji->dij", F, irp)  # -<d_d mu| F.r |nu>, PySCF get_hcore convention
    base = g.get_hcore
    g.get_hcore = lambda mol=None: base(mol) + dfield
    return g.kernel(), mf.make_rdm1()


def _worker(args):
    return field_gradient(*args)[0]


def dpol_ff(z, xyz, nproc=int(os.environ.get("DPOL_NPROC", 10)), threads=int(os.environ.get("DPOL_THREADS", 3)),
            F=FIELD, grid=(75, 302), log=print):
    t0 = time.time()
    g0, dm0 = field_gradient(z, xyz, np.zeros(3), nproc * threads, grid)
    fields = []
    for i in range(3):
        for s in (+1, -1):
            e = np.zeros(3)
            e[i] = s * F
            fields.append(e)
    for i, j in itertools.combinations(range(3), 2):
        for si, sj in itertools.product((+1, -1), repeat=2):
            e = np.zeros(3)
            e[i], e[j] = si * F, sj * F
            fields.append(e)
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc) as pool:
        gs = pool.map(_worker, [(z, xyz, e, threads, grid, dm0) for e in fields], chunksize=1)
    gs = dict(zip([tuple(np.sign(e).astype(int)) for e in fields], gs))
    n = len(z)
    dal = np.zeros((n, 3, 3, 3))  # atom, coord, i, j  in Bohr^3/Bohr
    for i in range(3):
        p, m = [0, 0, 0], [0, 0, 0]
        p[i], m[i] = 1, -1
        dal[:, :, i, i] = -(gs[tuple(p)] + gs[tuple(m)] - 2 * g0) / F ** 2
    for i, j in itertools.combinations(range(3), 2):
        def G(si, sj):
            k = [0, 0, 0]
            k[i], k[j] = si, sj
            return gs[tuple(k)]
        v = -(G(1, 1) - G(1, -1) - G(-1, 1) + G(-1, -1)) / (4 * F ** 2)
        dal[:, :, i, j] = dal[:, :, j, i] = v
    dal /= BOHR_A  # per Angstrom, as QM9S
    log(f"  polarizability derivatives, finite field ({time.time() - t0:.0f} s)")
    return np.stack([dal[:, :, i, j] for i, j in COMP], -1).reshape(n, 18)


def main():
    import qm9s_modes as qm
    for path in sys.argv[2:]:
        r = np.load(path)
        t0 = time.time()
        d1 = dpol_ff(r["z"], r["xyz"])
        d0 = r["dpol"]
        nu0, a0, _, _ = qm.modes(r["z"], r["xyz"], d0, r["hess"])
        _, a1, _, _ = qm.modes(r["z"], r["xyz"], d1, r["hess"])
        big = a0 > 0.01 * a0.max()
        print(f"{os.path.basename(path)}: {len(r['z'])} atoms, {time.time() - t0:.0f} s "
              f"(displacement route {float(r['seconds']):.0f} s total incl. opt+Hessian)")
        print(f"  dpol max |diff| {np.abs(d1 - d0).max():.4f} of max {np.abs(d0).max():.2f} Bohr^3/A; "
              f"activity rel diff max {np.max(np.abs(a1 - a0)[big] / a0[big]):.4f}, "
              f"median {np.median(np.abs(a1 - a0)[big] / a0[big]):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        main()
