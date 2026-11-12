"""Re-optimise any result that landed on a saddle point (imaginary frequency).

Displaces the geometry along each imaginary normal mode (largest atom move 0.1 A), re-optimises
and recomputes the Hessian and polarizability derivatives. Repeats up to 3 times.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_raman as dr  # noqa: E402
import qm9s_modes as qm  # noqa: E402


def imag_modes(z, xyz, hess):
    n = len(z)
    m3 = np.repeat([qm.MASS[a] for a in z], 3)
    hm = hess / np.sqrt(np.outer(m3, m3))
    hm = 0.5 * (hm + hm.T)
    U = qm.tr_basis(z, xyz)
    P = np.eye(3 * n) - U @ U.T
    w, L = np.linalg.eigh(P @ hm @ P)
    overlap = ((U.T @ L) ** 2).sum(0)
    keep = np.sort(np.argsort(overlap)[: 3 * n - U.shape[1]])
    w, L = w[keep], L[:, keep]
    nu = np.sign(w) * qm.EV_A2_AMU_TO_CM * np.sqrt(np.abs(w))
    return nu, [(L[:, k] / np.sqrt(m3)).reshape(n, 3) for k in np.flatnonzero(nu < -5)]


def main():
    for path in sys.argv[1:]:
        r = dict(np.load(path))
        for it in range(3):
            nu, disp = imag_modes(r["z"], r["xyz"], r["hess"])
            if not disp:
                break
            print(f"{os.path.basename(path)}: imaginary {np.round(nu[nu < -5], 1)} cm^-1, displacing (round {it + 1})", flush=True)
            xyz = r["xyz"].copy()
            for v in disp:
                xyz = xyz + 0.1 * v / np.abs(v).max()
            src = str(r["source"])
            r = dr.run(r["z"], xyz, True, int(os.environ.get("FIX_NPROC", 8)), 4, lambda s: print(s, flush=True))
            r["source"] = src
            np.savez(path, **r)
        nu, _ = imag_modes(r["z"], r["xyz"], r["hess"])
        print(f"{os.path.basename(path)}: lowest frequency {nu.min():.1f} cm^-1", flush=True)


if __name__ == "__main__":
    main()
