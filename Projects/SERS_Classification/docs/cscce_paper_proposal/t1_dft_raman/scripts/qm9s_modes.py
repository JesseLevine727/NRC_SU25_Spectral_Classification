"""Harmonic frequencies and Raman activities from QM9S per-molecule CSVs.

QM9S stores the Hessian (eV/A^2) and the Cartesian derivative of the
polarizability (Bohr^3/A, components xx, xy, yy, zx, zy, zz) for each molecule.
"""
import csv
import io
import sys
import zipfile

import numpy as np

MASS = {1: 1.00782503, 6: 12.0, 7: 14.00307401, 8: 15.99491462, 9: 18.99840316}
EV_A2_AMU_TO_CM = 521.4709  # sqrt(eV / (A^2 amu)) / (2 pi c) in cm^-1
BOHR3_TO_A3 = 0.14818471


def parse(text):
    rows = list(csv.reader(io.StringIO(text)))[1:]  # drop pandas header
    rows = [[c for c in r[1:] if c != ""] for r in rows]  # drop index column
    head = rows[0]
    qm9s_id, qm9_id, smiles, n = int(float(head[0])), int(float(head[1])), head[2], int(float(head[3]))
    geo = np.array(rows[10:10 + n], dtype=float)
    z = geo[:, 0].astype(int)
    xyz = geo[:, 1:4]
    dpol = np.array(rows[10 + 2 * n:10 + 3 * n], dtype=float)  # n x 18
    hess = np.array(rows[10 + 3 * n:10 + 6 * n], dtype=float)  # 3n x 3n
    return qm9s_id, qm9_id, smiles, z, xyz, dpol, hess


def tr_basis(z, xyz):
    m = np.array([MASS[a] for a in z])
    com = (m[:, None] * xyz).sum(0) / m.sum()
    r = xyz - com
    sm = np.sqrt(m)
    vecs = []
    for k in range(3):
        v = np.zeros((len(z), 3))
        v[:, k] = sm
        vecs.append(v.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        v = np.cross(e, r) * sm[:, None]
        vecs.append(v.ravel())
    B = np.array(vecs).T
    u, s, _ = np.linalg.svd(B, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]


def modes(z, xyz, dpol, hess, order="coord_major"):
    """Return wavenumbers (cm^-1) and Raman activities (A^4/amu)."""
    n = len(z)
    m3 = np.repeat([MASS[a] for a in z], 3)
    hm = hess / np.sqrt(np.outer(m3, m3))
    hm = 0.5 * (hm + hm.T)
    U = tr_basis(z, xyz)
    P = np.eye(3 * n) - U @ U.T
    w, L = np.linalg.eigh(P @ hm @ P)
    # drop the translation/rotation null space: the eigenvectors with largest overlap with U
    overlap = ((U.T @ L) ** 2).sum(0)
    keep = np.argsort(overlap)[: 3 * n - U.shape[1]]
    keep = np.sort(keep)
    w, L = w[keep], L[:, keep]
    nu = np.sign(w) * EV_A2_AMU_TO_CM * np.sqrt(np.abs(w))
    # polarizability derivative tensor d alpha_c / d x_(atom, k)
    if order == "coord_major":  # per atom: [x: 6 comps][y: 6][z: 6]
        d = dpol.reshape(n, 3, 6)
    else:  # per atom: [comp][coord]
        d = dpol.reshape(n, 6, 3).transpose(0, 2, 1)
    d = d.reshape(3 * n, 6) * BOHR3_TO_A3  # A^3 / A
    dq = (L / np.sqrt(m3)[:, None]).T @ d  # A^2 / sqrt(amu) per mode
    xx, xy, yy, zx, zy, zz = dq.T
    a = (xx + yy + zz) / 3.0
    g2 = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2 + 6 * (xy ** 2 + zy ** 2 + zx ** 2))
    act = 45 * a ** 2 + 7 * g2
    return nu, act, a, g2


if __name__ == "__main__":
    zp = zipfile.ZipFile(sys.argv[1])
    name = sys.argv[2]
    rec = parse(zp.read(f"qm9s_csv/{name}.csv").decode())
    print(rec[:3], "natoms", len(rec[3]))
    for order in ("coord_major", "comp_major"):
        nu, act, a, g2 = modes(*rec[3:], order=order)
        print(order)
        for f, s, aa, gg in zip(nu, act, a, g2):
            print(f"  {f:8.1f}  act {s:8.2f}  a'^2 {aa*aa:8.4f}  g'^2 {gg:8.4f}")
