"""Compare our PySCF Hessian / polarizability derivatives with the values stored in QM9S."""
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qm9s_modes as qm  # noqa: E402

ZIP = sys.argv[1]
for path in sys.argv[2:]:
    r = np.load(path)
    qid = int(str(r["source"]))
    rec = qm.parse(zipfile.ZipFile(ZIP).read(f"qm9s_csv/{qid:06d}.csv").decode())
    z, xyz, dpol, hess = rec[3:]
    assert np.allclose(xyz, r["xyz"])
    nu0, act0, _, _ = qm.modes(z, xyz, dpol, hess)
    nu1, act1, _, _ = qm.modes(z, xyz, r["dpol"], r["hess"])
    print(f"QM9S {qid} {rec[2]}: max|grad| {float(r['grad_max']):.1e} Ha/Bohr, {float(r['seconds']):.0f} s")
    print(f"  Hessian max |diff| {np.abs(r['hess'] - hess).max():.4f} eV/A^2 (max |H| {np.abs(hess).max():.1f}); "
          f"dpol max |diff| {np.abs(r['dpol'] - dpol).max():.4f} Bohr^3/A (max {np.abs(dpol).max():.2f})")
    print(f"  freq |diff| max {np.abs(nu1 - nu0).max():.2f} cm^-1, mean {np.abs(nu1 - nu0).mean():.2f}; "
          f"activity rel diff (modes with act > 1% of max) "
          f"max {np.max(np.abs(act1 - act0)[act0 > 0.01 * act0.max()] / act0[act0 > 0.01 * act0.max()]):.3f}")
    for f0, f1, a0, a1 in zip(nu0, nu1, act0, act1):
        print(f"    {f0:8.1f} {f1:8.1f}   act {a0:8.2f} {a1:8.2f}")
