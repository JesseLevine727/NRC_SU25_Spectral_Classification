"""Gaussian's 'TZVP' (Schaefer, Huber, Ahlrichs 1994): Ahlrichs TZV s/p plus one d on C-F.

H: (5s1p)/[3s1p], p 0.8, identical to the def2-TZVP hydrogen basis.
"""
import basis_set_exchange as bse
from pyscf import gto

D_EXP = {"C": 0.8, "N": 1.0, "O": 1.2, "F": 1.4}


def tzvp(elements=("H", "C", "N", "O", "F"), d_exp=D_EXP):
    out = {"H": gto.basis.parse(bse.get_basis("def2-TZVP", elements=["H"], fmt="nwchem"))}
    for el in elements:
        if el != "H":
            out[el] = gto.basis.parse(bse.get_basis("Ahlrichs TZV", elements=[el], fmt="nwchem")) + [[2, [d_exp[el], 1.0]]]
    return out
