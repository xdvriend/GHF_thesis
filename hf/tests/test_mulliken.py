"""
Testing the functions used to calculate the Mulliken charges.
===================================================================

Simple tests to check whether or not the functions return the correct value.
"""
from hf.HartreeFock import *
from hf.properties.mulliken import mulliken
import numpy as np
from pyscf import gto
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Creating the molecules.
h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
no = gto.M(atom = 'N 0 0 0; O 0 0 1', spin=1, basis = 'sto-3g')
hf = gto.M(atom = 'F 0 0 0; H 0 0 0.9', basis = 'sto-3g')


def test_uhf_mulliken():
    x = UHF.MF(no, 15)
    y = UHF.MF(h3, 3)
    z = UHF.MF(hf, 10)
    x.scf()
    y.scf()
    z.scf()
    #m = mulliken(no, x.get_dens(), x.get_ovlp())[0]
    m = x.calculate_mulliken()[0]
    l = y.calculate_mulliken()[0]
    k = z.calculate_mulliken()[0]
    assert np.isclose(m[0], 0.07271, atol=1e-3)
    assert np.isclose(m[1], -0.07271, atol=1e-3)
    assert np.isclose(l[0], -0.12265, atol=1e-3)
    assert np.isclose(l[1], 0.24507, atol=1e-3)
    assert np.isclose(l[2], -0.12265, atol=1e-3)
    assert np.isclose(k[0], -0.220092, atol=1e-3)
    assert np.isclose(k[1], 0.220092, atol=1e-3)


