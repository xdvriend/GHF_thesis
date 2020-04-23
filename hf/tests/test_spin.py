"""
Testing the functions used to calculate the spin expectation value.
===================================================================

Simple tests to check whether or not the functions return the correct value.
"""
from hf.HartreeFock import *
from hf.utilities import spin as spin
from hf.utilities import SCF_functions as Scf
import numpy as np
from pyscf import gto
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Creating the molecules.
h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin=1, basis='cc-pvdz')


def test_uhf_spin():
    x = UHF.MF(h3, 3)
    x.scf()
    s = spin.uhf(x.n_a, x.n_b, x.get_mo_coeff()[0], x.get_mo_coeff()[1], x.get_ovlp())
    assert np.isclose(0.773570840069661, s[0])
    assert np.isclose(0.5, s[1])
    assert np.isclose(2.02343355716926, s[2])


def test_ghf_spin():
    x = GHF.MF(h3, 3)
    g = x.random_guess()
    x.diis(g)
    s = spin.ghf(x.get_mo_coeff(), x.number_of_electrons, Scf.trans_matrix(x.get_ovlp()))
    assert np.isclose(-2.1576929460587202e-05, s[0])
    assert np.isclose(0.7790732218658856, s[1])
    assert np.isclose(2.02886492587938, s[2])
