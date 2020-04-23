"""
Testing the integral calculation methods
========================================

Simple tests to check whether or not using the pySCF and psi4 integral calculators yield the same results.
"""
from hf.HartreeFock import *
import numpy as np
from pyscf import gto
import psi4
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create the molecules.
h2o = gto.M(atom='O 0.000000000000 -0.143225816552 0.000000000000;'
                 'H 1.638036840407 1.136548822547 -0.000000000000;'
                 'H -1.638036840407 1.136548822547 -0.000000000000',
            unit='bohr',
            basis='sto-3g')

h2o_psi4 = psi4.geometry("""
# O 0.000000000000 -0.143225816552 0.000000000000
# H 1.638036840407 1.136548822547 -0.000000000000
# H -1.638036840407 1.136548822547 -0.000000000000
# units bohr
# symmetry c1
# """)

psi4.set_options({'basis': 'sto-3g'})


def test_overlap():
    """
    Test whether or not psi4 and pyscf give the same overlap integrals.
    """
    x = RHF.MF(h2o, 10)
    y = RHF.MF(h2o_psi4, 10, 'psi4')
    ovlp1 = np.sum(x.get_ovlp())
    ovlp2 = np.sum(y.get_ovlp())
    assert np.isclose(ovlp1, ovlp2)


def test_one_e():
    """
    Test whether or not psi4 and pyscf give the same core Hamiltonian integrals.
    :return:
    """
    x = RHF.MF(h2o, 10)
    y = RHF.MF(h2o_psi4, 10, 'psi4')
    one_e_1 = np.sum(x.get_one_e())
    one_e_2 = np.sum(y.get_one_e())
    assert np.isclose(one_e_1, one_e_2)


def test_two_e():
    """
    Test whether or not psi4 and pyscf give the same two electron integrals.
    """
    x = RHF.MF(h2o, 10)
    y = RHF.MF(h2o_psi4, 10, 'psi4')
    two_e_1 = np.sum(x.get_two_e())
    two_e_2 = np.sum(y.get_two_e())
    assert np.isclose(two_e_1, two_e_2)
