"""
Testing the RHF and UHF methods
===============================

Simple tests to check whether or not the functions return the correct value.
"""
from ghf.RHF import RHF
from ghf.UHF import UHF
from ghf.GHF import GHF
import numpy as np
from pyscf import *
import psi4
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin=1, basis='cc-pvdz')

h4 = gto.M(atom='h 0 0.707107 0; h 0.707107 0 0; h 0 -0.707107 0; h -0.707107 0 0', spin=2, basis='cc-pvdz')

h2o = gto.M(atom='O 0.000000000000 -0.143225816552 0.000000000000; H 1.638036840407 1.136548822547 -0.000000000000; '
                 'H -1.638036840407 1.136548822547 -0.000000000000', unit='bohr', basis='sto-3g')

h2o_psi4 = psi4.geometry("""
O 0.000000000000 -0.143225816552 0.000000000000 
H 1.638036840407 1.136548822547 -0.000000000000
H -1.638036840407 1.136548822547 -0.000000000000
units bohr
symmetry c1
""")

psi4.set_options({'basis': 'sto-3g'})


def test_rhf():
    """
    test_RHF will test whether or not the RHF method returns the wanted result. The accuracy is 10^-11.

    """
    x = RHF(h4, 4)
    assert -1.940359839283 <= x.get_scf_solution() <= -1.940359839282


def test_uhf():
    """
    test_UHF will test the regular UHF method, by checking whether or not it returns the expected result.
    The accuracy is 10^-6.
    """
    x = UHF(h3, 3)
    assert -1.506275 <= x.get_scf_solution() <= -1.506274


def test_extra_e():
    """
    test_extra_e will test the UHF method, with the added option of first adding 2 electrons to the system and using
    those coefficients for the actual system, by checking whether or not it returns the expected result.
    The accuracy is 10^-6.
    """
    x = UHF(h4, 4)
    guess = x.extra_electron_guess()
    assert -2.021089 <= x.get_scf_solution(guess) <= -2.021088


def test_stability():
    """
    test_stability will test the UHF method, with stability analysis, by checking whether or not it returns
    the expected result. The accuracy is 10^-6.
    """
    x = UHF(h4, 4)
    guess = x.stability()
    assert -2.021089 <= x.get_scf_solution(guess) <= -2.021088


def test_overlap():
    """
    Test whether or not psi4 and pyscf give the same overlap integrals.
    """
    x = RHF(h2o, 10)
    y = RHF(h2o_psi4, 10, 'psi4')
    ovlp1 = np.sum(x.get_ovlp())
    ovlp2 = np.sum(y.get_ovlp())
    assert np.isclose(ovlp1, ovlp2) == True


def test_one_e():
    """
    Test whether or not psi4 and pyscf give the same core Hamiltonian integrals.
    :return:
    """
    x = RHF(h2o, 10)
    y = RHF(h2o_psi4, 10, 'psi4')
    one_e_1 = np.sum(x.get_one_e())
    one_e_2 = np.sum(y.get_one_e())
    assert np.isclose(one_e_1, one_e_2) == True


def test_two_e():
    """
    Test whether or not psi4 and pyscf give the same two electron integrals.
    """
    x = RHF(h2o, 10)
    y = RHF(h2o_psi4, 10, 'psi4')
    two_e_1 = np.sum(x.get_two_e())
    two_e_2 = np.sum(y.get_two_e())
    assert np.isclose(two_e_1, two_e_2) == True


def test_pyscf_vs_psi4():
    """
    Test a pyscf energy calculation vs a psi4 energy calculation.
    """
    x = RHF(h2o, 10)
    y = RHF(h2o_psi4, 10, 'psi4')
    assert np.isclose(x.scf(convergence=1e-6)[0], y.scf(convergence=1e-6)[0])
    assert x.scf(convergence=1e-6)[1] == y.scf(convergence=1e-6)[1]


def test_diis_rhf():
    """
    Test whether diis gives the same energy in fewer iterations.
    """
    x = RHF(h2o, 10)
    assert np.isclose(x.scf(convergence=1e-6)[0], x.diis(convergence=1e-6)[0])
    assert x.scf(convergence=1e-6)[1] >= x.diis(convergence=1e-6)[1]


def test_diis_uhf():
    """
    Test whether diis gives the same energy in fewer iterations.
    """
    x = UHF(h2o, 10)
    assert np.isclose(x.scf(convergence=1e-6)[0], x.diis(convergence=1e-6)[0])
    assert x.scf(convergence=1e-6)[1] >= x.diis(convergence=1e-6)[1]


def test_diis_real_ghf():
    """
    Test whether diis gives the same energy in fewer iterations.
    """
    x = GHF(h2o, 10)
    assert np.isclose(x.scf(convergence=1e-6)[0], x.diis(convergence=1e-6)[0])
    assert x.scf(convergence=1e-6)[1] >= x.diis(convergence=1e-6)[1]
