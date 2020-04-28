"""
Testing the UHF method
=======================

Simple tests to check whether or not the functions return the correct value.
"""
from hf.HartreeFock import *
import numpy as np
from pyscf import gto
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Creating the molecules.
h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin=1, basis='sto-3g')
h4 = gto.M(atom='h 0 0.707107 0; h 0.707107 0 0; h 0 -0.707107 0; h -0.707107 0 0', spin=2, basis='cc-pvdz')
Be = gto.M(atom='Be 1 0 0', basis='4-31g')


def test_scf():
    """
    Test whether or not the UHF method returns the correct result when using regular scf.

    """
    x = UHF.MF(h3, 3)
    assert np.isclose(-1.335980054016416, x.scf()[0])


def test_diis():
    """
    Test whether or not the UHF method returns the correct result when using diis.

    """
    x = UHF.MF(h3, 3)
    assert np.isclose(-1.335980054016416, x.diis()[0])


def test_complex_method():
    """
    Most systems don't have a real/complex instability in UHF, but the complex method should still find the same
    energy as the real UHF state with complex MO coefficients.
    """
    x = UHF.MF(h4, 4)
    assert np.isclose(-1.956749178, x.scf(complex_method=True))
    c_i_a = x.get_mo_coeff()[0].imag
    c_i_b = x.get_mo_coeff()[1].imag
    assert np.sum(c_i_a) > 1e-3
    assert np.sum(c_i_b) > 1e-3


def test_extra_e():
    """
    test_extra_e will test the UHF method, with the added option of first adding 2 electrons to the system and using
    those coefficients for the actual system, by checking whether or not it returns the correct result.
    """
    x = UHF.MF(h4, 4)
    guess = x.extra_electron_guess()
    assert np.isclose(-2.021088247649845, x.get_scf_solution(guess))


def test_stability_analysis():
    """
    A test that checks whether the stability analysis correctly estimates a function's stability.
    """
    ber = UHF.MF(Be, 4)
    ber.scf()
    ber.stability_analysis('internal')
    ber.stability_analysis('external')
    assert ber.int_instability
    assert ber.ext_instability_rc is None
    assert ber.ext_instability_ug


def test_follow_stability_analysis():
    """
    A test that will perform an internal stability analysis, and check that the rotated coefficients return a lower
    energy state. The test molecule is H_4, which has an internal instbility in the UHF space.
    """
    x = UHF.MF(h4, 4)
    x.scf()
    guess = x.stability_analysis('internal')
    assert np.allclose(-2.021088247649845, x.get_scf_solution(guess))
