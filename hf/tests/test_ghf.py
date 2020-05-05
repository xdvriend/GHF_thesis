"""
Testing the GHF method
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
Be = gto.M(atom='Be 1 0 0', basis='4-31g')


def test_scf():
    """
    Test whether or not the GHF method returns the correct result when using regular scf.

    """
    x = GHF.MF(Be, 4)
    g = x.random_guess()
    assert np.isclose(-14.557798355720772, x.scf(g)[0])


def test_diis():
    """
    Test whether or not the GHF method returns the correct result when using diis.

    """
    b = GHF.MF(Be, 4)
    g = b.random_guess()
    g = b.unitary_rotation_guess(g)
    g = b.unitary_rotation_guess(g)
    g = b.unitary_rotation_guess(g)
    g = b.unitary_rotation_guess(g)
    assert np.isclose(-14.557798355720772, b.diis(g)[0])


def test_complex_method():
    """
    Beryllium in the 4-31G basis set is one of the few systems with a known complex GHF solution.
    This test checks whether the complex GHF algorithm can find it.
    """
    a = GHF.MF(Be, 4)
    a.scf()
    mos = a.stability_analysis('internal')
    b = a.scf(mos, complex_method=True)
    c_i = a.get_mo_coeff().imag
    assert np.isclose(-14.55781859592479, b[0])
    assert abs(np.sum(c_i)) > 1e-3


def test_int_stability():
    """
    test_int_stability will test the internal stability analysis on the GHF method by checking the stability of Be,
    which should be None, since the solution is internally stable.
    """
    x = GHF.MF(Be, 4)
    g = x.random_guess()
    x.scf(g)
    x.stability_analysis('internal')
    assert x.int_instability is None


def test_ext_stability():
    """
    test_int_stability will test the internal stability analysis on the GHF method by checking the stability of Be,
    which should be True, since the solution is externally unstable.
    """
    x = GHF.MF(Be, 4)
    g = x.random_guess()
    x.scf(g)
    x.stability_analysis('external')
    assert x.ext_instability
