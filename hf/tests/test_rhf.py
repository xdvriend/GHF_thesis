"""
Testing the RHF method
=======================

Simple tests to check whether or not the functions return the correct value.
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
O 0.000000000000 -0.143225816552 0.000000000000
H 1.638036840407 1.136548822547 -0.000000000000
H -1.638036840407 1.136548822547 -0.000000000000
units bohr
symmetry c1
""")

psi4.set_options({'basis': 'sto-3g'})

h4 = gto.M(atom='h 0 0.707107 0;'
                'h 0.707107 0 0;'
                'h 0 -0.707107 0;'
                'h -0.707107 0 0',
           spin=2,
           basis='sto-3g')


def test_scf():
    """
    Test whether or not the RHF method returns the correct result when using regular scf.

    """
    x = RHF.MF(h2o, 10)
    assert np.isclose(-74.94207992819219, x.scf()[0])


def test_diis():
    """
    Test whether or not the RHF method returns the correct result when using diis.

    """
    x = RHF.MF(h2o, 10)
    assert np.isclose(-74.94207992819219, x.diis()[0])


def test_complex_method():
    """
    Most systems don't have a real/complex instability in RHF, but the complex method should still find the same
    energy as the real RHF state with complex MO coefficients.
    """
    x = RHF.MF(h2o, 10)
    assert np.isclose(-74.9420799281921, x.scf(complex_method=True))
    c_i = x.get_mo_coeff().imag
    assert np.isclose(np.sum(c_i), 0) is False


def test_scf_vs_diis():
    """
    Test whether or not the RHF method returns the correct result when using regular scf.

    """
    x = RHF.MF(h2o, 10)
    assert x.scf(convergence=1e-6)[1] >= x.diis(convergence=1e-6)[1]


def test_pyscf_vs_psi4():
    """
    Test a pyscf energy calculation vs a psi4 energy calculation.
    """
    x = RHF.MF(h2o, 10)
    y = RHF.MF(h2o_psi4, 10, 'psi4')
    assert np.isclose(x.scf(convergence=1e-6)[0], y.scf(convergence=1e-6)[0])
    assert x.scf(convergence=1e-6)[1] == y.scf(convergence=1e-6)[1]


def test_stability():
    """
    A test for the RHF stability analysis.
    """
    x = RHF.MF(h2o, 10)
    x.diis()
    x.stability_analysis('internal')
    x.stability_analysis('external')
    assert x.int_instability is None
    assert x.ext_instability_rc is None
    assert x.ext_instability_ru is None


def test_follow_stability():
    """
    A test used to see if following the stability analysis can lead to a lower lying energy state and
    a more stable solution.
    """
    test = RHF.MF(h4, 4)
    test.get_scf_solution()

    x = test.stability_analysis('internal')
    while test.int_instability:
        test.get_scf_solution(x)
        x = test.stability_analysis('internal')
    assert test.int_instability is None
