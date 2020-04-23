"""
Testing the RHF method
=======================

Simple tests to check whether or not the functions return the correct value.
"""
from hf.HartreeFock import *
import numpy as np
from pyscf import *
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create the molecules.
h2o = gto.M(atom='O 0.000000000000 -0.143225816552 0.000000000000;'
                 'H 1.638036840407 1.136548822547 -0.000000000000;'
                 'H -1.638036840407 1.136548822547 -0.000000000000',
            unit='bohr',
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


def test_scf_vs_diis():
    """
    Test whether or not the RHF method returns the correct result when using regular scf.

    """
    x = RHF.MF(h2o, 10)
    assert x.scf(convergence=1e-6)[1] >= x.diis(convergence=1e-6)[1]
