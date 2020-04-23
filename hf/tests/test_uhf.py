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


def test_extra_e():
    """
    test_extra_e will test the UHF method, with the added option of first adding 2 electrons to the system and using
    those coefficients for the actual system, by checking whether or not it returns the correct result.
    """
    x = UHF.MF(h4, 4)
    guess = x.extra_electron_guess()
    assert np.isclose(-2.021088247649845, x.get_scf_solution(guess))


def test_stability():
    """
    test_stability will test the UHF method, with stability analysis, by checking whether or not it returns
    the correct result
    """
    x = UHF.MF(h4, 4)
    guess = x.stability()
    assert np.allclose(-2.021088247649845, x.get_scf_solution(guess))
