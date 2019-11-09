"""
Testing the RHF and UHF methods
===============================

Simple tests to check whether or not the functions return the correct value.
"""
from ghf.RHF import RHF
from ghf.UHF import UHF
from pyscf import *
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')

h4 = gto.M(atom = 'h 0 0.707107 0; h 0.707107 0 0; h 0 -0.707107 0; h -0.707107 0 0' ,spin = 2, basis = 'cc-pvdz')


def test_RHF():
    """
    test_RHF will test whether or not the RHF method returns the wanted result. The accuracy is 10^16.

    """
    x = RHF(h4, 4)
    assert x.get_scf_solution() == -1.9403598392831243


def test_UHF():
    """
    test_UHF will test the regular UHF method, by checking whether or not it returns the expected result. The accuracy is 10^-6.
    """
    x = UHF(h3, 3)
    assert -1.506275 <= x.get_scf_solution() <= -1.506274


def test_extra_e():
    """
    test_extra_e will test the UHF method, with the added option of first adding 2 electrons to the system and using those coefficients
    for the actual system, by checking whether or not it returns the expected result. The accuracy is 10^-6.
    """
    x = UHF(h4, 4)
    guess = x.extra_electron_guess()
    assert -2.021089 <= x.get_scf_solution(guess) <= -2.021088


def test_stability():
    """
    test_stability will test the UHF method, with stability analysis, by checking whether or not it returns the expected result. The accuracy is 10^-6.
    """
    x = UHF(h4, 4)
    guess = x.stability()
    assert -2.021089 <= x.get_scf_solution(guess) <= -2.021088


