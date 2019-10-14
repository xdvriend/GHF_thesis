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

h14 = gto.M(atom = 'h 0 2.24698 0; h 0.9749280841223709 2.024459026799378 0; h 1.7567597044760135 1.4009691149805374 0; '
                   'h 2.1906435201143144 0.5000000881811595 0; h 2.1906435201143144 -0.5000000881811593 0; '
                   'h 1.7567597044760133 -1.4009691149805379 0; h 0.9749280841223711 -2.0244590267993776 0;'
                   'h 0 -2.24698 0; h -0.9749280841223705 -2.024459026799378 0; h -1.7567597044760135 -1.4009691149805377 0; '
                   'h -2.1906435201143144 -0.5000000881811578 0; h -2.190643520114314 0.5000000881811608 0; '
                   'h -1.756759704476014 1.4009691149805368 0; h -0.9749280841223712 2.0244590267993776 0',spin = 0, basis = 'cc-pvdz')



def test_RHF():
    """
    test_RHF will test whether or not the RHF method returns the wanted result. The accuracy is 10^16.

    """
    assert RHF(h4, 2) == -1.9403598392831243

def test_UHF():
    """
    test_UHF will test the regular UHF method, by checking whether or not it returns the expected result. The accuracy is 10^-6.
    """
    assert -1.506275 <= UHF(h3, 2, 1) <= -1.506274

def test_extra_e():
    """
    test_extra_e will test the UHF method, with the added option of first adding 2 electrons to the system and using those coefficients
    for the actual system, by checking whether or not it returns the expected result. The accuracy is 10^-6.
    """
    assert -2.021089 <= UHF(h4, 2, 2, extra_e_coeff = True) <= -2.021088

def test_stability():
    """
    test_stability will test the UHF method, with stability analysis, by checking whether or not it returns the expected result. The accuracy is 10^-6.
    """
    assert -7.531852 <= UHF(h14, 7, 7, internal_stability_analysis=True) <= -7.531851


