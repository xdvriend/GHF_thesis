"""
Functions to calculate spin expectation values
==============================================

This file contains functions that calculate the expectation values of the different spin operators.
"""
import numpy as np
from scipy import linalg as la


def uhf(occ_a, occ_b, coeff_a, coeff_b, overlap):
    """
    A function used to calculate the spin expectation values in the unrestricted hartree fock formalism.

    :param occ_a: number of occupied alpha orbitals
    :param occ_b: number of occupied beta orbitals
    :param coeff_a: MO coefficients of alpha orbitals
    :param coeff_b: MO coefficients of beta orbitals
    :param overlap: overlap matrix of the molecule
    :return: S^2, S_z and spin multiplicity
    """
    occ_indx_a = np.arange(occ_a)  # indices of the occupied alpha orbitals
    occ_indx_b = np.arange(occ_b)  # indices of the occupied beta orbitals
    occ_a_orb = coeff_a[:, occ_indx_a]  # orbital coefficients associated with occupied alpha orbitals
    occ_b_orb = coeff_b[:, occ_indx_b]  # orbital coefficients associated with occupied beta orbitals
    s = occ_a_orb.conj().T @ overlap @ occ_b_orb  # Basically (alpha orbitals).T * S * (beta orbitals)
    ss_xy = (occ_a + occ_b) * 0.5 - np.einsum('ij,ij->', s.conj(), s)  # = S^2_x + S^2_y
    ss_z = (occ_b - occ_a)**2 * 0.25  # = S^2_z
    ss = (ss_xy + ss_z).real  # = S^2_total
    s_z = (occ_a - occ_b) / 2  # = S_z
    multiplicity = 2 * (np.sqrt(ss + 0.25) - 0.5) + 1  # = 2S+1
    return ss, s_z, multiplicity


def ghf(coeff, n_e, trans):
    """
    A function used to calculate the spin expectation values in the generalised hartree fock formalism.

    :param coeff: The generalised MO coefficients
    :param n_e: number of electrons
    :param trans: transformation matrix, eg.: S^(-1/2)
    :return: The expectation values of S_z, S^2 and the multiplicity (2S+1)
    """
    number_of_orbitals = coeff.shape[0] // 2
    mo_a = coeff[:number_of_orbitals]
    mo_b = coeff[number_of_orbitals:]
    mo_a = la.inv(trans) @ mo_a
    mo_b = la.inv(trans) @ mo_b
    mo_a_occ = mo_a[:, :n_e]
    mo_b_occ = mo_b[:, :n_e]

    ovlp_a = mo_a_occ.conj().T @ mo_a_occ
    ovlp_b = mo_b_occ.conj().T @ mo_b_occ
    ovlp_ab = mo_a_occ.conj().T @ mo_b_occ
    ovlp_ba = mo_b_occ.conj().T @ mo_a_occ

    number_occ_a = ovlp_a.trace()
    number_occ_b = ovlp_b.trace()
    s_z = (0.5 * (number_occ_a - number_occ_b)).real

    temp = ovlp_a - ovlp_b
    ss_z = (s_z ** 2) + 0.25 * ((number_occ_a + number_occ_b) - np.einsum('ij, ij', temp, temp.conj()))
    ss_mp = number_occ_b + ((ovlp_ba.trace() * ovlp_ab.trace()) - np.einsum('ij, ji', ovlp_ba, ovlp_ab))
    s_2 = (ss_mp + s_z + ss_z).real

    s = (np.sqrt(s_2 + .25) - .5).real
    return s_z, s_2, 2*s + 1
