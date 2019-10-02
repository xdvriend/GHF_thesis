"""
Useful functions for SCF procedure
===================================

A number of functions used throughout the UHF and RHF calculations are summarised here.
"""
from pyscf import *
import numpy as np
from numpy import linalg as la
from scipy import diag
from functools import reduce


def get_integrals(molecule):
    """
    A function to calculate your integrals & nuclear repulsion with pyscf.
    """
    overlap = molecule.intor('int1e_ovlp')
    one_electron = molecule.intor('int1e_nuc') + molecule.intor('int1e_kin')
    two_electron = molecule.intor('int2e')
    nuclear_repulsion = gto.mole.energy_nuc(molecule)
    return overlap, one_electron, two_electron, nuclear_repulsion


def trans_matrix(overlap):
    """
    - Define a transformation matrix X, used to orthogonalize different matrices throughout the calculation.
    - Input should be an overlap matrix.
    """
    eigenvalues, eigenvectors = la.eigh(overlap)  # calculate eigenvalues & eigenvectors of the overlap matrix
    diag_matrix = diag(eigenvalues)  # create a diagonal matrix from the eigenvalues of the overlap
    X = eigenvectors.dot(np.sqrt(la.inv(diag_matrix))).dot(eigenvectors.T)
    return X


def density_matrix(f_matrix, occ, trans):
    """
    - density() creates a density matrix from a fock matrix and the number of occupied orbitals.
    - Input is a fock matrix, the number of occupied orbitals, which can be separat for alpha and beta in case of UHF. And
    a transformation matrix X.
    """
    f_eigenvalues, f_eigenvectors = la.eigh(f_matrix)  # eigenvalues are initial orbital energies
    coefficients = trans.dot(f_eigenvectors)
    coefficients_r = coefficients[:, 0:occ]  # summation over occupied orbitals
    return np.einsum('ij,kj->ik', coefficients_r, coefficients_r)  # np.einsum represents Sum_j^occupied_orbitals(c_ij * c_kj)


def uhf_fock_matrix(density_matrix_1, density_matrix_2, one_electron, two_electron):
    """
    - calculate a fock matrix from a given alpha and beta density matrix

    - fock alpha if 1 = alpha and 2 = beta and vice versa
    - input is the density matrix for alpha and beta, a one electron matrix and a two electron tensor.
    """
    jk_integrals = two_electron - two_electron.transpose(0, 2, 1, 3)  # exchange and coulomb integrals
    jk_a = np.einsum('kl,ijkl->ij', density_matrix_1,
                     jk_integrals)  # double summation of density matrix * (coulomb - exchange)
    j_b = np.einsum('kl, ijkl->ij', density_matrix_2, two_electron)  # double summation of density matrix * exchange
    return one_electron + jk_a + j_b


def uhf_scf_energy(density_matrix_a, density_matrix_b, fock_a, fock_b, one_electron):
    """
    - calculate the scf energy value from a given density matrix and a given fock matrix for both alpha and beta, so 4 matrices in total
    - then calculate the initial electronic energy and put it into an array
    - input is the density matrices for alpha and beta, the fock matrices for alpha and beta and lastly a one electron matrix
    """
    scf_e = np.einsum('ij, ij->', density_matrix_a + density_matrix_b,
                      one_electron)  # np.einsum here is used to add all the matrix values together
    scf_e += np.einsum('ij, ij->', density_matrix_a, fock_a)
    scf_e += np.einsum('ij, ij->', density_matrix_b, fock_b)
    scf_e *= 0.5  # divide by two, since the summation technically adds the alpha and beta values twice
    return scf_e

def spin(occ_a, occ_b, coeff_a, coeff_b, overlap):
    occ_indx_a = np.arange(occ_a)  # indices of the occupied alpha orbitals
    occ_indx_b = np.arange(occ_b)  # indices of the occupied beta orbitals
    occ_a_orb = coeff_a[:, occ_indx_a]  # orbital coefficients associated with occupied alpha orbitals
    occ_b_orb = coeff_b[:, occ_indx_b]  # orbital coefficients associated with occupied beta orbitals
    s = reduce(np.dot, (occ_a_orb.T, overlap, occ_b_orb))
    #print(occ_a_orb, occ_b_orb)
    ss_xy = (occ_a + occ_b) * 0.5 - np.einsum('ij,ij->', s.conj(), s)
    ss_z = (occ_b - occ_a)**2 * 0.25
    ss = (ss_xy + ss_z).real
    s_z = (occ_a - occ_b) / 2
    multiplicity = 2 * (np.sqrt(ss + 0.25) - 0.5) + 1
    print("<S^2> = " + str(ss) + ", <S_z> = " + str(s_z) + ", Multiplicity = " + str(multiplicity))
    return ss, s_z, multiplicity


