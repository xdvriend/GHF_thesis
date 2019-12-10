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
import psi4


def get_integrals_pyscf(molecule):
    """
    A function to calculate your integrals & nuclear repulsion with pyscf.
    """
    overlap = molecule.intor('int1e_ovlp')
    one_electron = molecule.intor('int1e_nuc') + molecule.intor('int1e_kin')
    two_electron = molecule.intor('int2e')
    nuclear_repulsion = gto.mole.energy_nuc(molecule)
    return overlap, one_electron, two_electron, nuclear_repulsion


def get_integrals_psi4(mol):
    """
    A function to calculate your integrals & nuclear repulsion with psi4.
    :param mol: Psi4 instance
    :return: overlap, core hamiltonian, eri tensor and nuclear repulsion
    """
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
    mints = psi4.core.MintsHelper(wfn.basisset())
    overlap = np.asarray(mints.ao_overlap())
    one_electron = np.asarray(mints.ao_potential()) + np.asarray(mints.ao_kinetic())
    two_electron = np.asarray(mints.ao_eri())
    nuclear_repulsion = mol.nuclear_repulsion_energy()
    return overlap, one_electron, two_electron, nuclear_repulsion


def trans_matrix(overlap):
    """
    - Define a transformation matrix X, used to orthogonalize different matrices throughout the calculation.
    - Input should be an overlap matrix.
    """
    eigenvalues, eigenvectors = la.eigh(overlap)  # calculate eigenvalues & eigenvectors of the overlap matrix
    diag_matrix = diag(eigenvalues)  # create a diagonal matrix from the eigenvalues of the overlap
    x = eigenvectors.dot(np.sqrt(la.inv(diag_matrix))).dot(eigenvectors.conj().T)
    return x


def density_matrix(f_matrix, occ, trans):
    """
    - density() creates a density matrix from a fock matrix and the number of occupied orbitals.
    - Input is a fock matrix, the number of occupied orbitals, which can be separate for alpha and beta in case of UHF.
      And a transformation matrix X.

    """
    f_eigenvalues, f_eigenvectors = la.eigh(f_matrix)  # eigenvalues are initial orbital energies
    coefficients = trans.dot(f_eigenvectors)
    coefficients_r = coefficients[:, 0:occ]  # summation over occupied orbitals
    # np.einsum represents Sum_j^occupied_orbitals(c_ij * c_kj)
    density = np.einsum('ij,kj->ik', coefficients_r, coefficients_r, optimize=True)
    return density


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
    - calculate the scf energy value from a given density matrix and a given fock matrix for both alpha and beta,
      so 4 matrices in total.
    - then calculate the initial electronic energy and put it into an array
    - input is the density matrices for alpha and beta, the fock matrices for alpha and beta and lastly a one electron
      matrix.
    """
    scf_e = np.einsum('ij, ij->', density_matrix_a + density_matrix_b,
                      one_electron)  # np.einsum here is used to add all the matrix values together
    scf_e += np.einsum('ij, ij->', density_matrix_a, fock_a)
    scf_e += np.einsum('ij, ij->', density_matrix_b, fock_b)
    scf_e *= 0.5  # divide by two, since the summation technically adds the alpha and beta values twice
    return scf_e


def spin(occ_a, occ_b, coeff_a, coeff_b, overlap):
    """

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
    s = reduce(np.dot, (occ_a_orb.T, overlap, occ_b_orb))  # Basically (alpha orbitals).T * S * (beta orbitals)
    ss_xy = (occ_a + occ_b) * 0.5 - np.einsum('ij,ij->', s.conj(), s)  # = S^2_x + S^2_y
    ss_z = (occ_b - occ_a)**2 * 0.25  # = S^2_z
    ss = (ss_xy + ss_z).real  # = S^2_total
    s_z = (occ_a - occ_b) / 2  # = S_z
    multiplicity = 2 * (np.sqrt(ss + 0.25) - 0.5) + 1  # = 2S+1
    return ss, s_z, multiplicity


def expand_matrix(matrix):
    """
    :param matrix:
    :return: a matrix double the size, where blocks of zero's are added top right and bottom left.
    """
    # get the shape of the matrix you want to expand
    # create a zero-matrix with the same shape
    shape = np.shape(matrix)
    zero = np.zeros(shape)
    # create the top part of the expanded matrix by putting the matrix and zero-matrix together
    # create the bottom part of the expanded matrix by putting the matrix and zero-matrix together
    top = np.hstack((matrix, zero))
    bottom = np.hstack((zero, matrix))
    # Add top and bottom part together.
    return np.vstack((top, bottom))


def spin_blocked(block_1, block_2, block_3, block_4):
    """
    When creating the blocks of the density or fock matrix separately, this function is used to add them together,
    and create the total density or Fock matrix in spin Blocked notation.
    :return: a density matrix in the spin-blocked notation
    """
    top = np.hstack((block_1, block_2))
    bottom = np.hstack((block_3, block_4))
    return np.vstack((top, bottom))


def ghf_spin(mo_coeff, overlap):
    number_of_orbitals = mo_coeff.shape[0] // 2
    mo_a = mo_coeff[:number_of_orbitals]
    mo_b = mo_coeff[number_of_orbitals:]
    saa = mo_a.conj().T @ overlap @ mo_a
    sbb = mo_b.conj().T @ overlap @ mo_b
    sab = mo_a.conj().T @ overlap @ mo_b
    sba = mo_b.conj().T @ overlap @ mo_a
    number_occ_a = saa.trace()
    print(number_occ_a)
    number_occ_b = sbb.trace()
    print(number_occ_b)
    ss_xy = (number_occ_a + number_occ_b) * .5
    ss_xy += sba.trace() * sab.trace() - np.einsum('ij,ji->', sba, sab)
    print(ss_xy)
    ss_z = (number_occ_a + number_occ_b) * .25
    ss_z += (number_occ_a - number_occ_b) ** 2 * .25
    tmp = saa - sbb
    ss_z -= np.einsum('ij,ji', tmp, tmp) * .25
    print(ss_z)
    s_z = 1
    ss = (ss_xy + ss_z).real
    s = np.sqrt(ss + .25) - .5
    return ss, s_z, s * 2 + 1
