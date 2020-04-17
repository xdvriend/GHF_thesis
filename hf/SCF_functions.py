"""
Useful functions for SCF procedure
===================================

A number of functions used throughout the UHF and RHF calculations are summarised here.
"""
from pyscf import *
import numpy as np
from numpy import linalg as la
from scipy import diag
import psi4
import math as m


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
    density = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj(), optimize=True)
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
    s = occ_a_orb.conj().T @ overlap @ occ_b_orb  # Basically (alpha orbitals).T * S * (beta orbitals)
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
    if isinstance(matrix.any(), complex):
        zero = np.zeros(shape).astype(complex)
    else:
        zero = np.zeros(shape)
    # create the top part of the expanded matrix by putting the matrix and zero-matrix together
    # create the bottom part of the expanded matrix by putting the matrix and zero-matrix together
    top = np.hstack((matrix, zero))
    bottom = np.hstack((zero, matrix))
    # Add top and bottom part together.
    return np.vstack((top, bottom))


def expand_tensor(tensor, complexity=False):
    """
    Expand every matrix within the tensor, in the same way the expand matrix function works.
    :param tensor: The tensor, usually eri, that you wish to expand.
    :param complexity: Is your tensor complex or not? Default is false.
    :return: a tensor where each dimension is doubled.
    """
    dim = np.shape(tensor)[0]
    if complexity:
        tens_block = np.zeros((dim, dim, 2*dim, 2*dim)).astype(complex)
        zero = np.zeros((dim, dim, 2*dim, 2*dim)).astype(complex)
    else:
        tens_block = np.zeros((dim, dim, 2 * dim, 2 * dim))
        zero = np.zeros((dim, dim, 2 * dim, 2 * dim))
    for l in range(dim):
        for k in range(dim):
            tens_block[l][k] = expand_matrix(tensor[l][k])
    top = np.hstack((tens_block, zero))
    bottom = np.hstack((zero, tens_block))
    return np.vstack((top, bottom))


def eri_ao_to_mo(eri, coeff, complexity=False):
    """
    Transform the two electron tensor to MO basis. Scales as N^5.
    :param eri: Electron repulsion interaction, in ao notation, tensor in spinor basis
    :param coeff: coefficient matrix in spinor basis
    :param complexity: specify whether you are working with real or complex tensors. Default is real.
    :return: Electron repulsion interaction, in mo notation, tensor in spinor basis
    """
    dim = len(eri)
    if complexity:
        eri_mo = np.zeros((dim, dim, dim, dim)).astype(complex)
        mo_1 = np.zeros((dim, dim, dim, dim)).astype(complex)
        mo_2 = np.zeros((dim, dim, dim, dim)).astype(complex)
        mo_3 = np.zeros((dim, dim, dim, dim)).astype(complex)
    else:
        eri_mo = np.zeros((dim, dim, dim, dim))
        mo_1 = np.zeros((dim, dim, dim, dim))
        mo_2 = np.zeros((dim, dim, dim, dim))
        mo_3 = np.zeros((dim, dim, dim, dim))

    for s in range(0, len(coeff)):
        for sig in range(0, len(coeff)):
            mo_1[:, :, :, s] += coeff[sig, s] * eri[:, :, :, sig]

        for r in range(0, len(coeff)):
            for lam in range(0, len(coeff)):
                mo_2[:, :, r, s] += coeff[lam, r] * mo_1[:, :, lam, s]

            for q in range(0, len(coeff)):
                for nu in range(0, len(coeff)):
                    mo_3[:, q, r, s] += coeff[nu, q] * mo_2[:, nu, r, s]

                for p in range(0, len(coeff)):
                    for mu in range(0, len(coeff)):
                        eri_mo[p, q, r, s] += coeff[mu, p] * mo_3[mu, q, r, s]
    return eri_mo


def spin_blocked(block_1, block_2, block_3, block_4):
    """
    When creating the blocks of the density or fock matrix separately, this function is used to add them together,
    and create the total density or Fock matrix in spin Blocked notation.
    :return: a density matrix in the spin-blocked notation
    """
    top = np.hstack((block_1, block_2))
    bottom = np.hstack((block_3, block_4))
    return np.vstack((top, bottom))


def ghf_spin(coeff, n_e, trans):
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
    s_z = 0.5 * (number_occ_a - number_occ_b)

    temp = ovlp_a - ovlp_b
    ss_z = (s_z ** 2) + 0.25 * ((number_occ_a + number_occ_b) - np.einsum('ij, ij', temp, temp))
    ss_mp = number_occ_b + ((ovlp_ba.trace() * ovlp_ab.trace()) - np.einsum('ij, ji', ovlp_ba, ovlp_ab))
    s_2 = ss_mp + s_z + ss_z

    s = np.sqrt(s_2 + .25) - .5
    return s_z, s_2, 2*s + 1
