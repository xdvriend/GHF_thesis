from ghf.SCF_functions import *
import numpy
import numpy as np
from numpy import linalg as la
import scipy
from functools import reduce


def complex_GHF(molecule, number_of_electrons):
    overlap = get_integrals(molecule)[0].astype(complex)
    one_electron = get_integrals(molecule)[1].astype(complex)
    two_electron = get_integrals(molecule)[2].astype(complex)
    nuclear_repulsion = get_integrals(molecule)[3].astype(complex)

    def expand_matrix(matrix):
        """
        :param matrix:
        :return: a matrix double the size, where blocks of zero's are added top right and bottom left.
        """
        shape = np.shape(matrix)  # get the shape of the matrix you want to expand
        zero = np.zeros(shape, dtype=complex)  # create a zero-matrix with the same shape
        top = np.hstack((matrix, zero))  # create the top part of the expanded matrix by putting the matrix and zero-matrix together
        bottom = np.hstack((zero, matrix))# create the bottom part of the expanded matrix by putting the matrix and zero-matrix together
        return np.vstack((top, bottom))  # Add top and bottom part together.

    s_min_12 = trans_matrix(overlap).astype(complex)
    s_12_o = expand_matrix(s_min_12).astype(complex)
    c_ham = expand_matrix(one_electron).astype(complex) + 0.001j

    def coeff_matrix(orth_matrix, core_ham):
        """
        :param orth_matrix: an orthogonalisation matrix, created with the expand matrix function
        :param core_ham: the expanded core Hamiltonian matrix
        :return: The orthogonalised version of the core Hamiltonian matrix
        """
        return orth_matrix @ core_ham @ orth_matrix.conj().T

    c_init = coeff_matrix(s_12_o, c_ham)

    def density(fock):
        """
        :param fock: a fock matrix
        :return: one big density matrix
        """
        eigenval, eigenvec = la.eigh(fock)
        coeff = s_12_o @ eigenvec
        coeff_r = coeff[:, 0:number_of_electrons]
        return np.einsum('ij,kj->ik', coeff_r, coeff_r.conj())

    def spin_blocked(block_1, block_2, block_3, block_4):
        """
        When creating the blocks of the density separately, this function is used to add them together
        :return: a density matrix in the spin-blocked notation
        """
        top = np.hstack((block_1, block_2))
        bottom = np.hstack((block_3, block_4))
        return np.vstack((top, bottom))

    p_g = density(c_init)  # total guess density
    densities = [p_g]

    def coulomb(density_block, two_electron):
        """
        Calculate the coulomb integrals.
        """
        return np.einsum('kl, ijkl -> ij', density_block, two_electron)

    def exchange(density_block, two_electron):
        """
        Calculate the exchange integrals.
        """
        return np.einsum('kl, ijkl -> ij', density_block, two_electron.transpose(0, 2, 1, 3))

    def fock_block(sigma, tau, p):
        """
        :param sigma: Can be either 'a' for alpha or 'b' for beta.
        :param tau: Can be either 'a' for alpha or 'b' for beta.
        :param p: a complete density matrix
        :return: one of the four blocks of the fock matrix, depending on the sigma and tau values.
        """
        if sigma == tau:
            d = 1
        else:
            d = 0
        dim = int(np.shape(p)[0] / 2)  # determine the dimension of 1 block.
        # split the density matrix in it's four spin-blocks
        # calculate the coulomb and exchange integrals
        # depending on which block of the Fock matrix you're making, use the needed J & K values.
        p_aa = p[0:dim, 0:dim]
        p_ab = p[dim:2*dim, 0:dim]
        p_ba = p[0:dim, dim:2*dim]
        p_bb = p[dim:2*dim, dim:2*dim]
        h_st = one_electron
        j_aa = coulomb(p_aa, two_electron)
        j_bb = coulomb(p_bb, two_electron)
        k_aa = exchange(p_aa, two_electron)
        k_ab = exchange(p_ab, two_electron)
        k_ba = exchange(p_ba, two_electron)
        k_bb = exchange(p_bb, two_electron)
        if sigma == 'a' and tau == 'a':  # aa-block
            return d * h_st + d * (j_aa + j_bb) - k_aa
        if sigma == 'a' and tau == 'b':  # ab-block
            return d * h_st + d * (j_aa + j_bb) - k_ab
        if sigma == 'b' and tau == 'a':  # ba-block
            return d * h_st + d * (j_aa + j_bb) - k_ba
        if sigma == 'b' and tau == 'b':  # bb-block
            return d * h_st + d * (j_aa + j_bb) - k_bb

    def scf_e(dens, fock, one_electron):
        """
        Calculates the scf energy for the GHF method
        """
        return np.sum(dens * (one_electron + fock)) / 2

    electronic_e = scf_e(p_g, c_ham, c_ham)
    energies = [electronic_e]
    delta_e = []

    def iter():
        """
        This creates an iteration process to converge to the minimun energy.
        """
        # create the four spin blocks of the Fock matrix
        f_aa = fock_block('a', 'a', densities[-1])
        f_ab = fock_block('a', 'b', densities[-1])
        f_ba = fock_block('b', 'a', densities[-1])
        f_bb = fock_block('b', 'b', densities[-1])

        # Add them together to form the total Fock matrix in spin block notation
        # orthogonalise the Fock matrix
        f = spin_blocked(f_aa, f_ab, f_ba, f_bb)
        f_o = s_12_o @ f @ s_12_o.conj().T

        #p_new = spin_blocked(new_p1, new_p2, new_p3, new_p4)
        p_new = density(f_o)
        densities.append(p_new)

        energies.append(scf_e(densities[-1], f, c_ham))
        delta_e.append(energies[-1] - energies[-2])

    iter()
    i = 1
    while abs(delta_e[-1]) >= 1e-12:
        iter()
        i += 1
    scf_e = energies[-1] + nuclear_repulsion

    print("Number of iterations: " + str(i))
    print("Converged SCF energy in Hartree: " + str(scf_e) + " (complex GHF)")
    return scf_e
