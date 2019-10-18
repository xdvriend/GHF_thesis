"""
Generalised Hartree Fock, by means of SCF procedure
====================================================


This function calculates the GHF energy for a given molecule and the number of electrons in the system.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)

The function can do this in two ways.

- Create the general density matrix and work with this as a whole
- create the density matrix in spin-blocked notation
"""

from ghf.SCF_functions import *
import numpy as np
from numpy import linalg as la


def GHF(molecule, number_of_electrons):
    """
       calculate GHF energy.
       ---------------------
       Input is a molecule and the number of electrons.

       Molecules are made in pySCF, eg.:

       >>> h_2 = gto.M(atom = 'h 0 0 0; h 0 0 1', spin = 0, basis = 'cc-pvdz')
       >>> GHF(h_2, 2)

       prints and returns GHF energy of h_2

       """
    # get_integrals (from GHF.SCF_functions) calculates the overlap matrix, one_electron integrals,
    # two_electron_integrals
    # and the nuclear repulsion value.
    overlap = get_integrals(molecule)[0]
    one_electron = get_integrals(molecule)[1]
    two_electron = get_integrals(molecule)[2]
    nuclear_repulsion = get_integrals(molecule)[3]

    def expand_matrix(matrix):
        """
        :param matrix:
        :return: a matrix double the size, where blocks of zero's are added top right and bottom left.
        """
        shape = np.shape(matrix)  # get the shape of the matrix you want to expand
        zero = np.zeros(shape)  # create a zero-matrix with the same shape
        top = np.hstack((matrix, zero))  # create the top part of the expanded matrix by putting the matrix and zero-matrix together
        bottom = np.hstack((zero, matrix))# create the bottom part of the expanded matrix by putting the matrix and zero-matrix together
        return np.vstack((top, bottom))  # Add top and bottom part together.

    s_min_12 = trans_matrix(overlap)
    s_12_o = expand_matrix(s_min_12)
    c_ham = expand_matrix(one_electron)

    def coeff_matrix(orth_matrix, core_ham):
        """
        :param orth_matrix: an orthogonalisation matrix, created with the expand matrix function
        :param core_ham: the expanded core Hamiltonian matrix
        :return: The orthogonalised version of the core Hamiltonian matrix
        """
        return orth_matrix @ core_ham @ orth_matrix.T

    c_init = coeff_matrix(s_12_o, c_ham)

    def density_block(fock_t, sigma, tau):
        """
        :param fock_t: a fock matrix
        :param sigma: can be either 'a' for alpha or 'b' for beta
        :param tau: can be either 'a' for alpha or 'b' for beta
        :return: one of the four blocks in the density matrix, depending on the given sigma and tau
        """
        dim = int(np.shape(fock_t)[0] / 2)  # determine the dimension of 1 block.
        eigenval, eigenvec = la.eigh(fock_t)  # calculate the eigenvectors
        coeff = s_12_o @ eigenvec  # orthogonalise the eigenvectors
        coeff_a = coeff[:dim, :]  # determine the C^alpha coefficients
        coeff_b = coeff[dim:2*dim, :]  # determoine the C^beta coefficients
        if sigma == 'a' and tau == 'a':  # alpha-alpha block
            coeff_s = coeff_a[:, 0:number_of_electrons]
            coeff_t = coeff_a[:, 0:number_of_electrons].conj()
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'a' and tau == 'b':  # alpha-beta block
            coeff_s = coeff_a[:, 0:number_of_electrons]
            coeff_t = coeff_b[:, 0:number_of_electrons].conj()
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'b' and tau == 'a':  # beta-alpha block
            coeff_s = coeff_b[:, 0:number_of_electrons]
            coeff_t = coeff_a[:, 0:number_of_electrons].conj()
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'b' and tau == 'b':  # beta-beta block
            coeff_s = coeff_b[:, 0:number_of_electrons]
            coeff_t = coeff_b[:, 0:number_of_electrons].conj()
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)

    def density(fock):
        """
        :param fock: a fock matrix
        :return: one big density matrix
        """
        eigenval, eigenvec = la.eigh(fock)
        coeff = s_12_o @ eigenvec
        coeff_r = coeff[:, 0:number_of_electrons]
        return np.einsum('ij,kj->ik', coeff_r, coeff_r)

    p_block_aa_g = density_block(c_init, 'a', 'a')  # aa-block of guess density
    p_block_ab_g = density_block(c_init, 'a', 'b')  # ab-block of guess density
    p_block_ba_g = density_block(c_init, 'b', 'a')  # ba-block of guess density
    p_block_bb_g = density_block(c_init, 'b', 'b')  # bb-block of guess density

    def spin_blocked(block_1, block_2, block_3, block_4):
        """
        When creating the blocks of the density separately, this function is used to add them together
        :return: a density matrix in the spin-blocked notation
        """
        top = np.hstack((block_1, block_2))
        bottom = np.hstack((block_3, block_4))
        return np.vstack((top, bottom))

    #p_g = spin_blocked(p_block_aa_g, p_block_ab_g, p_block_ba_g, p_block_bb_g)
    p_g = density(c_ham)  # total guess density
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
        f_o = s_12_o @ f @ s_12_o.T

        # Create the new density matrix
        new_p1 = density_block(f_o, 'a', 'a')
        new_p2 = density_block(f_o, 'a', 'b')
        new_p3 = density_block(f_o, 'b', 'a')
        new_p4 = density_block(f_o, 'b', 'b')

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
    print("Converged SCF energy in Hartree: " + str(scf_e) + " (GHF)")
    return scf_e
