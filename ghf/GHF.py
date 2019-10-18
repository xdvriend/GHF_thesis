from ghf.SCF_functions import *
import numpy as np
from numpy import linalg as la


def GHF(molecule, number_of_electrons):
    overlap = get_integrals(molecule)[0]
    one_electron = get_integrals(molecule)[1]
    two_electron = get_integrals(molecule)[2]
    nuclear_repulsion = get_integrals(molecule)[3]

    def expand_matrix(matrix):
        shape = np.shape(matrix)
        zero = np.zeros(shape)
        top = np.hstack((matrix, zero))
        bottom = np.hstack((zero, matrix))
        return np.vstack((top, bottom))

    s_min_12 = trans_matrix(overlap)
    s_12_o = expand_matrix(s_min_12)
    c_ham = expand_matrix(one_electron)

    def coeff_matrix(orth_matrix, core_ham):
        return orth_matrix @ core_ham @ orth_matrix.T

    c_init = coeff_matrix(s_12_o, c_ham)

    def density_block(fock_t, sigma, tau):
        dim = int(np.shape(fock_t)[0] / 2)
        eigenval, eigenvec = la.eigh(fock_t)
        coeff = s_12_o @ eigenvec
        coeff_a = coeff[:dim, :]
        coeff_b = coeff[dim:2*dim, :]
        if sigma == 'a' and tau == 'a':
            coeff_s = coeff_a[:, 0:number_of_electrons]
            coeff_t = coeff_a[:, 0:number_of_electrons].conj()
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'a' and tau == 'b':
            coeff_s = coeff_a[:, 0:number_of_electrons]
            coeff_t = coeff_b[:, 0:number_of_electrons].conj()
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'b' and tau == 'a':
            coeff_s = coeff_b[:, 0:number_of_electrons]
            coeff_t = coeff_a[:, 0:number_of_electrons].conj()
            print(coeff_s, coeff_t)

            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'b' and tau == 'b':
            coeff_s = coeff_b[:, 0:number_of_electrons]
            coeff_t = coeff_b[:, 0:number_of_electrons].conj()
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)

    p_block_aa_g = density_block(c_init, 'a', 'a')
    p_block_ab_g = density_block(c_init, 'a', 'b')
    p_block_ba_g = density_block(c_init, 'b', 'a')
    p_block_bb_g = density_block(c_init, 'b', 'b')

    def spin_blocked(block_1, block_2, block_3, block_4):
        top = np.hstack((block_1, block_2))
        bottom = np.hstack((block_3, block_4))
        return np.vstack((top, bottom))

    p_g = spin_blocked(p_block_aa_g, p_block_ab_g, p_block_ba_g, p_block_bb_g)
    densities = [p_g]

    def coulomb(density_block, two_electron):
        return np.einsum('kl, ijkl -> ij', density_block, two_electron)

    def exchange(density_block, two_electron):
        return np.einsum('kl, ijkl -> ij', density_block, two_electron.transpose(0, 2, 1, 3))

    def fock_block(sigma, tau, p):
        if sigma == tau:
            d = 1
        else:
            d = 0
        dim = int(np.shape(p)[0] / 2)
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
        if sigma == 'a' and tau == 'a':
            return d * h_st + d * (j_aa + j_bb) - k_aa
        if sigma == 'a' and tau == 'b':
            return d * h_st + d * (j_aa + j_bb) - k_ab
        if sigma == 'b' and tau == 'a':
            return d * h_st + d * (j_aa + j_bb) - k_ba
        if sigma == 'b' and tau == 'b':
            return d * h_st + d * (j_aa + j_bb) - k_bb

    def scf_e(dens, fock, one_electron):
        return np.sum(dens * (one_electron + fock)) / 2

    electronic_e = scf_e(p_g, c_ham, c_ham)
    energies = [electronic_e]
    delta_e = []

    def iter():
        f_aa = fock_block('a', 'a', densities[-1])
        f_ab = fock_block('a', 'b', densities[-1])
        f_ba = fock_block('b', 'a', densities[-1])
        f_bb = fock_block('b', 'b', densities[-1])

        f = spin_blocked(f_aa, f_ab, f_ba, f_bb)
        f_o = s_12_o @ f @ s_12_o.T

        new_p1 = density_block(f_o, 'a', 'a')
        new_p2 = density_block(f_o, 'a', 'b')
        new_p3 = density_block(f_o, 'b', 'a')
        new_p4 = density_block(f_o, 'b', 'b')

        p_new = spin_blocked(new_p1, new_p2, new_p3, new_p4)
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
