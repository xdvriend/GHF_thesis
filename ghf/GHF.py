from ghf.SCF_functions import *
import numpy as np
from scipy import linalg as la


def GHF(molecule,  number_of_elecrons):
    overlap = get_integrals(molecule)[0]
    one_electron = get_integrals(molecule)[1]
    two_electron = get_integrals(molecule)[2]
    nuclear_repulsion = get_integrals(molecule)[3]

    if number_of_elecrons % 2 == 0:
        a = int(number_of_elecrons / 2)
        b = int(number_of_elecrons / 2)
    else:
        x = int(number_of_elecrons - 1)
        a = int((x / 2) + 1)
        b = int(x / 2)

    def orth_matrix(trans_matrix):
        v = la.inv(trans_matrix)
        shape = np.shape(v)
        zero = np.zeros(shape)
        top = np.hstack((v, zero))
        bottom = np.hstack((zero, v))
        return np.vstack((top, bottom))

    s_12 = la.inv(trans_matrix(overlap))
    v_inv_t = la.inv(s_12).T
    fock_o = orth_matrix(v_inv_t)
    print(np.shape(fock_o))

    def density_block(fock_t, sigma, tau, o_matrix):
        eigenval, eigenvec = la.eigh(fock_t, o_matrix)
        coeff = o_matrix @ eigenvec
        coeff_s = coeff[:, 0:sigma]
        coeff_t = coeff[:, 0:tau].conj()
        return np.einsum('ij, kj -> ik', coeff_s, coeff_t)

    one_electron = np.vstack((np.hstack((one_electron, one_electron)), np.hstack((one_electron, one_electron))))
    print(np.shape(one_electron))

    p_block_aa_g = density_block(one_electron, a, a, fock_o)
    p_block_ab_g = density_block(one_electron, a, b, fock_o)
    p_block_ba_g = density_block(one_electron, b, a, fock_o)
    p_block_bb_g = density_block(one_electron, b, b, fock_o)
    print(np.shape(p_block_aa_g))
    p_block_1 = [p_block_aa_g]
    p_block_2 = [p_block_ab_g]
    p_block_3 = [p_block_ba_g]
    p_block_4 = [p_block_bb_g]


    def spin_blocked(block_1, block_2, block_3, block_4):
        top = np.hstack((block_1, block_2))
        bottom = np.hstack((block_3, block_4))
        return np.vstack((top, bottom))

    p_g = spin_blocked(p_block_aa_g, p_block_ab_g, p_block_ba_g, p_block_bb_g)

    def coulomb(density_block, two_electron):
        return np.einsum('kl, ijkl -> ij', density_block, two_electron)

    def exchange(density_block, two_electron):
        return np.einsum('kl, ijkl -> ij', density_block, two_electron.transpose(0, 2, 1, 3))

    def fock_block(sigma, tau, p1, p4, p_st):
        h_st = one_electron
        j_aa = coulomb(p1, two_electron)
        j_bb = coulomb(p4, two_electron)
        k_st = exchange(p_st, two_electron)
        if sigma == tau:
            d = 1
        else:
            d = 0
        return d * h_st + d * (j_aa + j_bb) - k_st

    def scf_e(dens, fock, one_electron):
        print(np.shape(dens))
        print(np.shape(fock))
        print(np.shape(one_electron))
        return np.sum(dens * (one_electron + fock)) / 4

    one_e_total = np.vstack((np.hstack((one_electron, one_electron)), np.hstack((one_electron, one_electron))))
    electronic_e = scf_e(p_g, one_e_total, one_e_total)
    energies = [electronic_e]
    delta_e = []

    def iter():
        f_aa = fock_block(a, a, p_block_1[-1], p_block_4[-1], p_block_1[-1])
        f_ab = fock_block(a, b, p_block_1[-1], p_block_4[-1], p_block_2[-1])
        f_ba = fock_block(b, a, p_block_1[-1], p_block_4[-1], p_block_3[-1])
        f_bb = fock_block(b, b, p_block_1[-1], p_block_4[-1], p_block_4[-1])

        f = spin_blocked(f_aa, f_ab, f_ba, f_bb)

        new_p1 = density_block(f, a, a, fock_o)
        new_p2 = density_block(f, a, b, fock_o)
        new_p3 = density_block(f, b, a, fock_o)
        new_p4 = density_block(f, b, b, fock_o)

        p_block_1.append(new_p1)
        p_block_2.append(new_p2)
        p_block_3.append(new_p3)
        p_block_4.append(new_p4)

        p = spin_blocked(p_block_1[-1], p_block_2[-1], p_block_3[-1], p_block_4[-1])
        energies.append(scf_e(p, f, one_e_total))
        delta_e.append(energies[-1] - energies[-2])

    iter()
    i = 1
    while abs(delta_e[-1]) >= 1e-12:
        iter()
        i += 1

    scf_e = energies[-1] + nuclear_repulsion
    print("Number of iterations: " + str(i))
    print("Converged SCF energy in Hartree: " + str(scf_e) + " (RHF)")
    return scf_e