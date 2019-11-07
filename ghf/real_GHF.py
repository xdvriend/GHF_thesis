"""
Real generalised Hartree Fock, by means of SCF procedure
=========================================================


This function calculates the real GHF energy for a given molecule and the number of electrons in the system.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)

The function can do this in two ways.

- Create the general density matrix and work with this as a whole
- create the density matrix in spin-blocked notation
"""

from ghf.SCF_functions import *
import numpy as np
from numpy import linalg as la
import scipy
from scipy import linalg as LA
from functools import reduce
import matplotlib.pyplot as plt



def real_GHF(molecule, number_of_electrons):
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

    def unitary_rotation(coefficient_matrix):
        shape = np.shape(coefficient_matrix)
        one = np.full(shape, 1)
        neg = np.full(shape, -1)
        ones = np.tril(one)
        negative = np.triu(neg)
        exp = -1 * (ones + negative)
        U = LA.expm(exp)
        return U @ coefficient_matrix @ la.inv(U)

    def random_unitary_matrix(dimension):
        x = np.random.rand(dimension, dimension)
        x_t = x.T
        y = x + x_t
        val, vec = la.eigh(y)
        return vec

    dim = int(np.shape(c_ham)[0])
    c_init_rot = unitary_rotation(c_init)


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
        coeff_b = coeff[dim:2*dim, :]  # determine the C^beta coefficients
        if sigma == 'a' and tau == 'a':  # alpha-alpha block
            coeff_s = coeff_a[:, 0:number_of_electrons]
            coeff_t = coeff_a[:, 0:number_of_electrons]
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'a' and tau == 'b':  # alpha-beta block
            coeff_s = coeff_a[:, 0:number_of_electrons]
            coeff_t = coeff_b[:, 0:number_of_electrons]
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'b' and tau == 'a':  # beta-alpha block
            coeff_s = coeff_b[:, 0:number_of_electrons]
            coeff_t = coeff_a[:, 0:number_of_electrons]
            return np.einsum('ij, kj -> ik', coeff_s, coeff_t)
        elif sigma == 'b' and tau == 'b':  # beta-beta block
            coeff_s = coeff_b[:, 0:number_of_electrons]
            coeff_t = coeff_b[:, 0:number_of_electrons]
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

    p_block_aa_g = density_block(c_init_rot, 'a', 'a')  # aa-block of guess density
    p_block_ab_g = density_block(c_init_rot, 'a', 'b')  # ab-block of guess density
    p_block_ba_g = density_block(c_init_rot, 'b', 'a')  # ba-block of guess density
    p_block_bb_g = density_block(c_init_rot, 'b', 'b')  # bb-block of guess density

    def spin_blocked(block_1, block_2, block_3, block_4):
        """
        When creating the blocks of the density separately, this function is used to add them together
        :return: a density matrix in the spin-blocked notation
        """
        top = np.hstack((block_1, block_2))
        bottom = np.hstack((block_3, block_4))
        return np.vstack((top, bottom))

    #p_g = spin_blocked(p_block_aa_g, p_block_ab_g, p_block_ba_g, p_block_bb_g)
    p_g = density(c_init_rot)  # total guess density
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
    fock_o = []

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
        eigenval, eigenvec = la.eigh(f_o)
        coeff = s_12_o @ eigenvec
        #print(coeff)
        fock_o.append(f)


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
    while abs(delta_e[-1]) >= 1e-12 and i<5000:
        iter()
        i += 1

    print(i)
    print(energies[-1] + nuclear_repulsion)
    #print(energies + nuclear_repulsion)
    #plt.plot(energies + nuclear_repulsion)
    #plt.show()

    f_aa = fock_block('a', 'a', densities[-1])
    f_ab = fock_block('a', 'b', densities[-1])
    f_ba = fock_block('b', 'a', densities[-1])
    f_bb = fock_block('b', 'b', densities[-1])
    f = spin_blocked(f_aa, f_ab, f_ba, f_bb)

    dim = int(np.shape(f)[0])

    f_o = s_12_o.T @ f @ s_12_o
    val, vec = la.eigh(f_o)
    coeff = s_12_o @ vec

    val2, vec2 = la.eigh(coeff)
    #print(val2)
    #print(np.trace(coeff))
    #print(la.det(coeff))
    #print(coeff)


    def generate_g():
        total_orbitals = c_ham.shape[0]
        n_occ = number_of_electrons
        n_vir = int(total_orbitals - number_of_electrons)
        occ_indx = np.arange(number_of_electrons)
        vir_indx = np.arange(total_orbitals)[number_of_electrons:]
        occ_orb = coeff[:, occ_indx]
        vir_orb = coeff[:, vir_indx]


        fock_init = f
        fock_ao = reduce(np.dot, (coeff.conj().T, fock_init, coeff))


        fock_occ = fock_ao[occ_indx[:, None], occ_indx]
        fock_vir = fock_ao[vir_indx[:, None], vir_indx]


        g = fock_ao[vir_indx[:, None], occ_indx]
        h_diag = fock_vir.diagonal().real[:, None] - fock_occ.diagonal().real

        def h_op(x):
            x = x.reshape(n_vir, n_occ)
            x2 = np.einsum('ps,sq->pq', fock_vir, x)
            x2 -= np.einsum('ps,rp->rs', fock_occ, x)
            d1 = reduce(np.dot, (vir_orb, x, occ_orb.conj().T))
            dm1 = d1 + d1.conj().T

            def vind(dm1):
                vj, vk = scf.hf.get_jk(molecule, dm1, hermi=1)
                return vj - vk
            v1 = vind(dm1)
            x2 += reduce(np.dot, (vir_orb.conj().T, v1, occ_orb))
            return x2.ravel()

        def h_op_2(x):
            x = x.reshape(n_vir, n_occ)
            x2 = np.einsum('ps,sq->pq', fock_vir, x)
            x2 -= np.einsum('ps,rp->rs', fock_occ, x)
            d1 = reduce(np.dot, (vir_orb, x, occ_orb.conj().T))
            dm1 = d1 + d1.conj().T

            def vind(dm1):
                vj, vk = scf.hf.get_jk(molecule, dm1, hermi=1)
                return vj - vk
            v1 = vind(dm1)
            x2 += reduce(np.dot, (vir_orb.conj().T, v1, occ_orb))
            bottom_right = np.zeros((np.shape(x2)[0], np.shape(x2)[0]))
            top_left = np.zeros((np.shape(x2)[1], np.shape(x2)[1]))
            top = np.hstack((top_left, x2.T))
            bottom = np.hstack((x2, bottom_right))
            hess = np.vstack((top, bottom))
            return hess

        return g.reshape(-1), h_op, h_diag.reshape(-1)

    def internal_stability():
        g, hop, hdiag = generate_g()
        hdiag *= 2

        def precond(dx, e, x0):
            hdiagd = hdiag - e
            hdiagd[abs(hdiagd) < 1e-8] = 1e-8
            return dx / hdiagd

        def hessian_x(x):
            return hop(x).real * 2

        def uniq_variable_indices(mo_occ):
            occ_indx_a = mo_occ > 0  # indices of occupied alpha orbitals
            occ_indx_b = mo_occ == 2  # indices of occupied beta orbitals
            vir_indx_a = ~occ_indx_a  # indices of virtual (unoccupied) alpha orbitals, done with bitwise operator: ~ (negation)
            vir_indx_b = ~occ_indx_b  # indices of virtual (unoccupied) beta orbitals, done with bitwise operator: ~ (negation)
            # & and | are bitwise operators for 'and' and 'or'
            # each bit position is the result of the logical 'and' or 'or' of the bits in the corresponding position of the operands
            unique = (vir_indx_a[:, None] & occ_indx_a) | (vir_indx_b[:,
                                                           None] & occ_indx_b)  # determine the unique variable indices, by use of bitwise operators
            return unique

        def unpack_uniq_variables(dx, mo_occ):
            nmo = len(mo_occ)
            idx = uniq_variable_indices(mo_occ)
            #print(np.shape(idx))
            #idx2 = np.ravel(idx[0:number_of_electrons, number_of_electrons:2 * nmo])
            #print(np.shape(idx2))
            x1 = np.zeros((nmo, nmo), dtype=dx.dtype)
            x1[idx] = dx
            #print(np.shape(dx))
            return x1 - x1.conj().T

        def rotate_mo(mo_coeff, mo_occ, dx):
            dr = unpack_uniq_variables(dx, mo_occ)
            u = scipy.linalg.expm(dr)  # computes the matrix exponential
            return np.dot(mo_coeff, u)

        x0 = np.zeros_like(g)
        x0[g != 0] = 1. / hdiag[g != 0]
        #print(hessian_x(x0))
        #print(np.shape(hessian_x(x0)))
        #hess = hessian_x(x0)
        e, v = lib.davidson(hessian_x, x0, precond, tol=1e-4)
        #e, v = la.eigh(hess)
        #print(e)
        if e < -1e-5:
            print('yes')
            mo_occ = np.zeros(dim,)
            for i in range(number_of_electrons):
                mo_occ[i] = 1
            mo = rotate_mo(coeff, mo_occ, v)
        else:
            print('no')
            mo = coeff
        return mo

    #mf = scf.GHF(molecule).run()
    #mf.stability()
    #mo1 = internal_stability()
    #dm1 = mf.make_rdm1(mo1, mf.mo_occ)

    new_guess = internal_stability()
    coeff_r = new_guess[:, 0:number_of_electrons]
    guess_dens = np.einsum('ij,kj->ik', coeff_r, coeff_r)
    imp_dens = [guess_dens]

    electronic_e = scf_e(imp_dens[-1], c_ham, c_ham)
    new_energies = [electronic_e]
    delta_e = []

    def new_iter():
        # create the four spin blocks of the Fock matrix
        f_aa = fock_block('a', 'a', imp_dens[-1])
        f_ab = fock_block('a', 'b', imp_dens[-1])
        f_ba = fock_block('b', 'a', imp_dens[-1])
        f_bb = fock_block('b', 'b', imp_dens[-1])

        # Add them together to form the total Fock matrix in spin block notation
        # orthogonalise the Fock matrix
        f = spin_blocked(f_aa, f_ab, f_ba, f_bb)
        f_o = s_12_o @ f @ s_12_o.T

        # p_new = spin_blocked(new_p1, new_p2, new_p3, new_p4)
        p_new = density(f_o)
        imp_dens.append(p_new)

        new_energies.append(scf_e(imp_dens[-1], f, c_ham))
        delta_e.append(new_energies[-1] - new_energies[-2])

    new_iter()
    i = 1
    while abs(delta_e[-1]) >= 1e-12 and i<5000:
        new_iter()
        i += 1


    scf_e = new_energies[-1] + nuclear_repulsion

    print("Number of iterations: " + str(i))
    print("Converged SCF energy in Hartree: " + str(scf_e) + " (real GHF)")

    #plt.plot(np.real(new_energies + nuclear_repulsion))
    #plt.show()
    return scf_e, coeff
