"""
Complex generalised Hartree Fock, by means of SCF procedure
============================================================


This class creates a generalised Hartree-Fock object which can be used for scf calculations. Different initial guesses
are provided as well as the option to perform a stability analysis.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)

"""

from ghf.SCF_functions import *
import numpy as np
from numpy import linalg as la
import math as m
from scipy import linalg as la2
from functools import reduce
import collections as c
from ghf.UHF import UHF


class ComplexGHF:
    """
    Calculate the complex GHF energy.
    ----------------------------------
    Input is a molecule and the number of electrons.

    Molecules are made in pySCF and calculations are performed as follows, eg.:
    The following snippet prints and returns UHF energy of h3
    and the number of iterations needed to get this value.

    For a normal scf calculation your input looks like the following example:

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> x = ComplexGHF(h3, 3)
    >>> x.loop_calculations()
    """
    def __init__(self, molecule, number_of_electrons, int_method='pyscf'):
        """
        Initiate an instance of a complex GHF class.

        :param molecule: The molecule on which to perform the calculations, made in PySCF.
        :param number_of_electrons: The amount of electrons present in the system.
        :param int_method: method to calculate the integrals. pyscf and psi4 are supported.
        """
        # Set the molecule, the integrals, the mo coefficients, the number of electrons,
        # the energy, the last fock and the last density matrices as class parameters.
        self.molecule = molecule
        self.number_of_electrons = number_of_electrons
        if int_method == 'pyscf':
            self.integrals = get_integrals_pyscf(molecule)
        elif int_method == 'psi4':
            self.integrals = get_integrals_psi4(molecule)
        else:
            raise Exception('Unsupported method to calculate integrals. Supported methods are pyscf or psi4. '
                            'Make sure the molecule instance matches the method.')
        self.energy = None
        self.mo = None
        self.last_dens = None
        self.last_fock = None
        self.instability = None
        self.iterations = None

    # Get the overlap integrals of the given molecule
    def get_ovlp(self):
        """

        :return: The overlap matrix
        """
        return self.integrals[0]

    # Get the one electron integrals of the given molecule
    def get_one_e(self):
        """

        :return: The one electron integral matrix: T + V
        """
        return self.integrals[1]

    # Get the two electron integrals of the given molecule
    def get_two_e(self):
        """

        :return: The electron repulsion interaction tensor
        """
        return self.integrals[2]

    # Get the nuclear repulsion value of the given molecule
    def nuc_rep(self):
        """

        :return: The nuclear repulsion value
        """
        return self.integrals[3]

    def random_guess(self):
        """
        A function that creates a matrix with random values that can be used as an initial guess
        for the SCF calculations.

        IMPORTANT: It is recommended to use a random guess since the results are significantly better
        than those found when using the standard guess.

        To use this guess:

        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = ComplexGHF(h3, 3)
        >>> guess = x.random_guess()
        >>> x.get_scf_solution(guess)

        :return: A random hermitian matrix.
        """
        dim = int(np.shape(self.get_ovlp())[0] * 2)

        def random_hermitian_matrix(dimension):
            # fill a matrix of the given dimensions with random complex numbers.
            x = np.random.rand(dimension, dimension) + np.random.rand(dimension, dimension) * 1j
            # Make the matrix symmetric by adding it's transpose.
            # Get the eigenvectors to use them, since they form a hermitian matrix.
            x_t = x.T
            y = x + x_t
            val, vec = la.eigh(y)
            return vec
        return random_hermitian_matrix(dim)

    def mixed_ghf_guess(self, ghf, constant):
        mo = ghf.get_mo_coeff()
        guess = mo + mo * 1j * constant

        def gs(X):
            Q, R = np.linalg.qr(X)
            return Q

        return gs(guess)

    def scf(self, guess=None, convergence=1e-12):
        """
        This function performs the SCF calculation by using the generalised Hartree-Fock formulas. Since we're working
        in the complex GHF class, all values throughout are complex.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param guess: Initial guess for scf. If none is given, a unitary rotation on the core Hamiltonian is used.
        :return: scf_energy, iterations, mo coefficients, last density matrix & last Fock matrix
        """
        s_min_12 = trans_matrix(self.get_ovlp()).astype(complex)
        s_12_o = expand_matrix(s_min_12).astype(complex)
        c_ham = expand_matrix(self.get_one_e()).astype(complex)
        if guess is None:

            def coeff_matrix(orth_matrix, core_ham):
                """
                :param orth_matrix: an orthogonalisation matrix, created with the expand matrix function
                :param core_ham: the expanded core Hamiltonian matrix
                :return: The orthogonalised version of the core Hamiltonian matrix
                """
                return orth_matrix @ core_ham @ orth_matrix.conj().T

            # Define the initial coefficient matrix the same way as for the real GHF object.
            c_init = coeff_matrix(s_12_o, c_ham)

            def unitary_rotation(coefficient_matrix):
                """
                This function will create a complex unitary matrix, which will then transform the given
                coefficient matrix.
                :param coefficient_matrix: Initial coefficient matrix, to be rotated.
                :return: The complex rotated coefficient matrix.
                """
                dim = np.shape(coefficient_matrix)[0]
                matrix = np.zeros((dim, dim), dtype=complex)
                # The complex unitary matrix that is constructed is based on the Fourier transformation matrix.
                for j in range(dim):
                    for k in range(dim):
                        matrix[j][k] = m.e ** ((2j * m.pi * (j + 1) * (k + 1)) / dim)
                u = matrix / m.sqrt(dim)
                return u @ coefficient_matrix @ u.conj().T

            initial_guess = unitary_rotation(c_init)
        else:
            initial_guess = guess

        def density(fock):
            """
            :param fock: a fock matrix
            :return: one big density matrix
            """
            # Get the coefficients by diagonalising the fock/guess matrix and calculate the density wit C(C.T)
            eigenval, eigenvec = la.eigh(fock)
            coeff = s_12_o @ eigenvec
            coeff_r = coeff[:, 0:self.number_of_electrons].astype(complex)
            # np.einsum represents Sum_j^occupied_orbitals(c_ij * c_kj)
            return np.einsum('ij,kj->ik', coeff_r, coeff_r.conj())

        # Calculate the guess density from the given initial guess and put it in an array.
        p_g = density(initial_guess).astype(complex)
        densities = [p_g]

        # Defining functions to calculate the coulomb and exchange integrals will make it easier to create the
        # Fock matrix in it's spin-blocked notation.
        def coulomb(density_block):
            """
            Calculate the coulomb integrals.
            """
            return np.einsum('kl, ijkl -> ij', density_block, self.get_two_e()).astype(complex)

        def exchange(density_block):
            """
            Calculate the exchange integrals.
            """
            return np.einsum('kl, ijkl -> ij', density_block, self.get_two_e().transpose(0, 2, 1, 3)).astype(complex)

        def fock_block(sigma, tau, p):
            """
            :param sigma: Can be either 'a' for alpha or 'b' for beta.
            :param tau: Can be either 'a' for alpha or 'b' for beta.
            :param p: a complete density matrix
            :return: one of the four blocks of the fock matrix, depending on the sigma and tau values.
            """
            # define d as a Cronecker delta, which will be useful when creating the blocks.
            if sigma == tau:
                d = 1
            else:
                d = 0
            # determine the dimension of 1 block.
            dim = int(np.shape(p)[0] / 2)
            # split the density matrix in it's four spin-blocks
            p_aa = p[0:dim, 0:dim]
            p_ab = p[dim:2 * dim, 0:dim]
            p_ba = p[0:dim, dim:2 * dim]
            p_bb = p[dim:2 * dim, dim:2 * dim]
            # Calculate the one electron integrals
            h_st = self.get_one_e()
            # calculate the coulomb and exchange integrals
            j_aa = coulomb(p_aa)
            j_bb = coulomb(p_bb)
            k_aa = exchange(p_aa)
            k_ab = exchange(p_ab)
            k_ba = exchange(p_ba)
            k_bb = exchange(p_bb)
            # depending on which block of the Fock matrix you're making, use the needed J & K values.
            if sigma == 'a' and tau == 'a':  # aa-block
                return d * h_st + d * (j_aa + j_bb) - k_aa
            if sigma == 'a' and tau == 'b':  # ab-block
                return d * h_st + d * (j_aa + j_bb) - k_ab
            if sigma == 'b' and tau == 'a':  # ba-block
                return d * h_st + d * (j_aa + j_bb) - k_ba
            if sigma == 'b' and tau == 'b':  # bb-block
                return d * h_st + d * (j_aa + j_bb) - k_bb

        # The function that will calculate the energy value according to the GHF algorithm.
        def scf_e(dens, fock):
            """
            Calculates the scf energy for the GHF method
            """
            return np.sum(dens * (expand_matrix(self.get_one_e()) + fock)) / 2

        # Calculate the first electronic energy from the initial guess and the guess density that's calculated from it.
        # Create an array to store the energy values and another to store the energy differences.
        electronic_e = scf_e(p_g, c_ham)
        energies = [electronic_e]
        delta_e = []

        def iteration():
            """
            This creates an iteration process to converge to the minimun energy.
            """
            # create the four spin blocks of the Fock matrix
            f_aa = fock_block('a', 'a', densities[-1]).astype(complex)
            f_ab = fock_block('a', 'b', densities[-1]).astype(complex)
            f_ba = fock_block('b', 'a', densities[-1]).astype(complex)
            f_bb = fock_block('b', 'b', densities[-1]).astype(complex)

            # Add them together to form the total Fock matrix in spin block notation
            f = spin_blocked(f_aa, f_ab, f_ba, f_bb).astype(complex)

            # Calculate the new energy and add it to the energies array.
            # Calculate the energy difference and add it to the delta_e array.
            energies.append(scf_e(densities[-1], f))
            delta_e.append(energies[-1] - energies[-2])

            # orthogonalise the Fock matrix
            f_o = s_12_o @ f @ s_12_o.conj().T

            # Create the new density matrix from the Orthogonalised Fock matrix.
            # Add the new density matrix to the densities array.
            p_new = density(f_o).astype(complex)
            densities.append(p_new)

        iteration()
        i = 1
        while abs(delta_e[-1]) >= convergence and i < 5000:
            iteration()
            i += 1
        self.iterations = i

        # A function that gives the last density matrix of the scf procedure.
        # Then set the last_dens value of the class object to this density matrix.
        def last_dens():
            return densities[-1]

        self.last_dens = last_dens()

        # A function that returns the last Fock matrix of the scf procedure.
        # Then, set the last_fock value of the GHF object to this Fock matrix.
        def last_fock():
            # Create the 4 individual spin-blocks of the last Fock matrix.
            f_aa = fock_block('a', 'a', densities[-2])
            f_ab = fock_block('a', 'b', densities[-2])
            f_ba = fock_block('b', 'a', densities[-2])
            f_bb = fock_block('b', 'b', densities[-2])
            # Add the blocks together.
            f = spin_blocked(f_aa, f_ab, f_ba, f_bb)
            # Return the orthogonalised last Fock matrix.
            return s_12_o.conj().T @ f @ s_12_o

        self.last_fock = last_fock()

        # A function that calculates the MO's from the last needed Fock matrix in the scf calculation.
        def get_mo():
            # Get the last Fock matrix.
            last_f = last_fock()
            # Diagonalise the Fock matrix.
            val, vec = la.eigh(last_f)
            # calculate the coefficients.
            coeff = s_12_o @ vec
            return coeff

        self.mo = get_mo()

        # Calculate the final scf energy (electronic + nuclear repulsion)
        scf_e = energies[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution(self, guess=None, convergence=1e-12):
        """
        Prints the number of iterations and the converged scf energy.

        :param guess: The initial scf guess. None specified: core Hamiltonian unitarily rotated with complex U matrix.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The converged scf energy.
        """
        scf_values = self.scf(guess, convergence=convergence)
        e = scf_values[0]
        i = scf_values[1]
        if abs(np.imag(e)) > 1e-12:
            print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
        else:
            print("Number of iterations: " + str(i))
            print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex GHF)")
        return self.energy

    def get_mo_coeff(self):
        """
        Gets the mo coefficients of the converged solution.

        :return: The mo coefficients
        """
        return self.mo

    def get_last_dens(self):
        """
        Gets the last density matrix of the converged solution.

        :return: The last density matrix.
        """
        return self.last_dens

    def get_last_fock(self):
        """
        Gets the last fock matrix of the converged solution.

        :return: The last Fock matrix.
        """
        return self.last_fock

    def stability(self):
        """
        Performing a stability analysis checks whether or not the wave function is stable, by checking the lowest
        eigenvalue of the Hessian matrix. If there's an instability, the MO's will be rotated in the direction
        of the lowest eigenvalue. These new MO's can then be used to start a new scf procedure.

        To perform a stability analysis, use the following syntax, this will continue the analysis until there is
        no more instability:

        >>> h4 = gto.M(atom = 'h 0 0 0; h 1 0 0; h 0 1 0; h 1 1 0' , spin = 2, basis = 'cc-pvdz')
        >>> x = ComplexGHF(h4, 4)
        >>> x.scf()
        >>> guess = x.stability()
        >>> while x.instability:
        >>>     new_guess = x.stability()
        >>>     x.get_scf_solution(new_guess)

        :return: New and improved MO's.
        """

        # Calculate the original coefficients after the scf calculation.
        coeff = self.get_mo_coeff()
        dim = np.shape(coeff)[0]

        # the generate_g() function returns 3 values
        # - the gradient, g
        # - the result of h_op, the trial vector
        # - the diagonal of the Hessian matrix, h_diag
        def generate_g():
            total_orbitals = dim  # total number of orbitals = number of basis functions
            n_occ = self.number_of_electrons  # Number of occupied orbitals
            n_vir = int(total_orbitals - n_occ)  # number of virtual/unoccupied orbitals
            occ_indx = np.arange(n_occ)  # indices of the occupied orbitals
            vir_indx = np.arange(total_orbitals)[n_occ:]  # indices of the virtual/unoccupied orbitals.
            occ_orb = coeff[:, occ_indx]  # the occupied orbitals
            vir_orb = coeff[:, vir_indx]  # The virtual/unoccupied orbitals

            # Use the last Fock matrix of the scf procedure as the initial Fock matrix for the stability analysis.
            # Transform the Fock matrix to AO basis
            fock_init = self.get_last_fock()
            fock_ao = reduce(np.dot, (coeff.conj().T, fock_init, coeff))

            # Split the Fock matrix in the occupied and virtual parts.
            fock_occ = fock_ao[occ_indx[:, None], occ_indx]
            fock_vir = fock_ao[vir_indx[:, None], vir_indx]

            # Calculate the Gradient and the Hessian diagonal.
            g = fock_ao[vir_indx[:, None], occ_indx]
            h_diag = fock_vir.diagonal().real[:, None] - fock_occ.diagonal().real

            # h_op is the Hessian operator, which will return the useful block (occupied-virtual rotation)
            # of the Hessian matrix.
            def h_op(x):
                x = x.reshape(n_vir, n_occ)  # Set the dimensions
                x2 = np.einsum('ps,sq->pq', fock_vir, x)  # Add the virtual fock matrix
                x2 -= np.einsum('ps,rp->rs', fock_occ, x)  # Add the occupied Fock matrix
                d1 = reduce(np.dot, (vir_orb, x, occ_orb.conj().T))  # vir_orb @ x @ occ_orb.T
                dm1 = d1 + d1.conj().T  # determine a density matrix

                # A function to calculate the coulomb and exchange integrals
                def vind(dm_1):
                    vj, vk = scf.hf.get_jk(self.molecule, dm_1, hermi=1)
                    return vj - vk

                v1 = vind(dm1)  # Get Coulomb - Exchange
                x2 += reduce(np.dot, (vir_orb.conj().T, v1, occ_orb))  # vir_orb.T @ v1 @ occ_orb
                return x2.ravel()  # Unravel the matrix into a 1D array.

            return g.reshape(-1), h_op, h_diag.reshape(-1)

        # This function will check whether or not there is an internal instability,
        # and if there is one, it will calculate new and improved coefficients.
        def internal_stability():
            g, hop, hdiag = generate_g()
            hdiag *= 2

            # this function prepares for the conditions needed to use a davidson solver later on
            def precond(dx, z, x0):
                hdiagd = hdiag - z
                hdiagd[abs(hdiagd) < 1e-8] = 1e-8
                return dx / hdiagd

            # The overall Hessian for internal rotation is x2 + x2.T.conj().
            # This is the reason we apply (.real * 2)
            def hessian_x(x):
                return hop(x).real * 2

            # Find the unique indices of the variables
            # in this function, bitwise operators are used.
            # They treat each operand as a sequence of binary digits and operate on them bit by bit
            def uniq_variable_indices(occ_mo):
                # indices of occupied alpha orbitals
                # indices of occupied beta orbitals
                occ_indx_a = occ_mo > 0
                occ_indx_b = occ_mo == 2
                # indices of virtual (unoccupied) alpha orbitals, done with bitwise operator: ~ (negation)
                # indices of virtual (unoccupied) beta orbitals, done with bitwise operator: ~ (negation)
                vir_indx_a = ~occ_indx_a
                vir_indx_b = ~occ_indx_b
                # & and | are bitwise operators for 'and' and 'or'
                # each bit position is the result of the logical 'and' or 'or' of the bits in the
                # corresponding position of the operands
                # determine the unique variable indices, by use of bitwise operators
                unique = (vir_indx_a[:, None] & occ_indx_a) | (vir_indx_b[:, None] & occ_indx_b)
                return unique

            # put the unique variables in a new matrix used later to create a rotation matrix.
            def unpack_uniq_variables(dx, occ_mo):
                nmo = len(occ_mo)
                idx = uniq_variable_indices(occ_mo)
                # print(np.shape(idx))
                # idx2 = np.ravel(idx[0:number_of_electrons, number_of_electrons:2 * nmo])
                # print(np.shape(idx2))
                x1 = np.zeros((nmo, nmo), dtype=dx.dtype)
                x1[idx] = dx
                # print(np.shape(dx))
                return x1 - x1.conj().T

            # A function to apply a rotation on the given coefficients
            def rotate_mo(mo_coeff, occ_mo, dx):
                dr = unpack_uniq_variables(dx, occ_mo)
                u = la2.expm(dr)  # computes the matrix exponential
                return np.dot(mo_coeff, u)

            x0 = np.zeros_like(g)  # like returns a matrix of the same shape as the argument given
            x0[g != 0] = 1. / hdiag[g != 0]  # create initial guess for davidson solver
            # use the davidson solver to find the eigenvalues and eigenvectors
            # needed to determine an internal instability
            e, v = lib.davidson(hessian_x, x0, precond, tol=1e-4)
            if e < -1e-5:  # this points towards an internal instability
                print("There is an instability in the complex GHF wave function.")
                mo_occ = np.zeros(dim, )  # total number of basis functions
                # create representation of alpha orbitals by adding an electron (= 1) to each occupied orbital
                for i in range(self.number_of_electrons):
                    mo_occ[i] = 1
                # create new orbitals by rotating the old ones
                mo = rotate_mo(coeff, mo_occ, v)
                self.instability = True
            else:
                # in the case where no instability is present
                print("There is no instability in the complex GHF wave function.")
                mo = coeff
                self.instability = False
            return mo

        return internal_stability()

    def diis(self, guess=None, convergence=1e-12):
        """
        The DIIS method is an alternative to the standard scf procedure. It reduces the number of iterations needed to
        find a solution. The same guesses can be used as for a standard scf calculation. Stability analysis can be
        done as well.

        :param guess: The initial guess matrix, if none is specified: expanded core Hamiltonian unitarily rotated.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: scf_energy, iterations, mo coefficients, last density matrix & last Fock matrix
        """
        # Get the transformation matrix, S^1/2, and write it in spin blocked notation.
        # Also define the core Hamiltonian matrix in it's spin-blocked notation.
        s_min_12 = trans_matrix(self.get_ovlp()).astype(complex)
        s_12_o = expand_matrix(s_min_12).astype(complex)
        c_ham = expand_matrix(self.get_one_e()).astype(complex)

        if guess is None:

            def coeff_matrix(orth_matrix, core_ham):
                """
                :param orth_matrix: an orthogonalisation matrix, created with the expand matrix function
                :param core_ham: the expanded core Hamiltonian matrix
                :return: The orthogonalised version of the core Hamiltonian matrix
                """
                return orth_matrix @ core_ham @ orth_matrix.conj().T

            # Define the initial coefficient matrix the same way as for the real GHF object.
            c_init = coeff_matrix(s_12_o, c_ham)

            def unitary_rotation(coefficient_matrix):
                """
                This function will create a complex unitary matrix, which will then transform the given
                coefficient matrix.
                :param coefficient_matrix: Initial coefficient matrix, to be rotated.
                :return: The complex rotated coefficient matrix.
                """
                dim = np.shape(coefficient_matrix)[0]
                matrix = np.zeros((dim, dim), dtype=complex)
                # The complex unitary matrix that is constructed is based on the Fourier transformation matrix.
                for j in range(dim):
                    for k in range(dim):
                        matrix[j][k] = m.e ** ((2j * m.pi * (j + 1) * (k + 1)) / dim)
                u = matrix / m.sqrt(dim)
                return u @ coefficient_matrix @ u.conj().T

            initial_guess = unitary_rotation(c_init)
        else:
            initial_guess = guess

        def density(fock):
            """
            :param fock: a fock matrix
            :return: one big density matrix
            """
            # Get the coefficients by diagonalising the fock/guess matrix and calculate the density wit C(C.T)
            eigenval, eigenvec = la.eigh(fock)
            coeff = s_12_o @ eigenvec
            coeff_r = coeff[:, 0:self.number_of_electrons].astype(complex)
            # np.einsum represents Sum_j^occupied_orbitals(c_ij * c_kj)
            return np.einsum('ij,kj->ik', coeff_r, coeff_r)

        # Calculate the guess density from the given initial guess and put it in an array.
        p_g = density(initial_guess).astype(complex)
        densities_diis = [p_g]

        # Defining functions to calculate the coulomb and exchange integrals will make it easier to create the
        # Fock matrix in it's spin-blocked notation.
        def coulomb(density_block):
            """
            Calculate the coulomb integrals.
            """
            return np.einsum('kl, ijkl -> ij', density_block, self.get_two_e()).astype(complex)

        def exchange(density_block):
            """
            Calculate the exchange integrals.
            """
            return np.einsum('kl, ijkl -> ij', density_block, self.get_two_e().transpose(0, 2, 1, 3)).astype(complex)

        def fock_block(sigma, tau, p):
            """
            :param sigma: Can be either 'a' for alpha or 'b' for beta.
            :param tau: Can be either 'a' for alpha or 'b' for beta.
            :param p: a complete density matrix
            :return: one of the four blocks of the fock matrix, depending on the sigma and tau values.
            """
            # define d as a Cronecker delta, which will be useful when creating the blocks.
            if sigma == tau:
                d = 1
            else:
                d = 0
            # determine the dimension of 1 block.
            dim = int(np.shape(p)[0] / 2)
            # split the density matrix in it's four spin-blocks: aa, ab, ba and bb.
            p_aa = p[0:dim, 0:dim]
            p_ab = p[dim:2 * dim, 0:dim]
            p_ba = p[0:dim, dim:2 * dim]
            p_bb = p[dim:2 * dim, dim:2 * dim]
            # Calculate the one_electron integrals.
            h_st = self.get_one_e()
            # calculate the coulomb and exchange integrals, needed for each of the spin-blocks in the Fock matrix.
            j_aa = coulomb(p_aa)
            j_bb = coulomb(p_bb)
            k_aa = exchange(p_aa)
            k_ab = exchange(p_ab)
            k_ba = exchange(p_ba)
            k_bb = exchange(p_bb)
            # depending on which block of the Fock matrix you're making, use the needed J & K values.
            if sigma == 'a' and tau == 'a':  # aa-block
                return d * h_st + d * (j_aa + j_bb) - k_aa
            if sigma == 'a' and tau == 'b':  # ab-block
                return d * h_st + d * (j_aa + j_bb) - k_ab
            if sigma == 'b' and tau == 'a':  # ba-block
                return d * h_st + d * (j_aa + j_bb) - k_ba
            if sigma == 'b' and tau == 'b':  # bb-block
                return d * h_st + d * (j_aa + j_bb) - k_bb

        def residual(dens, fock):
            """
            Function that calculates the error matrix for the DIIS algorithm
            :param dens: density matrix
            :param fock: fock matrix
            :return: a value that should be zero and a fock matrix
            """
            return s_12_o @ (fock @ dens @ expand_matrix(self.get_ovlp()) -
                             expand_matrix(self.get_ovlp()) @ dens @ fock) @ s_12_o.conj().T

        # Create a list to store the errors
        # create a list to store the fock matrices
        # deque is used as a high performance equivalent of an array
        error_list = c.deque(maxlen=6)
        fock_list = c.deque(maxlen=6)

        def diis_fock(focks, residuals):
            dim = len(focks) + 1

            b = np.empty((dim, dim)).astype(complex)
            b[-1, :] = -1
            b[:, -1] = -1
            b[-1, -1] = 0

            for k in range(len(focks)):
                for l in range(len(focks)):
                    b[k, l] = np.einsum('kl,kl->', residuals[k], residuals[l])

            res_vec = np.zeros(dim).astype(complex)
            res_vec[-1] = -1

            coeff = np.linalg.solve(b, res_vec)

            fock = np.zeros(focks[0].shape).astype(complex)
            for x in range(coeff.shape[0] - 1):
                fock += coeff[x] * focks[x]
            return fock.astype(complex)

        # The function that will calculate the energy value according to the GHF algorithm.
        def scf_e(dens, fock):
            """
            Calculates the scf energy for the GHF method
            """
            return np.sum(dens * (expand_matrix(self.get_one_e()) + fock)) / 2

        # Calculate the first electronic energy from the initial guess and the guess density that's calculated from it.
        # Create an array to store the energy values and another to store the energy differences.
        electronic_e = scf_e(p_g, initial_guess)
        energies_diis = [electronic_e]
        delta_e_diis = []

        def iteration_diis(number_of_iterations):
            # create the four spin blocks of the Fock matrix
            f_aa = fock_block('a', 'a', densities_diis[-1]).astype(complex)
            f_ab = fock_block('a', 'b', densities_diis[-1]).astype(complex)
            f_ba = fock_block('b', 'a', densities_diis[-1]).astype(complex)
            f_bb = fock_block('b', 'b', densities_diis[-1]).astype(complex)

            # Add them together to form the total Fock matrix in spin block notation
            f = spin_blocked(f_aa, f_ab, f_ba, f_bb).astype(complex)

            # Calculate the residual
            resid = residual(densities_diis[-1], f).astype(complex)

            # Add them to the arrays
            fock_list.append(f)
            error_list.append(resid)

            # Calculate the new energy and add it to the energies array.
            # Calculate the energy difference and add it to the delta_e array.
            energies_diis.append(scf_e(densities_diis[-1], f))
            delta_e_diis.append(energies_diis[-1] - energies_diis[-2])

            if number_of_iterations >= 2:
                f = diis_fock(fock_list, error_list)

            # orthogonalise the Fock matrix
            f_o = s_12_o.T @ f @ s_12_o

            # Create the new density matrix from the Orthogonalised Fock matrix.
            # Add the new density matrix to the densities array.
            p_new = density(f_o).astype(complex)
            densities_diis.append(p_new)

        i = 1
        iteration_diis(i)
        while abs(delta_e_diis[-1]) >= convergence and i < 5000:
            iteration_diis(i)
            i += 1
        self.iterations = i

        # A function that gives the last density matrix of the scf procedure.
        # Then set the last_dens value of the class object to this density matrix.
        def last_dens():
            return densities_diis[-1]

        self.last_dens = last_dens()

        # A function that returns the last Fock matrix of the scf procedure.
        # Then, set the last_fock value of the GHF object to this Fock matrix.
        def last_fock():
            # Create the 4 individual spin-blocks of the last Fock matrix.
            f_aa = fock_block('a', 'a', densities_diis[-2])
            f_ab = fock_block('a', 'b', densities_diis[-2])
            f_ba = fock_block('b', 'a', densities_diis[-2])
            f_bb = fock_block('b', 'b', densities_diis[-2])
            # Add the blocks together.
            f = spin_blocked(f_aa, f_ab, f_ba, f_bb)
            # Return the orthogonalised last Fock matrix.
            return s_12_o.T.conj() @ f @ s_12_o

        self.last_fock = last_fock()

        # A function that calculates the MO's from the last needed Fock matrix in the scf calculation.
        def get_mo():
            # Get the last Fock matrix.
            last_f = last_fock()
            # Diagonalise the Fock matrix.
            val, vec = la.eigh(last_f)
            # calculate the coefficients.
            coeff = s_12_o @ vec
            return coeff

        self.mo = get_mo()

        # Calculate the final scf energy (electronic + nuclear repulsion)
        scf_e = energies_diis[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution_diis(self, guess=None, convergence=1e-12):
        """
        Prints the number of iterations and the converged energy after a diis calculation. Guesses can also be specified
        just like with a normal scf calculation.

        Example:

        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = ComplexGHF(h3, 3)
        >>> guess = x.random_guess()
        >>> x.get_scf_solution_diis(guess)

        :param guess: Initial scf guess. None specified: core HAmiltonian unitarily rotated with complex U matrix.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The converged scf energy.
        """
        scf_values = self.diis(guess, convergence=convergence)
        e = scf_values[0]
        i = scf_values[1]
        if abs(np.imag(e)) > 1e-12:
            print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
        else:
            print("Number of iterations: " + str(i))
            print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex GHF)")

    def loop_calculations(self, number_of_loops, guess=None, convergence=1e-12):
        """
        This function is specifically catered to the random guess method. Since it is hard to predict the seed of the
        correct random matrix, a simple solution is to repeat the scf calculation a certain number of times, starting
        from different random guesses and returning the lowest value of all the different calculations. The loops will
        automatically perform a stability analysis until there is no more instability in the wave function.

        :param number_of_loops: The amount of times you want to repeat the scf + stability procedure.
        :param guess: The guess used for the scf procedure.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The scf energy after the loops.
        """
        # Create the needed arrays.
        energy = []
        iterations = []
        # Loop the scf calculation + stability analysis
        for i in range(number_of_loops):
            self.scf(guess, convergence=convergence)
            self.stability()
            while self.instability:
                new_guess = self.stability()
                self.scf(new_guess, convergence=convergence)
            energy.append(self.energy)
            iterations.append(self.iterations)
        e = np.amin(energy)
        i = iterations[energy.index(e)]
        self.energy = e
        # Print the resulting energy.
        if abs(np.imag(e)) > 1e-12:
            print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
        else:
            print("Number of iterations: " + str(i))
            print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex GHF)")
        return self.energy

    def loop_calculations_diis(self, number_of_loops, guess=None, convergence=1e-12):
        """
        This function is specifically catered to the random guess method. Since it is hard to predict the seed of the
        correct random matrix, a simple solution is to repeat the scf calculation a certain number of times, starting
        from different random guesses and returning the lowest value of all the different calculations. The loops will
        automatically perform a stability analysis until there is no more instability in the wave function. This option
        uses the DIIS iteration so that convergence is generally reached faster.

        :param number_of_loops: The amount of times you want to repeat the DIIS + stability procedure.
        :param guess: The guess used for the DIIS procedure.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The energy after the loops.
        """
        # Create the needed arrays.
        energy = []
        iterations = []
        # Loop the scf calculation + stability analysis
        for i in range(number_of_loops):
            self.diis(guess, convergence=convergence)
            self.stability()
            while self.instability:
                new_guess = self.stability()
                self.diis(new_guess, convergence=convergence)
            energy.append(self.energy)
            iterations.append(self.iterations)
        e = np.amin(energy)
        i = iterations[energy.index(e)]
        self.energy = e
        # Print the resulting energy.
        if abs(np.imag(e)) > 1e-12:
            print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
        else:
            print("Number of iterations: " + str(i))
            print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex GHF)")
        return self.energy
