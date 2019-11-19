"""
Real generalised Hartree Fock, by means of SCF procedure
=========================================================


This class creates a generalised Hartree-Fock object which can be used for scf calculations. Different initial guesses
are provided as well as the option to perform a stability analysis.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)

"""

from ghf.SCF_functions import *
import numpy as np
from numpy import linalg as la
from scipy import linalg as la2
from functools import reduce


class RealGHF:
    """
    Calculate the real GHF energy.
    --------------------------------
    Input is a molecule and the number of electrons.

    Molecules are made in pySCF and calculations are performed as follows, eg.:
    The following snippet prints and returns UHF energy of h3
    and the number of iterations needed to get this value.

    For a normal scf calculation your input looks like the following example:

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> x = RealGHF(h3, 3)
    >>> x. get_scf_solution()
    Number of iterations: 82
    Converged SCF energy in Hartree: -1.5062743202607725 (Real GHF)
    """
    def __init__(self, molecule, number_of_electrons):
        """
        Initiate an instance of a real GHF class.

        :param molecule: The molecule on which to perform the calculations, made in PySCF.
        :param number_of_electrons: The amount of electrons present in the system.
        """
        # Set the molecule, the integrals, the mo coefficients, the number of electrons,
        # the energy, the last fock and the last density matrices as class parameters.
        self.molecule = molecule
        self.number_of_electrons = number_of_electrons
        self.integrals = get_integrals(molecule)
        self.energy = None
        self.mo = None
        self.last_dens = None
        self.last_fock = None
        self.iterations = None
        self.instability = None

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

    def unitary_rotation_guess(self):
        """
        A function that creates an initial guess matrix by performing a unitary transformation on the core Hamiltonian
        matrix.

        To use this guess:

        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = RealGHF(h3, 3)
        >>> guess = x.unitary_rotation_guess()
        >>> x.get_scf_solution(guess)

        :return: A rotated guess matrix.
        """
        c_ham = expand_matrix(self.get_one_e())

        def unitary_rotation(coefficient_matrix):
            """
            Perform a unitary transformation on the given coefficient matrix.

            :param coefficient_matrix: The initial coefficient matrix, most often the core hamiltonian.
            :return: The rotated coefficient matrix.
            """
            # Make sure the unitary matrix is the same size as the coefficient matrix.
            shape = np.shape(coefficient_matrix)
            # Create the unitary matrix as an exponential of a dense 1/-1 matrix, with all diagonal elements 0.
            one = np.full(shape, 1)
            neg = np.full(shape, -1)
            ones = np.tril(one)
            negative = np.triu(neg)
            exp = -1 * (ones + negative)
            u = la2.expm(exp)
            # Return UCU^1 as the transformed matrix.
            return u @ coefficient_matrix @ la.inv(u)

        return unitary_rotation(c_ham)

    def random_guess(self):
        """
        A function that creates a matrix with random values that can be used as an initial guess
        for the SCF calculations.

        To use this guess:

        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = RealGHF(h3, 3)
        >>> guess = x.random_guess()
        >>> x.get_scf_solution(guess)

        :return: A random hermitian matrix.
        """
        dim = int(np.shape(self.get_ovlp())[0] * 2)

        def random_hermitian_matrix(dimension):
            # fill a matrix of the given dimensions with random numbers.
            np.random.seed(2)
            x = np.random.rand(dimension, dimension)
            # Make the matrix symmetric by adding it's transpose.
            # Get the eigenvectors to use them, since they form a hermitian matrix.
            x_t = x.T
            y = x + x_t
            val, vec = la.eigh(y)
            return vec

        return random_hermitian_matrix(dim)

    def scf(self, guess=None):
        """
        This function performs the SCF calculation by using the generalised Hartree-Fock formulas. Since we're working
        in the real class, all values throughout are real. For complex, see the "complex_GHF" class.

        :param guess: Initial guess to start SCF. If none is given, core hamiltonian will be used.
        :return: scf_energy, iterations, mo coefficients, last density matrix & last Fock matrix
        """
        # Get the transformation matrix, S^1/2, and write it in spin blocked notation.
        # Also define the core Hamiltonian matrix in it's spin-blocked notation.
        s_min_12 = trans_matrix(self.get_ovlp())
        s_12_o = expand_matrix(s_min_12)
        c_ham = expand_matrix(self.get_one_e())

        if guess is None:

            def coeff_matrix(orth_matrix, core_ham):
                """
                :param orth_matrix: an orthogonalisation matrix, created with the expand matrix function
                :param core_ham: the expanded core Hamiltonian matrix
                :return: The orthogonalised version of the core Hamiltonian matrix
                """
                return orth_matrix @ core_ham @ orth_matrix.T
            # Define the initial guess as the orthogonalised core Hamiltonian matrix
            initial_guess = coeff_matrix(s_12_o, c_ham)
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
            coeff_r = coeff[:, 0:self.number_of_electrons]
            # np.einsum represents Sum_j^occupied_orbitals(c_ij * c_kj)
            return np.einsum('ij,kj->ik', coeff_r, coeff_r)

        # Calculate the guess density from the given initial guess and put it in an array.
        p_g = density(initial_guess)
        densities = [p_g]

        # Defining functions to calculate the coulomb and exchange integrals will make it easier to create the
        # Fock matrix in it's spin-blocked notation.
        def coulomb(density_block):
            """
            Calculate the coulomb integrals.
            """
            return np.einsum('kl, ijkl -> ij', density_block, self.get_two_e())

        def exchange(density_block):
            """
            Calculate the exchange integrals.
            """
            return np.einsum('kl, ijkl -> ij', density_block, self.get_two_e().transpose(0, 2, 1, 3))

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

        # The function that will calculate the energy value according to the GHF algorithm.
        def scf_e(dens, fock):
            """
           Calculates the scf energy for the GHF method
            """
            return np.sum(dens * (expand_matrix(self.get_one_e()) + fock)) / 2

        # Calculate the first electronic energy from the initial guess and the guess density that's calculated from it.
        # Create an array to store the energy values and another to store the energy differences.
        electronic_e = scf_e(p_g, initial_guess)
        energies = [electronic_e]
        delta_e = []

        def iteration():
            """
            This creates an iteration process to converge to the minimum energy.
            """
            # create the four spin blocks of the Fock matrix
            f_aa = fock_block('a', 'a', densities[-1])
            f_ab = fock_block('a', 'b', densities[-1])
            f_ba = fock_block('b', 'a', densities[-1])
            f_bb = fock_block('b', 'b', densities[-1])

            # Add them together to form the total Fock matrix in spin block notation
            # orthogonalise the Fock matrix
            f = spin_blocked(f_aa, f_ab, f_ba, f_bb)
            f_o = s_12_o.T @ f @ s_12_o

            # Create the new density matrix from the Orthogonalised Fock matrix.
            # Add the new density matrix to the densities array.
            p_new = density(f_o)
            densities.append(p_new)

            # Calculate the new energy and add it to the energies array.
            # Calculate the energy difference and add it to the delta_e array.
            energies.append(scf_e(densities[-1], f))
            delta_e.append(energies[-1] - energies[-2])

        iteration()
        i = 1
        while abs(delta_e[-1]) >= 1e-12 and i < 5000:
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
            return s_12_o.T @ f @ s_12_o
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

    def get_scf_solution(self, guess=None):
        """
        Prints the number of iterations and the converged scf energy.

        :return: The converged scf energy.
        """
        scf_values = self.scf(guess)
        #s_values = ghf_spin(self.get_mo_coeff(), self.get_ovlp())
        print("Number of iterations: " + str(scf_values[1]))
        print("Converged SCF energy in Hartree: " + str(scf_values[0]) + " (Real GHF)")
        #print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) + ", Multiplicity = " + str(s_values[2]))
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
        >>> x = RealGHF(h4, 4)
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
                print("There is an internal instability in the real GHF wave function.")
                mo_occ = np.zeros(dim, )  # total number of basis functions
                # create representation of alpha orbitals by adding an electron (= 1) to each occupied orbital
                for i in range(self.number_of_electrons):
                    mo_occ[i] = 1
                # create new orbitals by rotating the old ones
                mo = rotate_mo(coeff, mo_occ, v)
                self.instability = True
            else:
                # in the case where no instability is present
                print("There is no internal instability in the real GHF wave function.")
                mo = coeff
                self.instability = False
            return mo
        return internal_stability()
