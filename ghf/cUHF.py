"""
Constrained unrestricted Hartree Fock
=====================================
This class is used to calculate the ROHF energy for a given molecule and the number of electrons of that molecule,
using a constrained version of unrestricted Hartree Fock.
Several options are available to make sure you get the lowest energy from your calculation, as well as some useful
functions to get intermediate values such as MO coefficients, density and fock matrices.
"""

from ghf.SCF_functions import *
from pyscf import *
import scipy.linalg as la
from functools import reduce



class CUHF:
    """
    Calculate UHF energy.
    ---------------------
    Input is a molecule and the number of electrons.

    Molecules are made in pySCF/psi4 and calculations are performed as follows, eg.:
    The following snippet prints and returns UHF energy of h_3
    and the number of iterations needed to get this value.

    For a normal scf calculation your input looks like the following example:

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> x = CUHF(h3, 3)
    >>> x.get_scf_solution()
    """
    def __init__(self, molecule, number_of_electrons, int_method='pyscf'):
        """
        Initiate the UHF instance by specifying the molecule in question with pyscf and the total number of electrons.

        :param molecule: The molecule on which to perform the calculations
        :param number_of_electrons: The total number of electrons in the system
        :param int_method: method to calculate the integrals. pyscf and psi4 are supported.
        """
        # Set the molecule, the integrals, the mo coefficients, the number of alpha and beta
        # electrons, the energy, the last fock and the last density matrices as class parameters.
        # If the number of electrons is even, an equal amount of alpha and beta electrons are set.
        # If there's an uneven number of electrons, there will be one more alpha than beta electron.
        self.molecule = molecule
        if int_method == 'pyscf':
            self.integrals = get_integrals_pyscf(molecule)
        elif int_method == 'psi4':
            self.integrals = get_integrals_psi4(molecule)
        else:
            raise Exception('Unsupported method to calculate integrals. Supported methods are pyscf or psi4. '
                            'Make sure the molecule instance matches the method and is gives as a string.')
        # Several dictionaries are created, as to be able to check these properties for a certain iteration
        # The dictionaries are filled while iterating.
        self.energy = None
        self.energy_list = {}
        self.mo = None
        self.last_dens = None
        self.coeff_list = {}
        self.constrained_coeff_list = {}
        self.density_list = {}
        self.last_fock = None
        self.fock_list = {}
        self.fock_orth_list = {}
        self.iterations = None
        if number_of_electrons % 2 == 0:
            self.n_a = int(number_of_electrons / 2)
            self.n_b = int(number_of_electrons / 2)
        else:
            self.n_a = int(np.ceil(number_of_electrons / 2))
            self.n_b = int(np.floor(number_of_electrons / 2))

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

        To use this guess:

        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = CUHF(h3, 3)
        >>> guess = x.random_guess()
        >>> x.get_scf_solution(guess)

        :return: A random hermitian matrix.
        """
        dim = int(np.shape(self.get_ovlp())[0])

        def random_unitary_matrix(dimension):
            # fill a matrix of the given dimensions with random numbers.
            np.random.seed(2)
            x = np.random.rand(dimension, dimension)
            # Make the matrix symmetric by adding it's transpose.
            # Get the eigenvectors to use them, since they form a unitary matrix.
            x_t = x.T
            y = x + x_t
            val, vec = la.eigh(y)
            return vec

        return random_unitary_matrix(dim), random_unitary_matrix(dim)

    def scf(self, initial_guess=None, convergence=1e-12):
        """
        Performs a self consistent field calculation to find the lowest UHF energy.

        :param initial_guess: Set the convergence criterion. If none is given, 1e-12 is used.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The scf energy, number of iterations, the mo coefficients, the last density and the last fock matrices
        """
        # calculate the transformation matrix
        s_12 = trans_matrix(self.get_ovlp())
        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        if initial_guess is None:
            # create guess density matrix from core guess, separate for alpha and beta and put them into an array
            initial_guess = s_12.conj().T @ self.get_one_e() @ s_12
            guess_density_a = density_matrix(initial_guess, self.n_a, s_12)
            guess_density_b = density_matrix(initial_guess, self.n_b, s_12)
        else:
            # Make the coefficients orthogonal in the correct basis.
            coeff_a = s_12 @ initial_guess[0]
            coeff_b = s_12 @ initial_guess[1]

            # Get C_alpha and C_beta
            coeff_r_a = coeff_a[:, 0:self.n_a]
            coeff_r_b = coeff_b[:, 0:self.n_b]

            guess_density_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a)
            guess_density_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b)

        # Store the density matrices in an array
        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        # Create a second array to study the difference in energy between iterations.
        energies = [0.0]
        delta_e = []

        # Calculate your mo coefficients
        def mo_coeff(fock_o):
            f_eigenvalues, f_eigenvectors = la.eigh(fock_o)
            coefficients = s_12.dot(f_eigenvectors)
            return coefficients

        # define your constraint
        def constrain_mo(mo_a, mo_b):
            mo_b[:, :self.n_b] = mo_a[:, :self.n_b]
            return mo_a, mo_b

        # calculate your density matrix
        def dens(mo, occ):
            coefficients_r = mo[:, 0:occ]
            density = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj(), optimize=True)
            return density

        def constrain_dens(dens_a, dens_b):
            dens_b[:, :self.n_b] = dens_a[:, :self.n_b]
            return dens_a, dens_b

        # create an iteration procedure
        def iteration(n_i):
            self.density_list[n_i] = [densities_a[-1], densities_b[-1]]
            # create a fock matrix for alpha from last alpha density
            fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], self.get_one_e(), self.get_two_e())
            # create a fock matrix for beta from last beta density
            fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], self.get_one_e(), self.get_two_e())
            self.fock_list[n_i] = [fock_a, fock_b]

            # calculate the improved scf energy and add it to the array
            energies.append(uhf_scf_energy(densities_a[-1], densities_b[-1], fock_a, fock_b, self.get_one_e()))
            self.energy_list[n_i] = uhf_scf_energy(densities_a[-1], densities_b[-1], fock_a, fock_b,
                                                   self.get_one_e()) + self.nuc_rep()
            # calculate the energy difference and add it to the delta_E array
            delta_e.append(energies[-1] - energies[-2])

            # orthogonalize both fock matrices
            fock_orth_a = s_12.conj().T.dot(fock_a).dot(s_12)
            fock_orth_b = s_12.conj().T.dot(fock_b).dot(s_12)
            self.fock_orth_list[n_i] = [fock_orth_a, fock_orth_b]

            # calculate new mo coefficients
            mo_a = mo_coeff(fock_orth_a)
            mo_b = mo_coeff(fock_orth_b)
            self.coeff_list[n_i] = [mo_a, mo_b]

            # constrain the mo's
            c_mo_a, c_mo_b = constrain_mo(mo_a, mo_b)
            self.constrained_coeff_list[n_i] = [c_mo_a, c_mo_b]

            # create a new alpha and beta density matrix
            new_density_a = dens(c_mo_a, self.n_a)
            new_density_b = dens(c_mo_b, self.n_b)
            #new_density_a, new_density_b = constrain_dens(new_density_a, new_density_b)

            # put the density matrices in their respective arrays
            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        i = 1
        iteration(i)
        while abs(delta_e[-1]) >= convergence and i < 1000:
            iteration(i)
            i += 1
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure, both for alpha and beta
        def last_dens():
            return densities_a[-1], densities_b[-1]
        self.last_dens = last_dens()

        def last_fock():
            return self.fock_list[self.iterations-1]
        self.last_fock = last_fock()

        def get_mo():
            return self.constrained_coeff_list[self.iterations-1]
        self.mo = get_mo()

        # Calculate the final scf energy (electronic + nuclear repulsion)
        scf_e = energies[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution(self, guess=None, convergence=1e-12):
        """
        Prints the number of iterations and the converged scf energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :param guess: Initial scf guess
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The converged scf energy.
        """
        self.scf(guess, convergence=convergence)
        s_values = spin(self.n_a, self.n_b, CUHF.get_mo_coeff(self)[0], CUHF.get_mo_coeff(self)[1], self.get_ovlp())
        print("Number of iterations: " + str(self.iterations))
        print("Converged SCF energy in Hartree: " + str(self.energy) + " (Constrained UHF)")
        print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) + ", Multiplicity = " + str(s_values[2]))
        return self.energy

    def get_mo_coeff(self):
        """
        Gets the mo coefficients of the converged solution.
        Alpha coefficients in the first matrix, beta coefficients in the second.

        :return: The mo coefficients
        """
        return self.mo

    def get_last_dens(self):
        """
        Gets the last density matrix of the converged solution.
        Alpha density in the first matrix, beta density in the second.

        :return: The last density matrix.
        """
        return self.last_dens

    def get_last_fock(self):
        """
        Gets the last fock matrix of the converged solution.
        Alpha Fock matrix first, beta Fock matrix second.

        :return: The last Fock matrix.
        """
        return self.last_fock

    def stability(self):
        """
        Performing a stability analysis checks whether or not the wave function is stable, by checking the lowest eigen-
        value of the Hessian matrix. If there's an instability, the MO's will be rotated in the direction
        of the lowest eigenvalue. These new MO's can then be used to start a new scf procedure.

        To perform a stability analysis, use the following syntax:

        >>> h4 = gto.M(atom = 'h 0 0 0; h 1 0 0; h 0 1 0; h 1 1 0' , spin = 2, basis = 'cc-pvdz')
        >>> x = CUHF(h4, 4)
        >>> guess = x.stability()
        >>> x.get_scf_solution(guess)
        There is an internal instability in the UHF wave function.
        Number of iterations: 66
        Converged SCF energy in Hartree: -2.0210882477030716 (UHF)
        <S^2> = 1.056527700105677, <S_z> = 0.0, Multiplicity = 2.2860688529488145

        :return: New and improved MO's.
        """
        # The trans_matrix function calculates the orthogonalisation matrix from a given overlap matrix.
        # core_guess = the guess in the case where the electrons don't interact with each other
        s_12 = trans_matrix(self.get_ovlp())
        core_guess = s_12.T.dot(self.get_one_e()).dot(s_12)

        # create guess density matrix from core guess, separate for alpha and beta and put it into an array
        # Switch the spin state by adding one alpha and removing one beta electron
        guess_density_a = density_matrix(core_guess, self.n_a + 1, s_12)
        guess_density_b = density_matrix(core_guess, self.n_b - 1, s_12)
        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        # create an array to check the differences between density matrices in between iterations
        delta_dens = []

        # create an iteration procedure, based on the densities
        def iteration():
            # create a fock matrix for alpha from last alpha density
            # create a fock matrix for beta from last alpha density
            fock_matrix_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], self.get_one_e(), self.get_two_e())
            fock_matrix_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], self.get_one_e(), self.get_two_e())

            # orthogonalize the fock matrices
            orth_fock_a = s_12.T.dot(fock_matrix_a).dot(s_12)
            orth_fock_b = s_12.T.dot(fock_matrix_b).dot(s_12)

            # create a new alpha density matrix
            # create a new beta density matrix
            # And add the density matrices to an array.
            new_density_a = density_matrix(orth_fock_a, self.n_a + 1, s_12)
            new_density_b = density_matrix(orth_fock_b, self.n_b - 1, s_12)
            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

            # calculate the difference between the last two density matrices
            delta_dens.append(np.sum(densities_a[-1] + densities_b[-1]) - np.sum(densities_a[-2] + densities_b[-2]))

        # start and continue the iteration process as long as the difference between densities is larger than 1e-12
        iteration()
        while abs(delta_dens[-1]) >= 1e-12:
            iteration()

        # Now that the system has converged, calculate the system's orbital coefficients from the last calculated
        # density matrix. First calculate the Fock matrices.
        fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], self.get_one_e(), self.get_two_e())
        fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], self.get_one_e(), self.get_two_e())
        # orthogonalize the fock matrices
        fock_orth_a = s_12.T.dot(fock_a).dot(s_12)
        fock_orth_b = s_12.T.dot(fock_b).dot(s_12)
        # Diagonalise the Fock matrices.
        val_a, vec_a = la.eigh(fock_orth_a)
        val_b, vec_b = la.eigh(fock_orth_b)
        # Calculate the MO coefficients
        coeff_a = s_12.dot(vec_a)
        coeff_b = s_12.dot(vec_b)

        # the generate_g() function returns 3 values
        # - the gradient, g
        # - the result of h_op, the trial vector
        # - the diagonal of the Hessian matrix, h_diag
        def generate_g():
            total_orbitals = self.get_ovlp().shape[0]  # total number of orbitals = number of basis functions
            n_vir_a = int(total_orbitals - self.n_a)  # number of virtual (unoccupied) alpha orbitals
            n_vir_b = int(total_orbitals - self.n_b)  # number of virtual (unoccupied) beta orbitals
            occ_indx_a = np.arange(self.n_a)  # indices of the occupied alpha orbitals
            occ_indx_b = np.arange(self.n_b)  # indices of the occupied beta orbitals
            vir_indx_a = np.arange(total_orbitals)[self.n_a:]  # indices of the virtual (unoccupied) alpha orbitals
            vir_indx_b = np.arange(total_orbitals)[self.n_b:]  # indices of the virtual (unoccupied) beta orbitals
            occ_a_orb = coeff_a[:, occ_indx_a]  # orbital coefficients associated with occupied alpha orbitals
            occ_b_orb = coeff_b[:, occ_indx_b]  # orbital coefficients associated with occupied beta orbitals
            # orbital coefficients associated with virtual (unoccupied) alpha orbitals
            # orbital coefficients associated with virtual (unoccupied) beta orbitals
            vir_a_orb = coeff_a[:, vir_indx_a]
            vir_b_orb = coeff_b[:, vir_indx_b]

            # initial fock matrix for stability analysis is the last fock matrix from the first iteration process,
            # for both alpha and beta
            fock_a_init = uhf_fock_matrix(densities_a[-1], densities_b[-1], self.get_one_e(), self.get_two_e())
            fock_b_init = uhf_fock_matrix(densities_b[-1], densities_a[-1], self.get_one_e(), self.get_two_e())

            # orthogonolize the initial fock matrix with the coefficients, calculated from the first iteration process
            # reduce() is a short way to write calculations, the first argument is an operation,
            # the second one are the values on which to apply the operation
            fock_matrix_a = reduce(np.dot, (coeff_a.T, fock_a_init, coeff_a))
            fock_matrix_b = reduce(np.dot, (coeff_b.T, fock_b_init, coeff_b))

            # specify the fock matrix for only occupied alpha and beta orbitals
            # specify the fock matrix for only virtual (unoccupied) alpha and beta orbitals
            fock_occ_a = fock_matrix_a[occ_indx_a[:, None], occ_indx_a]
            fock_vir_a = fock_matrix_a[vir_indx_a[:, None], vir_indx_a]
            fock_occ_b = fock_matrix_b[occ_indx_b[:, None], occ_indx_b]
            fock_vir_b = fock_matrix_b[vir_indx_b[:, None], vir_indx_b]

            # create the gradient
            # This is done by combining the necessary parts of the alpha and beta fock matrix with np.hstack and then
            # using np.ravel() to create a 1D matrix
            # np.hstack() adds together arrays horizontally e.g.:
            #                               a = np.array([[1], [2], [3]]))
            #                               b = np.array([[2], [3], [4]]))
            #                               np.hstack((a,b)) = array([1, 2],
            #                                                        [2, 3],
            #                                                        [3, 4])
            # np.ravel() pulls all array values into 1 long 1D array
            g = np.hstack(
                (fock_a[vir_indx_a[:, None], occ_indx_a].ravel(), fock_b[vir_indx_b[:, None], occ_indx_b].ravel()))

            # Create the diagonal alpha and beta hessian respectively from the virtual and the occupied fock matrices
            # Use np.hstack() to combine the diagonal alpha and beta hessians to create
            # the total diagonal hessian matrix
            h_diag_a = fock_vir_a.diagonal().real[:, None] - fock_occ_a.diagonal().real
            h_diag_b = fock_vir_b.diagonal().real[:, None] - fock_occ_b.diagonal().real
            h_diag = np.hstack((h_diag_a.reshape(-1), h_diag_b.reshape(-1)))

            # The result of h_op is the displacement vector.
            def h_op(x):
                x1a = x[:n_vir_a * self.n_a].reshape(n_vir_a, self.n_a)  # create a trial vector for alpha orbitals
                x1b = x[n_vir_a * self.n_a:].reshape(n_vir_b, self.n_b)  # create a trial vector for beta orbitals
                x2a = np.einsum('pr,rq->pq', fock_vir_a, x1a)  # summation from fock_vir_a * x1a
                x2a -= np.einsum('sq,ps->pq', fock_occ_a, x1a)  # subtract summation from fock_occ_a * x1a
                x2b = np.einsum('pr,rq->pq', fock_vir_b, x1b)  # summation from fock_vir_b * x1b
                x2b -= np.einsum('sq,ps->pq', fock_occ_b, x1b)  # subtract summation from fock_occ_b * x1b

                d1a = reduce(np.dot, (vir_a_orb, x1a, occ_a_orb.conj().T))  # diagonalise x1a
                d1b = reduce(np.dot, (vir_b_orb, x1b, occ_b_orb.conj().T))  # diagonalise x1b
                dm1 = np.array((d1a + d1a.conj().T, d1b + d1b.conj().T))  # create a density matrix from d1a and d1b
                v1 = -scf.hf.get_jk(self.molecule, dm1, hermi=1)[
                    1]  # calculate the exchange integrals in the case where dm1 is used as a density matrix
                # add the matrix product from the virtual alpha orbitals (conjugate transpose),
                # the exchange integrals, and the occupied alpha orbitals to the final trial vector
                x2a += reduce(np.dot, (vir_a_orb.conj().T, v1[0], occ_a_orb))
                # add the matrix product from the virtual beta orbitals (conjugate transpose),
                # the exchange integrals, and the occupied beta orbitals to the final trial vector
                x2b += reduce(np.dot, (vir_b_orb.conj().T, v1[1], occ_b_orb))
                x2 = np.hstack((x2a.ravel(), x2b.ravel()))  # merge x2a and x2b together to create the trial vector
                return x2
            return g, h_op, h_diag

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
            def uniq_variable_indices(mo_occ):
                occ_indx_a = mo_occ > 0  # indices of occupied alpha orbitals
                occ_indx_b = mo_occ == 2  # indices of occupied beta orbitals
                # indices of virtual (unoccupied) alpha orbitals, done with bitwise operator: ~ (negation)
                # indices of virtual (unoccupied) beta orbitals, done with bitwise operator: ~ (negation)
                vir_indx_a = ~occ_indx_a
                vir_indx_b = ~occ_indx_b
                # & and | are bitwise operators for 'and' and 'or'
                # each bit position is the result of the logical 'and' or 'or' of the bits
                # in the corresponding position of the operands
                # determine the unique variable indices, by use of bitwise operators
                unique = (vir_indx_a[:, None] & occ_indx_a) | (vir_indx_b[:, None] & occ_indx_b)
                return unique

            # put the unique variables in a new matrix used later to create a rotation matrix.
            def unpack_uniq_variables(dx, mo_occ):
                nmo = len(mo_occ)
                idx = uniq_variable_indices(mo_occ)
                x1 = np.zeros((nmo, nmo), dtype=dx.dtype)
                x1[idx] = dx
                return x1 - x1.conj().T

            # A function to apply a rotation on the given coefficients
            def rotate_mo(mo_coeff, mo_occ, dx):
                dr = unpack_uniq_variables(dx, mo_occ)
                u = la.expm(dr)  # computes the matrix exponential
                return np.dot(mo_coeff, u)

            x0 = np.zeros_like(g)  # like returns a matrix of the same shape as the argument given
            x0[g != 0] = 1. / hdiag[g != 0]  # create initial guess for davidson solver
            # use the davidson solver to find the eigenvalues and eigenvectors
            # needed to determine an internal instability
            e, v = lib.davidson(hessian_x, x0, precond, tol=1e-4)
            if e < -1e-5:  # this points towards an internal instability
                total_orbitals = self.get_ovlp().shape[0]  # total number of basis functions
                n_vir_a = int(total_orbitals - self.n_a)  # number of virtual (unoccupied) alpha orbitals
                mo_a = np.zeros(total_orbitals)
                mo_b = np.zeros(total_orbitals)
                # create representation of alpha orbitals by adding an electron (a 1) to each occupied alpha orbital
                # create representation of beta orbitals by adding an electron (b 1) to each occupied beta orbital
                for j in range(self.n_a):
                    mo_a[j] = 1
                for k in range(self.n_b):
                    mo_b[k] = 1
                    # create new orbitals by rotating the old ones
                new_orbitals = (rotate_mo(coeff_a, mo_a, v[:self.n_a * n_vir_a]),
                                rotate_mo(coeff_b, mo_b, v[self.n_b * n_vir_a:]))
                print("There is an instability in the UHF wave function.")
            else:  # in the case where no instability is present
                new_orbitals = (coeff_a, coeff_b)
                print('There is no instability in the UHF wave function.')
            return new_orbitals
        return internal_stability()
