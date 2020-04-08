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
import collections as c


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

    def diis(self, initial_guess=None, convergence=1e-12):
        """
        When needed, DIIS can be used to speed up the UHF calculations by reducing the needed iterations.

        :param initial_guess: Initial guess for the scf procedure. None specified: core Hamiltonian.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: scf energy, number of iterations, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        # create guess density matrix from core guess, separate for alpha and beta and put them into an array
        if initial_guess is None:
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

        def uhf_fock(density_matrix_1, density_matrix_2):
            """
            - calculate a fock matrix from a given alpha and beta density matrix
            - fock alpha if 1 = alpha and 2 = beta and vice versa
            - input is the density matrix for alpha and beta, a one electron matrix and a two electron tensor.
            """
            j_1 = np.einsum('pqrs,rs->pq', self.get_two_e(), density_matrix_1)
            j_2 = np.einsum('pqrs,rs->pq', self.get_two_e(), density_matrix_2)
            k_1 = np.einsum('prqs,rs->pq', self.get_two_e(), density_matrix_1)
            fock = self.get_one_e() + (j_1 + j_2) - k_1
            return fock

        def residual(density, fock):
            """
            Function that calculates the error matrix for the DIIS algorithm
            :param density: density matrix
            :param fock: fock matrix
            :return: a value that should be zero and a fock matrix
            """
            return s_12 @ (fock @ density @ self.get_ovlp() - self.get_ovlp() @ density @ fock) @ s_12.conj().T

        # Create a list to store the errors
        # create a list to store the fock matrices
        # deque is used as a high performance equivalent of an array
        error_list_a = c.deque(maxlen=6)
        error_list_b = c.deque(maxlen=6)

        fock_list_a = c.deque(maxlen=6)
        fock_list_b = c.deque(maxlen=6)

        def diis_fock(focks, residuals):
            # Dimensions
            dim = len(focks) + 1

            # Create the empty B matrix
            b = np.empty((dim, dim))
            b[-1, :] = -1
            b[:, -1] = -1
            b[-1, -1] = 0

            # Fill the B matrix: ei * ej, with e the errors
            for k in range(len(focks)):
                for l in range(len(focks)):
                    b[k, l] = np.einsum('kl,kl->', residuals[k], residuals[l])

            # Create the residual vector
            res_vec = np.zeros(dim)
            res_vec[-1] = -1

            # Solve the pulay equation to get the coefficients
            coeff = np.linalg.solve(b, res_vec)

            # Create a fock as a linear combination of previous focks
            fock = np.zeros(focks[0].shape)
            for x in range(coeff.shape[0] - 1):
                fock += coeff[x] * focks[x]
            return fock

        # Create the necessary arrays to perform an iterative diis procedure
        densities_diis_a = [guess_density_a]
        densities_diis_b = [guess_density_b]
        energies_diis = [0.0]
        delta_e_diis = []

        def iteration_diis(n_i):
            self.density_list[n_i] = [densities_diis_a[-1], densities_diis_b[-1]]
            # Create the alpha and beta fock matrices
            f_a = uhf_fock(densities_diis_a[-1], densities_diis_b[-1])
            f_b = uhf_fock(densities_diis_b[-1], densities_diis_a[-1])
            self.fock_list[n_i] = [f_a, f_b]

            # Calculate the residuals from both
            resid_a = residual(densities_diis_a[-1], f_a)
            resid_b = residual(densities_diis_b[-1], f_b)

            # Add everything to their respective list
            fock_list_a.append(f_a)
            fock_list_b.append(f_b)
            error_list_a.append(resid_a)
            error_list_b.append(resid_b)

            # Calculate the energy and energy difference
            energies_diis.append(uhf_scf_energy(densities_diis_a[-1], densities_diis_b[-1], f_a, f_b, self.get_one_e()))
            delta_e_diis.append(energies_diis[-1] - energies_diis[-2])

            # Starting at two iterations, use the DIIS acceleration
            if n_i >= 2:
                f_a = diis_fock(fock_list_a, error_list_a)
                f_b = diis_fock(fock_list_b, error_list_b)

            # Orthogonalise the fock matrices
            f_orth_a = s_12.conj().T @ f_a @ s_12
            f_orth_b = s_12.conj().T @ f_b @ s_12
            self.fock_orth_list[n_i] = [f_orth_a, f_orth_b]

            # calculate new mo coefficients
            mo_a = mo_coeff(f_orth_a)
            mo_b = mo_coeff(f_orth_b)
            self.coeff_list[n_i] = [mo_a, mo_b]

            # constrain the mo's
            c_mo_a, c_mo_b = constrain_mo(mo_a, mo_b)
            self.constrained_coeff_list[n_i] = [c_mo_a, c_mo_b]

            # Calculate the new density matrices
            new_density_a = dens(c_mo_a, self.n_a)
            new_density_b = dens(c_mo_b, self.n_b)

            # Add them to their respective lists
            densities_diis_a.append(new_density_a)
            densities_diis_b.append(new_density_b)

        # Let the process iterate until the energy difference is smaller than 10e-12
        i = 1
        iteration_diis(i)
        while abs(delta_e_diis[-1]) >= convergence:
            iteration_diis(i)
            i += 1

        self.iterations = i

        # a function that gives the last density matrix of the scf procedure
        def last_dens():
            return densities_diis_a[-1], densities_diis_b[-1]

        self.last_dens = last_dens()

        # a function that gives the last Fock matrix of the scf procedure
        def last_fock():
            last_fock_a = fock_list_a[-1]
            last_fock_b = fock_list_b[-1]
            return last_fock_a, last_fock_b

        self.last_fock = last_fock()

        # A function that returns the converged mo coefficients
        def get_mo():
            return self.constrained_coeff_list[self.iterations-1]

        self.mo = get_mo()

        # calculate the total energy, taking nuclear repulsion into account
        scf_e = energies_diis[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution_diis(self, guess=None, convergence=1e-12):
        """
        Prints the number of iterations and the converged diis energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :param guess: The initial guess. If none is specified, core Hamiltonian.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The converged diis energy.
        """
        self.diis(guess, convergence=convergence)
        s_values = spin(self.n_a, self.n_b, CUHF.get_mo_coeff(self)[0], CUHF.get_mo_coeff(self)[1], self.get_ovlp())

        print("Number of iterations: " + str(self.iterations))
        print("Converged SCF energy in Hartree: " + str(self.energy) + " (Constrained UHF, DIIS)")
        print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) + ", Multiplicity = " + str(s_values[2]))
        return self.energy
