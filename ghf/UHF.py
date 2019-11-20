"""
Unrestricted Hartree Fock, by means of SCF procedure
====================================================
This class is used to calculate the UHF energy for a given molecule and the number of electrons of that molecule.
Several options are available to make sure you get the lowest energy from your calculation, as well as some usefull
functions to get intermediate values such as MO coefficients, density and fock matrices.
"""

from ghf.SCF_functions import *
from functools import reduce
from pyscf import *
import scipy.linalg as la
import collections as c


class UHF:
    """
    Calculate UHF energy.
    ---------------------
    Input is a molecule and the number of electrons.

    Molecules are made in pySCF and calculations are performed as follows, eg.:
    The following snippet prints and returns UHF energy of h_3
    and the number of iterations needed to get this value.

    For a normal scf calculation your input looks like the following example:

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> x = UHF(h3, 3)
    >>> x.get_scf_solution()
    Number of iterations: 62
    Converged SCF energy in Hartree: -1.5062743202681235 (UHF)
    <S^2> = 0.7735672504295973, <S_z> = 0.5, Multiplicity = 2.023430009098014
    """
    def __init__(self, molecule, number_of_electrons):
        """
        Initiate the UHF instance by specifying the molecule in question with pyscf and the total number of electrons.

        :param molecule: The molecule on which to perform the calculations
        :param number_of_electrons: The total number of electrons in the system
        """
        # Set the molecule, the integrals, the mo coefficients, the number of alpha and beta
        # electrons, the energy, the last fock and the last density matrices as class parameters.
        # If the number of electrons is even, an equal amount of alpha and beta electrons are set.
        # If there's an uneven number of electrons, there will be one more alpha than beta electron.
        self.molecule = molecule
        self.integrals = get_integrals(molecule)
        self.energy = None
        self.mo = None
        self.last_dens = None
        self.last_fock = None
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

    def scf(self, initial_guess=None):
        """
        Performs a self consistent field calculation to find the lowest UHF energy.

        :param initial_guess: A tuple of an alpha and beta guess matrix. If none, the core hamiltonian will be used.
        :return: The scf energy, number of iterations, the mo coefficients, the last density and the last fock matrices
        """
        # calculate the transformation matrix
        s_12 = trans_matrix(self.get_ovlp())
        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        # create guess density matrix from core guess, separate for alpha and beta and put them into an array
        if initial_guess is None:
            initial_guess = s_12.T @ self.get_one_e() @ s_12
            guess_density_a = density_matrix(initial_guess, self.n_a, s_12)
            guess_density_b = density_matrix(initial_guess, self.n_b, s_12)
        else:
            # Make the coefficients orthogonal in the correct basis.
            coeff_a = s_12 @ initial_guess[0]
            coeff_b = s_12 @ initial_guess[1]

            # Get C_alpha and C_beta
            coeff_r_a = coeff_a[:, 0:self.n_a]
            coeff_r_b = coeff_b[:, 0:self.n_b]

            # Create the guess density matrices from the given coefficients
            guess_density_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a)
            guess_density_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b)

        # Store the density matrices in an array
        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        # Calculate the initial electronic energy and create an array to store the energy values.
        # Also, create a second array to study the difference in energy between iterations.
        electronic_e = uhf_scf_energy(guess_density_a, guess_density_b, self.get_one_e(), self.get_one_e(),
                                      self.get_one_e())
        energies = [electronic_e]
        delta_e = []

        # create an iteration procedure
        def iteration():
            # create a fock matrix for alpha from last alpha density
            fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], self.get_one_e(), self.get_two_e())
            # create a fock matrix for beta from last beta density
            fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], self.get_one_e(), self.get_two_e())

            # orthogonalize both fock matrices
            fock_orth_a = s_12.T.dot(fock_a).dot(s_12)
            fock_orth_b = s_12.T.dot(fock_b).dot(s_12)

            # create a new alpha and beta density matrix
            new_density_a = density_matrix(fock_orth_a, self.n_a, s_12)
            new_density_b = density_matrix(fock_orth_b, self.n_b, s_12)

            # put the density matrices in their respective arrays
            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

            # calculate the improved scf energy and add it to the array
            energies.append(uhf_scf_energy(new_density_a, new_density_b, fock_a, fock_b, self.get_one_e()))
            # calculate the energy difference and add it to the delta_E array
            delta_e.append(energies[-1] - energies[-2])

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        iteration()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration()
            i += 1
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure, both for alpha and beta
        def last_dens():
            return densities_a[-1], densities_b[-1]
        self.last_dens = last_dens()

        def last_fock():
            last_fock_a = uhf_fock_matrix(densities_a[-2], densities_b[-2], self.get_one_e(), self.get_two_e())
            last_fock_b = uhf_fock_matrix(densities_b[-2], densities_a[-2], self.get_one_e(), self.get_two_e())
            return s_12.T @ last_fock_a @ s_12, s_12.T @ last_fock_b @ s_12
        self.last_fock = last_fock()

        def get_mo():
            # Calculate the last fock matrix for both alpha and beta
            fock_a = uhf_fock_matrix(densities_a[-2], densities_b[-2], self.get_one_e(), self.get_two_e())
            fock_b = uhf_fock_matrix(densities_b[-2], densities_a[-2], self.get_one_e(), self.get_two_e())

            # orthogonalize both fock matrices
            fock_a_ = s_12.T.dot(fock_a).dot(s_12)
            fock_b_ = s_12.T.dot(fock_b).dot(s_12)

            # Calculate the eigenvectors of both Fock matrices
            # Orthogonalise both sets of eigenvectors to get the mo coefficients
            val_a, vec_a = la.eigh(fock_a_)
            val_b, vec_b = la.eigh(fock_b_)
            coefficient_a = s_12 @ vec_a
            coefficient_b = s_12 @ vec_b
            return coefficient_a, coefficient_b
        self.mo = get_mo()

        # Calculate the final scf energy (electronic + nuclear repulsion)
        scf_e = energies[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution(self, guess=None):
        """
        Prints the number of iterations and the converged scf energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :return: The converged scf energy.
        """
        scf_values = self.scf(guess)
        s_values = spin(self.n_a, self.n_b, UHF.get_mo_coeff(self)[0], UHF.get_mo_coeff(self)[1], self.get_ovlp())
        print("Number of iterations: " + str(scf_values[1]))
        print("Converged SCF energy in Hartree: " + str(scf_values[0]) + " (UHF)")
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

    def extra_electron_guess(self):
        """
        This method adds two electrons to the system in order to get coefficients that can be used as a better guess
        for the scf procedure. This essentially forces the system into it's <S_z> = 0 state.

        To perform a calculation with this method, you will have to work as follows:

        >>> h4 = gto.M(atom = 'h 0 0 0; h 1 0 0; h 0 1 0; h 1 1 0' , spin = 2, basis = 'cc-pvdz')
        >>> x = UHF(h4, 4)
        >>> guess = x.extra_electron_guess()
        >>> x.get_scf_solution(guess)
        Number of iterations: 74
        Converged SCF energy in Hartree: -2.0210882477030547 (UHF)
        <S^2> = 1.0565277001056579, <S_z> = 0.0, Multiplicity = 2.2860688529487976

        :return: A new guess matrix to use for the scf procedure.
        """
        # Add two electrons to the molecule and build the new test system.
        self.molecule.nelectron = self.n_a + self.n_b + 2
        self.molecule.build()

        # calculate the integrals for the new test system (_t)
        overlap_t = get_integrals(self.molecule)[0]
        one_electron_t = get_integrals(self.molecule)[1]
        two_electron_t = get_integrals(self.molecule)[2]

        # Calculate the orthogonalisation matrix and the core guess for the new test system (_t)
        s_12_t = trans_matrix(overlap_t)
        core_guess_t = s_12_t.T.dot(one_electron_t).dot(s_12_t)

        # Calculate a guess density for the test system, both for alpha and beta.
        # add the two extra electrons to alpha
        # add the density matrices to respective arrays
        guess_density_a_t = density_matrix(core_guess_t, self.n_a + 2, s_12_t)
        guess_density_b_t = density_matrix(core_guess_t, self.n_b, s_12_t)
        densities_a = [guess_density_a_t]
        densities_b = [guess_density_b_t]

        # create an array to check the differences between density matrices in between iterations
        delta_dens = []

        # create an iteration procedure, based on the densities, for the test system
        def iteration_t():
            # create a fock matrix for alpha from last alpha density, and the integrals from the test system
            # create a fock matrix for beta from last beta density, and the integrals from the test system
            fock_matrix_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron_t, two_electron_t)
            fock_matrix_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron_t, two_electron_t)

            # orthogonalize both fock matrices
            orth_fock_a = s_12_t.T.dot(fock_matrix_a).dot(s_12_t)
            orth_fock_b = s_12_t.T.dot(fock_matrix_b).dot(s_12_t)

            # create a new alpha density matrix, with the test system's orthogonalisation matrix
            # create a new beta density matrix, with the test system's orthogonalisation matrix
            # add the densities to an array
            new_density_a = density_matrix(orth_fock_a, self.n_a + 2, s_12_t)
            new_density_b = density_matrix(orth_fock_b, self.n_b, s_12_t)
            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

            # calculate the difference between the last two density matrices
            delta_dens.append(np.sum(densities_a[-1] + densities_b[-1]) - np.sum(densities_a[-2] + densities_b[-2]))

        # start and continue the iteration process as long as the difference between densities is larger than 1e-12
        iteration_t()
        while abs(delta_dens[-1]) >= 1e-12:
            iteration_t()

        # Now that the test system has converged, we use the last calculated density matrices
        # to calculate new orbital coefficients. These coefficients can then be used to start a new scf procedure.
        # for both alpha and beta
        fock_a = uhf_fock_matrix(densities_a[-2], densities_b[-2], one_electron_t, two_electron_t)
        fock_b = uhf_fock_matrix(densities_b[-2], densities_a[-2], one_electron_t, two_electron_t)

        # orthogonalize both fock matrices
        fock_orth_a = s_12_t.T @ fock_a @ s_12_t
        fock_orth_b = s_12_t.T @ fock_b @ s_12_t

        val_a, vec_a = la.eigh(fock_orth_a)
        val_b, vec_b = la.eigh(fock_orth_b)

        coeff_a = vec_a @ s_12_t
        coeff_b = vec_b @ s_12_t

        # Remove the added electrons from the system and reset the system to the original molecule.
        self.molecule.nelectron = self.n_a + self.n_b
        self.molecule.build()

        return coeff_a, coeff_b

    def stability(self):
        """
        Performing a stability analysis checks whether or not the wave function is stable, by checking the lowest eigen-
        value of the Hessian matrix. If there's an instability, the MO's will be rotated in the direction
        of the lowest eigenvalue. These new MO's can then be used to start a new scf procedure.

        To perform a stability analysis, use the following syntax:

        >>> h4 = gto.M(atom = 'h 0 0 0; h 1 0 0; h 0 1 0; h 1 1 0' , spin = 2, basis = 'cc-pvdz')
        >>> x = UHF(h4, 4)
        >>> guess = x.stability()
        >>> x.get_scf_solution(guess)
        There is an internal instability in the UHF wave function.
        Number of iterations: 78
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
                x1a = x[:n_vir_a * self.n_a].reshape(n_vir_a, self.n_b)  # create a trial vector for alpha orbitals
                x1b = x[n_vir_a * self.n_a:].reshape(n_vir_b, self.n_b)  # create a trial vector for beta orbitals
                x2a = np.einsum('pr,rq->pq', fock_vir_a, x1a)  # summation from fock_vir_a * x1a
                x2a -= np.einsum('sq,ps->pq', fock_occ_a, x1a)  # subtract summation from fock_occ_a * x1a
                x2b = np.einsum('pr,rq->pq', fock_vir_b, x1b)  # summation from fock_vir_b * x1b
                x2b -= np.einsum('sq,ps->pq', fock_occ_b, x1b)  # subtract summation from fock_occ_b * x1b

                d1a = reduce(np.dot, (vir_a_orb, x1a, occ_a_orb.conj().T))  # diagonolise x1a
                d1b = reduce(np.dot, (vir_b_orb, x1b, occ_b_orb.conj().T))  # diagonolise x1b
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
                print("There is an internal instability in the UHF wave function.")
            else:  # in the case where no instability is present
                new_orbitals = (coeff_a, coeff_b)
                print('There is no internal instability in the UHF wave function.')
            return new_orbitals
        return internal_stability()

    def diis(self, initial_guess=None):
        """
        When needed, DIIS can be used to speed up the UHF calculations by reducing the needed iterations.

        :return: scf energy, number of iterations, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        # create guess density matrix from core guess, separate for alpha and beta and put them into an array
        if initial_guess is None:
            initial_guess = s_12.T @ self.get_one_e() @ s_12
            guess_density_a = density_matrix(initial_guess, self.n_a, s_12)
            guess_density_b = density_matrix(initial_guess, self.n_b, s_12)
        else:
            # Make the coefficients orthogonal in the correct basis.
            coeff_a = s_12 @ initial_guess[0]
            coeff_b = s_12 @ initial_guess[1]

            # Get C_alpha and C_beta
            coeff_r_a = coeff_a[:, 0:self.n_a]
            coeff_r_b = coeff_b[:, 0:self.n_b]

            # Create the guess density matrices from the given coefficients
            guess_density_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a)
            guess_density_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b)

        def uhf_fock(density_matrix_1, density_matrix_2):
            """
            - calculate a fock matrix from a given alpha and beta density matrix
            - fock alpha if 1 = alpha and 2 = beta and vice versa
            - input is the density matrix for alpha and beta, a one electron matrix and a two electron tensor.
            """
            jk_integrals = self.get_two_e() - self.get_two_e().transpose(0, 2, 1, 3)  # exchange and coulomb integrals
            jk_a = np.einsum('kl,ijkl->ij', density_matrix_1,
                             jk_integrals)  # double summation of density matrix * (coulomb - exchange)
            j_b = np.einsum('kl, ijkl->ij', density_matrix_2,
                            self.get_two_e())  # double summation of density matrix * exchange
            return self.get_one_e() + jk_a + j_b

        def error_matrix(density_a, density_b):
            """
            Calculates the error matrices for alpha and beta
            :param density_a: alpha density matrix
            :param density_b: beta density matrix
            :return: error matrices for alpha and beta
            """
            fock_a = uhf_fock(density_a, density_b)
            fock_b = uhf_fock(density_b, density_a)
            a = fock_a @ density_a @ self.get_ovlp() - self.get_ovlp() @ density_a @ fock_a
            b = fock_b @ density_b @ self.get_ovlp() - self.get_ovlp() @ density_b @ fock_b
            return a, b

        # Create a list to store the errors
        # create a list to store the fock matrices
        # deque is used as a high performance equivalent of an array
        error_list_a = c.deque(maxlen=6)
        error_list_b = c.deque(maxlen=6)

        fock_list_a = c.deque(maxlen=6)
        fock_list_b = c.deque(maxlen=6)

        def build_b_matrix(density_a, density_b):
            """
            Build the B matrix for both the alpha and beta orbitals
            :param density_a: alpha density matrix
            :param density_b: beta density matrix
            :return: B matrix for alpha and beta orbitals
            """
            # Get the error and fock matrices
            # add them to their respective lists
            error_a, error_b = error_matrix(density_a, density_b)
            error_list_a.append(error_a)
            error_list_b.append(error_b)

            fock_a = uhf_fock(density_a, density_b)
            fock_b = uhf_fock(density_b, density_a)

            fock_list_a.append(fock_a)
            fock_list_b.append(fock_b)

            # Determine the dimensions of the B matrix
            m = len(error_list_a)
            print(m)
            o = len(error_list_b)
            print(o)
            n = np.shape(self.get_ovlp())[0]

            # Create and return the B-matrix: B_ij = e_i * e_j (product of error matrices)
            error_a = np.array(list(error_list_a) * m).reshape(m, m, n, n)
            error_b = np.array(list(error_list_b) * m).reshape(o, o, n, n)
            return np.einsum('ijkl, jilk->ij', error_a, error_a), np.einsum('ijkl, jilk->ij', error_b, error_b)

        def coefficients(density_a, density_b):
            # Calculate the B matrix with the function above
            b_matrix_a = build_b_matrix(density_a, density_b)[0]
            b_matrix_b = build_b_matrix(density_a, density_b)[1]
            print(b_matrix_a, b_matrix_b)

            # Determine matrix dimensions
            length_a = len(fock_list_a)
            length_b = len(fock_list_b)
            print(length_a, length_b)

            # Create the needed matrices for the linear equation that results in the coefficients
            p = np.full((1, length_a), -1)
            q = np.full((length_a + 1, 1), -1)

            # alpha coefficients
            a = np.append(b_matrix_a, p, axis=0)
            a = np.append(a, q, axis=1)
            a[-1][-1] = 0
            b = np.zeros((length_a + 1, 1))
            b[length_a][0] = -1
            print(a, b)

            # Create the needed matrices for the linear equation that results in the coefficients
            y = np.full((1, length_b), -1)
            z = np.full((length_b + 1, 1), -1)

            # beta coefficients
            e = np.append(b_matrix_b, y, axis=0)
            e = np.append(e, z, axis=1)
            e[-1][-1] = 0
            f = np.zeros((length_b + 1, 1))
            f[length_b][0] = -1
            print(e, f)

            # Solve the linear equation (using a scipy solver)
            x_a = la.solve(a, b)
            x_b = la.solve(e, f)
            return np.array(x_a[:-1]), np.array(x_b[:-1])

        def new_fock_matrix(density_a, density_b):
            """
            Creates new fock matrices
            :param density_a: alpha density matrix
            :param density_b: beta density matrix
            :return:
            """
            c_a = coefficients(density_a, density_b)[0]
            c_b = coefficients(density_a, density_b)[1]
            f_a = c_a[0].reshape((len(fock_list_a), 1, 1)) * fock_list_a
            f_b = c_b[1].reshape((len(fock_list_b), 1, 1)) * fock_list_b
            return np.sum(f_a, 0), np.sum(f_b, 0)

        # Calculate the guess electronic energy
        # calculate the initial energy, using the guess
        electronic_e = uhf_scf_energy(guess_density_a, guess_density_b, self.get_one_e(), self.get_one_e(),
                                      self.get_one_e())

        # Create the necessary arrays to perform an iterative diis procedure
        densities_diis_a = [guess_density_a]
        densities_diis_b = [guess_density_b]
        energies_diis = [electronic_e]
        delta_e_diis = []

        def iteration_diis():
            f_a = new_fock_matrix(densities_diis_a[-1], densities_diis_b[-1])[0]
            f_b = new_fock_matrix(densities_diis_a[-1], densities_diis_b[-1])[1]

            f_orth_a = s_12.T @ f_a @ s_12
            f_orth_b = s_12.T @ f_b @ s_12

            new_density_a = density_matrix(f_orth_a, self.n_a, s_12)
            new_density_b = density_matrix(f_orth_b, self.n_b, s_12)

            densities_diis_a.append(new_density_a)
            densities_diis_b.append(new_density_b)

            energies_diis.append(uhf_scf_energy(densities_diis_a[-1], densities_diis_b[-1], f_a, f_b, self.get_one_e()))
            delta_e_diis.append(energies_diis[-1] - energies_diis[-2])

        # Let the process iterate until the energy difference is smaller than 10e-12
        iteration_diis()
        i = 1
        while abs(delta_e_diis[-1]) >= 1e-12:
            iteration_diis()
            i += 1

        # a function that gives the last density matrix of the scf procedure
        def last_dens():
            return densities_diis_a[-1], densities_diis_b[-1]

        self.last_dens = last_dens()

        # a function that gives the last Fock matrix of the scf procedure
        def last_fock():
            last_fock_a = uhf_fock_matrix(densities_diis_a[-2], densities_diis_b[-2],
                                          self.get_one_e(), self.get_two_e())
            last_fock_b = uhf_fock_matrix(densities_diis_b[-2], densities_diis_a[-2],
                                          self.get_one_e(), self.get_two_e())
            return s_12.T @ last_fock_a @ s_12, s_12.T @ last_fock_b @ s_12

        self.last_fock = last_fock()

        # A function that returns the converged mo coefficients
        def get_mo():
            # Calculate the last fock matrix for both alpha and beta
            fock_a = uhf_fock_matrix(densities_diis_a[-2], densities_diis_b[-2], self.get_one_e(), self.get_two_e())
            fock_b = uhf_fock_matrix(densities_diis_b[-2], densities_diis_a[-2], self.get_one_e(), self.get_two_e())

            # orthogonalize both fock matrices
            fock_a_ = s_12.T.dot(fock_a).dot(s_12)
            fock_b_ = s_12.T.dot(fock_b).dot(s_12)

            # Calculate the eigenvectors of both Fock matrices
            # Orthogonalise both sets of eigenvectors to get the mo coefficients
            val_a, vec_a = la.eigh(fock_a_)
            val_b, vec_b = la.eigh(fock_b_)
            coefficient_a = s_12 @ vec_a
            coefficient_b = s_12 @ vec_b
            return coefficient_a, coefficient_b

        self.mo = get_mo()

        # calculate the total energy, taking nuclear repulsion into account
        scf_e = energies_diis[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution_diis(self, guess=None):
        """
        Prints the number of iterations and the converged diis energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :return: The converged diis energy.
        """
        scf_values = self.diis(guess)
        s_values = spin(self.n_a, self.n_b, UHF.get_mo_coeff(self)[0], UHF.get_mo_coeff(self)[1], self.get_ovlp())
        print("Number of iterations: " + str(scf_values[1]))
        print("Converged SCF energy in Hartree: " + str(scf_values[0]) + " (UHF)")
        print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) + ", Multiplicity = " + str(s_values[2]))
        return self.energy
