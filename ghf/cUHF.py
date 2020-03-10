"""
Constrained unrestricted Hartree Fock
=====================================
This class is used to calculate the ROHF energy for a given molecule and the number of electrons of that molecule,
using a constraied version of unrestricted Hartree Fock.
Several options are available to make sure you get the lowest energy from your calculation, as well as some useful
functions to get intermediate values such as MO coefficients, density and fock matrices.
"""

from ghf.SCF_functions import *
from pyscf import *
import scipy.linalg as la


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
        self.energy = None
        self.energy_list = {}
        self.mo = None
        self.last_dens = None
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

    def scf(self, convergence=1e-12):
        """
        Performs a self consistent field calculation to find the lowest UHF energy.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The scf energy, number of iterations, the mo coefficients, the last density and the last fock matrices
        """
        # calculate the transformation matrix
        s_12 = trans_matrix(self.get_ovlp())
        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        # create guess density matrix from core guess, separate for alpha and beta and put them into an array
        initial_guess = s_12.conj().T @ self.get_one_e() @ s_12
        guess_density_a = density_matrix(initial_guess, self.n_a, s_12)
        guess_density_b = density_matrix(initial_guess, self.n_b, s_12)

        # Store the density matrices in an array
        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        # Create a second array to study the difference in energy between iterations.
        energies = [0.0]
        delta_e = []

        def constrained_fock(fock_a, fock_b):
            fock_cs = (fock_a + fock_b) / 2
            delta_uhf = fock_cs - fock_a

            core = self.n_a - self.n_b
            open = self.n_b

            lambda_ij_cv_a = delta_uhf[:core, core+open:]
            lambda_ij_cv_b = delta_uhf[:core, core:]

            lambda_ij_a = np.zeros_like(delta_uhf)
            lambda_ij_a[:core, core+open:] = lambda_ij_cv_a
            lambda_ij_a[core+open:, :core] = lambda_ij_cv_a.T

            lambda_ij_b = np.zeros_like(delta_uhf)
            lambda_ij_b[:core, core:] = lambda_ij_cv_b
            lambda_ij_b[core:, :core] = lambda_ij_cv_b.T

            delta_cuhf_a = delta_uhf - lambda_ij_a
            delta_cuhf_b = delta_uhf - lambda_ij_b

            cfa = fock_cs - delta_cuhf_a
            cfb = fock_cs + delta_cuhf_b

            return cfa, cfb

        # create an iteration procedure
        def iteration(n_i):
            self.density_list[n_i] = [densities_a[-1], densities_b[-1]]
            # create a fock matrix for alpha from last alpha density
            fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], self.get_one_e(), self.get_two_e())
            # create a fock matrix for beta from last beta density
            fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], self.get_one_e(), self.get_two_e())

            c_fock_a, c_fock_b = constrained_fock(fock_a, fock_b)
            self.fock_list[n_i] = [c_fock_a, c_fock_b]

            # calculate the improved scf energy and add it to the array
            energies.append(uhf_scf_energy(densities_a[-1], densities_b[-1], c_fock_a, c_fock_b, self.get_one_e()))
            self.energy_list[n_i] = uhf_scf_energy(densities_a[-1], densities_b[-1], c_fock_a, c_fock_b,
                                                   self.get_one_e()) + self.nuc_rep()
            # calculate the energy difference and add it to the delta_E array
            delta_e.append(energies[-1] - energies[-2])

            # orthogonalize both fock matrices
            fock_orth_a = s_12.conj().T.dot(c_fock_a).dot(s_12)
            fock_orth_b = s_12.conj().T.dot(c_fock_b).dot(s_12)
            self.fock_orth_list[n_i] = [fock_orth_a, fock_orth_b]

            # create a new alpha and beta density matrix
            new_density_a = density_matrix(fock_orth_a, self.n_a, s_12)
            new_density_b = density_matrix(fock_orth_b, self.n_b, s_12)

            # put the density matrices in their respective arrays
            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        i = 1
        iteration(i)
        while abs(delta_e[-1]) >= convergence:
            iteration(i)
            i += 1
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure, both for alpha and beta
        def last_dens():
            return densities_a[-1], densities_b[-1]
        self.last_dens = last_dens()

        def last_fock():
            last_fock_a = uhf_fock_matrix(densities_a[-2], densities_b[-2], self.get_one_e(), self.get_two_e())
            last_fock_b = uhf_fock_matrix(densities_b[-2], densities_a[-2], self.get_one_e(), self.get_two_e())
            last_c_fock_a, last_c_fock_b = constrained_fock(last_fock_a, last_fock_b)
            return last_c_fock_a, last_c_fock_b
        self.last_fock = last_fock()

        def get_mo():
            # Calculate the last fock matrix for both alpha and beta
            fock_a = uhf_fock_matrix(densities_a[-2], densities_b[-2], self.get_one_e(), self.get_two_e())
            fock_b = uhf_fock_matrix(densities_b[-2], densities_a[-2], self.get_one_e(), self.get_two_e())

            c_fock_a, c_fock_b = constrained_fock(fock_a, fock_b)

            # orthogonalize both fock matrices
            fock_a_ = s_12.T.dot(c_fock_a).dot(s_12)
            fock_b_ = s_12.T.dot(c_fock_b).dot(s_12)

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

    def get_scf_solution(self, convergence=1e-12):
        """
        Prints the number of iterations and the converged scf energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The converged scf energy.
        """
        self.scf(convergence=convergence)
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
