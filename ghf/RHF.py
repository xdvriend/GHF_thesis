"""
Restricted Hartree Fock, by means of SCF procedure
==================================================
This class is used to calculate the RHF energy of a given molecule and the number of electrons.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)
"""

from ghf.SCF_functions import *
import collections as c
from scipy import linalg as lina


class RHF:
    """
        calculate RHF energy.
        ---------------------
        Input is a molecule and the number of electrons.

        Molecules are made in pySCF and calculations are performed as follows, eg.:
        The following snippet prints and returns RHF energy of h_2
        and the number of iterations needed to get this value.

        >>> h_2 = gto.M(atom = 'h 0 0 0; h 0 0 1', spin = 0, basis = 'sto-3g')
        >>> x = RHF(h_2, 2)
        >>> x.get_scf_solution()
        Number of iterations: 2
        Converged SCF energy in Hartree: -1.0661086493179357 (RHF)
        """
    def __init__(self, molecule, number_of_electrons, int_method='pyscf'):
        """
        Initiate the RHF instance by specifying the molecule in question with pyscf and the total number of electrons.

        :param molecule: The molecule on which to perform the calculations
        :param number_of_electrons: The total number of electrons in the system
        :param int_method: method to calculate the integrals. pyscf and psi4 are supported.
        """
        self.molecule = molecule
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
        self.iterations = None
        # For closed shell calculations the number of electrons should be a multiple of 2.
        # If this is not the case, a message is printed telling you to adjust the parameter.
        if number_of_electrons % 2 == 0:
            self.occupied = int(number_of_electrons/2)
        else:
            raise Exception('Number of electrons has to be even for a closed shell calculation. Given number of '
                            'electrons was {}.'.format(number_of_electrons))

    def get_ovlp(self):
        """

        :return: The overlap matrix
        """
        return self.integrals[0]  # Get the overlap integrals of the given molecule

    def get_one_e(self):
        """

        :return: The one electron integral matrix: T + V
        """
        return self.integrals[1]  # Get the one electron integrals of the given molecule

    def get_two_e(self):
        """

        :return: The electron repulsion interaction tensor
        """
        return self.integrals[2]  # Get the two electron integrals of the given molecule

    def nuc_rep(self):
        """

        :return: The nuclear repulsion value
        """
        return self.integrals[3]  # Get the nuclear repulsion value of the given molecule

    def scf(self, convergence=1e-12):
        """
        Performs a self consistent field calculation to find the lowest RHF energy.

        :param convergence: Convergence criterion. If none is specified, 1e-12 is used.
        :return: number of iterations, scf energy, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        core_guess = s_12 @ self.get_one_e() @ s_12.T  # orthogonalise the transformation matrix.
        guess_density = density_matrix(core_guess, self.occupied, s_12)  # calculate the guess density

        densities = [guess_density]  # put the guess density in an array

        def rhf_scf_energy(dens_matrix, fock):
            """calculate the scf energy value from a given density matrix and a given fock matrix"""
            return np.einsum('pq, pq->', (self.get_one_e() + fock), dens_matrix)

        energies = [0.0]

        def rhf_fock_matrix(dens_matrix):
            """calculate a fock matrix from a given density matrix"""
            # jk_integrals = coulomb - exchange
            jk_integrals = 2 * self.get_two_e() - self.get_two_e().transpose(0, 2, 1, 3)
            # double summation of density matrix * (coulomb - exchange)
            return self.get_one_e() + np.einsum('kl,ijkl->ij', dens_matrix, jk_integrals)

        delta_e = []  # create an energy difference array

        def iteration():
            """create an iteration procedure, calculate fock from density,
            orthogonalise, new density from new fock,..."""
            fock = rhf_fock_matrix(densities[-1])  # calculate fock matrix from the newest density matrix
            # calculate the electronic energy and put it into the energies array
            energies.append(rhf_scf_energy(densities[-1], fock) + self.nuc_rep())
            # calculate the energy difference and add it to the correct array
            delta_e.append(energies[-1] - energies[-2])
            # orthogonalize the new fock matrix
            # calculate density matrix from the new fock matrix
            fock_orth = s_12.T.dot(fock).dot(s_12)
            new_density = density_matrix(fock_orth, self.occupied, s_12)
            # put new density matrix in the densities array
            densities.append(new_density)

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        iteration()
        i = 1
        while abs(delta_e[-1]) >= convergence:
            iteration()
            i += 1
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure
        def last_dens():
            return densities[-1]
        self.last_dens = last_dens()

        # a function that gives the last Fock matrix of the scf procedure
        def last_fock():
            return rhf_fock_matrix(densities[-2])
        self.last_fock = last_fock()

        # A function that returns the converged mo coefficients
        def get_mo():
            last_f = rhf_fock_matrix(densities[-2])  # get the last fock matrix
            f_eigenvalues, f_eigenvectors = la.eigh(last_f)  # eigenvalues are initial orbital energies
            coefficients = s_12.dot(f_eigenvectors)  # transform to mo basis
            return coefficients
        self.mo = get_mo()

        # calculate the total energy, taking nuclear repulsion into account
        scf_e = energies[-1]
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution(self, convergence=1e-12):
        """
        Prints the number of iterations and the converged scf energy.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: the converged energy
        """
        self.scf(convergence=convergence)
        print("Number of iterations: " + str(self.iterations))
        print("Converged SCF energy in Hartree: " + str(self.energy) + " (RHF)")
        return self.energy

    def get_mo_coeff(self):
        """
        Returns mo coefficients of the converged solution.

        :return: The mo coefficients
        """
        return self.mo

    def get_last_dens(self):
        """
        Returns the last density matrix of the converged solution.

        :return: The last density matrix.
        """
        return self.last_dens

    def get_last_fock(self):
        """
        Returns the last fock matrix of the converged solution.

        :return: The last Fock matrix.
        """
        return self.last_fock

    def diis(self, convergence=1e-12):
        """
        When needed, DIIS can be used to speed up the RHF calculations by reducing the needed iterations.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: scf energy, number of iterations, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        core_guess = s_12.T @ self.get_one_e() @ s_12  # orthogonalise the transformation matrix.
        guess_density = density_matrix(core_guess, self.occupied, s_12)  # calculate the guess density

        def rhf_fock_matrix(dens_matrix):
            """calculate a fock matrix from a given density matrix"""
            # jk_integrals = coulomb - exchange
            jk_integrals = 2 * self.get_two_e() - self.get_two_e().transpose(0, 2, 1, 3)
            # double summation of density matrix * (coulomb - exchange)
            return self.get_one_e() + np.einsum('kl,ijkl->ij', dens_matrix, jk_integrals)

        def residual(density, fock):
            """
            Function that calculates the error matrix for the DIIS algorithm
            :param density: density matrix
            :param fock: fock matrix
            :return: a value that should be zero and a fock matrix
            """
            return s_12 @ (fock @ density @ self.get_ovlp() - self.get_ovlp() @ density @ fock) @ s_12.T

        # Create a list to store the errors
        # create a list to store the fock matrices
        # deque is used as a high performance equivalent of an array
        error_list = c.deque(maxlen=6)
        fock_list = c.deque(maxlen=6)

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

        def rhf_scf_energy(dens_matrix, fock):
            """calculate the scf energy value from a given density matrix and a given fock matrix"""
            return np.einsum('pq, pq->', (self.get_one_e() + fock), dens_matrix)

        # Create the necessary arrays to perform an iterative diis procedure
        densities_diis = [guess_density]
        energies_diis = [0.0]
        delta_e = []

        def diis_iteration(number_of_iterations):
            # Calculate the fock matrix and the residual
            fock = rhf_fock_matrix(densities_diis[-1])
            resid = residual(densities_diis[-1], fock)

            # Add them to their respective lists
            fock_list.append(fock)
            error_list.append(resid)

            # Calculate the energy and the energy difference
            energies_diis.append(rhf_scf_energy(densities_diis[-1], fock) + self.nuc_rep())
            delta_e.append(energies_diis[-1] - energies_diis[-2])

            # Starting at two iterations, use the DIIS acceleration
            if number_of_iterations >= 2:
                fock = diis_fock(fock_list, error_list)

            # orthogonalize the new fock matrix
            # calculate density matrix from the new fock matrix
            fock_orth = s_12.T.dot(fock).dot(s_12)
            new_density = density_matrix(fock_orth, self.occupied, s_12)

            # put new density matrix in the densities array
            densities_diis.append(new_density)

        i = 1
        diis_iteration(i)
        while abs(delta_e[-1]) >= convergence:
            diis_iteration(i)
            i += 1
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure
        def last_dens():
            return densities_diis[-1]

        self.last_dens = last_dens()

        # a function that gives the last Fock matrix of the scf procedure
        def last_fock():
            return fock_list[-1]

        self.last_fock = last_fock()

        # A function that returns the converged mo coefficients
        def get_mo():
            last_f = last_fock()  # get the last fock matrix
            f_eigenvalues, f_eigenvectors = la.eigh(last_f)  # eigenvalues are initial orbital energies
            coeff = s_12.dot(f_eigenvectors)  # transform to mo basis
            return coeff

        self.mo = get_mo()

        # calculate the total energy, taking nuclear repulsion into account
        scf_e = energies_diis[-1]
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution_diis(self, convergence=1e-12):
        """
        Prints the number of iterations and the converged DIIS energy. The number of iterations will be lower than with
        a normal scf, but the energy value will be the same. Example:

        >>> h2 = gto.M(atom = 'h 0 0 0; h 1 0 0', basis = 'cc-pvdz')
        >>> x = RHF(h2, 2)
        >>> x.get_scf_solution_diis()
        Number of iterations: 9
        Converged SCF energy in Hartree: -1.100153764878446 (RHF)

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :return: The converged scf energy, using DIIS.
        """
        self.diis(convergence=convergence)
        print("Number of iterations: " + str(self.iterations))
        print("Converged SCF energy in Hartree: " + str(self.energy) + " (RHF)")
        return self.energy
