"""
Restricted Hartree Fock, by means of SCF procedure
==================================================
This class is used to calculate the RHF energy of a given molecule and the number of electrons.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)
"""

from ghf.SCF_functions import *


class RHF:
    """
        calculate RHF energy.
        ---------------------
        Input is a molecule and the number of electrons.

        Molecules are made in pySCF and calculations are performed as follows, eg.:
        The following snippet prints and returns RHF energy of h_2
        and the number of iterations needed to get this value.

        >>> h_2 = gto.M(atom = 'h 0 0 0; h 0 0 1', spin = 0, basis = 'cc-pvdz')
        >>> x = RHF(h_2, 2)
        >>> x.get_scf_solution()
        Number of iterations: 109
        Converged SCF energy in Hartree: -1.9403598392831243 (RHF)
        """
    def __init__(self, molecule, number_of_electrons):
        """
        Initiate the RHF instance by specifying the molecule in question with pyscf and the total number of electrons.

        :param molecule: The molecule on which to perform the calculations
        :param number_of_electrons: The total number of electrons in the system
        """
        self.molecule = molecule
        self.integrals = get_integrals(molecule)
        self.energy = None
        self.mo = None
        self.last_dens = None
        self.last_fock = None
        # For closed shell calculations the number of electrons should be a multiple of 2.
        # If this is not the case, a message is printed telling you to adjust the parameter.
        if number_of_electrons % 2 == 0:
            self.occupied = int(number_of_electrons/2)
        else:
            print('Number of electrons has to be even for a closed shell calculation.')

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

    def scf(self):
        """
        Performs a self consistent field calculation to find the lowest RHF energy.

        :return: number of iterations, scf energy, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        core_guess = s_12.T @ self.get_one_e() @ s_12  # orthogonalise the transformation matrix.
        guess_density = density_matrix(core_guess, self.occupied, s_12)  # calculate the guess density
        densities = [guess_density]  # put the guess density in an array

        def rhf_scf_energy(dens_matrix, fock):
            """calculate the scf energy value from a given density matrix and a given fock matrix"""
            return np.sum(dens_matrix * (self.get_one_e() + fock))

        electronic_e = np.sum(guess_density * self.get_one_e() * 2)  # calculate the initial energy, using the guess
        energies = [electronic_e]  # create an electronic energy array

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
            fock_orth = s_12.T.dot(fock).dot(s_12)  # orthogonalize the new fock matrix
            new_density = density_matrix(fock_orth, self.occupied,
                                         s_12)  # calculate density matrix from the new fock matrix
            densities.append(new_density)  # put new density matrix in the densities array
            energies.append(
                rhf_scf_energy(new_density, fock))  # calculate the electronic energy and put it into the energies array
            delta_e.append(
                energies[-1] - energies[-2])  # calculate the energy difference and add it to the correct array

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        iteration()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration()
            i += 1

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
        scf_e = energies[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), last_dens(), last_fock()

    def get_scf_solution(self):
        """
        Prints the number of iterations and the converged scf energy.

        :return: The converged scf energy.
        """
        print("Number of iterations: " + str(self.scf()[1]))
        print("Converged SCF energy in Hartree: " + str(self.scf()[0]) + " (RHF)")
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
