"""
Constrained unrestricted Hartree Fock by Scuseria
==================================================
This class is used to calculate the ROHF energy for a given molecule and the number of electrons of that molecule,
using a constrained version of unrestricted Hartree Fock, according to Scuseria..
Several options are available to make sure you get the lowest energy from your calculation, as well as some useful
functions to get intermediate values such as MO coefficients, density and fock matrices.
"""

import hf.utilities.SCF_functions as Scf
import hf.properties.spin as spin
from hf.properties.mulliken import mulliken
import numpy as np
from pyscf import *
import numpy.linalg as la
import collections as c


class MF:
    """
    Calculate cUHF energy.
    -----------------------
    Input is a molecule and the number of electrons.

    Molecules are made in pySCF/psi4 and calculations are performed as follows, eg.:
    The following snippet prints and returns UHF energy of h_3
    and the number of iterations needed to get this value.

    For a normal scf calculation your input looks like the following example:

    >>> from hf.HartreeFock import *
    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> x = cUHF_s.MF(h3, 3)
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
            self.integrals = Scf.get_integrals_pyscf(molecule)
        elif int_method == 'psi4':
            self.integrals = Scf.get_integrals_psi4(molecule)
        else:
            raise Exception('Unsupported method to calculate integrals. Supported methods are pyscf or psi4. '
                            'Make sure the molecule instance matches the method and is gives as a string.')
        # Several dictionaries are created, as to be able to check these properties for a certain iteration
        # The dictionaries are filled while iterating.
        self.energy = None
        self.mo = None
        self.dens = None
        self.fock = None
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

        >>> from hf.HartreeFock import *
        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = cUHF_s.MF(h3, 3)
        >>> guess = x.random_guess()
        >>> x.get_scf_solution(guess)

        :return: A random unitary matrix.
        """
        dim = int(np.shape(self.get_ovlp())[0])

        def random_unitary_matrix(dimension):
            # fill a matrix of the given dimensions with random numbers.
            np.random.seed(2)
            x = np.random.rand(dimension, dimension)
            # Make the matrix symmetric by adding it's transpose.
            # Get the eigenvectors, since they form a unitary matrix.
            x_t = x.T
            y = x + x_t
            val, vec = la.eigh(y)
            return vec

        return random_unitary_matrix(dim), random_unitary_matrix(dim)

    def scf(self, initial_guess=None, convergence=1e-12, diis=True):
        """
        Performs a self consistent field calculation to find the lowest UHF energy.


        :param initial_guess: Random initial guess, if none is given the Core Hamiltonian is used.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param diis: Accelerates the convergence, default is true.
        :return: The scf energy, number of iterations, the mo coefficients, the last density and the last fock matrices
        """
        # calculate the transformation matrix (X_)
        s_12 = Scf.trans_matrix(self.get_ovlp())

        def density(f, occ):
            f_o = s_12.conj().T @ f @ s_12
            e, c_p = la.eigh(f_o)
            coeff = s_12 @ c_p
            coeff_occ = coeff[:, :occ]
            d = np.einsum('pi, qi->pq', coeff_occ, coeff_occ, optimize=True)
            return coeff, d

        def two_electron(d_a, d_b):
            j_a = np.einsum('pqrs,rs->pq', self.get_two_e(), d_a, optimize=True)
            j_b = np.einsum('pqrs,rs->pq', self.get_two_e(), d_b, optimize=True)
            k_a = np.einsum('prqs,rs->pq', self.get_two_e(), d_a, optimize=True)
            k_b = np.einsum('prqs,rs->pq', self.get_two_e(), d_b, optimize=True)
            return j_a, j_b, k_a, k_b

        def fock(j_a, j_b, k_a, k_b):
            f_a = self.get_one_e() + (j_a + j_b) - k_a
            f_b = self.get_one_e() + (j_a + j_b) - k_b
            return f_a, f_b

        def energy(d_a, d_b, f_a, f_b):
            e = np.einsum('pq,pq->', (d_a + d_b), self.get_one_e(), optimize=True)
            e += np.einsum('pq,pq->', d_a, f_a, optimize=True)
            e += np.einsum('pq,pq->', d_b, f_b, optimize=True)
            e *= 0.5
            e += self.nuc_rep()
            return float(e)

        def residual(d, f):
            """
            Function that calculates the error matrix for the DIIS algorithm
            :param d: density matrix
            :param f: fock matrix
            :return: a value that should be zero and a fock matrix
            """
            return s_12 @ (f @ d @ self.get_ovlp() - self.get_ovlp() @ d @ f) @ s_12.conj().T

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

            # Solve the Pulay equation to get the coefficients
            coeff = np.linalg.solve(b, res_vec)

            # Create a fock as a linear combination of previous focks
            f = np.zeros(focks[0].shape)
            for x in range(coeff.shape[0] - 1):
                f += coeff[x] * focks[x]
            return f

        # constraint function
        def constrain1(j_a, j_b, k_a, k_b, d_a, d_b, x):
            f_p = 0.5 * (2 * (j_a + j_b) - k_a - k_b)
            f_m = -0.5 * (k_a - k_b)

            p = (d_a + d_b) / 2
            p = la.inv(x) @ p @ la.inv(x.T)
            nat_occ_num, nat_occ_vec = la.eigh(p)

            f_m = la.inv(nat_occ_vec) @ la.inv(x) @ f_m @ la.inv(x.T) @ la.inv(nat_occ_vec.T)
            f_m[:self.n_b, self.n_a:] = 0.0
            f_m[self.n_a:, :self.n_b] = 0.0
            f_m = x @ nat_occ_vec @ f_m @ nat_occ_vec.T @ x.T

            f_a = self.get_one_e() + f_p + f_m
            f_b = self.get_one_e() + f_p - f_m
            return f_a, f_b

        def constrain2(d_a, d_b, f_a, f_b, x):
            p = (d_a + d_b) / 2.0
            f_cs = (f_a + f_b) / 2.0
            delta_uhf = (f_a - f_b) / 2.0

            p = la.inv(x) @ p @ la.inv(x.T)
            nat_occ_num, nat_occ_vec = np.linalg.eigh(p)
            nat_occ_vec = np.flip(nat_occ_vec, axis=1)

            delta_uhf_no = la.inv(nat_occ_vec) @ la.inv(x) @ delta_uhf @ la.inv(x.T) @ la.inv(nat_occ_vec.T)

            delta_cuhf = np.copy(delta_uhf_no)
            delta_cuhf[:self.n_b, self.n_a:] = 0.0
            delta_cuhf[self.n_a:, :self.n_b] = 0.0

            delta_cuhf = x @ nat_occ_vec @ delta_cuhf @ nat_occ_vec.T @ x.T

            f_a = f_cs + delta_cuhf
            f_b = f_cs - delta_cuhf
            return f_a, f_b

        # core Hamiltonian guess

        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        if initial_guess is None:
            # create guess density matrix from core guess, separate for alpha and beta and put them into an array
            c_a, guess_d_a = density(self.get_one_e(), self.n_a)
            c_b, guess_d_b = density(self.get_one_e(), self.n_b)
        else:
            # Make the coefficients orthogonal in the correct basis.
            c_a = s_12 @ initial_guess[0]
            c_b = s_12 @ initial_guess[1]

            # Get C_alpha and C_beta
            coeff_r_a = c_a[:, 0:self.n_a]
            coeff_r_b = c_b[:, 0:self.n_b]

            guess_d_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a)
            guess_d_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b)

        dens_a = [guess_d_a]
        dens_b = [guess_d_b]
        mo_a = [c_a]
        mo_b = [c_b]

        # energy check
        energies = [0.0]
        delta_e = []

        # fock list
        focks_a = []
        focks_b = []

        def iterate(n_i):
            j_a, j_b, k_a, k_b = two_electron(dens_a[-1], dens_b[-1])
            f_a, f_b = constrain1(j_a, j_b, k_a, k_b, dens_a[-1], dens_b[-1], s_12)
            energies.append(energy(dens_a[-1], dens_b[-1], f_a, f_b))
            delta_e.append(energies[-1] - energies[-2])

            focks_a.append(f_a)
            focks_b.append(f_b)

            if diis is True:
                # Calculate the residuals from both
                resid_a = residual(dens_a[-1], f_a)
                resid_b = residual(dens_b[-1], f_b)

                # Add everything to their respective list
                fock_list_a.append(f_a)
                fock_list_b.append(f_b)
                error_list_a.append(resid_a)
                error_list_b.append(resid_b)

                # Starting at two iterations, use the DIIS acceleration
                if n_i >= 1:
                    f_a = diis_fock(fock_list_a, error_list_a)
                    f_b = diis_fock(fock_list_b, error_list_b)

            # Compute new orbital guess
            c_a_new, d_a_new = density(f_a, self.n_a)
            c_b_new, d_b_new = density(f_b, self.n_b)

            dens_a.append(d_a_new)
            dens_b.append(d_b_new)
            mo_a.append(c_a_new)
            mo_b.append(c_b_new)

        i = 0
        iterate(i)
        while abs(delta_e[-1]) >= convergence and i < 1000:
            i += 1
            iterate(i)
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure, both for alpha and beta
        def dens():
            return dens_a, dens_b
        self.dens = dens()

        def fock():
            return focks_a, focks_b
        self.fock = fock()

        def get_mo():
            return mo_a, mo_b
        self.mo = get_mo()

        self.energy = energies[-1]

        return energies[-1], i

    def calculate_mulliken(self):
        """
        Calculates Mulliken charges for each atom in the pyscf molecule.

        !!!IMPORTANT!!! Only supported with pyscf.

        :print: The Mulliken charges and their corresponding atoms.
        """
        x = mulliken(self.molecule, (self.dens[0][-1], self.dens[1][-1]), self.integrals[0])
        print('Mulliken charges: {}\tCorresponding atoms: {}'.format(x[0], x[1]))

    def get_scf_solution(self, guess=None, convergence=1e-12, diis=True):
        """
        Prints the number of iterations and the converged scf energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :param guess: Initial scf guess
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param diis: Accelerates the convergence, default is true.
        :return: The converged scf energy.
        """
        self.scf(guess, convergence=convergence, diis=diis)
        s_values = spin.uhf(self.n_a, self.n_b, MF.get_mo_coeff(self)[0], MF.get_mo_coeff(self)[1], self.get_ovlp())
        print("Number of iterations: " + str(self.iterations))
        print("Converged SCF energy in Hartree: " + str(self.energy) + " (Constrained UHF)")
        print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) + ", Multiplicity = " + str(s_values[2]))
        return self.energy

    def get_mo_coeff(self, i=-1):
        """
        Gets the mo coefficients of the converged solution.
        Alpha coefficients in the first matrix, beta coefficients in the second.

        :return: The mo coefficients
        """
        return self.mo[0][i], self.mo[1][i]

    def get_dens(self, i=-1):
        """
        Gets the last density matrix of the converged solution.
        Alpha density in the first matrix, beta density in the second.

        :return: The last density matrix.
        """
        return self.dens[0][i], self.dens[1][i]

    def get_fock(self, i=-1):
        """
        Gets the last fock matrix of the converged solution.
        Alpha Fock matrix first, beta Fock matrix second.

        :return: The last Fock matrix.
        """
        return self.fock[0][i], self.fock[1][i]
