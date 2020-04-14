"""
Generalised Hartree Fock, by means of SCF procedure
====================================================


This class creates a generalised Hartree-Fock object which can be used for scf calculations. Different initial guesses
are provided as well as the option to perform a stability analysis.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)

"""

from hf.SCF_functions import *
import numpy as np
from numpy import linalg as la
from scipy import linalg as la2
import collections as c


class GHF:
    """
    Calculate the GHF energy.
    --------------------------------
    Input is a molecule and the number of electrons.

    Molecules are made in pySCF and calculations are performed as follows, eg.:
    The following snippet prints and returns UHF energy of h3
    and the number of iterations needed to get this value.

    For a normal scf calculation your input looks like the following example:

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> x = GHF(h3, 3)
    >>> x. get_scf_solution()
    Number of iterations: 81
    Converged SCF energy in Hartree: -1.5062743202607725 (Real GHF)
    """
    def __init__(self, molecule, number_of_electrons, int_method='pyscf'):
        """
        Initiate an instance of a real GHF class.

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
                            'Make sure the molecule instance matches the method and is given as a string.')
        self.energy = None
        self.mo = None
        self.last_dens = None
        self.last_fock = None
        self.iterations = None
        self.hessian = None
        self.int_instability = None
        self.ext_instability = None

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

    def unitary_rotation_guess(self, init=None):
        """
        A function that creates an initial guess matrix by performing a unitary transformation on the core Hamiltonian
        matrix.

        To use this guess:

        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = GHF(h3, 3)
        >>> guess = x.unitary_rotation_guess()
        >>> x.get_scf_solution(guess)

        :return: A rotated guess matrix.
        """
        if init is None:
            c_ham = expand_matrix(self.get_one_e())
        else:
            if np.shape(init)[0] == np.shape(self.get_ovlp())[0]:
                c_ham = expand_matrix(init)
            else:
                c_ham = init

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
        >>> x = GHF(h3, 3)
        >>> guess = x.random_guess()
        >>> x.get_scf_solution(guess)

        :return: A random hermitian matrix.
        """
        dim = int(np.shape(self.get_ovlp())[0] * 2)

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

        return random_unitary_matrix(dim)

    def scf(self, guess=None, convergence=1e-12, complex_method=False):
        """
        This function performs the SCF calculation by using the generalised Hartree-Fock formulas. Since we're working
        in the real class, all values throughout are real. For complex, see the "complex_GHF" class.

        :param guess: Initial guess to start SCF. If none is given, core hamiltonian will be used.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: scf_energy, iterations, mo coefficients, last density matrix & last Fock matrix
        """
        # Get the transformation matrix, S^1/2, and write it in spin blocked notation.
        # Also define the core Hamiltonian matrix in it's spin-blocked notation.
        s_min_12 = trans_matrix(self.get_ovlp())
        s_12_o = expand_matrix(s_min_12)
        c_ham = expand_matrix(self.get_one_e())

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
            return np.einsum('ij,kj->ik', coeff_r, coeff_r.conj())

        if guess is None:

            def coeff_matrix(orth_matrix, core_ham):
                """
                :param orth_matrix: an orthogonalisation matrix, created with the expand matrix function
                :param core_ham: the expanded core Hamiltonian matrix
                :return: The orthogonalised version of the core Hamiltonian matrix
                """
                return orth_matrix @ core_ham @ orth_matrix.conj().T
            # Define the initial guess as the orthogonalised core Hamiltonian matrix
            initial_guess = coeff_matrix(s_12_o, c_ham)

            if complex_method:
                if not isinstance(initial_guess[0][0], complex):
                    p_g = density(initial_guess) + 0j
                    p_g[0, :] += .1j
                    p_g[:, 0] -= .1j
                else:
                    p_g = density(initial_guess)
            else:
                p_g = density(initial_guess)

        else:
            coefficients = guess
            coefficients_r = coefficients[:, 0:self.number_of_electrons]

            if complex_method:
                if not isinstance(guess[0][0], complex):
                    p_g = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj()) + 0j
                    p_g[0, :] += .1j
                    p_g[:, 0] -= .1j
                else:
                    p_g = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj())
            else:
                p_g = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj())

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
            k_ab = exchange(p_ba)
            k_ba = exchange(p_ab)
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
        energies = [0.0]
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
            f = spin_blocked(f_aa, f_ab, f_ba, f_bb)

            # Calculate the new energy and add it to the energies array.
            # Calculate the energy difference and add it to the delta_e array.
            energies.append(scf_e(densities[-1], f))
            delta_e.append(energies[-1] - energies[-2])

            # orthogonalise the Fock matrix
            f_o = s_12_o.conj().T @ f @ s_12_o

            # Create the new density matrix from the Orthogonalised Fock matrix.
            # Add the new density matrix to the densities array.
            p_new = density(f_o)
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
            return f
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

    def get_scf_solution(self, guess=None, convergence=1e-12, complex_method=False):
        """
        Prints the number of iterations and the converged scf energy.

        :param guess: Initial guess for scf. If none is specified: expanded core Hamiltonian.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: The converged scf energy.
        """
        self.scf(guess, convergence=convergence, complex_method=complex_method)
        e = self.energy
        # s_values = ghf_spin(self.get_mo_coeff(), expand_matrix(self.get_ovlp()), self.number_of_electrons)
        if complex_method:
            if abs(np.imag(e)) > 1e-12:
                print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
            else:
                print("Number of iterations: " + str(self.iterations))
                print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex GHF)")
        else:
            print("Number of iterations: " + str(self.iterations))
            print("Converged SCF energy in Hartree: " + str(self.energy) + " (Real GHF)")
    # print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) + ", Multiplicity = " + str(s_values[1]))
        return self.energy

    def get_mo_coeff(self):
        """
        Gets the mo coefficients of the converged solution.

        :return: The mo coefficients
        """
        if self.mo is None:
            raise Exception('Perform an scf calculation first.')
        else:
            return self.mo

    def get_last_dens(self):
        """
        Gets the last density matrix of the converged solution.

        :return: The last density matrix.
        """
        if self.last_dens is None:
            raise Exception('Perform an scf calculation first.')
        else:
            return self.last_dens

    def get_last_fock(self):
        """
        Gets the last fock matrix of the converged solution.

        :return: The last Fock matrix.
        """
        if self.last_fock is None:
            raise Exception('Perform an scf calculation first.')
        else:
            return self.last_fock

    def stability_analysis(self, method, step_size=1e-4):
        """
        Internal stability analysis to verify whether the wave function is stable within the space of the used method.
        :param method: Indicate whether you want to check the internal or external stability of the wave function. Can
        be internal or external.
        :param step_size: Step size for orbital rotation. standard is 1e-4.
        :return: In case of internal stability analysis, it returns a new set of coefficients.
        """
        # Calculate the A & B blocks needed for stability analysis.
        # Determine the number of occupied and virtual orbitals.
        occ = int(self.number_of_electrons)
        vir = int(np.shape(expand_matrix(self.get_ovlp()))[0] - self.number_of_electrons)

        # Determine the Fock matrices needed.
        coeff = self.get_mo_coeff()
        fock_init = self.get_last_fock()
        fock_mo = coeff.conj().T @ fock_init @ coeff

        # Determine the two electron integrals in spinor basis.
        eri_ao = self.get_two_e()
        eri_ao_spinor = expand_tensor(eri_ao)
        if isinstance(self.energy, complex):
            eri_mo = eri_ao_to_mo(eri_ao_spinor, coeff, complexity=True)
        else:
            eri_mo = eri_ao_to_mo(eri_ao_spinor, coeff)

        eri_spinor_anti_abrs = eri_mo - eri_mo.transpose(0, 2, 1, 3)
        eri_spinor_anti_asrb = eri_mo.transpose(0, 1, 3, 2) - eri_mo.transpose(0, 3, 1, 2)

        # Create the tensors A and B.
        if isinstance(self.energy, complex):
            a_arbs = np.zeros((occ, vir, occ, vir)).astype(complex)
            b_arbs = np.zeros((occ, vir, occ, vir)).astype(complex)
        else:
            a_arbs = np.zeros((occ, vir, occ, vir))
            b_arbs = np.zeros((occ, vir, occ, vir))

        # Fill the A and B tensors with the correct elements.
        for a in range(occ):
            for r in range(occ, occ + vir):
                for b in range(occ):
                    for s in range(occ, occ + vir):
                        if a == b and s == r:
                            a_arbs[a][r-occ][b][s-occ] = fock_mo[s][r] - fock_mo[a][b]\
                                                         + eri_spinor_anti_asrb[a][s][r][b]
                        elif a == b and s != r:
                            a_arbs[a][r-occ][b][s-occ] = fock_mo[s][r] + eri_spinor_anti_asrb[a][s][r][b]
                        elif s == r and a != b:
                            a_arbs[a][r-occ][b][s-occ] = -1 * fock_mo[a][b] + eri_spinor_anti_asrb[a][s][r][b]
                        else:
                            a_arbs[a][r-occ][b][s-occ] = eri_spinor_anti_asrb[a][s][r][b]
        for a in range(occ):
            for r in range(occ, occ + vir):
                for b in range(occ):
                    for s in range(occ, occ + vir):
                        b_arbs[a][r-occ][b][s-occ] = eri_spinor_anti_abrs[a][b][r][s]

        # Reshape the tensors pairwise to create the matrix representation.
        a = a_arbs.reshape((occ*vir, occ*vir), order='F')
        b = b_arbs.reshape((occ*vir, occ*vir), order='F')

        # Create a function to rotate the orbitals in case of internal instability
        def rotate_to_eigenvec(eigenvec):
            if isinstance(self.energy, complex):
                indx = int(np.shape(eigenvec)[0] / 2)
                eigenvec = eigenvec[:indx]

            block_ba = eigenvec.reshape((occ, vir), order='F')
            block_bb = np.zeros((occ, occ))
            block_ab = -1 * block_ba.conj().T
            block_aa = np.zeros((vir, vir))
            k = spin_blocked(block_aa, block_ab, block_ba, block_bb)
            coeff_init = self.get_mo_coeff()
            exp = la2.expm(-1 * step_size * k)
            return coeff_init @ exp

        # Check the different stability matrices to verify the stability.
        if not isinstance(self.energy, complex):
            if method == 'internal':
                # the stability matrix for the real sub problem consists of a + b
                stability_matrix = a + b
                self.hessian = spin_blocked(a, b, b.conj(), a.conj())

                # Calculate the eigenvalues of the stability matrix to asses stability
                e, v = la.eigh(stability_matrix)
                if np.amin(e) < -1e-5:  # this points towards an instability
                    print("There is an internal instability in the real GHF wave function.")
                    self.int_instability = True
                    lowest_eigenvec = v[:, 0]
                    return rotate_to_eigenvec(lowest_eigenvec)
                else:
                    print('The wave function is stable within the real GHF space.')
                    self.int_instability = None

            elif method == 'external':
                # the stability matrix for the complex sub problem consists of a - b
                stability_matrix = a - b
                self.hessian = spin_blocked(a, b, b.conj(), a.conj())

                # Calculate the eigenvalues of the stability matrix to asses stability
                e, v = la.eigh(stability_matrix)
                if np.amin(e) < -1e-5:  # this points towards an instability
                    print("There is an external real/complex instability in the real GHF wave function.")
                    self.ext_instability = True
                else:
                    print('The wave function is stable within the real/complex space.')
                    self.ext_instability = None
            else:
                raise Exception('Only internal and external stability analysis are possible. '
                                'Please enter a valid type.')
        else:
            if method == 'internal':
                # The total stability matrix consists of a & b in the upper corners, and b* and a* in the lower corners
                stability_matrix = spin_blocked(a, b, b.conj(), a.conj())
                self.hessian = stability_matrix

                # Calculate the eigenvalues of the stability matrix to asses stability
                e, v = la.eigh(stability_matrix)
                if np.amin(e) < -1e-5:  # this points towards an instability
                    print("There is an internal instability in the complex GHF wave function.")
                    self.int_instability = True
                    lowest_eigenvec = v[:, 0]
                    return rotate_to_eigenvec(lowest_eigenvec)
                else:
                    print('The wave function is stable in the complex GHF space.')
                    self.int_instability = None
            else:
                raise Exception('Only internal stability analysis is possible for complex GHF.')

    def get_hessian(self):
        """
        After stability analysis is performed, the hessian is stored and can be used for further studying.
        :return: The Hessian matrix
        """
        if self.hessian is None:
            raise Exception('The Hessian matrix has not been calculated yet. To do so, perform a stability analysis'
                            ' on your wave function.')
        else:
            return self.hessian

    def diis(self, guess=None, convergence=1e-12, complex_method=False):
        """
        The DIIS method is an alternative to the standard scf procedure. It reduces the number of iterations needed to
        find a solution. The same guesses can be used as for a standard scf calculation. Stability analysis can be
        done as well.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param guess: The initial guess matrix, if none is specified, the spin blocked core Hamiltonian is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: scf_energy, iterations, mo coefficients, last density matrix & last Fock matrix
        """
        # Get the transformation matrix, S^1/2, and write it in spin blocked notation.
        # Also define the core Hamiltonian matrix in it's spin-blocked notation.
        s_min_12 = trans_matrix(self.get_ovlp())
        s_12_o = expand_matrix(s_min_12)
        c_ham = expand_matrix(self.get_one_e())

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
            return np.einsum('ij,kj->ik', coeff_r, coeff_r.conj())

        if guess is None:

            def coeff_matrix(orth_matrix, core_ham):
                """
                :param orth_matrix: an orthogonalisation matrix, created with the expand matrix function
                :param core_ham: the expanded core Hamiltonian matrix
                :return: The orthogonalised version of the core Hamiltonian matrix
                """
                return orth_matrix @ core_ham @ orth_matrix.conj().T

            # Define the initial guess as the orthogonalised core Hamiltonian matrix
            initial_guess = coeff_matrix(s_12_o, c_ham)

            if complex_method:
                if not isinstance(initial_guess[0][0], complex):
                    p_g = density(initial_guess).astype(complex)
                    p_g[0, :] += .1j
                    p_g[:, 0] -= .1j
                else:
                    p_g = density(initial_guess)
            else:
                p_g = density(initial_guess)

        else:
            coefficients = guess
            coefficients_r = coefficients[:, 0:self.number_of_electrons]

            if complex_method:
                if not isinstance(guess[0][0], complex):
                    p_g = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj()).astype(complex)
                    p_g[0, :] += .1j
                    p_g[:, 0] -= .1j
                else:
                    p_g = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj())
            else:
                p_g = np.einsum('ij,kj->ik', coefficients_r, coefficients_r.conj())

        densities_diis = [p_g]

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
            k_ab = exchange(p_ba)
            k_ba = exchange(p_ab)
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
            # Dimensions
            dim = len(focks) + 1

            # Create the empty B matrix
            b = np.empty((dim, dim))
            b[-1, :] = -1
            b[:, -1] = -1
            b[-1, -1] = 0

            # Fill the B matrix: ei * ej, with e the errors
            if complex_method:
                for k in range(len(focks)):
                    for l in range(len(focks)):
                        b = b.astype(complex)
                        b[k, l] = np.einsum('kl,kl->', residuals[k], residuals[l])
            else:
                for k in range(len(focks)):
                    for l in range(len(focks)):
                        b[k, l] = np.einsum('kl,kl->', residuals[k], residuals[l])

            # Create the residual vector
            res_vec = np.zeros(dim)
            res_vec[-1] = -1

            # Solve the pulay equation to get the coefficients
            coeff = np.linalg.solve(b, res_vec)

            # Create a fock as a linear combination of previous focks
            if complex_method:
                fock = np.zeros(focks[0].shape).astype(complex)
            else:
                fock = np.zeros(focks[0].shape)
            for x in range(coeff.shape[0] - 1):
                fock += coeff[x] * focks[x]
            return fock

        # The function that will calculate the energy value according to the GHF algorithm.
        def scf_e(dens, fock):
            """
            Calculates the scf energy for the GHF method
            """
            return np.sum(dens * (expand_matrix(self.get_one_e()) + fock)) / 2

        # Calculate the first electronic energy from the initial guess and the guess density that's calculated from it.
        # Create an array to store the energy values and another to store the energy differences.
        energies_diis = [0.0]
        delta_e_diis = []

        def iteration_diis(number_of_iterations):
            # create the four spin blocks of the Fock matrix
            f_aa = fock_block('a', 'a', densities_diis[-1])
            f_ab = fock_block('a', 'b', densities_diis[-1])
            f_ba = fock_block('b', 'a', densities_diis[-1])
            f_bb = fock_block('b', 'b', densities_diis[-1])

            # Add them together to form the total Fock matrix in spin block notation
            f = spin_blocked(f_aa, f_ab, f_ba, f_bb)

            # Calculate the residual
            resid = residual(densities_diis[-1], f)

            # Add them to the arrays
            fock_list.append(f)
            error_list.append(resid)

            # Calculate the new energy and add it to the energies array.
            # Calculate the energy difference and add it to the delta_e array.
            energies_diis.append(scf_e(densities_diis[-1], f))
            delta_e_diis.append(energies_diis[-1] - energies_diis[-2])

            # Starting at two iterations, use the DIIS acceleration
            if number_of_iterations >= 2:
                f = diis_fock(fock_list, error_list)

            # orthogonalise the Fock matrix
            f_o = s_12_o.conj().T @ f @ s_12_o

            # Create the new density matrix from the Orthogonalised Fock matrix.
            # Add the new density matrix to the densities array.
            p_new = density(f_o)
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
            f = fock_list[-1]
            return f

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

    def get_scf_solution_diis(self, guess=None, convergence=1e-12, complex_method=False):
        """
        Prints the number of iterations and the converged energy after a diis calculation. Guesses can also be specified
        just like with a normal scf calculation.

        Example:

        >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
        >>> x = GHF(h3, 3)
        >>> guess = x.random_guess()
        >>> x.get_scf_solution_diis(guess)
        Number of iterations: 23
        Converged SCF energy in Hartree: -1.5062743202915496 (Real GHF)

        Without DIIS, 81 iterations are needed to find this solution.

        :param guess: Initial guess for scf. None specified: expanded core Hamiltonian
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.cd
        :return: The converged scf energy.
        """
        self.diis(guess, convergence=convergence, complex_method=complex_method)
        e = self.energy
        if complex_method:
            if abs(np.imag(e)) > 1e-12:
                print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
            else:
                print("Number of iterations: " + str(self.iterations))
                print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex GHF, DIIS)")
        else:
            print("Number of iterations: " + str(self.iterations))
            print("Converged SCF energy in Hartree: " + str(self.energy) + " (Real GHF, DIIS)")
        return self.energy
