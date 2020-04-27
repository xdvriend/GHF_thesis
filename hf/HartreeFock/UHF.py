"""
Unrestricted Hartree Fock, by means of SCF procedure
====================================================
This class is used to calculate the UHF energy for a given molecule and the number of electrons of that molecule.
Several options are available to make sure you get the lowest energy from your calculation, as well as some useful
functions to get intermediate values such as MO coefficients, density and fock matrices.
"""

import hf.utilities.SCF_functions as Scf
import hf.properties.spin as spin
import hf.utilities.transform as t
import numpy as np
from pyscf import *
import scipy.linalg as la
import collections as c


class MF:
    """
    Calculate UHF energy.
    ---------------------
    Input is a molecule and the number of electrons.

    Molecules are made in pySCF and calculations are performed as follows, eg.:
    The following snippet prints and returns UHF energy of h_3
    and the number of iterations needed to get this value.

    For a normal scf calculation your input looks like the following example:

    >>> from hf.HartreeFock import *
    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> x = UHF.MF(h3, 3)
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
        self.energy = None
        self.mo = None
        self.dens = None
        self.focks = None
        self.fock_orth = None
        self.iterations = None
        self.hessian_p = None
        self.hessian_pp = None
        self.int_instability = None
        self.ext_instability_rc = None
        self.ext_instability_ug = None
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

    def scf(self, initial_guess=None, convergence=1e-12, complex_method=False):
        """
        Performs a self consistent field calculation to find the lowest UHF energy.

        :param initial_guess: A tuple of an alpha and beta guess matrix. If none, the core hamiltonian will be used.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: The scf energy, number of iterations, the mo coefficients, the last density and the last fock matrices
        """
        # calculate the transformation matrix
        s_12 = Scf.trans_matrix(self.get_ovlp())
        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        # create guess density matrix from core guess, separate for alpha and beta and put them into an array
        if initial_guess is None:
            if complex_method:
                initial_guess = s_12.conj().T @ self.get_one_e() @ s_12
                initial_guess = initial_guess.astype(complex)
                guess_density_a = Scf.density_matrix(initial_guess, self.n_a, s_12)
                guess_density_a[0, :] += 0.1j
                guess_density_a[:, 0] -= 0.1j
                guess_density_b = Scf.density_matrix(initial_guess, self.n_b, s_12)
                guess_density_b[0, :] += 0.1j
                guess_density_b[:, 0] -= 0.1j
            else:
                initial_guess = s_12.conj().T @ self.get_one_e() @ s_12
                guess_density_a = Scf.density_matrix(initial_guess, self.n_a, s_12)
                guess_density_b = Scf.density_matrix(initial_guess, self.n_b, s_12)

        else:
            # Make the coefficients orthogonal in the correct basis.
            coeff_a = s_12 @ initial_guess[0]
            coeff_b = s_12 @ initial_guess[1]

            # Get C_alpha and C_beta
            coeff_r_a = coeff_a[:, 0:self.n_a]
            coeff_r_b = coeff_b[:, 0:self.n_b]

            if complex_method:
                # Create the guess density matrices from the given coefficients
                guess_density_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a).astype(complex)
                guess_density_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b).astype(complex)
            else:
                guess_density_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a)
                guess_density_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b)

        # Store the density matrices in an array
        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        # Create a second array to study the difference in energy between iterations.
        energies = [0.0]
        delta_e = []
        f_list = []
        fo_list = []
        mo_list = []

        # create an iteration procedure
        def iteration():
            # create a fock matrix for alpha from last alpha density
            fock_a = Scf.uhf_fock_matrix(densities_a[-1], densities_b[-1], self.get_one_e(), self.get_two_e())
            # create a fock matrix for beta from last beta density
            fock_b = Scf.uhf_fock_matrix(densities_b[-1], densities_a[-1], self.get_one_e(), self.get_two_e())
            f_list.append([fock_a, fock_b])
            # calculate the improved scf energy and add it to the array
            energies.append(Scf.uhf_scf_energy(densities_a[-1], densities_b[-1], fock_a, fock_b, self.get_one_e()))
            # calculate the energy difference and add it to the delta_E array
            delta_e.append(energies[-1] - energies[-2])

            # orthogonalize both fock matrices
            fock_orth_a = s_12.conj().T.dot(fock_a).dot(s_12)
            fock_orth_b = s_12.conj().T.dot(fock_b).dot(s_12)
            fo_list.append([fock_orth_a, fock_orth_b])

            # Calculate MO's
            mos_a = Scf.calc_mo(fock_orth_a, s_12)
            mos_b = Scf.calc_mo(fock_orth_b, s_12)
            mo_list.append([mos_a, mos_b])
            # create a new alpha and beta density matrix
            new_density_a = Scf.density_matrix(fock_orth_a, self.n_a, s_12)
            new_density_b = Scf.density_matrix(fock_orth_b, self.n_b, s_12)

            # put the density matrices in their respective arrays
            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        iteration()
        i = 1
        while abs(delta_e[-1]) >= convergence:
            iteration()
            i += 1
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure, both for alpha and beta
        def dens():
            return densities_a, densities_b
        self.dens = dens()

        def fock():
            return f_list
        self.focks = fock()
        self.fock_orth = fo_list

        def get_mo():
            return mo_list
        self.mo = get_mo()

        # Calculate the final scf energy (electronic + nuclear repulsion)
        scf_e = energies[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), dens(), fock()

    def get_scf_solution(self, guess=None, convergence=1e-12, complex_method=False):
        """
        Prints the number of iterations and the converged scf energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :param guess: The initial guess for the scf procedure. If none is given: core Hamiltonian.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: The converged scf energy.
        """
        self.scf(guess, convergence=convergence, complex_method=complex_method)
        e = self.energy
        s_values = spin.uhf(self.n_a, self.n_b, MF.get_mo_coeff(self)[0], MF.get_mo_coeff(self)[1], self.get_ovlp())
        if complex_method:
            if abs(np.imag(e)) > 1e-12:
                print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
            else:
                print("Number of iterations: " + str(self.iterations))
                print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex UHF)")
                print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) +
                      ", Multiplicity = " + str(s_values[2]))
        else:
            print("Number of iterations: " + str(self.iterations))
            print("Converged SCF energy in Hartree: " + str(self.energy) + " (Real UHF)")
            print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) +
                  ", Multiplicity = " + str(s_values[2]))
        return self.energy

    def get_mo_coeff(self, i=-1):
        """
        Gets the mo coefficients of the converged solution.
        Alpha coefficients in the first matrix, beta coefficients in the second.

        :return: The mo coefficients
        """
        return self.mo[i][0], self.mo[i][1]

    def get_fock_orth(self, i=-1):
        """
        Get the fock matrices in the orthonormal basis. Defaults to the last one
        :param i: index of the matrix you want.
        :return: a and b orthonormal fock matrix
        """
        return self.fock_orth[i][0], self.fock_orth[i][1]

    def get_mo_energy(self, i=-1):
        """
        Returns the MO energies of the converged solution.
        :return: an array of MO energies.
        """
        e_a = Scf.calc_mo_e(self.get_fock_orth(i)[0])
        e_b = Scf.calc_mo_e(self.get_fock_orth(i)[1])
        return e_a, e_b

    def get_dens(self, i=-1):
        """
        Gets the (last) density matrix of the converged solution.
        Alpha density in the first matrix, beta density in the second.

        :return: The last density matrix.
        """
        return self.dens[0][i], self.dens[1][i]

    def get_fock(self, i=-1):
        """
        Gets the (last) fock matrix of the converged solution.
        Alpha Fock matrix first, beta Fock matrix second.

        :return: The requested Fock matrices.
        """
        return self.focks[i][0], self.focks[i][1]

    def extra_electron_guess(self):
        """
        This method adds two electrons to the system in order to get coefficients that can be used as a better guess
        for the scf procedure. This essentially forces the system into it's <S_z> = 0 state.

        !!!IMPORTANT!!! Only supported with pyscf.

        To perform a calculation with this method, you will have to work as follows:

        >>> from hf.HartreeFock import *
        >>> h4 = gto.M(atom = 'h 0 0 0; h 1 0 0; h 0 1 0; h 1 1 0' , spin = 2, basis = 'cc-pvdz')
        >>> x = UHF.MF(h4, 4)
        >>> guess = x.extra_electron_guess()
        >>> x.get_scf_solution(guess)

        :return: A new guess matrix to use for the scf procedure.
        """
        # Add two electrons to the molecule and build the new test system.
        self.molecule.nelectron = self.n_a + self.n_b + 2
        self.molecule.build()

        # calculate the integrals for the new test system (_t)
        overlap_t = Scf.get_integrals_pyscf(self.molecule)[0]
        one_electron_t = Scf.get_integrals_pyscf(self.molecule)[1]
        two_electron_t = Scf.get_integrals_pyscf(self.molecule)[2]

        # Calculate the orthogonalisation matrix and the core guess for the new test system (_t)
        s_12_t = Scf.trans_matrix(overlap_t)
        core_guess_t = s_12_t.T.dot(one_electron_t).dot(s_12_t)

        # Calculate a guess density for the test system, both for alpha and beta.
        # add the two extra electrons to alpha
        # add the density matrices to respective arrays
        guess_density_a_t = Scf.density_matrix(core_guess_t, self.n_a + 2, s_12_t)
        guess_density_b_t = Scf.density_matrix(core_guess_t, self.n_b, s_12_t)
        densities_a = [guess_density_a_t]
        densities_b = [guess_density_b_t]

        # create an array to check the differences between density matrices in between iterations
        delta_dens = []

        # create an iteration procedure, based on the densities, for the test system
        def iteration_t():
            # create a fock matrix for alpha from last alpha density, and the integrals from the test system
            # create a fock matrix for beta from last beta density, and the integrals from the test system
            fock_matrix_a = Scf.uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron_t, two_electron_t)
            fock_matrix_b = Scf.uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron_t, two_electron_t)

            # orthogonalize both fock matrices
            orth_fock_a = s_12_t.T.dot(fock_matrix_a).dot(s_12_t)
            orth_fock_b = s_12_t.T.dot(fock_matrix_b).dot(s_12_t)

            # create a new alpha density matrix, with the test system's orthogonalisation matrix
            # create a new beta density matrix, with the test system's orthogonalisation matrix
            # add the densities to an array
            new_density_a = Scf.density_matrix(orth_fock_a, self.n_a + 2, s_12_t)
            new_density_b = Scf.density_matrix(orth_fock_b, self.n_b, s_12_t)
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
        fock_a = Scf.uhf_fock_matrix(densities_a[-2], densities_b[-2], one_electron_t, two_electron_t)
        fock_b = Scf.uhf_fock_matrix(densities_b[-2], densities_a[-2], one_electron_t, two_electron_t)

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

    def stability_analysis(self, method):
        """
        Internal stability analysis to verify whether the wave function is stable within the space of the used method.
        :param method: Indicate whether you want to check the internal or external stability of the wave function. Can
        be internal or external.
        :return: In case of internal stability analysis, it returns a new set of coefficients.
        """
        # Get the mo coefficients
        mo_a = self.get_mo_coeff()[0]
        mo_b = self.get_mo_coeff()[1]

        # Get MO energies
        mo_e_a = self.get_mo_energy()[0]
        mo_e_b = self.get_mo_energy()[1]

        # Get the total amount of orbitals
        # Number of occupied alpha and beta orbitals
        # number of virtual (unoccupied) alpha and beta orbitals
        n_orb = np.shape(self.get_ovlp())[0]
        occ_a = self.n_a
        occ_b = self.n_b
        vir_a = int(n_orb - self.n_a)
        vir_b = int(n_orb - self.n_b)

        # Determine the two electron integrals in MO basis.
        eri_ao = self.get_two_e()
        eri_mo_aa = t.tensor_basis_transform(eri_ao, mo_a)
        eri_mo_bb = t.tensor_basis_transform(eri_ao, mo_b)
        eri_mo_ab = t.mix_tensor_to_basis_transform(eri_ao, mo_a, mo_a, mo_b, mo_b)

        # Create the alpha->alpha part of the a'+b' stability matrix
        h_aa = np.einsum('aijb->iajb', eri_mo_aa[occ_a:, :occ_a, :occ_a, occ_a:]) * 2
        h_aa -= np.einsum('abji->iajb', eri_mo_aa[occ_a:, occ_a:, :occ_a, :occ_a])
        h_aa -= np.einsum('ajbi->iajb', eri_mo_aa[occ_a:, :occ_a, occ_a:, :occ_a])
        for a in range(vir_a):
            for i in range(occ_a):
                h_aa[i, a, i, a] += mo_e_a[occ_a + a] - mo_e_a[i]

        # Create the beta->beta part of the a'+b' stability matrix
        h_bb = np.einsum('aijb->iajb', eri_mo_bb[occ_b:, :occ_b, :occ_b, occ_b:]) * 2
        h_bb -= np.einsum('abji->iajb', eri_mo_bb[occ_b:, occ_b:, :occ_b, :occ_b])
        h_bb -= np.einsum('ajbi->iajb', eri_mo_bb[occ_b:, :occ_b, occ_b:, :occ_b])
        for a in range(vir_b):
            for i in range(occ_b):
                h_bb[i, a, i, a] += mo_e_b[occ_b + a] - mo_e_b[i]

        # Create the (alpha->alpha, beta->beta) part of the a'+b' stability matrix
        h_ab = np.einsum('aijb->iajb', eri_mo_ab[occ_a:, :occ_a, :occ_b, occ_b:]) * 2

        # Create the complete a'+b' stability matrix
        dim_a = occ_a * vir_a
        dim_b = occ_b * vir_b
        h_a_plus_b = np.empty((dim_a + dim_b, dim_a + dim_b))
        h_a_plus_b[:dim_a, :dim_a] = h_aa.reshape(dim_a, dim_a)
        h_a_plus_b[dim_a:, dim_a:] = h_bb.reshape(dim_b, dim_b)
        h_a_plus_b[:dim_a, dim_a:] = h_ab.reshape(dim_a, dim_b)
        h_a_plus_b[dim_a:, :dim_a] = h_ab.reshape(dim_a, dim_b).T

        # Create the alpha->alpha part of the a'-b' stability matrix
        h_aa_m = - np.einsum('abji->iajb', eri_mo_aa[occ_a:, occ_a:, :occ_a, :occ_a])
        h_aa_m += np.einsum('ajbi->iajb', eri_mo_aa[occ_a:, :occ_a, occ_a:, :occ_a])
        for a in range(vir_a):
            for i in range(occ_a):
                h_aa_m[i, a, i, a] += mo_e_a[occ_a + a] - mo_e_a[i]

        # Create the beta->beta part of the a'-b' stability matrix
        h_bb_m = - np.einsum('abji->iajb', eri_mo_bb[occ_b:, occ_b:, :occ_b, :occ_b])
        h_bb_m += np.einsum('ajbi->iajb', eri_mo_bb[occ_b:, :occ_b, occ_b:, :occ_b])
        for a in range(vir_b):
            for i in range(occ_b):
                h_bb_m[i, a, i, a] += mo_e_b[occ_b + a] - mo_e_b[i]

        # There is no need to create a mixed part since these terms become zero in the equations.
        # Create the complete a'-b' stability matrix
        h_a_min_b = np.zeros((dim_a + dim_b, dim_a + dim_b))
        h_a_min_b[:dim_a, :dim_a] = h_aa_m.reshape(dim_a, dim_a)
        h_a_min_b[dim_a:, dim_a:] = h_bb_m.reshape(dim_b, dim_b)

        # For complex methods we must look at the entire hessian so we need A' and B'
        b_p = (h_a_plus_b - h_a_min_b)/2
        a_p = h_a_plus_b - b_p

        # From here we will start calculating the H'' components. These are used to check UHF->GHF instabilities.
        # We will follow the same approach, so first, A''+B'' is calculated.
        # first: the (alpha->beta, alpha->beta) and (beta->alpha, beta->alpha) part.
        # This part only gets calculated once, since it's the same for A''-B''
        h_abab = - np.einsum('abji->iajb', eri_mo_ab[occ_a:, occ_a:, :occ_b, :occ_b])
        for a in range(vir_a):
            for i in range(occ_b):
                h_abab[i, a, i, a] += mo_e_a[occ_a + a] - mo_e_b[i]
        h_baba = - np.einsum('jiab->iajb', eri_mo_ab[:occ_a, :occ_a, occ_b:, occ_b:])
        for a in range(vir_b):
            for i in range(occ_a):
                h_baba[i, a, i, a] += mo_e_b[occ_b + a] - mo_e_a[i]

        # Create the (alpha->beta, beta->alpha) and (beta->alpha, alpha->beta) part of A''+B''
        h_abba = - np.einsum('ajbi->iajb', eri_mo_ab[occ_a:, :occ_a, occ_b:, :occ_b])
        h_baab = - np.einsum('biaj->iajb', eri_mo_ab[occ_a:, :occ_a, occ_b:, :occ_b])

        # Create the complete A''+B'' stability matrix
        dim_mix_1 = occ_b * vir_a
        dim_mix_2 = occ_a * vir_b
        h_a_plus_b_p = np.empty((dim_mix_1 + dim_mix_2, dim_mix_1 + dim_mix_2))
        h_a_plus_b_p[:dim_mix_1, :dim_mix_1] = h_abab.reshape(dim_mix_1, dim_mix_1)
        h_a_plus_b_p[dim_mix_1:, dim_mix_1:] = h_baba.reshape(dim_mix_2, dim_mix_2)
        h_a_plus_b_p[:dim_mix_1, dim_mix_1:] = h_abba.reshape(dim_mix_1, dim_mix_2)
        h_a_plus_b_p[dim_mix_1:, :dim_mix_1] = h_baab.reshape(dim_mix_2, dim_mix_1)

        # Create the (alpha->beta, beta->alpha) and (beta->alpha, alpha->beta) part of A''-B''
        h_abba_m = np.einsum('ajbi->iajb', eri_mo_ab[occ_a:, :occ_a, occ_b:, :occ_b])
        h_baab_m = np.einsum('biaj->iajb', eri_mo_ab[occ_a:, :occ_a, occ_b:, :occ_b])

        # Create the complete A''-B'' stability matrix
        h_a_min_b_p = np.empty((dim_mix_1 + dim_mix_2, dim_mix_1 + dim_mix_2))
        h_a_min_b_p[:dim_mix_1, :dim_mix_1] = h_abab.reshape(dim_mix_1, dim_mix_1)
        h_a_min_b_p[dim_mix_1:, dim_mix_1:] = h_baba.reshape(dim_mix_2, dim_mix_2)
        h_a_min_b_p[:dim_mix_1, dim_mix_1:] = h_abba_m.reshape(dim_mix_1, dim_mix_2)
        h_a_min_b_p[dim_mix_1:, :dim_mix_1] = h_baab_m.reshape(dim_mix_2, dim_mix_1)

        # To get the separate A'' and B'' components
        b_pp = (h_a_plus_b_p - h_a_min_b_p)/2
        a_pp = h_a_plus_b_p - b_pp

        # Check the different stability matrices to verify the stability.
        if not isinstance(self.energy, complex):
            if method == 'internal':
                # the stability matrix for the real sub problem consists of a + b
                stability_matrix = h_a_plus_b
                self.hessian_p = t.spin_blocked(a_p, b_p, b_p.conj(), a_p.conj())
                # Calculate the eigenvalues of the stability matrix to asses stability
                e, v = la.eigh(stability_matrix)
                lowest_eigenvec = v[:, 0]
                lowest_eigenvec_a = lowest_eigenvec[:occ_a * vir_a]
                lowest_eigenvec_b = lowest_eigenvec[occ_b * vir_b:]
                if np.amin(e) < -1e-5:  # this points towards an instability
                    print("There is an internal instability in the real UHF wave function.")
                    self.int_instability = True
                    return t.rotate_to_eigenvec(lowest_eigenvec_a, mo_a, occ_a, n_orb), \
                        t.rotate_to_eigenvec(lowest_eigenvec_b, mo_b, occ_b, n_orb)

                else:
                    print('The wave function is stable within the real UHF space.')
                    self.int_instability = None

            elif method == 'external':
                # the stability matrix for the complex sub problem consists of a - b
                stability_matrix_1 = h_a_min_b
                self.hessian_p = t.spin_blocked(a_p, b_p, b_p.conj(), a_p.conj())
                stability_matrix_2 = h_a_plus_b_p
                self.hessian_pp = t.spin_blocked(a_pp, b_pp, b_pp.conj(), a_pp.conj())

                # Calculate the eigenvalues of the stability matrix to asses stability
                e_1, v_1 = la.eigh(stability_matrix_1)
                e_2, v_2 = la.eigh(stability_matrix_2)
                if np.amin(e_1) < -1e-5:  # this points towards an instability
                    print("There is an external real/complex instability in the real UHF wave function.")
                    self.ext_instability_rc = True
                if np.amin(e_2) < -1e-5:  # this points towards an instability
                    print("There is an external unrestricted/generalised instability in the real UHF wave function.")
                    self.ext_instability_ug = True
                else:
                    print('The wave function is stable within the real/complex & unrestricted/generalised space.')
                    self.ext_instability_rc = None
                    self.ext_instability_ug = None
            else:
                raise Exception('Only internal and external stability analysis are possible. '
                                'Please enter a valid type.')
        else:
            if method == 'internal':
                # The total stability matrix consists of a & b in the upper corners, and b* and a* in the lower corners
                stability_matrix_3 = t.spin_blocked(a_p, b_p, b_p.conj(), a_p.conj())
                self.hessian_p = stability_matrix_3

                # Calculate the eigenvalues of the stability matrix to asses stability
                e_3, v_3 = la.eigh(stability_matrix_3)
                lowest_eigenvec = v_3[:, 0]
                real_part = lowest_eigenvec[:int(len(lowest_eigenvec) / 2)]
                lowest_real_a = real_part[:occ_a * vir_a]
                lowest_real_b = real_part[occ_b * vir_b:]
                if np.amin(e_3) < -1e-5:  # this points towards an instability
                    print("There is an internal instability in the complex UHF wave function.")
                    self.int_instability = True
                    return t.rotate_to_eigenvec(lowest_real_a, mo_a, occ_a, n_orb), \
                        t.rotate_to_eigenvec(lowest_real_b, mo_b, occ_b, n_orb)
                else:
                    print('The wave function is stable in the complex UHF space.')
                    self.int_instability = None

            elif method == 'external':
                # The total stability matrix consists of a'' & b'' in the upper corners, and b''* and a''*
                # in the lower corners
                stability_matrix_4 = t.spin_blocked(a_pp, b_pp, b_pp.conj(), a_pp.conj())
                self.hessian_pp = stability_matrix_4
                e_4, v_4 = la.eigh(stability_matrix_4)

                if np.amin(e_4) < -1e-5:  # this points towards an instability
                    print("There is an external unrestricted/generalised instability in the complex UHF wave function.")
                    self.ext_instability_ug = True
                else:
                    print('The wave function is stable within the complex UHF space.')
                    self.ext_instability_ug = None
            else:
                raise Exception('Only internal and external stability analysis are possible. '
                                'Please enter a valid type.')

    def get_hessian(self, prime):
        """
        Get the Hessian matrix after performing a stability analysis.
        :param: specify whether you want to look at H' or H'' by putting 1 or 2
        :return: The hessian matrix
        """
        if prime == 1:
            return self.hessian_p
        elif prime == 2:
            return self.hessian_pp
        else:
            raise Exception('Specify whether you want H prime or H double-prime, '
                            'by giving either 1 or 2 to the function. Other numbers are invalid. ')

    def diis(self, initial_guess=None, convergence=1e-12, complex_method=False):
        """
        When needed, DIIS can be used to speed up the UHF calculations by reducing the needed iterations.

        :param initial_guess: Initial guess for the scf procedure. None specified: core Hamiltonian.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: scf energy, number of iterations, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = Scf.trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        # If no initial guess is given, use the orthogonalised core Hamiltonian
        # Else, use the given initial guess.
        # create guess density matrix from core guess, separate for alpha and beta and put them into an array
        if initial_guess is None:
            if complex_method:
                initial_guess = s_12.conj().T @ self.get_one_e() @ s_12
                initial_guess = initial_guess.astype(complex)
                guess_density_a = Scf.density_matrix(initial_guess, self.n_a, s_12)
                guess_density_a[0, :] += 0.1j
                guess_density_a[:, 0] -= 0.1j
                guess_density_b = Scf.density_matrix(initial_guess, self.n_b, s_12)
                guess_density_b[0, :] += 0.1j
                guess_density_b[:, 0] -= 0.1j
            else:
                initial_guess = s_12.conj().T @ self.get_one_e() @ s_12
                guess_density_a = Scf.density_matrix(initial_guess, self.n_a, s_12)
                guess_density_b = Scf.density_matrix(initial_guess, self.n_b, s_12)

        else:
            # Make the coefficients orthogonal in the correct basis.
            coeff_a = s_12 @ initial_guess[0]
            coeff_b = s_12 @ initial_guess[1]

            # Get C_alpha and C_beta
            coeff_r_a = coeff_a[:, 0:self.n_a]
            coeff_r_b = coeff_b[:, 0:self.n_b]

            if complex_method:
                # Create the guess density matrices from the given coefficients
                guess_density_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a).astype(complex)
                guess_density_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b).astype(complex)
            else:
                guess_density_a = np.einsum('ij,kj->ik', coeff_r_a, coeff_r_a)
                guess_density_b = np.einsum('ij,kj->ik', coeff_r_b, coeff_r_b)

        def uhf_fock(density_matrix_1, density_matrix_2):
            """
            - calculate a fock matrix from a given alpha and beta density matrix
            - fock alpha if 1 = alpha and 2 = beta and vice versa
            - input is the density matrix for alpha and beta, a one electron matrix and a two electron tensor.
            """
            j_1 = np.einsum('pqrs,rs->pq', self.get_two_e(), density_matrix_1)
            j_2 = np.einsum('pqrs,rs->pq', self.get_two_e(), density_matrix_2)
            k_1 = np.einsum('prqs,rs->pq', self.get_two_e(), density_matrix_1)
            f = self.get_one_e() + (j_1 + j_2) - k_1
            return f

        def residual(density, f):
            """
            Function that calculates the error matrix for the DIIS algorithm
            :param density: density matrix
            :param f: fock matrix
            :return: a value that should be zero and a fock matrix
            """
            return s_12 @ (f @ density @ self.get_ovlp() - self.get_ovlp() @ density @ f) @ s_12.conj().T

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
                f = np.zeros(focks[0].shape).astype(complex)
            else:
                f = np.zeros(focks[0].shape)
            for x in range(coeff.shape[0] - 1):
                f += coeff[x] * focks[x]
            return f

        # Create the necessary arrays to perform an iterative diis procedure
        densities_diis_a = [guess_density_a]
        densities_diis_b = [guess_density_b]
        energies_diis = [0.0]
        delta_e_diis = []
        f_list = []
        fo_list = []
        mo_list = []

        def iteration_diis(number_of_iterations):
            # Create the alpha and beta fock matrices
            f_a = uhf_fock(densities_diis_a[-1], densities_diis_b[-1])
            f_b = uhf_fock(densities_diis_b[-1], densities_diis_a[-1])

            # Calculate the residuals from both
            resid_a = residual(densities_diis_a[-1], f_a)
            resid_b = residual(densities_diis_b[-1], f_b)

            # Add everything to their respective list
            fock_list_a.append(f_a)
            fock_list_b.append(f_b)
            error_list_a.append(resid_a)
            error_list_b.append(resid_b)

            # Calculate the energy and energy difference
            energies_diis.append(Scf.uhf_scf_energy(densities_diis_a[-1], densities_diis_b[-1], f_a, f_b,
                                                    self.get_one_e()))
            delta_e_diis.append(energies_diis[-1] - energies_diis[-2])

            # Starting at two iterations, use the DIIS acceleration
            if number_of_iterations >= 2:
                f_a = diis_fock(fock_list_a, error_list_a)
                f_b = diis_fock(fock_list_b, error_list_b)

            # fill the arrays
            f_list.append([f_a, f_b])

            # Orthogonalise the fock matrices
            f_orth_a = s_12.conj().T @ f_a @ s_12
            f_orth_b = s_12.conj().T @ f_b @ s_12
            fo_list.append([f_orth_a, f_orth_b])

            # MO coefficients
            mos_a = Scf.calc_mo(f_orth_a, s_12)
            mos_b = Scf.calc_mo(f_orth_b, s_12)
            mo_list.append([mos_a, mos_b])

            # Calculate the new density matrices
            new_density_a = Scf.density_matrix(f_orth_a, self.n_a, s_12)
            new_density_b = Scf.density_matrix(f_orth_b, self.n_b, s_12)

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
        def dens():
            return densities_diis_a, densities_diis_b
        self.dens = dens()

        # a function that gives the last Fock matrix of the scf procedure
        def fock():
            return f_list
        self.focks = fock()
        self.fock_orth = fo_list

        # A function that returns the converged mo coefficients
        def get_mo():
            return mo_list
        self.mo = get_mo()

        # calculate the total energy, taking nuclear repulsion into account
        scf_e = energies_diis[-1] + self.nuc_rep()
        self.energy = scf_e

        return scf_e, i, get_mo(), dens(), fock()

    def get_scf_solution_diis(self, guess=None, convergence=1e-12, complex_method=False):
        """
        Prints the number of iterations and the converged diis energy.
        Also prints the expectation value of S_z, S^2 and the multiplicity.

        :param guess: The initial guess. If none is specified, core Hamiltonian.
        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: The converged diis energy.
        """
        self.diis(guess, convergence=convergence, complex_method=complex_method)
        e = self.energy
        s_values = spin.uhf(self.n_a, self.n_b, MF.get_mo_coeff(self)[0], MF.get_mo_coeff(self)[1], self.get_ovlp())
        if complex_method:
            if abs(np.imag(e)) > 1e-12:
                print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
            else:
                print("Number of iterations: " + str(self.iterations))
                print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex UHF, DIIS)")
                print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) +
                      ", Multiplicity = " + str(s_values[2]))
        else:
            print("Number of iterations: " + str(self.iterations))
            print("Converged SCF energy in Hartree: " + str(self.energy) + " (Real UHF, DIIS)")
            print("<S^2> = " + str(s_values[0]) + ", <S_z> = " + str(s_values[1]) +
                  ", Multiplicity = " + str(s_values[2]))
        return self.energy
