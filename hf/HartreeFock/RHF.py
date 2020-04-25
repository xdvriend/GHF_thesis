"""
Restricted Hartree Fock, by means of SCF procedure
==================================================
This class is used to calculate the RHF energy of a given molecule and the number of electrons.
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)
"""

import hf.utilities.SCF_functions as Scf
import hf.utilities.transform as t
import numpy as np
from scipy import linalg as la
import collections as c
from pyscf import *


class MF:
    """
        calculate RHF energy.
        ---------------------
        Input is a molecule and the number of electrons.

        Molecules are made in pySCF and calculations are performed as follows, eg.:
        The following snippet prints and returns RHF energy of h_2
        and the number of iterations needed to get this value.

        >>> from hf.HartreeFock import *
        >>> h_2 = gto.M(atom = 'h 0 0 0; h 0 0 1', spin = 0, basis = 'sto-3g')
        >>> x = RHF.MF(h_2, 2)
        >>> x.get_scf_solution()
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
            self.integrals = Scf.get_integrals_pyscf(molecule)
        elif int_method == 'psi4':
            self.integrals = Scf.get_integrals_psi4(molecule)
        else:
            raise Exception('Unsupported method to calculate integrals. Supported methods are pyscf or psi4. '
                            'Make sure the molecule instance matches the method and is gives as a string.')
        self.energy = None
        self.mo = None
        self.dens = None
        self.fock = None
        self.fock_orth = None
        self.iterations = None
        self.int_instability = None
        self.ext_instability = None
        self.hessian = None
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

    def scf(self, convergence=1e-12, complex_method=False):
        """
        Performs a self consistent field calculation to find the lowest RHF energy.

        :param convergence: Convergence criterion. If none is specified, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: number of iterations, scf energy, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = Scf.trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        if complex_method:
            core_guess = s_12 @ self.get_one_e() @ s_12.conj().T  # orthogonalise the transformation matrix.
            core_guess = core_guess.astype(complex)
            guess_density = Scf.density_matrix(core_guess, self.occupied, s_12)  # calculate the guess density
            guess_density[0, :] += 0.1j
            guess_density[:, 0] -= 0.1j
        else:
            core_guess = s_12 @ self.get_one_e() @ s_12.conj().T  # orthogonalise the transformation matrix.
            guess_density = Scf.density_matrix(core_guess, self.occupied, s_12)

        densities = [guess_density]  # put the guess density in an array

        def rhf_scf_energy(dens_matrix, f):
            """calculate the scf energy value from a given density matrix and a given fock matrix"""
            return np.einsum('pq, pq->', (self.get_one_e() + f), dens_matrix)

        energies = [0.0]

        def rhf_fock_matrix(dens_matrix):
            """calculate a fock matrix from a given density matrix"""
            # jk_integrals = coulomb - exchange
            jk_integrals = 2 * self.get_two_e() - self.get_two_e().transpose(0, 2, 1, 3)
            # double summation of density matrix * (coulomb - exchange)
            return self.get_one_e() + np.einsum('kl,ijkl->ij', dens_matrix, jk_integrals)

        # create arrays to store the wanted components
        delta_e = []
        f_list = []
        f_o_list = []
        mo_list = []

        def iteration():
            """create an iteration procedure, calculate fock from density,
            orthogonalise, new density from new fock,..."""
            f = rhf_fock_matrix(densities[-1])  # calculate fock matrix from the newest density matrix
            f_list.append(f)
            # calculate the electronic energy and put it into the energies array
            energies.append(rhf_scf_energy(densities[-1], f) + self.nuc_rep())
            # calculate the energy difference and add it to the correct array
            delta_e.append(energies[-1] - energies[-2])
            # orthogonalize the new fock matrix
            # calculate density matrix from the new fock matrix
            fock_orth = s_12.conj().T.dot(f).dot(s_12)
            mos = Scf.calc_mo(fock_orth, s_12)
            mo_list.append(mos)
            f_o_list.append(fock_orth)
            new_density = Scf.density_matrix(fock_orth, self.occupied, s_12)
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
        def dens():
            return densities
        self.dens = dens()

        # a function that gives the last Fock matrix of the scf procedure
        def fock():
            return f_list
        self.fock = fock()
        self.fock_orth = f_o_list

        # A function that returns the converged mo coefficients
        def get_mo():
            return mo_list
        self.mo = get_mo()

        # calculate the total energy, taking nuclear repulsion into account
        scf_e = energies[-1]
        self.energy = scf_e

        return scf_e, i, get_mo(), dens(), fock()

    def get_scf_solution(self, convergence=1e-12, complex_method=False):
        """
        Prints the number of iterations and the converged scf energy.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: the converged energy
        """
        self.scf(convergence=convergence, complex_method=complex_method)
        e = self.energy
        if complex_method:
            if abs(np.imag(e)) > 1e-12:
                print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
            else:
                print("Number of iterations: " + str(self.iterations))
                print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex RHF)")
        else:
            print("Number of iterations: " + str(self.iterations))
            print("Converged SCF energy in Hartree: " + str(self.energy) + " (Real RHF)")
        return self.energy

    def get_mo_coeff(self, i=-1):
        """
        Returns mo coefficients.

        :return: The mo coefficients
        """
        return self.mo[i]

    def get_dens(self, i=-1):
        """
        Returns the (last) density matrix.

        :return: The last density matrix.
        """
        return self.dens[i]

    def get_fock(self, i=-1):
        """
        Returns the (last) fock matrix.

        :return: The last Fock matrix.
        """
        return self.fock[i]

    def get_fock_orth(self, i=-1):
        """
        Returns the (last) fock matrix in the orthonormal basis.

        :return: The last Fock matrix.
        """
        return self.fock_orth[i]

    def get_mo_energy(self):
        """
        Returns the MO energies.
        :return: an array of MO energies.
        """
        e = Scf.calc_mo_e(self.get_fock_orth())
        return e

    def diis(self, convergence=1e-12, complex_method=False):
        """
        When needed, DIIS can be used to speed up the RHF calculations by reducing the needed iterations.

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: scf energy, number of iterations, mo coefficients, last density matrix, last fock matrix
        """
        s_12 = Scf.trans_matrix(self.get_ovlp())  # calculate the transformation matrix
        if complex_method:
            core_guess = s_12 @ self.get_one_e() @ s_12.conj().T  # orthogonalise the transformation matrix.
            core_guess = core_guess.astype(complex)
            guess_density = Scf.density_matrix(core_guess, self.occupied, s_12)  # calculate the guess density
            guess_density[0, :] += 0.1j
            guess_density[:, 0] -= 0.1j
        else:
            core_guess = s_12 @ self.get_one_e() @ s_12.conj().T  # orthogonalise the transformation matrix.
            guess_density = Scf.density_matrix(core_guess, self.occupied, s_12)

        def rhf_fock_matrix(dens_matrix):
            """calculate a fock matrix from a given density matrix"""
            # jk_integrals = coulomb - exchange
            jk_integrals = 2 * self.get_two_e() - self.get_two_e().transpose(0, 2, 1, 3)
            # double summation of density matrix * (coulomb - exchange)
            return self.get_one_e() + np.einsum('kl,ijkl->ij', dens_matrix, jk_integrals)

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
                f = np.zeros(focks[0].shape).astype(complex)
            else:
                f = np.zeros(focks[0].shape)
            for x in range(coeff.shape[0] - 1):
                f += coeff[x] * focks[x]
            return f

        def rhf_scf_energy(dens_matrix, f):
            """calculate the scf energy value from a given density matrix and a given fock matrix"""
            return np.einsum('pq, pq->', (self.get_one_e() + f), dens_matrix)

        # Create the necessary arrays to perform an iterative diis procedure
        densities_diis = [guess_density]
        energies_diis = [0.0]
        delta_e = []
        f_list = []
        f_o_list = []
        mo_list = []

        def diis_iteration(number_of_iterations):
            # Calculate the fock matrix and the residual
            f = rhf_fock_matrix(densities_diis[-1])
            resid = residual(densities_diis[-1], f)

            # Add them to their respective lists
            fock_list.append(f)
            error_list.append(resid)

            # Calculate the energy and the energy difference
            energies_diis.append(rhf_scf_energy(densities_diis[-1], f) + self.nuc_rep())
            delta_e.append(energies_diis[-1] - energies_diis[-2])

            # Starting at two iterations, use the DIIS acceleration
            if number_of_iterations >= 2:
                f = diis_fock(fock_list, error_list)

            # fill the arrays
            f_list.append(f)

            # orthogonalize the new fock matrix
            # calculate density matrix from the new fock matrix
            fock_orth = s_12.conj().T.dot(f).dot(s_12)
            f_o_list.append(fock_orth)
            mos = Scf.calc_mo(fock_orth, s_12)
            mo_list.append(mos)
            new_density = Scf.density_matrix(fock_orth, self.occupied, s_12)

            # put new density matrix in the densities array
            densities_diis.append(new_density)

        i = 1
        diis_iteration(i)
        while abs(delta_e[-1]) >= convergence:
            diis_iteration(i)
            i += 1
        self.iterations = i

        # a function that gives the last density matrix of the scf procedure
        def dens():
            return densities_diis

        self.dens = dens()

        # a function that gives the last Fock matrix of the scf procedure
        def fock():
            return f_list

        self.fock = fock()
        self.fock_orth = f_o_list

        # A function that returns the converged mo coefficients
        def get_mo():
            return mo_list

        self.mo = get_mo()

        # calculate the total energy, taking nuclear repulsion into account
        scf_e = energies_diis[-1]
        self.energy = scf_e

        return scf_e, i, get_mo(), dens(), fock()

    def get_scf_solution_diis(self, convergence=1e-12, complex_method=False):
        """
        Prints the number of iterations and the converged DIIS energy. The number of iterations will be lower than with
        a normal scf, but the energy value will be the same. Example:

        >>> from hf.HartreeFock import *
        >>> h2 = gto.M(atom = 'h 0 0 0; h 1 0 0', basis = 'cc-pvdz')
        >>> x = RHF.MF(h2, 2)
        >>> x.get_scf_solution_diis()

        :param convergence: Set the convergence criterion. If none is given, 1e-12 is used.
        :param complex_method: Specify whether or not you want to work in the complex space. Default is real.
        :return: The converged scf energy, using DIIS.
        """
        self.diis(convergence=convergence, complex_method=complex_method)
        e = self.energy
        if complex_method:
            if abs(np.imag(e)) > 1e-12:
                print("Energy value is complex." + " (" + str(np.imag(e)) + "i)")
            else:
                print("Number of iterations: " + str(self.iterations))
                print("Converged SCF energy in Hartree: " + str(np.real(e)) + " (Complex RHF, DIIS)")
        else:
            print("Number of iterations: " + str(self.iterations))
            print("Converged SCF energy in Hartree: " + str(self.energy) + " (Real RHF, DIIS)")
        return self.energy

    def stability_analysis(self, method):
        """
        Internal stability analysis to verify whether the wave function is stable within the space of the used method.
        :param method: Indicate whether you want to check the internal or external stability of the wave function. Can
        be internal or external.
        :param step_size: Step size for orbital rotation. standard is 1e-4.
        :return: In case of internal stability analysis, it returns a new set of coefficients.
        """
        # Determine the number of occupied and virtual orbitals.
        # Determine coefficients, mo energies, and other needed parameters.
        occ = int(self.occupied)
        vir = int(np.shape(self.get_ovlp())[0] - occ)
        coeff = self.get_mo_coeff()
        mo_e = self.get_mo_energy()
        mo_e_vir = mo_e[occ:]
        mo_e_occ = mo_e[:occ]

        # Determine the two electron integrals in MO basis.
        eri_ao = self.get_two_e()
        eri_mo = t.tensor_basis_transform(eri_ao, coeff)

        # Create A_singlet
        a1 = np.einsum('ckld->kcld', eri_mo[occ:, :occ, :occ, occ:]) * 2
        a1 -= np.einsum('cdlk->kcld', eri_mo[occ:, occ:, :occ, :occ])

        e_values = np.zeros((int(len(mo_e_vir)), int(len(mo_e_occ))))
        for i in range(vir):
            for j in range(occ):
                e_values[i][j] = mo_e_vir[i] + (-1 * mo_e_occ[j])

        for a in range(vir):
            for i in range(occ):
                a1[i, a, i, a] += e_values[a, i]

        # Create B_singlet
        b1 = np.einsum('ckdl->kcld', eri_mo[occ:, :occ, occ:, :occ]) * 2
        b1 -= np.einsum('cldk->kcld', eri_mo[occ:, :occ, occ:, :occ])

        # Create A_triplet
        a3 = - np.einsum('cdlk->kcld', eri_mo[occ:, occ:, :occ, :occ])
        for a in range(vir):
            for i in range(occ):
                a3[i, a, i, a] += e_values[a, i]

        # Create B_triplet
        b3 = np.einsum('cldk->kcld', eri_mo[occ:, :occ, occ:, :occ])

        # reshape to matrices
        a1 = a1.reshape((occ * vir, occ * vir))
        b1 = b1.reshape((occ * vir, occ * vir))
        a3 = a3.reshape((occ * vir, occ * vir))
        b3 = b3.reshape((occ * vir, occ * vir))

        # # Create a function to rotate the orbitals in case of internal instability
        # def rotate_to_eigenvec(eigenvec):
        #     if isinstance(self.energy, complex):
        #         indx = int(np.shape(eigenvec)[0] / 2)
        #         eigenvec = eigenvec[:indx]
        #
        #     block_ba = eigenvec.reshape((occ, vir), order='F')
        #     block_bb = np.zeros((occ, occ))
        #     block_ab = block_ba.conj().T
        #     block_aa = np.zeros((vir, vir))
        #     k = t.spin_blocked(block_aa, block_ab, block_ba, block_bb)
        #     coeff_init = self.get_mo_coeff()
        #     exp = la.expm(-1 * step_size * k)
        #     return coeff_init @ exp

        # Check the different stability matrices to verify the stability.
        if not isinstance(self.energy, complex):
            if method == 'internal':
                # the stability matrix for the real sub problem consists of a + b
                stability_matrix = a1 + b1  # real restricted internal

                # Calculate the eigenvalues of the stability matrix to asses stability
                e, v = la.eigh(stability_matrix)
                # lowest_eigenvec = v[:, 0]
                if np.amin(e) < -1e-5:  # this points towards an instability
                    print("There is an internal instability in the real RHF wave function.")
                    self.int_instability = True
                else:
                    print('The wave function is stable within the real RHF space.')
                    self.int_instability = None

            elif method == 'external':
                # the stability matrix for the complex sub problem consists of a - b
                stability_matrix_1 = a1 - b1  # real -> complex
                stability_matrix_3 = a3 - b3  # restricted -> unrestricted

                # Calculate the eigenvalues of the stability matrix to asses stability
                e_1, v_1 = la.eigh(stability_matrix_1)
                e_3, v_3 = la.eigh(stability_matrix_3)
                if np.amin(e_1) < -1e-5:  # this points towards an instability
                    print("There is an external real/complex instability in the real RHF wave function.")
                    self.ext_instability = True
                if np.amin(e_3) < -1e-5:
                    print("There is an external restricted/unrestricted instability in the real RHF wave function.")
                    self.ext_instability = True
                else:
                    print('The wave function is stable within the real/complex & RHF/UHF space.')
                    self.ext_instability = None
            else:
                raise Exception('Only internal and external stability analysis are possible. '
                                'Please enter a valid type.')
        else:
            if method == 'internal':
                # The total stability matrix consists of a & b in the upper corners, and b* and a* in the lower corners
                stability_matrix = t.spin_blocked(a1, b1, b1.conj(), a1.conj())

                # Calculate the eigenvalues of the stability matrix to asses stability
                e, v = la.eigh(stability_matrix)
                if np.amin(e) < -1e-5:  # this points towards an instability
                    print("There is an internal instability in the complex RHF wave function.")
                    self.int_instability = True

                else:
                    print('The wave function is stable within the complex RHF space.')
                    self.int_instability = None
            elif method == 'external':
                # The total stability matrix consists of a & b in the upper corners, and b* and a* in the lower corners
                stability_matrix = t.spin_blocked(a3, b3, b3.conj(), a3.conj())

                # Calculate the eigenvalues of the stability matrix to asses stability
                e, v = la.eigh(stability_matrix)
                if np.amin(e) < -1e-5:  # this points towards an instability
                    print("There is an external RHF/UHF instability in the complex RHF wave function.")
                    self.ext_instability = True

                else:
                    print('The wave function is stable within the complex RHF space.')
                    self.ext_instability = None
            else:
                raise Exception('Only internal and external stability analysis are possible. '
                                'Please enter a valid type.')
