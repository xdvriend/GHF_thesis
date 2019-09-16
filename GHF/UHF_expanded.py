from GHF.SCF_functions import *

"""Unrestricted Hartree Fock by means of SCF procedure"""

def UHF(molecule, occ_a, occ_b, extra_e_coeff=False ,stability_analysis=False):
    if extra_e_coeff==False and stability_analysis==False:
        overlap = get_integrals(molecule)[0]
        one_electron = get_integrals(molecule)[1]
        two_electron = get_integrals(molecule)[2]
        nuclear_repulsion = get_integrals(molecule)[3]

        X = trans_matrix(overlap)
        core_guess = X.T.dot(one_electron).dot(X)

        def density(f_matrix, occ):
            f_eigenvalues, f_eigenvectors = la.eigh(f_matrix)  # eigenvalues are initial orbital energies
            coefficients = X.dot(f_eigenvectors)
            coefficients_r = coefficients[:, 0:occ]  # summation over occupied orbitals
            return np.einsum('ij,kj->ik', coefficients_r, coefficients_r)

        guess_density_a = density(core_guess, occ_a)
        guess_density_b = density(core_guess, occ_b)

        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        def scf_energy(density_matrix_a, density_matrix_b, fock_a, fock_b):
            scf_e = np.einsum('ij, ij->', density_matrix_a + density_matrix_b, one_electron)
            scf_e += np.einsum('ij, ij->', density_matrix_a, fock_a)
            scf_e += np.einsum('ij, ij->', density_matrix_b, fock_b)
            scf_e *= 0.5
            return scf_e

        electronic_e = scf_energy(guess_density_a, guess_density_b, one_electron, one_electron)
        energies = [electronic_e]

        def fock_matrix(density_matrix_a, density_matrix_b):
            jk_integrals = two_electron - two_electron.transpose(0, 2, 1, 3)
            jk_a = np.einsum('kl,ijkl->ij', density_matrix_a, jk_integrals)
            j_b = np.einsum('kl, ijkl->ij', density_matrix_b, two_electron)
            return one_electron + jk_a + j_b

        delta_e = []

        def iteration():
            fock_a = fock_matrix(densities_a[-1], densities_b[-1])
            fock_b = fock_matrix(densities_b[-1], densities_a[-1])

            fock_orth_a = X.T.dot(fock_a).dot(X)
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density(fock_orth_a, occ_a)
            new_density_b = density(fock_orth_b, occ_b)

            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

            energies.append(scf_energy(new_density_a, new_density_b, fock_a, fock_b))

            delta_e.append(energies[-1] - energies[-2])

        iteration()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration()
            i += 1

        total_e = energies[-1] + nuclear_repulsion

        print("Number of iterations: " + str(i))
        print("Converged SCF energy in Hartree: " + str(total_e) + " (UHF)")

        return total_e

    if extra_e_coeff:
        molecule.nelectron = occ_a + occ_b + 2
        molecule.build()

        overlap_t = get_integrals(molecule)[0]
        one_electron_t = get_integrals(molecule)[1]
        two_electron_t = get_integrals(molecule)[2]

        X_t = trans_matrix(overlap_t)
        core_guess_t = X_t.T.dot(one_electron_t).dot(X_t)

        def density(f_matrix, occ, trans):
            f_eigenvalues, f_eigenvectors = la.eigh(f_matrix)  # eigenvalues are initial orbital energies
            coefficients = trans.dot(f_eigenvectors)
            coefficients_r = coefficients[:, 0:occ]  # summation over occupied orbitals
            return np.einsum('ij,kj->ik', coefficients_r, coefficients_r), coefficients

        guess_density_a_t = density(core_guess_t, occ_a + 2, X_t)[0]
        guess_density_b_t = density(core_guess_t, occ_b, X_t)[0]

        densities_a = [guess_density_a_t]
        densities_b = [guess_density_b_t]

        def fock_matrix(density_matrix_a, density_matrix_b, one_e, two_e):
            jk_integrals = two_electron_t - two_electron_t.transpose(0, 2, 1, 3)
            jk_a = np.einsum('kl,ijkl->ij', density_matrix_a, jk_integrals)
            j_b = np.einsum('kl, ijkl->ij', density_matrix_b, two_e)
            return one_e + jk_a + j_b

        delta_dens = []

        def iteration_t():
            fock_a = fock_matrix(densities_a[-1], densities_b[-1], one_electron_t, two_electron_t)
            fock_b = fock_matrix(densities_b[-1], densities_a[-1], one_electron_t, two_electron_t)

            fock_orth_a = X_t.T.dot(fock_a).dot(X_t)
            fock_orth_b = X_t.T.dot(fock_b).dot(X_t)

            new_density_a = density(fock_orth_a, occ_a + 2, X_t)[0]
            new_density_b = density(fock_orth_b, occ_b, X_t)[0]

            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

            delta_dens.append(np.sum(densities_a[-1] + densities_b[-1]) - np.sum(densities_a[-2] + densities_b[-2]))


        iteration_t()
        while abs(delta_dens[-1]) >= 1e-12:
            iteration_t()

        val_a, coeff_a = la.eigh(densities_a[-1])
        val_b, coeff_b = la.eigh(densities_b[-1])

        molecule.nelectron = None
        molecule.build()

        overlap = get_integrals(molecule)[0]
        one_electron = get_integrals(molecule)[1]
        two_electron = get_integrals(molecule)[2]
        nuclear_repulsion = get_integrals(molecule)[3]

        X = trans_matrix(overlap)

        imp_guess_density_a = coeff_a
        imp_guess_density_b = coeff_b

        imp_dens_a = [imp_guess_density_a]
        imp_dens_b = [imp_guess_density_b]

        def scf_energy(density_matrix_a, density_matrix_b, fock_a, fock_b):
            scf_e = np.einsum('ij, ij->', density_matrix_a + density_matrix_b, one_electron)
            scf_e += np.einsum('ij, ij->', density_matrix_a, fock_a)
            scf_e += np.einsum('ij, ij->', density_matrix_b, fock_b)
            scf_e *= 0.5
            return scf_e

        energies = [scf_energy(imp_dens_a[-1], imp_dens_b[-1], one_electron, one_electron)]
        delta_e = []

        def iteration():
            fock_a = fock_matrix(imp_dens_a[-1], imp_dens_b[-1], one_electron, two_electron)
            fock_b = fock_matrix(imp_dens_b[-1], imp_dens_a[-1], one_electron, two_electron)

            fock_orth_a = X.T.dot(fock_a).dot(X)
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density(fock_orth_a, occ_a, X)[0]
            new_density_b = density(fock_orth_b, occ_b, X)[0]

            imp_dens_a.append(new_density_a)
            imp_dens_b.append(new_density_b)

            energies.append(scf_energy(new_density_a, new_density_b, fock_a, fock_b) + nuclear_repulsion)
            delta_e.append(energies[-1] - energies[-2])

        iteration()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration()
            i += 1

        total_e = energies[-1]

        print("Number of iterations: " + str(i))
        print("Converged SCF energy in Hartree: " + str(total_e) + " (UHF)")

        return total_e
