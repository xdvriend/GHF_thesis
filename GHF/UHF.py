from GHF.SCF_functions import *

"""Unrestricted Hartree Fock by means of SCF procedure"""

def UHF(molecule, occ_a, occ_b):

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
        j_b = np.einsum('kl, ijkl->ij', density_matrix_b, two_electron )
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



