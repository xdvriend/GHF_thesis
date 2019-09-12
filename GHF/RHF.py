from GHF.SCF_functions import *

"""Restricted Hartree Fock, by means of SCF procedure"""

def RHF(molecule, number_occupied_orbitals):

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

    guess_density = density(core_guess, number_occupied_orbitals)
    densities = [guess_density]

    def scf_energy(density_matrix, fock):
        return np.sum(density_matrix * (one_electron + fock))

    electronic_e = np.sum(guess_density * one_electron * 2)
    energies = [electronic_e]

    def fock_matrix(density_matrix):
        jk_integrals = 2 * two_electron - two_electron.transpose(0, 2, 1, 3)
        return one_electron + np.einsum('kl,ijkl->ij', density_matrix, jk_integrals)

    delta_e = []

    def iteration():
        fock = fock_matrix(densities[-1])
        fock_orth = X.T.dot(fock).dot(X)
        new_density = density(fock_orth, number_occupied_orbitals)
        densities.append(new_density)
        energies.append(scf_energy(new_density, fock))
        delta_e.append(energies[-1] - energies[-2])

    iteration()
    i = 1
    while abs(delta_e[-1]) >= 1e-12:
        iteration()
        i += 1
    
    scf_e = energies[-1] + nuclear_repulsion
    print("Number of iterations: " + str(i))
    print("Converged SCF energy in Hartree: " + str(scf_e) + " (RHF)")
    return scf_e
