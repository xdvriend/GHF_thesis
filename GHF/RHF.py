from GHF.SCF_functions import *

"""Restricted Hartree Fock, by means of SCF procedure

This function calculates the RHF energy of a given molecule and the number of occupied orbitals. 
The molecule has to be created in pySCF: 
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)

number of occupied orbitals:
number of doubly occupied orbitals (1 for H_2, 2 for H_4,...)

The function prints the number of iterations and the converged SCF energy, while also returning the energy value
for eventual subsequent calculations"""

def RHF(molecule, number_occupied_orbitals):
# get_integrals (from GHF.SCF_functions) calculates the overlap matrix, one_electron integrals, two_electron_integrals
# and the nuclear repulsion value.
    overlap = get_integrals(molecule)[0]
    one_electron = get_integrals(molecule)[1]
    two_electron = get_integrals(molecule)[2]
    nuclear_repulsion = get_integrals(molecule)[3]

# The trans_matrix function calculates the orthogonalisation matrix from a given overlap matrix.
    X = trans_matrix(overlap)

# core_guess = the guess in the case where the electrons don't interact with each other
    core_guess = X.T.dot(one_electron).dot(X)

# density() creates a density matrix from a fock matrix and the number of occupied orbitals
    def density(f_matrix, occ):
        f_eigenvalues, f_eigenvectors = la.eigh(f_matrix)  # eigenvalues are initial orbital energies
        coefficients = X.dot(f_eigenvectors)
        coefficients_r = coefficients[:, 0:occ]  # summation over occupied orbitals
        return np.einsum('ij,kj->ik', coefficients_r, coefficients_r)  # np.einsum represents Sum_j^occupied_orbitals(c_ij * c_kj)

# create guess density matrix from core guess and put it into an array
    guess_density = density(core_guess, number_occupied_orbitals)
    densities = [guess_density]

# calculate the scf energy value from a given density matrix and a given fock matrix
# then calculate the initial electronic energy and put it into an array
    def scf_energy(density_matrix, fock):
        return np.sum(density_matrix * (one_electron + fock))

    electronic_e = np.sum(guess_density * one_electron * 2)
    energies = [electronic_e]

# calculate a fock matrix from a given density matrix
    def fock_matrix(density_matrix):
        jk_integrals = 2 * two_electron - two_electron.transpose(0, 2, 1, 3) # jk_integrals = coulomb - exchange
        return one_electron + np.einsum('kl,ijkl->ij', density_matrix, jk_integrals) # double summation of density matrix * (coulomb - exchange)

# create an array to check the energy differences between iterations
    delta_e = []

# create an iteration procedure
    def iteration():
        fock = fock_matrix(densities[-1]) # calculate fock matrix from the newest density matrix
        fock_orth = X.T.dot(fock).dot(X)  # orthogonalize the new fock matrix
        new_density = density(fock_orth, number_occupied_orbitals)  # calculate density matrix from the new fock matrix
        densities.append(new_density)  # put new density matrix in the densities array
        energies.append(scf_energy(new_density, fock)) # calculate the electronic energy and put it into the energies array
        delta_e.append(energies[-1] - energies[-2])  # calculate the energy difference and add it to the correct array

# start and continue the iteration process as long as the energy difference is larger than 1e-12
    iteration()
    i = 1
    while abs(delta_e[-1]) >= 1e-12:
        iteration()
        i += 1

# calculate the total energy by adding the nuclear repulsion value and print the results
    scf_e = energies[-1] + nuclear_repulsion
    print("Number of iterations: " + str(i))
    print("Converged SCF energy in Hartree: " + str(scf_e) + " (RHF)")
    return scf_e
