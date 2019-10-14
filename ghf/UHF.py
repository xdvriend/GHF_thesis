"""
Unrestricted Hartree Fock, by means of SCF procedure
====================================================


This function calculates the UHF energy for a given molecule and the number of occupied alpha and beta orbitals. 
The molecule has to be created in pySCF:
molecule = gto.M(atom = geometry, spin = diff. in alpha and beta electrons, basis = basis set)

number of occupied orbitals:
first the number of occupied alpha orbitals
second the number of occupied beta orbitals

There are two extra options added into the function:

- extra_e_coeff: when true, the program will add two extra alpha electrons to the system and calculate when the new system's energy converges. From the last density matrix in this calculation, new coefficients for the alpha and beta orbitals will be calculated and used as new guess density for the original system.

- internal_stability_analysis: when true, an internal stability analysis will be performed. When the wave finction is deemed unstable, new coefficients will be calculated that are closer in value to the stable condition, which is done by varying the Hessian by means of a trial vector. These will then be used as a guess density for the energy calculation.

The function prints the number of iterations and the converged SCF energy, while also returning the energy value
for eventual subsequent calculations
"""
from ghf.SCF_functions import *
from functools import reduce
from pyscf import *
import scipy.linalg


def UHF(molecule, occ_a, occ_b, extra_e_coeff=False, internal_stability_analysis=False):
    """
    Calculate UHF energy, with or without adding extra electrons to improve coefficients or internal stability analysis.
    --------------------------------------------------------------------------------------------------------------------

    Without extra steps
    +++++++++++++++++++

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> UHF(h3, 2, 1)


    With extra electrons added to calculate coefficients
    ++++++++++++++++++++++++++++++++++++++++++++++++++++

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> UHF(h3, 2, 1, extra_e_coeff=True)

    With internal stability analysis
    ++++++++++++++++++++++++++++++++

    >>> h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')
    >>> UHF(h3, 2, 1, internal_stability_analysis=True)

     prints and returns UHF energy of h3



    """

    # first case: no extra electrons and no stability analysis
    if extra_e_coeff==False and internal_stability_analysis==False:

        # get_integrals (from GHF.SCF_functions) calculates the overlap matrix, one_electron integrals,
        # two_electron_integrals
        # and the nuclear repulsion value.
        overlap = get_integrals(molecule)[0]
        one_electron = get_integrals(molecule)[1]
        two_electron = get_integrals(molecule)[2]
        nuclear_repulsion = get_integrals(molecule)[3]

        # The trans_matrix function calculates the orthogonalisation matrix from a given overlap matrix.
        # core_guess = the guess in the case where the electrons don't interact with each other
        X = trans_matrix(overlap)
        core_guess = X.T.dot(one_electron).dot(X)

        # create guess density matrix from core guess, separate for alpha and beta and put it into an array
        guess_density_a = density_matrix(core_guess, occ_a, X)
        guess_density_b = density_matrix(core_guess, occ_b, X)

        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        electronic_e = uhf_scf_energy(guess_density_a, guess_density_b, one_electron, one_electron,one_electron)
        energies = [electronic_e]

        # create an array to check the energy differences between iterations
        delta_e = []

        # create an iteration procedure
        def iteration():
            fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron, two_electron) # create a fock matrix for alpha from last alpha density
            fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron, two_electron) # create a fock matrix for beta from last beta density

            fock_orth_a = X.T.dot(fock_a).dot(X) # orthogonalize both fock matrices
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density_matrix(fock_orth_a, occ_a, X) # create a new alpha density matrix
            new_density_b = density_matrix(fock_orth_b, occ_b, X) # create a new beta density matrix

            densities_a.append(new_density_a) # put the density matrices in their respective arrays
            densities_b.append(new_density_b)

            energies.append(uhf_scf_energy(new_density_a, new_density_b, fock_a, fock_b, one_electron)) # calculate the improved scf energy and add it to the array

            delta_e.append(energies[-1] - energies[-2]) # calculate the energy difference and add it to the delta_E array

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        iteration()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration()
            i += 1

        # calculate the total energy by adding the nuclear repulsion value and print the results
        total_e = energies[-1] + nuclear_repulsion
        print("Number of iterations: " + str(i))
        print("Converged SCF energy in Hartree: " + str(total_e) + " (UHF)")

        # calculate and return the spin (multiplicity)
        # first calculate and orthogonalise the fock matrix
        # calculate the coefficients from the orthogonal fock matrix
        fock_a = uhf_fock_matrix(densities_a[-2], densities_b[-2], one_electron,
                                 two_electron)
        fock_b = uhf_fock_matrix(densities_b[-2], densities_a[-2], one_electron, two_electron)

        fock_a_ = X.T.dot(fock_a).dot(X)  # orthogonalize both fock matrices
        fock_b_ = X.T.dot(fock_b).dot(X)

        val_a, vec_a = la.eigh(fock_a_)
        val_b, vec_b = la.eigh(fock_b_)
        coeff_a = X @ vec_a
        coeff_b = X @ vec_b

        spin(occ_a, occ_b, coeff_a, coeff_b, overlap)


        return total_e

    # Second case: push the iteration out of a local minimum by adding two electrons to the system and using those coefficients
    # for the energy calculation.
    if extra_e_coeff:

        # increase the number of electrons by two and rebuild the molecule
        molecule.nelectron = occ_a + occ_b + 2
        molecule.build()

        # calculate the integrals for the new test system (_t)
        overlap_t = get_integrals(molecule)[0]
        one_electron_t = get_integrals(molecule)[1]
        two_electron_t = get_integrals(molecule)[2]

        # Calculate the orthogonalisation matrix and the core guess for the new test system (_t)
        X_t = trans_matrix(overlap_t)
        core_guess_t = X_t.T.dot(one_electron_t).dot(X_t)


        # Calculate a guess density for the test system, both for alpha and beta.
        # add the two extra electrons to alpha
        # add the density matrices to respective arrays
        guess_density_a_t = density_matrix(core_guess_t, occ_a + 2, X_t)
        guess_density_b_t = density_matrix(core_guess_t, occ_b, X_t)

        densities_a = [guess_density_a_t]
        densities_b = [guess_density_b_t]

        # create an array to check the differences between density matrices in between iterations
        delta_dens = []

        # create an iteration procedure, based on the densities, for the test system
        def iteration_t():
            # create a fock matrix for alpha from last alpha density, and the integrals from the test system
            fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron_t, two_electron_t)
            # create a fock matrix for beta from last beta density, and the integrals from the test system
            fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron_t, two_electron_t)

            fock_orth_a = X_t.T.dot(fock_a).dot(X_t) # orthogonalize both fock matrices
            fock_orth_b = X_t.T.dot(fock_b).dot(X_t)

            new_density_a = density_matrix(fock_orth_a, occ_a + 2, X_t) # create a new alpha density matrix, with the test system's orthogonalisation matrix
            new_density_b = density_matrix(fock_orth_b, occ_b, X_t) # create a new beta density matrix, with the test system's orthogonalisation matrix

            densities_a.append(new_density_a) # add the densities to an array
            densities_b.append(new_density_b)

            # calculate the difference between the last two density matrices
            delta_dens.append(np.sum(densities_a[-1] + densities_b[-1]) - np.sum(densities_a[-2] + densities_b[-2]))

        # start and continue the iteration process as long as the difference between densities is larger than 1e-12
        iteration_t()
        while abs(delta_dens[-1]) >= 1e-12:
            iteration_t()

        # Now that the test system has converged, we use the last calculated density matrices to calculate new orbital coefficients
        # for both alpha and beta
        fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron_t, two_electron_t)
        fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron_t, two_electron_t)

        fock_orth_a = X_t.T.dot(fock_a).dot(X_t)  # orthogonalize both fock matrices
        fock_orth_b = X_t.T.dot(fock_b).dot(X_t)

        val_a, vec_a = la.eigh(fock_orth_a)
        val_b, vec_b = la.eigh(fock_orth_b)

        coeff_a = X_t.dot(vec_a)
        coeff_b = X_t.dot(vec_b)

        # reset the number of electrons in the system and rebuild the molecule
        molecule.nelectron = None
        molecule.build()

        # calculate the integrals of the original system
        overlap = get_integrals(molecule)[0]
        one_electron = get_integrals(molecule)[1]
        two_electron = get_integrals(molecule)[2]
        nuclear_repulsion = get_integrals(molecule)[3]

        # Calculate the new orthogonalisation matrix, for the original system
        X = trans_matrix(overlap)

        # the improved guess density is set equal to the coefficients calculated from the test system
        # both are added to an array
        imp_guess_density_a = coeff_a
        imp_guess_density_b = coeff_b

        imp_dens_a = [imp_guess_density_a]
        imp_dens_b = [imp_guess_density_b]

        energies = [uhf_scf_energy(imp_dens_a[-1], imp_dens_b[-1], one_electron, one_electron, one_electron)]
        delta_e = []

        # define a new iteration procedure based on the energy difference
        def iteration():
            # create a fock matrix for alpha from last alpha density, and the integrals from the original system
            fock_a = uhf_fock_matrix(imp_dens_a[-1], imp_dens_b[-1], one_electron, two_electron)
            # create a fock matrix for beta from last beta density, and the integrals from the original system
            fock_b = uhf_fock_matrix(imp_dens_b[-1], imp_dens_a[-1], one_electron, two_electron)

            fock_orth_a = X.T.dot(fock_a).dot(X) # orthogonalize both fock matrices
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density_matrix(fock_orth_a, occ_a, X) # create a new alpha density matrix, with the original system's orthogonalisation matrix
            new_density_b = density_matrix(fock_orth_b, occ_b, X) # create a new beta density matrix, with the original system's orthogonalisation matrix

            imp_dens_a.append(new_density_a) # add the densities to an array
            imp_dens_b.append(new_density_b)

            # calculate the improved scf energy and add it to the array
            energies.append(uhf_scf_energy(new_density_a, new_density_b, fock_a, fock_b, one_electron) + nuclear_repulsion)
            delta_e.append(energies[-1] - energies[-2]) # calculate the energy difference and add it to the delta_E array

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        iteration()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration()
            i += 1

        # calculate the total energy by adding the nuclear repulsion value and print the results
        total_e = energies[-1]
        print("Number of iterations: " + str(i))
        print("Converged SCF energy in Hartree: " + str(total_e) + " (UHF)")

        # calculate and return the spin (multiplicity)
        # first calculate and orthogonalise the fock matrix
        # calculate the coefficients from the orthogonal fock matrix
        fock_a = uhf_fock_matrix(imp_dens_a[-2], imp_dens_b[-2], one_electron,
                                 two_electron)
        fock_b = uhf_fock_matrix(imp_dens_b[-2], imp_dens_a[-2], one_electron, two_electron)

        fock_a_ = X.T.dot(fock_a).dot(X)  # orthogonalize both fock matrices
        fock_b_ = X.T.dot(fock_b).dot(X)

        val_a, vec_a = la.eigh(fock_a_)
        val_b, vec_b = la.eigh(fock_b_)
        coeff_a = X @ vec_a
        coeff_b = X @ vec_b

        spin(occ_a, occ_b, coeff_a, coeff_b, overlap)

        return total_e

    # Third case: Use an internal stability analysis to find coefficients closer to the stable condition and use these for
    # the subsequent energy calculation
    if internal_stability_analysis:

        # get_integrals (from GHF.SCF_functions) calculates the overlap matrix, one_electron integrals, two_electron_integrals
        # and the nuclear repulsion value.
        overlap = get_integrals(molecule)[0]
        one_electron = get_integrals(molecule)[1]
        two_electron = get_integrals(molecule)[2]
        nuclear_repulsion = get_integrals(molecule)[3]

        # The trans_matrix function calculates the orthogonalisation matrix from a given overlap matrix.
        # core_guess = the guess in the case where the electrons don't interact with each other
        X = trans_matrix(overlap)
        core_guess = X.T.dot(one_electron).dot(X)

        # create guess density matrix from core guess, separate for alpha and beta and put it into an array
        # Switch the spin state by adding one alpha and removing one beta electron
        guess_density_a = density_matrix(core_guess, occ_a+1, X)
        guess_density_b = density_matrix(core_guess, occ_b-1, X)

        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        # create an array to check the differences between density matrices in between iterations
        delta_dens = []

        # create an iteration procedure, based on the densities
        def iteration():
            fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron, two_electron) # create a fock matrix for alpha from last alpha density
            fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron, two_electron) # create a fock matrix for beta from last alpha density

            fock_orth_a = X.T.dot(fock_a).dot(X) # orthogonalize the fock matrices
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density_matrix(fock_orth_a, occ_a+1, X) # create a new alpha density matrix
            new_density_b = density_matrix(fock_orth_b, occ_b-1, X) # create a new beta density matrix

            densities_a.append(new_density_a) # add density matrices to an array
            densities_b.append(new_density_b)

            # calculate the difference between the last two density matrices
            delta_dens.append(np.sum(densities_a[-1] + densities_b[-1]) - np.sum(densities_a[-2] + densities_b[-2]))

        # start and continue the iteration process as long as the difference between densities is larger than 1e-12
        iteration()
        while abs(delta_dens[-1]) >= 1e-12:
            iteration()

        # now that the system has converged, calculate the system's orbital coefficients from the last calculated density matrix

        fock_a = uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron, two_electron)
        fock_b = uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron, two_electron)

        fock_orth_a = X.T.dot(fock_a).dot(X)  # orthogonalize the fock matrices
        fock_orth_b = X.T.dot(fock_b).dot(X)

        val_a, vec_a = la.eigh(fock_orth_a)
        val_b, vec_b = la.eigh(fock_orth_b)

        coeff_a = X.dot(vec_a)
        coeff_b = X.dot(vec_b)

        # the generate_g() function returns 3 values
        # - the gradient, g
        # - the result of h_op, the trial vector
        # - the diagonal of the Hessian matrix, h_diag
        def generate_g():
            total_orbitals = overlap.shape[0] # total number of orbitals = number of basis functions
            n_vir_a = int(total_orbitals - occ_a) # number of virtual (unoccupied) alpha orbitals
            n_vir_b = int(total_orbitals - occ_b) # number of virtual (unoccupied) beta orbitals
            occ_indx_a = np.arange(occ_a) # indices of the occupied alpha orbitals
            occ_indx_b = np.arange(occ_b) # indices of the occupied beta orbitals
            vir_indx_a = np.arange(total_orbitals)[occ_a:] # indices of the virtual (unoccupied) alpha orbitals
            vir_indx_b = np.arange(total_orbitals)[occ_b:] # indices of the virtual (unoccupied) beta orbitals

            occ_a_orb = coeff_a[:, occ_indx_a] # orbital coefficients associated with occupied alpha orbitals
            occ_b_orb = coeff_b[:, occ_indx_b] # orbital coefficients associated with occupied beta orbitals
            vir_a_orb = coeff_a[:, vir_indx_a] # orbital coefficients associated with virtual (unoccupied) alpha orbitals
            vir_b_orb = coeff_b[:, vir_indx_b] # orbital coefficients associated with virtual (unoccupied) beta orbitals

            # initial fock matrix for stability analysis is the last fock matrix from the first iteration process, for both alpha and beta
            fock_a_init = uhf_fock_matrix(densities_a[-1], densities_b[-1], one_electron, two_electron)
            fock_b_init = uhf_fock_matrix(densities_b[-1], densities_a[-1], one_electron, two_electron)

            # orthogonolize the initial fock matrix with the coefficients, calculated from the first iteration process
            # reduce() is a short way to write calculations, the first argument is an operation, the second one are the values
            # on which to apply the operation
            fock_a = reduce(np.dot, (coeff_a.T, fock_a_init, coeff_a))
            fock_b = reduce(np.dot, (coeff_b.T, fock_b_init, coeff_b))

            fock_occ_a = fock_a[occ_indx_a[:, None], occ_indx_a] # specify the fock matrix for only occupied alpha orbitals
            fock_vir_a = fock_a[vir_indx_a[:, None], vir_indx_a] # specify the fock matrix for only virtual (unoccupied) alpha orbitals
            fock_occ_b = fock_b[occ_indx_b[:, None], occ_indx_b] # specify the fock matrix for only occupied beta orbitals
            fock_vir_b = fock_b[vir_indx_b[:, None], vir_indx_b] # specify the fock matrix for only virtual (unoccupied) beta orbitals

            # create the gradient
            # This is done by combining the necessary parts of the alpha and beta fock matrix with np.hstack and then
            # using np.ravel() to create a 1D matrix
            # np.hstack() adds together arrays horizontally e.g.:
            #                               a = np.array([[1], [2], [3]]))
            #                               b = np.array([[2], [3], [4]]))
            #                               np.hstack((a,b)) = array([1, 2],
            #                                                        [2, 3],
            #                                                        [3, 4])
            # np.ravel() pulls all array values into 1 long 1D array
            g = np.hstack((fock_a[vir_indx_a[:, None], occ_indx_a].ravel(), fock_b[vir_indx_b[:, None], occ_indx_b].ravel()))


            # Creat the diagonal alpha and beta hessian respectively from the virtual and the occupied fock matrices
            # Use np.hstack() to combine the diagonal alpha and beta hessians to create the total diagonal hessian metrix
            h_diag_a = fock_vir_a.diagonal().real[:, None] - fock_occ_a.diagonal().real
            h_diag_b = fock_vir_b.diagonal().real[:, None] - fock_occ_b.diagonal().real
            h_diag = np.hstack((h_diag_a.reshape(-1), h_diag_b.reshape(-1)))

            # The result of h_op is the displacement vector.
            def h_op(x):
                x1a = x[:n_vir_a * occ_a].reshape(n_vir_a, occ_a) # create a trial vector for alpha orbitals
                x1b = x[n_vir_a * occ_a:].reshape(n_vir_b, occ_b) # create a trial vector for beta orbitals
                x2a = np.einsum('pr,rq->pq', fock_vir_a, x1a) # summation from fock_vir_a * x1a
                x2a -= np.einsum('sq,ps->pq', fock_occ_a, x1a) # subtract summation from fock_occ_a * x1a
                x2b = np.einsum('pr,rq->pq', fock_vir_b, x1b) # summation from fock_vir_b * x1b
                x2b -= np.einsum('sq,ps->pq', fock_occ_b, x1b) # subtract summation from fock_occ_b * x1b

                d1a = reduce(np.dot, (vir_a_orb, x1a, occ_a_orb.conj().T)) # diagonolize x1a
                d1b = reduce(np.dot, (vir_b_orb, x1b, occ_b_orb.conj().T)) # diagonolize x1b
                dm1 = np.array((d1a + d1a.conj().T, d1b + d1b.conj().T)) # create a density matrix from d1a and d1b
                v1 = -scf.hf.get_jk(molecule, dm1, hermi = 1)[1] # calculate the exchange integrals in the case where dm1 is used as a density matrix
                # add the matrix product from the virtual alpha orbitals (conjugate transpose), the exchange integrals, and the occupied alpha orbitals to the final trial vector
                x2a += reduce(np.dot, (vir_a_orb.conj().T, v1[0], occ_a_orb))
                # add the matrix product from the virtual beta orbitals (conjugate transpose), the exchange integrals, and the occupied beta orbitals to the final trial vector
                x2b += reduce(np.dot, (vir_b_orb.conj().T, v1[1], occ_b_orb))

                x2 = np.hstack((x2a.ravel(), x2b.ravel())) # merge x2a and x2b together to create the trial vector
                return x2

            return g, h_op, h_diag

        # This function will check whether or not there is an internal instability, and if there is one, it will calculate new and improved coefficients.
        def internal_stability():
            g, hop, hdiag = generate_g()
            hdiag *= 2

            # this function prepares for the conditions needed to use a davidson solver later on
            def precond(dx, e, x0):
                hdiagd = hdiag - e
                hdiagd[abs(hdiagd) < 1e-8] = 1e-8
                return dx / hdiagd

            # The overall Hessian for internal rotation is x2 + x2.T.conj().
            # This is the reason we apply (.real * 2)
            def hessian_x(x):
                return hop(x).real * 2

            # Find the unique indices of the variables
            # in this function, bitwise operators are used. They treat each operand as a sequence of binary digits and operate on them bit by bit
            def uniq_variable_indices(mo_occ):
                occ_indx_a = mo_occ > 0 # indices of occupied alpha orbitals
                occ_indx_b = mo_occ == 2 # indices of occupied beta orbitals
                vir_indx_a = ~occ_indx_a # indices of virtual (unoccupied) alpha orbitals, done with bitwise operator: ~ (negation)
                vir_indx_b = ~occ_indx_b # indices of virtual (unoccupied) beta orbitals, done with bitwise operator: ~ (negation)
                # & and | are bitwise operators for 'and' and 'or'
                # each bit position is the result of the logical 'and' or 'or' of the bits in the corresponding position of the operands
                unique = (vir_indx_a[:, None] & occ_indx_a) | (vir_indx_b[:, None] & occ_indx_b) # determine the unique variable indices, by use of bitwise operators
                return unique

            # put the unique variables in a new matrix used later to create a rotation matrix.
            def unpack_uniq_variables(dx, mo_occ):
                nmo = len(mo_occ)
                idx = uniq_variable_indices(mo_occ)
                x1 = np.zeros((nmo, nmo), dtype=dx.dtype)
                x1[idx] = dx
                return x1 - x1.conj().T

            # A function to apply a rotation on the given coefficients
            def rotate_mo(mo_coeff, mo_occ, dx):
                dr = unpack_uniq_variables(dx, mo_occ)
                u = scipy.linalg.expm(dr) # computes the matrix exponential
                return np.dot(mo_coeff, u)

            x0 = np.zeros_like(g) # like returns a matrix of the same shape as the argument given
            x0[g != 0] = 1. / hdiag[g != 0] # create initial guess for davidson solver
            # use the davidson solver to find the eigenvalues and eigenvectors needed to determine an internal instability
            e, v = lib.davidson(hessian_x, x0, precond, tol=1e-4)
            if e < -1e-5: # this points towards an internal instability
                total_orbitals = overlap.shape[0] # total number of basis functions
                n_vir_a = int(total_orbitals - occ_a) # number of virtual (unoccupied) alpha orbitals
                mo_a = np.zeros(total_orbitals)
                mo_b = np.zeros(total_orbitals)
                for j in range(occ_a):
                    mo_a[j] = 1 # create representation of alpha orbitals by adding an electron (a 1) to each occupied alpha orbital
                for k in range(occ_b):
                    mo_b[k] = 1 # create representation of beta orbitals by adding an electron (a 1) to each occupied beta orbital
                new_orbitals = (rotate_mo(coeff_a, mo_a, v[:occ_a * n_vir_a]),
                      rotate_mo(coeff_b, mo_b, v[occ_a * n_vir_a:])) # create new orbitals by rotating the old ones
                print("There is an internal instability in the UHF wave function.")
            else: # in the case where no instability is present
                new_orbitals = (coeff_a, coeff_b)
                print('There is no internal instability in the UHF wave function.')
            return new_orbitals

        # The improved guess densities for alpha and beta are taken as the new coefficients, calculated from the stability analysis
        # the improved guess densities are put into respective arrays for alpha and beta (imp_ = improved)
        imp_guess_density_a = internal_stability()[0]
        imp_guess_density_b = internal_stability()[1]

        imp_dens_a = [imp_guess_density_a]
        imp_dens_b = [imp_guess_density_b]

        # Calculate the initial electronic energy and put it into an array
        energies = [uhf_scf_energy(imp_dens_a[-1], imp_dens_b[-1], one_electron, one_electron, one_electron)]
        delta_e = []

        # define an iteration process
        def iteration_2():
            fock_a = uhf_fock_matrix(imp_dens_a[-1], imp_dens_b[-1], one_electron, two_electron)  # create a fock matrix for alpha from last improved alpha density
            fock_b = uhf_fock_matrix(imp_dens_b[-1], imp_dens_a[-1], one_electron, two_electron) # create a fock matrix for beta from last improved alpha density

            fock_orth_a = X.T.dot(fock_a).dot(X) # orthogonalize the fock matrices
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density_matrix(fock_orth_a, occ_a, X) # create a new alpha density matrix
            new_density_b = density_matrix(fock_orth_b, occ_b, X) # create a new beta density matrix

            imp_dens_a.append(new_density_a) # put the density matrices into their respective arrays
            imp_dens_b.append(new_density_b)

            # calculate the improved energy and add it to the array
            # calculate the energy difference and add it to the delta_e array
            energies.append(uhf_scf_energy(new_density_a, new_density_b, fock_a, fock_b, one_electron) + nuclear_repulsion)
            delta_e.append(energies[-1] - energies[-2])

        # start and continue the iteration process as long as the energy difference is larger than 1e-12
        iteration_2()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration_2()
            i += 1

        # calculate the total energy by adding the nuclear repulsion value and print the results
        total_e = energies[-1]
        print("Number of iterations: " + str(i))
        print("Converged SCF energy in Hartree: " + str(total_e) + " (UHF)")

        # calculate and return the spin (multiplicity)
        # first calculate and orthogonalise the fock matrix
        # calculate the coefficients from the orthogonal fock matrix
        fock_a = uhf_fock_matrix(imp_dens_a[-2], imp_dens_b[-2], one_electron,
                                 two_electron)
        fock_b = uhf_fock_matrix(imp_dens_b[-2], imp_dens_a[-2], one_electron, two_electron)

        fock_a_ = X.T.dot(fock_a).dot(X)  # orthogonalize both fock matrices
        fock_b_ = X.T.dot(fock_b).dot(X)

        val_a, vec_a = la.eigh(fock_a_)
        val_b, vec_b = la.eigh(fock_b_)
        coeff_a = X @ vec_a
        coeff_b = X @ vec_b

        spin(occ_a, occ_b, coeff_a, coeff_b, overlap)

        return total_e