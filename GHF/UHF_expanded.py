from GHF.SCF_functions import *
from functools import reduce
from pyscf import *
import scipy.linalg



"""Unrestricted Hartree Fock by means of SCF procedure"""

def UHF(molecule, occ_a, occ_b, extra_e_coeff=False, internal_stability_analysis=False):
    if extra_e_coeff==False and internal_stability_analysis==False:
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
            return np.einsum('ij,kj->ik', coefficients_r, coefficients_r)

        guess_density_a_t = density(core_guess_t, occ_a + 2, X_t)
        guess_density_b_t = density(core_guess_t, occ_b, X_t)

        densities_a = [guess_density_a_t]
        densities_b = [guess_density_b_t]

        def fock_matrix(density_matrix_a, density_matrix_b, one_e, two_e):
            jk_integrals = two_e - two_e.transpose(0, 2, 1, 3)
            jk_a = np.einsum('kl, ijkl->ij', density_matrix_a, jk_integrals)
            j_b = np.einsum('kl, ijkl->ij', density_matrix_b, two_e)
            return one_e + jk_a + j_b

        delta_dens = []

        def iteration_t():
            fock_a = fock_matrix(densities_a[-1], densities_b[-1], one_electron_t, two_electron_t)
            fock_b = fock_matrix(densities_b[-1], densities_a[-1], one_electron_t, two_electron_t)

            fock_orth_a = X_t.T.dot(fock_a).dot(X_t)
            fock_orth_b = X_t.T.dot(fock_b).dot(X_t)

            new_density_a = density(fock_orth_a, occ_a + 2, X_t)
            new_density_b = density(fock_orth_b, occ_b, X_t)

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

            new_density_a = density(fock_orth_a, occ_a, X)
            new_density_b = density(fock_orth_b, occ_b, X)

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

    if internal_stability_analysis:
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

        guess_density_a = density(core_guess, occ_a+1)
        guess_density_b = density(core_guess, occ_b-1)

        densities_a = [guess_density_a]
        densities_b = [guess_density_b]

        def scf_energy(density_matrix_a, density_matrix_b, fock_a, fock_b):
            scf_e = np.einsum('ij, ij->', density_matrix_a + density_matrix_b, one_electron)
            scf_e += np.einsum('ij, ij->', density_matrix_a, fock_a)
            scf_e += np.einsum('ij, ij->', density_matrix_b, fock_b)
            scf_e *= 0.5
            return scf_e

        def fock_matrix(density_matrix_a, density_matrix_b):
            jk_integrals = two_electron - two_electron.transpose(0, 2, 1, 3)
            jk_a = np.einsum('kl,ijkl->ij', density_matrix_a, jk_integrals)
            j_b = np.einsum('kl, ijkl->ij', density_matrix_b, two_electron)
            return one_electron + jk_a + j_b

        delta_dens = []

        def iteration():
            fock_a = fock_matrix(densities_a[-1], densities_b[-1])
            fock_b = fock_matrix(densities_b[-1], densities_a[-1])

            fock_orth_a = X.T.dot(fock_a).dot(X)
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density(fock_orth_a, occ_a+1)
            new_density_b = density(fock_orth_b, occ_b-1)

            densities_a.append(new_density_a)
            densities_b.append(new_density_b)

            delta_dens.append(np.sum(densities_a[-1] + densities_b[-1]) - np.sum(densities_a[-2] + densities_b[-2]))

        iteration()
        while abs(delta_dens[-1]) >= 1e-12:
            iteration()

        val_a, coeff_a = la.eigh(densities_a[-1])
        val_b, coeff_b = la.eigh(densities_b[-1])

        def generate_g():
            total_orbitals = overlap.shape[0]
            n_vir_a = int(total_orbitals - occ_a)
            n_vir_b = int(total_orbitals - occ_b)
            occ_indx_a = np.arange(occ_a)
            occ_indx_b = np.arange(occ_b)
            vir_indx_a = np.arange(total_orbitals)[occ_a:]
            vir_indx_b = np.arange(total_orbitals)[occ_b:]

            occ_a_orb = coeff_a[:, occ_indx_a]
            occ_b_orb = coeff_b[:, occ_indx_b]
            vir_a_orb = coeff_a[:, vir_indx_a]
            vir_b_orb = coeff_b[:, vir_indx_b]

            fock_a_init = fock_matrix(densities_a[-1], densities_b[-1])
            fock_b_init = fock_matrix(densities_b[-1], densities_a[-1])

            fock_a = reduce(np.dot, (coeff_a.T, fock_a_init, coeff_a))
            fock_b = reduce(np.dot, (coeff_b.T, fock_b_init, coeff_b))

            fock_occ_a = fock_a[occ_indx_a[:, None], occ_indx_a]
            fock_vir_a = fock_a[vir_indx_a[:, None], vir_indx_a]
            fock_occ_b = fock_b[occ_indx_b[:, None], occ_indx_b]
            fock_vir_b = fock_b[vir_indx_b[:, None], vir_indx_b]

            g = np.hstack((fock_a[vir_indx_a[:, None], occ_indx_a].ravel(), fock_b[vir_indx_b[:, None], occ_indx_b].ravel()))

            h_diag_a = fock_vir_a.diagonal().real[:, None] - fock_occ_a.diagonal().real
            h_diag_b = fock_vir_b.diagonal().real[:, None] - fock_occ_b.diagonal().real
            h_diag = np.hstack((h_diag_a.reshape(-1), h_diag_b.reshape(-1)))

            def h_op(x):
                x1a = x[:n_vir_a * occ_a].reshape(n_vir_a, occ_a)
                x1b = x[n_vir_a * occ_a:].reshape(n_vir_b, occ_b)
                x2a = np.einsum('pr,rq->pq', fock_vir_a, x1a)
                x2a -= np.einsum('sq,ps->pq', fock_occ_a, x1a)
                x2b = np.einsum('pr,rq->pq', fock_vir_b, x1b)
                x2b -= np.einsum('sq,ps->pq', fock_occ_b, x1b)

                d1a = reduce(np.dot, (vir_a_orb, x1a, occ_a_orb.conj().T))
                d1b = reduce(np.dot, (vir_b_orb, x1b, occ_b_orb.conj().T))
                dm1 = np.array((d1a + d1a.conj().T, d1b + d1b.conj().T))
                v1 = -scf.hf.get_jk(molecule, dm1, hermi = 1)[1]
                x2a += reduce(np.dot, (vir_a_orb.conj().T, v1[0], occ_a_orb))
                x2b += reduce(np.dot, (vir_b_orb.conj().T, v1[1], occ_b_orb))

                x2 = np.hstack((x2a.ravel(), x2b.ravel()))
                return x2

            return g, h_op, h_diag

        def internal_stability():
            g, hop, hdiag = generate_g()
            hdiag *= 2

            def precond(dx, e, x0):
                hdiagd = hdiag - e
                hdiagd[abs(hdiagd) < 1e-8] = 1e-8
                return dx / hdiagd

            def hessian_x(x):
                return hop(x).real * 2

            def uniq_variable_indices(mo_occ):
                occidxa = mo_occ > 0
                occidxb = mo_occ == 2
                viridxa = ~occidxa
                viridxb = ~occidxb
                mask = (viridxa[:, None] & occidxa) | (viridxb[:, None] & occidxb)
                return mask

            def unpack_uniq_variables(dx, mo_occ):
                nmo = len(mo_occ)
                idx = uniq_variable_indices(mo_occ)
                x1 = np.zeros((nmo, nmo), dtype=dx.dtype)
                x1[idx] = dx
                return x1 - x1.conj().T

            def rotate_mo(mo_coeff, mo_occ, dx):
                dr = unpack_uniq_variables(dx, mo_occ)
                u = scipy.linalg.expm(dr)
                return np.dot(mo_coeff, u)

            x0 = np.zeros_like(g)
            x0[g != 0] = 1. / hdiag[g != 0]
            e, v = lib.davidson(hessian_x, x0, precond, tol=1e-4)
            if e < -1e-5:
                total_orbitals = overlap.shape[0]
                n_vir_a = int(total_orbitals - occ_a)
                mo_a = np.zeros(total_orbitals)
                mo_b = np.zeros(total_orbitals)
                for j in range(occ_a):
                    mo_a[j] = 1
                for k in range(occ_b):
                    mo_b[k] = 1
                new_orbitals = (rotate_mo(coeff_a, mo_a, v[:occ_a * n_vir_a]),
                      rotate_mo(coeff_b, mo_b, v[occ_a * n_vir_a:]))
            else:
                new_orbitals = (coeff_a, coeff_b)
            return new_orbitals

        imp_guess_density_a = internal_stability()[0]
        imp_guess_density_b = internal_stability()[1]

        imp_dens_a = [imp_guess_density_a]
        imp_dens_b = [imp_guess_density_b]

        energies = [scf_energy(imp_dens_a[-1], imp_dens_b[-1], one_electron, one_electron)]
        delta_e = []

        def iteration_2():
            fock_a = fock_matrix(imp_dens_a[-1], imp_dens_b[-1])
            fock_b = fock_matrix(imp_dens_b[-1], imp_dens_a[-1])

            fock_orth_a = X.T.dot(fock_a).dot(X)
            fock_orth_b = X.T.dot(fock_b).dot(X)

            new_density_a = density(fock_orth_a, occ_a)
            new_density_b = density(fock_orth_b, occ_b)

            imp_dens_a.append(new_density_a)
            imp_dens_b.append(new_density_b)

            energies.append(scf_energy(new_density_a, new_density_b, fock_a, fock_b) + nuclear_repulsion)
            delta_e.append(energies[-1] - energies[-2])

        iteration_2()
        i = 1
        while abs(delta_e[-1]) >= 1e-12:
            iteration_2()
            i += 1

        total_e = energies[-1]

        print("Number of iterations: " + str(i))
        print("Converged SCF energy in Hartree: " + str(total_e) + " (UHF)")

        return total_e














