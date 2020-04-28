"""
Functions to deal with matrix and tensor transformations.
=========================================================

This file contains functions that calculate the expectation values of the different spin operators.
"""

import numpy as np
from scipy import linalg as la
import math


def expand_matrix(matrix):
    """
    Expand a matrix to spinor basis.

    :param matrix: a matrix.
    :return: a matrix double the size, where blocks of zero's are added top right and bottom left.
    """
    # get the shape of the matrix you want to expand
    # create a zero-matrix with the same shape
    shape = np.shape(matrix)
    if isinstance(matrix.any(), complex):
        zero = np.zeros(shape).astype(complex)
    else:
        zero = np.zeros(shape)
    # create the top part of the expanded matrix by putting the matrix and zero-matrix together
    # create the bottom part of the expanded matrix by putting the matrix and zero-matrix together
    top = np.hstack((matrix, zero))
    bottom = np.hstack((zero, matrix))
    # Add top and bottom part together.
    return np.vstack((top, bottom))


def expand_tensor(tensor, complexity=False):
    """
    Expand a given tensor to spinor basis representation.

    :param tensor: The tensor, usually eri, that you wish to expand.
    :param complexity: Is your tensor complex or not? Default is false.
    :return: a tensor where each dimension is doubled.
    """
    dim = np.shape(tensor)[0]
    if complexity:
        tens_block = np.zeros((dim, dim, 2*dim, 2*dim)).astype(complex)
        zero = np.zeros((dim, dim, 2*dim, 2*dim)).astype(complex)
    else:
        tens_block = np.zeros((dim, dim, 2 * dim, 2 * dim))
        zero = np.zeros((dim, dim, 2 * dim, 2 * dim))
    for l in range(dim):
        for k in range(dim):
            tens_block[l][k] = expand_matrix(tensor[l][k])
    top = np.hstack((tens_block, zero))
    bottom = np.hstack((zero, tens_block))
    return np.vstack((top, bottom))


def spin_blocked(block_1, block_2, block_3, block_4):
    """
    When creating the blocks of the density or fock matrix separately, this function is used to add them together,
    and create the total density or Fock matrix in spin Blocked notation. Transforms four separate matrices to 1 big
    matrix in spin-blocked notation.

    :return: a density matrix in the spin-blocked notation
    """
    top = np.hstack((block_1, block_2))
    bottom = np.hstack((block_3, block_4))
    return np.vstack((top, bottom))


def tensor_basis_transform(tensor, matrix):
    """
    Transform the two electron tensor to MO basis. Scales as N^5.

    :param tensor: A tensor in it's initial basis.
    :param matrix: the transformation matrix.
    :return: The tensor in the basis of the transformation matrix.
    """
    # Old method: Slow. Left in as comments to provide clarity on how the transformation works.
    # -----------------------------------------------------------------------------------------
    #     eri_mo = np.zeros((dim, dim, dim, dim))
    #     mo_1 = np.zeros((dim, dim, dim, dim))
    #     mo_2 = np.zeros((dim, dim, dim, dim))
    #     mo_3 = np.zeros((dim, dim, dim, dim))

    # for s in range(0, len(coeff)):
    #     for sig in range(0, len(coeff)):
    #         mo_1[:, :, :, s] += coeff[sig, s] * eri[:, :, :, sig]
    #
    #     for r in range(0, len(coeff)):
    #         for lam in range(0, len(coeff)):
    #             mo_2[:, :, r, s] += coeff[lam, r] * mo_1[:, :, lam, s]
    #
    #         for q in range(0, len(coeff)):
    #             for nu in range(0, len(coeff)):
    #                 mo_3[:, q, r, s] += coeff[nu, q] * mo_2[:, nu, r, s]
    #
    #             for p in range(0, len(coeff)):
    #                 for mu in range(0, len(coeff)):
    #                     eri_mo[p, q, r, s] += coeff[mu, p] * mo_3[mu, q, r, s]

    # New method: fast, but formula's might be less clear.
    # ----------------------------------------------------
    temp_1 = np.einsum('as,pqra->pqrs', matrix, tensor)
    temp_2 = np.einsum('lr,pqls->pqrs', matrix, temp_1)
    temp_3 = np.einsum('nq,pnrs->pqrs', matrix, temp_2)
    tensor_nb = np.einsum('mp,mqrs->pqrs', matrix, temp_3)

    return tensor_nb


def mix_tensor_to_basis_transform(tensor, matrix1, matrix2, matrix3, matrix4):
    """
    Transform a tensor to a mixed basis. Each index can have a separate transformation matrix.
    Mostly useful in UHF calculations where tensors must be transformed to an alpha-beta basis.
    :param tensor: The tensor you wish to transform
    :param matrix1: matrix to transform index 1
    :param matrix2: matrix to transform index 2
    :param matrix3: matrix to transform index 3
    :param matrix4: matrix to transform index 4
    :return: The transformed tensor in the mixed basis.
    """
    temp_1 = np.einsum('as,pqra->pqrs', matrix4, tensor)
    temp_2 = np.einsum('lr,pqls->pqrs', matrix3, temp_1)
    temp_3 = np.einsum('nq,pnrs->pqrs', matrix2, temp_2)
    tensor_nb = np.einsum('mp,mqrs->pqrs', matrix1, temp_3)

    return tensor_nb


def rotate_to_eigenvec(eigenvec, mo_coeff, occ, number_of_orbitals):
    """
    A function used to rotate a given set of coefficients to a given eigenvector.
    This is done by making an irreducible representation of the eigenvector and creating the exponential matrix
    of the result.
    :param eigenvec: The eigenvector to which you wish to rotate
    :param mo_coeff: The MO coefficients you wish to rotate
    :param occ: the number of occupied orbitals
    :param number_of_orbitals: the total number of orbitals
    :return: A rotated set of coefficients.
    """
    occ_mos = np.zeros(number_of_orbitals)
    for j in range(occ):
        occ_mos[j] = 1

    def unique_variable_indices(mo_occ):
        occ_indx_a = mo_occ > 0  # indices of occupied alpha orbitals
        occ_indx_b = mo_occ == 2  # indices of occupied beta orbitals
        # indices of virtual (unoccupied) alpha orbitals, done with bitwise operator: ~ (negation)
        # indices of virtual (unoccupied) beta orbitals, done with bitwise operator: ~ (negation)
        vir_indx_a = ~occ_indx_a
        vir_indx_b = ~occ_indx_b
        # & and | are bitwise operators for 'and' and 'or'
        # each bit position is the result of the logical 'and' or 'or' of the bits
        # in the corresponding position of the operands
        # determine the unique variable indices, by use of bitwise operators
        unique = (vir_indx_a[:, None] & occ_indx_a) | (vir_indx_b[:, None] & occ_indx_b)
        return unique

    # put the unique variables in a new matrix used to create a irreducible presentation.
    def unpack_unique_variables(vec, mo_occ):
        nmo = len(mo_occ)
        idx = unique_variable_indices(mo_occ)
        x1 = np.zeros((nmo, nmo), dtype=eigenvec.dtype)
        x1[idx] = vec
        return x1 - x1.conj().T

    # A function to apply a rotation on the given coefficients
    def rotate_mo(mo_occ, dx):
        dr = unpack_unique_variables(dx, mo_occ)
        u = la.expm(dr)  # computes the matrix exponential
        return np.dot(mo_coeff, u)

    return rotate_mo(occ_mos, eigenvec)


def mix_mo_coeff(coeff, n_a, n_b, angle=(math.pi/4)):
    """
    A function that mixes the HOMO and LUMO of a given coefficient matrix

    :param coeff: The mo coefficients with the alpha coefficents in the first matrix and beta coefficients in the second
    :param n_a: Number of alpha electrons
    :param n_b: Number of beta electrons
    :param angle: This is rotation angle in radians and is it's default value is pi/4
    """
    # Determine HOMO and LUMO
    if n_a == n_b:
        homo = coeff[1][:, n_b - 1]
        lumo = coeff[0][:, n_a]

    else:
        homo = coeff[0][:, n_a - 1]
        lumo = coeff[1][:, n_b]

    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])

    # Creating a matrix with the HOMO in the first column and the LUMO in the second column
    matrix = np.array([homo, lumo]).T
    # Mixing HOMO and LUMO
    matrix = matrix @ rotation_matrix
    # Putting new MO's back in the coefficient matrix
    if n_a == n_b:
        coeff[1][:, n_b - 1] = matrix[:, 0]
        coeff[0][:, n_a] = matrix[:, 1]

    else:
        coeff[0][:, n_a - 1] = matrix[:, 0]
        coeff[1][:, n_b] = matrix[:, 1]

    return coeff
