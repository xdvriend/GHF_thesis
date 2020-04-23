"""
Functions to deal with matrix and tensor transformations.
=========================================================

This file contains functions that calculate the expectation values of the different spin operators.
"""

import numpy as np


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
