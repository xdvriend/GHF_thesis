from pyscf import *
import numpy as np
from numpy import linalg as la
from scipy import diag

"""A function to calculate your integrals & nuclear repulsion with pyscf"""

def get_integrals(molecule):
    overlap = molecule.intor('int1e_ovlp')
    one_electron = molecule.intor('int1e_nuc') + molecule.intor('int1e_kin')
    two_electron = molecule.intor('int2e')
    nuclear_repulsion = gto.mole.energy_nuc(molecule)
    return overlap, one_electron, two_electron, nuclear_repulsion

"""Define a transformation matrix X"""

def trans_matrix(overlap):
    eigenvalues, eigenvectors = la.eigh(overlap)  # calculate eigenvalues & eigenvectors of the overlap matrix
    diag_matrix = diag(eigenvalues)  # create a diagonal matrix from the eigenvalues of the overlap
    X = eigenvectors.dot(np.sqrt(la.inv(diag_matrix))).dot(eigenvectors.T)
    return X


