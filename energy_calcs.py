from GHF.RHF import RHF
from GHF.UHF import UHF
from pyscf import *

"""Create a molecule with pyscf"""
# This part needs to be ran in pyscf

h2o = gto.M(atom ='O 0.000000000000 -0.143225816552 0.000000000000; H 1.638036840407 1.136548822547 -0.000000000000; '
                 'H -1.638036840407 1.136548822547 -0.000000000000', unit = 'bohr', basis = 'sto-3g')

h2 = gto.M(atom = 'H 0 0 0; H 0 0 1', unit = 'bohr', basis = 'cc-pvdz')

h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1,  unit = 'bohr', basis = 'cc-pvdz')

h4 = gto.M(atom = 'h 0 0 0; h 0 1 0; h 0 0 1; h 0 1 1', unit = 'bohr', basis = 'cc-pvdz')

"""calculate RHF energies of the 4 molecules"""
# input: molecule, number of occupied orbitals

RHF(h2, 1)
RHF(h3, 1)
RHF(h4, 2)
RHF(h2o, 5)

"""calculate UHF energies of the 4 molecules"""
# input: molecule, number of occupied alpha, number of occupied beta

UHF(h2, 1, 1)
UHF(h3, 2, 1)
UHF(h4, 3, 1)
UHF(h2o, 5, 5)
