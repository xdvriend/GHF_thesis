from hf.HartreeFock import lowest_HF as Lowest
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1',
           spin=1,
           unit='angstrom',
           basis='cc-pvdz')

# The "lowest_HF class" Will cascade through the HF ranks and determine which is the lowest HF energy state.
# first you will have to make an object, using "Find"
# The arguments are a molecule and the number of electrons.
# In order to search the lowest solution, use the command 'run_algorithm()'
Lowest.Find(h3, 3).run_algorithm()
