from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1',
           spin=1,
           unit='angstrom',
           basis='cc-pvdz')

# Now you have to make your GHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h3 = GHF.MF(h3, 3)

# To find a lower lying GHF solution, you should always use a random guess, in order to activate the off-diagonal
# blocks of your matrices.
guess = h3.random_guess()

# To perform a calculation and print out the result, simply run:
h3.get_scf_solution_diis(guess)
