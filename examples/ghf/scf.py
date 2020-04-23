from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
Be = gto.M(atom='Be 1 0 0', basis='4-31g')


# Now you have to make your GHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
be = GHF.MF(Be, 4)

# To find a lower lying GHF solution, you should always use a random guess, in order to activate the off-diagonal
# blocks of your matrices.
guess = be.random_guess()

# To perform a calculation and print out the result, simply run:
be.get_scf_solution(guess)
