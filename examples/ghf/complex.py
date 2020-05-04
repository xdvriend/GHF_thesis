from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
Be = gto.M(atom='Be 1 0 0', basis='4-31g')


# Now you have to make your GHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
be = GHF.MF(Be, 4)

# To perform a complex calculation and print out the result, simply run:
be.get_scf_solution(complex_method=True)

# The complex method has the same options as the real method
# Guesses, stability analysis,... are all possible without change.
# Let's look at stability analysis for example.
# Complex GHF has only internal stability analysis, since there's no symmetry left to be broken.
be.stability_analysis('internal')
