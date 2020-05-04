from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
Be = gto.M(atom='Be 1 0 0', basis='4-31g')


# Now you have to make your GHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
be = GHF.MF(Be, 4)

# To perform a calculation and print out the result, simply run:
be.get_scf_solution()

# Now to check the stability of the solution
# We can check internal and external (real/complex) stability
# The internal stability also returns a new set of coefficients that can be used as a guess in order to follow the
# negative eigenvalue of the Hessian.
be.stability_analysis('internal')
be.stability_analysis('external')
