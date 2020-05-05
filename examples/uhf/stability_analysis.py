from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
h4 = gto.M(atom='h 0 0.707107 0;'
                'h 0.707107 0 0;'
                'h 0 -0.707107 0;'
                'h -0.707107 0 0',
           spin=2,
           basis='cc-pvdz')

# Now you have to make your UHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h4 = UHF.MF(h4, 4)

# To perform stability analysis, we first need to do a regular calculation.
h4.get_scf_solution()

# Then, we run the stability analysis
next_guess = h4.stability_analysis('internal')

# Then, you run a new calculation, starting from the guess you got from the stability analysis.
h4.get_scf_solution(next_guess)

# Once the solution is stable, you can also check the external stability analysis.
h4.stability_analysis('external')
