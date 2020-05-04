from hf.HartreeFock import *
from pyscf import *
import psi4


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
h4 = gto.M(atom='h 0 0.707107 0;'
                'h 0.707107 0 0;'
                'h 0 -0.707107 0;'
                'h -0.707107 0 0',
           spin=2,
           basis='cc-pvdz')

# Now you have to make your RHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h4 = RHF.MF(h4, 4)

# To perform a complex calculation and print out the result, simply run:
# H_4 is actually a molecule with a lower lying complex RHF solution.
# To verify this, one can use stability analysis. (see example stability)
h4.get_scf_solution(complex_method=True)
