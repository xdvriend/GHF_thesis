from hf.HartreeFock import *
from pyscf import *
import psi4


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
h3 = gto.M(atom='h 0 0 0; '
                'h 0 0.86602540378 0.5; '
                'h 0 0 1',
           spin=1,
           unit='angstrom',
           basis='cc-pvdz')

# Now you have to make your UHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h3 = UHF.MF(h3, 3)

# To perform a calculation and print out the result, simply run:
h3.get_scf_solution(complex_method=True)

# You can also edit the convergence criterion, or use other things like stability analysis, when using a complex method
h3.get_scf_solution(convergence=1e-6, complex_method=True)
h3.stability_analysis('internal')
