from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# Here, we use PySCF:
h4 = gto.M(atom='h 0 0.707107 0;'
                'h 0.707107 0 0;'
                'h 0 -0.707107 0;'
                'h -0.707107 0 0',
           spin=2,
           basis='sto-3g')

# Now you have to make your RHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h4 = RHF.MF(h4, 4)

# Perform an SCF/DIIS calculation
h4.get_scf_solution()


# RHF functions can have both internal and external instabilities and both can be checked.
# Here, we'll only take a look at the internal stability.
# Internal stability analysis can be followed, meaning it returns a set of MO coefficients
# which may result in a lower lying energy/more stable solution.
h4.stability_analysis('internal')

# Initially, this yields an unstable solution. This can be solved by running the stability analysis
# and using the newly found coefficients to start a new scf procedure,
# until the instability disappears.
x = h4.stability_analysis('internal')
while h4.int_instability:
    h4.get_scf_solution(x)
    x = h4.stability_analysis('internal')
