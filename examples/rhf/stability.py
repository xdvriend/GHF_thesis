from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# Here, we use PySCF:
h2o = gto.M(atom='O 0.000000000000 -0.143225816552 0.000000000000; '
                 'H 1.638036840407 1.136548822547 -0.000000000000; '
                 'H -1.638036840407 1.136548822547 -0.000000000000',
            unit='bohr',
            basis='sto-3g')

h4 = gto.M(atom='h 0 0.707107 0;'
                'h 0.707107 0 0;'
                'h 0 -0.707107 0;'
                'h -0.707107 0 0',
           spin=2,
           basis='cc-pvdz')

# Now you have to make your RHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h2o = RHF.MF(h2o, 10)

# Perform an SCF/DIIS calculation
h2o.get_scf_solution_diis()


# RHF functions can have both internal and external instabilities and both can be checked.
# As for now, stability analysis can't be followed, meaning it does not yet return a set of MO coefficients
# which results in a lower lying energy/more stable solution
h2o.stability_analysis('internal')
h2o.stability_analysis('external')

# H2O is an example of a stable RHF solution. Now let's look at an unstable one.
# H4 in the given geometry is a triplet, which means you should be able to find a lower lying energy state with UHF
# This is told by the stability analysis.
h4 = RHF.MF(h4, 4)
h4.get_scf_solution()

h4.stability_analysis('internal')
h4.stability_analysis('external')
