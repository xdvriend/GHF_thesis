from hf.HartreeFock import *
from pyscf import *


# Make a molecule, this can either be done with pySCF or psi4
# Here, we use PySCF:
h2o = gto.M(atom='O 0.000000000000 -0.143225816552 0.000000000000; '
                 'H 1.638036840407 1.136548822547 -0.000000000000; '
                 'H -1.638036840407 1.136548822547 -0.000000000000',
            unit='bohr',
            basis='sto-3g')

# Now you have to make your RHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h2o = RHF.MF(h2o, 10)

# Perform an SCF/DIIS calculation
h2o.get_scf_solution_diis()

# Now you can ask for the different matrices used during the calculation
# The default matrix will be the last one, but you can specify which iteration you want to look at
last_fock = h2o.get_fock()
print(last_fock)

second_fock = h2o.get_fock(1)
print(second_fock)

# The same can be done for the density matrices, the fock matrices in the orthonormal basis and the MO coefficients
last_dens = h2o.get_dens()
last_mo = h2o.get_mo_coeff()
last_fock_orth = h2o.get_fock_orth()
print(last_dens, last_mo, last_fock_orth)
