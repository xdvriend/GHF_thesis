from hf.HartreeFock import *
from pyscf import *

# Make a molecule in pyscf (works only in pyscf)
no = gto.M(atom = 'N 0 0 0; O 0 0 1', spin=1, basis = 'sto-3g')

# Now you have to make your UHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.

no = UHF.MF(no, 15)
no.scf()
# You are now able to calculate the mulliken charge for each atom in the pyscf molecule.
mulliken_charges, atoms = no.calculate_mulliken()

