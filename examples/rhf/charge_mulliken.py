from hf.HartreeFock import *
from pyscf import *

# Make a molecule in pyscf (works only in pyscf)
hf = gto.M(atom = 'F 0 0 0; H 0 0 0.9', basis = 'sto-3g')

# Now you have to make your UHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.

hf = RHF.MF(hf, 10)
hf.scf()
# You are now able to calculate the mulliken charge for each atom in the pyscf molecule.
mulliken_charges, atoms = hf.calculate_mulliken()

