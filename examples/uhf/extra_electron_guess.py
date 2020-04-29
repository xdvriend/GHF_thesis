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
# The default integral calculation method is pySCF.`
h4 = UHF.MF(h4, 4)

# Sometimes, to force a system into its <S_z> = 0 state, we can use a trick
# By adding 2 electrons to the system and using this new test system's coefficients as a new guess
# we can sometimes find a lower lying solution.
extra_e_guess = h4.extra_electron_guess()
h4.get_scf_solution(extra_e_guess)
