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

# Psi4:
h3_psi4 = psi4.geometry("""
0 2
H 0 0 0
H 0 0.86602540378 0.5
H 0 0 1
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz'})

# Now you have to make your UHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h3 = UHF.MF(h3, 3)
h3_psi4 = UHF.MF(h3_psi4, 3, int_method='psi4')

# To perform a calculation and print out the result, simply run:
h3.get_scf_solution_diis()
h3_psi4.get_scf_solution_diis()

# You can also edit the convergence criterion
# The default is 1e-12
h3.get_scf_solution_diis(convergence=1e-6)
h3_psi4.get_scf_solution_diis(convergence=1e-6)
