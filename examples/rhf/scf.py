from hf.HartreeFock import *
from pyscf import *
import psi4


# Make a molecule, this can either be done with pySCF or psi4
# PySCF:
h2o = gto.M(atom='O 0.000000000000 -0.143225816552 0.000000000000; '
                 'H 1.638036840407 1.136548822547 -0.000000000000; '
                 'H -1.638036840407 1.136548822547 -0.000000000000',
            unit='bohr',
            basis='sto-3g')

# Psi4:
h2o_psi4 = psi4.geometry("""
O 0.000000000000 -0.143225816552 0.000000000000
H 1.638036840407 1.136548822547 -0.000000000000
H -1.638036840407 1.136548822547 -0.000000000000
units bohr
symmetry c1
""")
psi4.set_options({'basis': 'sto-3g'})

# Now you have to make your RHF mean field "object". This is what you use to perform calculations
# The arguments are a molecule, the number of electrons and the method of integral calculation.
# The default integral calculation method is pySCF.
h2o = RHF.MF(h2o, 10)
h2o_psi4 = RHF.MF(h2o_psi4, 10, int_method='psi4')

# To perform a calculation and print out the result, simply run:
h2o.get_scf_solution()
h2o_psi4.get_scf_solution()

# You can also edit the convergence criterion
# The default is 1e-12
h2o.get_scf_solution(convergence=1e-6)
h2o_psi4.get_scf_solution(convergence=1e-6)
