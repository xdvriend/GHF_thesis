from hf.HartreeFock import *
from pyscf import *
import numpy as np
import matplotlib.pyplot as plt

h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin=1, basis='STO-3G')
x = cUHF_s.MF(h3, 3)
E = []
for i in np.arange(0, 2, 0.1):
    E.append(x.scf(contype='thesis', lag=i)[0])

P = plt.scatter(np.arange(0,2,0.1),E)
plt.show()

