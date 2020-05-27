from hf.HartreeFock import *
from pyscf import *
import numpy as np
import matplotlib.pyplot as plt

h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin=1, basis='cc-pvdz')
x = cUHF_s.MF(h3, 3)
na = 2
nb = 1
E_list = []
S_list = []
lam = np.arange(0,5.5,0.1)

for i in lam:
    E, S = x.get_scf_solution(contype='thesis', lag=i)
    ds = S - (0.5*(na-nb))**2 - 0.5*(na-nb)
    E_list.append(E)
    S_list.append(ds)

uhf_ref = -1.5062743202605442
rohf_ref =-1.5031118630938636

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Lagrange multiplier \u03BB')
ax1.set_ylabel('Energy (Hartree)', color=color)
ax1.plot(lam, E_list, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(lam, len(lam)*[uhf_ref], color = 'grey', label='UHF energy')
ax1.plot(lam, len(lam)*[rohf_ref], color = 'black', label = 'ROHF energy')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
          ncol=3, fancybox=True, shadow=True)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('$\u03B4_S$', color=color)  # we already handled the x-label with ax1
ax2.plot(lam, S_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()