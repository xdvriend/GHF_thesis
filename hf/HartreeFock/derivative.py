from hf.HartreeFock import *
from pyscf import *
import numpy as np
import matplotlib.pyplot as plt

h3 = gto.M(atom='h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin=1, basis='cc-pvdz')
x = cUHF_s.MF(h3, 3)
na = 2
nb = 1
S_list = []
lam = np.arange(0.5,5.5,0.1)

for i in lam:
    S = x.get_scf_solution(contype='thesis', lag=i)[1]
    ds = S - (0.5*(na-nb))**2 - 0.5*(na-nb)
    S_list.append(ds)

der = np.gradient(S_list)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Lagrange multiplier \u03BB')
ax1.set_ylabel('$\u03B4_S$', color=color)
ax1.plot(lam, S_list, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel("$\u03B4_S'$", color=color)  # we already handled the x-label with ax1
ax2.plot(lam, der, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.plot(lam, len(lam)*[0], color='black')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()