from GHF.RHF import RHF
from GHF.UHF import UHF
from GHF.SCF_functions import *
from pyscf import *
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')

h4 = gto.M(atom = 'h 0 0.707107 0; h 0.707107 0 0; h 0 -0.707107 0; h -0.707107 0 0' ,spin = 2, basis = 'cc-pvdz')


def test_RHF():
    assert RHF(h4, 2) == scf.UHF(h4)[1]

def test_UHF():
    assert UHF(h3, 2, 1) == scf.UHF(h3)[1]

mf = scf.UHF(h4).run()
mf.stability()
mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)

def test_stability():
    assert UHF(h4, 2, 2, internal_stability_analysis=True) == mf[1]


