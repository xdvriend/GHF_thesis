from GHF.RHF import RHF
from GHF.UHF import UHF
from pyscf import *


"""Create a molecule with pyscf"""
# This part needs to be ran in pyscf
# Create the molecules as specified in the RHF/UHF code


h = gto.M(atom = 'h 0 0 0', spin = 1, basis = 'cc-pvdz')

h3 = gto.M(atom = 'h 0 0 0; h 0 0.86602540378 0.5; h 0 0 1', spin = 1, basis = 'cc-pvdz')

h4 = gto.M(atom = 'h 0 0.707107 0; h 0.707107 0 0; h 0 -0.707107 0; h -0.707107 0 0' ,spin = 2, basis = 'cc-pvdz')

h5 = gto.M(atom = 'h 0 0.850651 0; h 0.8090171766429887 0.26286561528204344 0; h 0.5000001126478447, -0.6881911152820434, 0;'
                  ' h -0.5000001126478445, -0.6881911152820435, 0; h -0.8090171766429888 0.2628656152820433 0', spin = 1, basis = 'cc-pvdz')

h6 = gto.M(atom = 'h 0 0 0; h 0 2 0; h 0.866025 0.5 0; h -0.866025 0.5 0; h 0.866025 1.5 0; h -0.866025 1.5 0',
           basis = 'cc-pvdz')

h7 = gto.M(atom = 'h 0 1.15238 0; h 0.900966963766508 0.7184971778659673 0; h 1.12348742744009 -0.2564286738725775 0;'
                  'h 0.49999894328429173 -1.0382585039933896 0; h -0.4999989432842915 -1.0382585039933898 0;'
                  'h -1.12348742744009 -0.2564286738725768 0; h -0.9009669637665083 0.7184971778659671 0', spin = 1, basis = 'cc-pvdz')

h8 = gto.M(atom = 'h 0 1.30656 0; h 0.9238774360270954 0.9238774360270956 0; h 1.30656 0 0; h 0.9238774360270956 -0.9238774360270954 0;'
                  'h 0 -1.30656 0; h -0.9238774360270954 -0.9238774360270957 0; h -1.30656 0 0; h -0.9238774360270957 0.9238774360270954 0;',spin = 0, basis = 'cc-pvdz')

h9 = gto.M(atom = 'h 0 1.4619 0; h 0.9396912066007517 1.119880371395634 0; h 1.439690454128547 0.25385627093128554 0;'
                  'h 1.2660425377924711 -0.7309499999999997 0; h 0.49999924752779534 -1.3737366423269193 0; '
                  'h -0.499999247527795 -1.3737366423269195 0; h -1.2660425377924704 -0.7309500000000007 0; '
                  'h -1.4396904541285471 0.25385627093128493 0; h -0.9396912066007522 1.1198803713956338 0', spin = 1, basis = 'cc-pvdz')

h10 = gto.M(atom = 'h 0 1.61803 0; h 0.9510541717667903 1.3090137674084963 0; h 1.5388379750610472 0.49999876740849625 0;'
                   'h 1.5388379750610475 -0.4999987674084961 0; h 0.9510541717667905 -1.309013767408496 0; '
                   'h 0 -1.61803 0; h -0.9510541717667902 -1.3090137674084963 0; h -1.5388379750610472 -0.4999987674084964 0;'
                   'h -1.5388379750610475 0.4999987674084959 0; h -0.9510541717667905 1.309013767408496 0', basis = 'cc-pvdz')

h11 = gto.M(atom = 'h 0 1.77473 0; h 0.9594914779629726 1.4929978823214822 0; h 1.6143511911155242 0.7372494860248379 0;'
                   'h 1.7566658075493475 -0.2525704129287475 0; h 1.341251442093733 -1.1622009903547155 0; '
                   'h 0.49999922060319113 -1.7028409650628566 0; h -0.49999922060319074 -1.7028409650628569 0;'
                   'h -1.3412514420937325 -1.162200990354716 0; h -1.7566658075493475 -0.25257041292874755 0; '
                   'h -1.614351191115524 0.7372494860248385 0; h -0.9594914779629737 1.4929978823214813 0', spin = 1, basis = 'cc-pvdz')

h12 = gto.M(atom = 'h 0 1.93185 0; h 0.9659249999999999 1.673031176300968 0; h 1.6730311763009678 0.9659250000000003 0; '
                   'h 1.93185 0 0; h 1.6730311763009682 -0.9659249999999996 0; h 0.9659249999999999 -1.673031176300968 0; h 0 -1.93185 0;'
                   'h -0.9659250000000003 -1.6730311763009678 0; h -1.6730311763009673 -0.9659250000000009 0; h -1.93185 0 0;'
                   'h -1.6730311763009678  0.965925 0; h -0.9659250000000009 1.6730311763009673 0',spin = 0, basis = 'cc-pvdz')

h13 = gto.M(atom = 'h 0 2.08929 0; h 0.9709414761193252 1.849974419836995 0; h 1.7194519611729573 1.1868519946979366 0; '
                   'h 2.0740567235643232 0.25183608069064384 0; h 1.9535200856802104 -0.7408724444490994 0; '
                   'h 1.3854555386359115 -1.5638560210463994 0; h 0.49999982423935135 -2.0285790297300763 0; '
                   'h -0.4999998242393518 -2.0285790297300763 0; h -1.3854555386359113 -1.5638560210463999 0; '
                   'h -1.9535200856802102 -0.7408724444490998 0; h -2.0740567235643237 0.2518360806906424 0; '
                   'h -1.7194519611729568 1.1868519946979375 0; h -0.9709414761193248 1.8499744198369952 0', spin = 1, basis = 'cc-pvdz')

h14 = gto.M(atom = 'h 0 2.24698 0; h 0.9749280841223709 2.024459026799378 0; h 1.7567597044760135 1.4009691149805374 0; '
                   'h 2.1906435201143144 0.5000000881811595 0; h 2.1906435201143144 -0.5000000881811593 0; '
                   'h 1.7567597044760133 -1.4009691149805379 0; h 0.9749280841223711 -2.0244590267993776 0;'
                   'h 0 -2.24698 0; h -0.9749280841223705 -2.024459026799378 0; h -1.7567597044760135 -1.4009691149805377 0; '
                   'h -2.1906435201143144 -0.5000000881811578 0; h -2.190643520114314 0.5000000881811608 0; '
                   'h -1.756759704476014 1.4009691149805368 0; h -0.9749280841223712 2.0244590267993776 0',spin = 0, basis = 'cc-pvdz')

h15 = gto.M(atom = 'h 0 2.40487 0; h 0.9781487508336996 2.1969580647209614 0; h 1.787166696445821 1.6091721213142274 0; '
                   'h 2.2871672843427255 0.7431456992624799 0; h 2.391695870514299 -0.251377365458482 0; h 2.0826785127990832 -1.2024349999999995 0;'
                   'h 1.4135471196806 -1.9455806992624796 0; h 0.5000005878969047 -2.3523178205767072 0; '
                   'h -0.5000005878969053 -2.352317820576707 0; h -1.4135471196805995 -1.9455806992624798 0; '
                   'h -2.0826785127990823 -1.202435000000001 0; h -2.3916958705142997 -0.2513773654584815 0; '
                   'h -2.287167284342726 0.7431456992624793 0; h -1.7871666964458222 1.6091721213142263 0; '
                   'h -0.9781487508336995 2.196958064720962 0', spin = 1, basis = 'cc-pvdz')


"""calculate RHF energies of the 4 molecules"""
# input: molecule, number of occupied orbitals

#RHF(h4, 2)
#RHF(h6, 3)
#RHF(h8, 4)
#RHF(h10, 5)
#RHF(h12, 6)
#RHF(h14, 7)

"""calculate UHF energies of the 4 molecules"""
# input: molecule, number of occupied alpha, number of occupied beta,
# extra electron method, internal stability analysis

#q = UHF(h, 1, 0)
a = UHF(h3, 2, 1)
#b = UHF(h4, 2, 2, extra_e_coeff=True)
#c = UHF(h5, 3, 2)
#d = UHF(h6, 3, 3)
#e = UHF(h7, 4, 3)
#f = UHF(h8, 4, 4, internal_stability_analysis=True)
#g = UHF(h9, 5, 4)
#h = UHF(h10, 5, 5)
#i = UHF(h11, 6, 5)
#j = UHF(h12, 6, 6, internal_stability_analysis=True)
#k = UHF(h13, 7, 6)
l = UHF(h14, 7, 7,internal_stability_analysis=True)
#m = UHF(h15, 8, 7)

mf = scf.UHF(h3).run()




"""Calculate Delta E in kcal/mol"""


#d3 = (a - 3 * q) * 627.5
#d4 = (b - 4 * q) * 627.5
#d5 = (c - 5 * q) * 627.5
#d6 = (d - 6 * q) * 627.5
#d7 = (e - 7 * q) * 627.5
#d8 = (f - 8 * q) * 627.5
#d9 = (g - 9 * q) * 627.5
#d10 = (h - 10 * q) * 627.5
#d11 = (i - 11 * q) * 627.5
#d12 = (j - 12 * q) * 627.5
#d13 = (k - 13 * q) * 627.5
#d14 = (l - 14 * q) * 627.5
#d15 = (m - 15 * q) * 627.5


#print(d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15)



