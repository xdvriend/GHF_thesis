"""
Finding the lowest Hartree-Fock solution.
=========================================
This class uses all other Hartree fock methods to find the lowest possible HF energy,
all while breaking as little symmetry as possible. ROHF is not included in this method, so for open shell molecules
UHF is the starting point.
"""


from hf.HartreeFock import *


class Find:
    """
    Finding the lowest HF energy
    ----------------------------
    This class uses stability analysis and the ability to follow the lowest eigenvector in order to find the lowest
    Hartree-Fock energy.
    """
    def __init__(self, molecule, number_of_electrons, int_method='pyscf'):
        """
        Molecules are made in a way that they can be used by the other classes, so either pySCF or psi4.
        First, the algorithm looks for a stable solution within the method. Next, external stabilities are checked.
        Externally, first symmetry breaking solutions are followed, finally real/complex solutions.
        :param molecule: The molecule made as stated previously.
        :param number_of_electrons: number of electrons in the system.
        """
        self.molecule = molecule
        self.n_e = number_of_electrons
        self.int_method = int_method

    def run_algorithm(self):
        """
        This will start your calculation at the lowest possible level (RHF or UHF) and cascade downwards,
        breaking more and more symmetry along the way.
        :return: None
        """
        if self.n_e % 2 == 0:
            print('====================================')
            print('Checking real RHF internal stability')
            print('====================================')
            rhf = RHF.MF(self.molecule, self.n_e, self.int_method)
            rhf.get_scf_solution()
            imp_mo = rhf.stability_analysis('internal')
            while rhf.int_instability:
                rhf.get_scf_solution(imp_mo)
                imp_mo = rhf.stability_analysis('internal')
            print('====================================')
            print('Checking real RHF external stability')
            print('====================================')
            rhf.stability_analysis('external')
            if rhf.ext_instability_ru is None and rhf.ext_instability_rc is None:
                print("======================================")
                print('Real RHF converges to lowest HF energy')
                print("======================================")
                return None

            elif rhf.ext_instability_ru is None and rhf.ext_instability_rc:
                print('=======================================')
                print('Checking complex RHF internal stability')
                print('=======================================')
                rhf.get_scf_solution(complex_method=True)
                imp_mo = rhf.stability_analysis('internal')
                while rhf.int_instability:
                    rhf.get_scf_solution(imp_mo, complex_method=True)
                    imp_mo = rhf.stability_analysis('internal')
                print('=======================================')
                print('Checking complex RHF external stability')
                print('=======================================')
                rhf.stability_analysis('external')
                if rhf.ext_instability_ru is None:
                    print("=========================================")
                    print('Complex RHF converges to lowest HF energy')
                    print("=========================================")
                    return None

        print('====================================')
        print('Checking real UHF internal stability')
        print('====================================')
        uhf = UHF.MF(self.molecule, self.n_e, self.int_method)
        uhf.get_scf_solution()
        imp_mo_u = uhf.stability_analysis('internal')
        while uhf.int_instability:
            uhf.get_scf_solution(imp_mo_u)
            imp_mo_u = uhf.stability_analysis('internal')
        print('====================================')
        print('Checking real UHF external stability')
        print('====================================')
        uhf.stability_analysis('external')
        if uhf.ext_instability_ug is None and uhf.ext_instability_rc is None:
            print("======================================")
            print('Real UHF converges to lowest HF energy')
            print("======================================")
            return None

        elif uhf.ext_instability_ug is None and uhf.ext_instability_rc:
            print('=======================================')
            print('Checking complex UHF internal stability')
            print('=======================================')
            uhf.get_scf_solution(complex_method=True)
            imp_mo_u = uhf.stability_analysis('internal')
            while uhf.int_instability:
                uhf.get_scf_solution(imp_mo_u, complex_method=True)
                imp_mo_u = uhf.stability_analysis('internal')
            print('=======================================')
            print('Checking complex UHF external stability')
            print('=======================================')
            uhf.stability_analysis('external')
            if uhf.ext_instability_ug is None and uhf.ext_instability_rc:
                print("=========================================")
                print('Complex UHF converges to lowest HF energy')
                print("=========================================")
                return None

        print('====================================')
        print('Checking real GHF internal stability')
        print('====================================')
        ghf = GHF.MF(self.molecule, self.n_e, self.int_method)
        ghf.get_scf_solution()
        imp_mo_g = ghf.stability_analysis('internal')
        while ghf.int_instability:
            ghf.get_scf_solution(imp_mo_g)
            imp_mo_g = ghf.stability_analysis('internal')
        print('====================================')
        print('Checking real GHF external stability')
        print('====================================')
        ghf.stability_analysis('external')
        if ghf.ext_instability is None:
            print("======================================")
            print('Real GHF converges to lowest HF energy')
            print("======================================")
            return None
        else:
            print('=======================================')
            print('Checking complex GHF internal stability')
            print('=======================================')
            ghf.get_scf_solution(complex_method=True)
            imp_mo_g = ghf.stability_analysis('internal')
            i = 0
            while ghf.int_instability and i < 25:
                ghf.get_scf_solution(imp_mo_g, complex_method=True)
                imp_mo_g = ghf.stability_analysis('internal')
                i += 1
            print("=========================================")
            print('Complex GHF converges to lowest HF energy')
            print("=========================================")
            if i == 25:
                print('The lower lying complex GHF energy could not be found. (Stability analysis followed 25 times.)')
            return None
