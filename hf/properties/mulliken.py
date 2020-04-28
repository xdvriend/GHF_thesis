"""
Function to calculate Mulliken charges
==============================================

This file contain a function that calculates the Mulliken charges of the atoms.
"""


def mulliken(molecule, dens, overlap):
    """
        A function used to calculate the Mulliken charges in the (un)restricted hartree fock formalism.

        !!! NOTE: molecules must be analysed in pyscf in order for the function to work !!!
        
        :param molecule: geometry of the molecule in pyscf
        :param dens: density matrix of the molecule, either a tuple (UHF) or a single parameter (RHF)
        :param overlap: overlap matrix of the molecule
        :return: Mulliken charges, their corresponding atoms
        """
    if len(dens) > 1:   # UHF
        dens = dens[0] + dens[1]
    else:   # RHF
        dens = 2 * dens
    p = dens @ overlap  # population matrix of alpha and beta
    trace = []
    for i in range(p.shape[0]):   # gross orbital product
        for j in range(p.shape[0]):   # store the diagonal elements
            if i == j:
                trace.append(p[i][j])
    goptot, atoms = [], []
    ao = [1,1,3,1,3,1,5,3,1,5,3,1,7,5,3,1,7,5,3] # subshells and their amount of electrons pairs
    f = 0                                        # 1s2s2p3s3p4s3d4p5s4d5p6s4f5d6p7s5f6d7p
    for i in range(molecule.natm):  # gross atomic population (GAP): sums the trace elements of each atom according to their amount of shells
        atoms.append(molecule.atom_symbol(i))
        sum = 0
        shells = molecule.atom_nshells(i)
        ao_atom = ao[:shells]
        for j in ao_atom:
            for k in range(j):
                sum += trace[f]
                f += 1
        goptot.append(sum)
    charge_mulliken = []    # Mulliken charges
    for c, value in enumerate(goptot):
        charge_mulliken.append(molecule.atom_charge(c) - value) # Z - GAP
    return charge_mulliken, atoms
