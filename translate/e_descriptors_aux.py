from rdkit import Chem

from rdkit.Chem.rdForceFieldHelpers import UFFHasAllMoleculeParams
from rdkit.Chem import AllChem, rdMolDescriptors


def compute_ihd(mol):
    """
    Calculate the index of hydrogen deficiency (IHD) by looping over the atoms
    and counting how many carbons, hydrogens, nitrogens, and halogens are present.
    """

    # # Example usage:
    # mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
    # print("IHD:", compute_ihd(mol))

    # Make hydrogens explicit (if they aren't already)
    mol = Chem.AddHs(mol)
    nC = 0
    nH = 0
    nN = 0
    nHal = 0

    # Atomic numbers for halogens: F (9), Cl (17), Br (35), I (53)
    halogens = {9, 17, 35, 53}

    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()
        if Z == 1:
            nH += 1
        elif Z == 6:
            nC += 1
        elif Z == 7:
            nN += 1
        elif Z in halogens:
            nHal += 1

    # IHD = (2C + 2 + N - H - X) / 2
    ihd = (2 * nC + 2 + nN - nH - nHal) / 2
    return ihd


def get_symmetry_number(mol):
    """
    Approximates molecular symmetry by counting unique canonical ranks of atoms.
    (A higher number indicates lower overall symmetry.)
    """
    sym_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    return len(set(sym_classes))


def get_chiral_centers(mol):
    """
    Returns the number of chiral centers in the molecule.
    Chiral centers are stereogenic atoms that can lead to non-superimposable mirror images.
    """
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))


def count_diastereotopic_protons(mol):
    """
    Counts diastereotopic hydrogen atoms.
    Diastereotopic protons reside in non-equivalent chemical environments (determined here via CIP assignments).
    """
    return sum(1 for atom in mol.GetAtoms()
               if atom.GetAtomicNum() == 1 and atom.HasProp('_CIPRank'))


def get_avg_gasteiger_charge(mol):
    """
    Computes the average Gasteiger charge over all atoms.
    Gasteiger charges are an approximation of the electron distribution.
    """
    # Compute Gasteiger charges (make sure AllChem is imported)
    AllChem.ComputeGasteigerCharges(mol)
    charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
    return sum(charges) / (len(charges) or 1)


def get_fused_ring_count(mol):
    """
    Counts the number of rings that are fused with at least one other ring.
    A ring is considered fused if it shares at least two atoms with another ring.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()  # Returns a tuple of atom index tuples
    fused_rings = 0
    # For each ring, check if it shares two or more atoms with any other ring.
    for i, ring in enumerate(atom_rings):
        for j, other_ring in enumerate(atom_rings):
            if i < j and len(set(ring) & set(other_ring)) >= 2:
                fused_rings += 1
                break  # Count each ring only once
    return fused_rings


def count_bridgehead_protons(mol):
    """
    Counts hydrogen atoms attached to bridgehead heavy atoms.
    Bridgehead atoms are those that are shared by at least two rings (i.e. fused rings).
    This function first identifies heavy atoms that are bridgeheads via ring intersection.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    bridgehead_indices = set()
    # Identify heavy atoms (indices) that are shared between rings by at least 2 atoms.
    for i, ring in enumerate(atom_rings):
        for j, other_ring in enumerate(atom_rings):
            if i < j and len(set(ring) & set(other_ring)) >= 2:
                bridgehead_indices.update(set(ring) & set(other_ring))
    count = 0
    # Count hydrogen neighbors attached to each bridgehead heavy atom.
    for idx in bridgehead_indices:
        atom = mol.GetAtomWithIdx(idx)
        count += sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
    return count


def get_rotatable_bonds(mol):
    """
    Returns the number of rotatable bonds in the molecule.
    Rotatable bonds (typically single bonds not in rings) influence molecular flexibility.
    """
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def count_internal_hbonds(mol):
    """
    Estimates the number of potential internal hydrogen bonding interactions.
    Uses a crude approximation by pairing donor atoms (N/O with attached H) with acceptor atoms (N/O with degree > 1).
    Note: This does not account for geometric constraints.
    """
    # UFFHasAllMoleculeParams is assumed to be defined/imported; otherwise, you may remove the check.
    if not Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        return 0  # Skip molecules without UFF parameters
    hb_donors = [atom for atom in mol.GetAtoms()
                 if atom.GetAtomicNum() in [7, 8] and atom.GetTotalNumHs() > 0]
    hb_acceptors = [atom for atom in mol.GetAtoms()
                    if atom.GetAtomicNum() in [7, 8] and atom.GetTotalDegree() > 1]
    return sum(1 for d in hb_donors for a in hb_acceptors if d.GetIdx() != a.GetIdx())


def get_alpha_heteroatom_protons(mol):
    """
    Counts the number of protons attached to carbons that are adjacent to heteroatoms (O or N).
    This avoids double-counting by iterating over carbons rather than heteroatoms.
    """
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # Carbon atom
            # Check if any neighbor is a heteroatom (O or N)
            if any(nbr.GetAtomicNum() in [7, 8] for nbr in atom.GetNeighbors()):
                # Count hydrogen neighbors on this carbon.
                count += sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
    return count


def compute_nmr_difficulty_descriptors(smiles):
    """ Computes a set of molecular descriptors relevant to ¹H-NMR prediction difficulty. """

    mol = Chem.MolFromSmiles(smiles)
    assert mol

    return {
        # Molecular symmetry number: counts unique rotations that map the molecule onto itself,
        # affecting thermodynamic properties and entropy calculations.
        'Symmetry number': get_symmetry_number(mol),

        # Number of chiral centers: indicates stereogenic centers that lead to non-superimposable mirror images,
        # important for stereochemistry.
        'Num chiral centers': get_chiral_centers(mol),

        # Number of diastereotopic protons: measures protons in distinct chemical environments (not interconvertible by symmetry),
        # influencing NMR signal splitting.
        'Num diastereotopic protons': count_diastereotopic_protons(mol),

        # Average Gasteiger charge: provides the mean partial atomic charge (using the Gasteiger method),
        # which reflects electron distribution in the molecule.
        'Avg Gasteiger charge': get_avg_gasteiger_charge(mol),

        # Fused ring count: counts rings that share common atoms (fused together),
        # affecting molecular rigidity and aromaticity.
        'Fused ring count': get_fused_ring_count(mol),

        # Bridgehead protons: counts protons on bridgehead atoms (atoms shared by multiple rings),
        # which can influence steric effects and molecular stability.
        'Bridgehead protons': count_bridgehead_protons(mol),

        # Rotatable bonds: enumerates bonds that can rotate (typically single bonds not in rings),
        # determining molecular flexibility.
        'Rotatable bonds': get_rotatable_bonds(mol),

        # Internal hydrogen bonds: identifies potential intramolecular hydrogen bonds,
        # which can stabilize specific conformations.
        'Internal hydrogen bonds': count_internal_hbonds(mol),

        # Alpha heteroatom protons: counts protons on carbons adjacent to heteroatoms,
        # impacting reactivity and chemical shifts in NMR spectra.
        'Alpha heteroatom protons': get_alpha_heteroatom_protons(mol),

        # # Index of hydrogen deficiency: calculates the IHD based on the molecular formula,
        # # which is related to the degree of unsaturation.
        # 'IHD': compute_ihd(mol),
    }


if __name__ == "__main__":
    smiles = "N # C C C C c 1 n c c c c 1 N".replace(" ", "")

    # Count the number of atoms in the SMILES using RDKit
    mol = Chem.MolFromSmiles(smiles)
    print(f"Number of atoms: {mol.GetNumAtoms()}")

    print(compute_nmr_difficulty_descriptors(smiles))
