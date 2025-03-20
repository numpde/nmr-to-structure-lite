from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdForceFieldHelpers import UFFHasAllMoleculeParams


def get_symmetry_number(mol):
    """ Approximates molecular symmetry by counting unique atomic ranks. """
    sym_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    return len(set(sym_classes))


def get_chiral_centers(mol):
    """ Returns the number of chiral centers in a molecule. """
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))


def count_diastereotopic_protons(mol):
    """ Counts diastereotopic hydrogen atoms using RDKit's CIP assignments. """
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.HasProp('_CIPRank'))


def get_avg_gasteiger_charge(mol):
    """ Computes the average Gasteiger charge as an approximation of electron density. """
    ComputeGasteigerCharges(mol)
    charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
    return sum(charges) / len(charges) if charges else 0


def get_fused_ring_count(mol):
    """ Counts the number of fused rings in the molecule. """
    sssr = Chem.GetSSSR(mol)  # Smallest set of smallest rings
    return sum(1 for ring in sssr if any(mol.GetAtomWithIdx(i).IsInRing() for i in ring))


def count_bridgehead_protons(mol):
    """ Counts bridgehead hydrogen atoms, which can have unique NMR shifts. """
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetIsBridgehead())


def get_rotatable_bonds(mol):
    """ Returns the number of rotatable bonds. """
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def count_internal_hbonds(mol):
    """ Estimates internal hydrogen bonding interactions within the molecule. """
    if not UFFHasAllMoleculeParams(mol):
        return 0  # Skip molecules without parameters
    hb_donors = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8] and atom.GetTotalNumHs() > 0]
    hb_acceptors = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8] and atom.GetTotalDegree() > 1]
    return sum(1 for d in hb_donors for a in hb_acceptors if d.GetIdx() != a.GetIdx())


def get_alpha_heteroatom_protons(mol):
    """ Counts the number of protons in α-position to heteroatoms (O/N), affecting chemical shifts. """
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8]:  # Oxygen or Nitrogen
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 6:  # Carbon
                    for h_neighbor in neighbor.GetNeighbors():
                        if h_neighbor.GetAtomicNum() == 1:
                            count += 1
    return count


def compute_nmr_difficulty_descriptors(smiles):
    """ Computes a set of molecular descriptors relevant to ¹H-NMR prediction difficulty. """

    mol = Chem.MolFromSmiles(smiles)
    assert mol

    return {
        "Symmetry Number": get_symmetry_number(mol),
        "Num Chiral Centers": get_chiral_centers(mol),
        "Num Diastereotopic Protons": count_diastereotopic_protons(mol),
        "Avg Gasteiger Charge": get_avg_gasteiger_charge(mol),
        "Fused Ring Count": get_fused_ring_count(mol),
        "Bridgehead Protons": count_bridgehead_protons(mol),
        "Rotatable Bonds": get_rotatable_bonds(mol),
        "Internal Hydrogen Bonds": count_internal_hbonds(mol),
        "Alpha Heteroatom Protons": get_alpha_heteroatom_protons(mol),
    }


def compute_descriptors(smiles):
    mol = MolFromSmiles(smiles)
    assert mol  # Ensure molecule is valid

    return {
        # Hydrophobicity & Polarity
        "LogP": Descriptors.MolLogP(mol),  # LogP (Hydrophobicity)
        "TPSA": Descriptors.TPSA(mol),  # Topological Polar Surface Area

        # Hydrogen-related properties
        "HBD": rdMolDescriptors.CalcNumHBD(mol),  # Hydrogen Bond Donors
        "HBA": rdMolDescriptors.CalcNumHBA(mol),  # Hydrogen Bond Acceptors
        "Num_H": Chem.AddHs(mol).GetNumAtoms() - mol.GetNumHeavyAtoms(),  # Total Hydrogen Count

        # Ring-related properties
        "Num_Rings": rdMolDescriptors.CalcNumRings(mol),  # Total number of rings
        "Num_AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),  # Aromatic rings
        "Num_Heterocycles": rdMolDescriptors.CalcNumHeterocycles(mol),  # Heterocycles (non-carbon rings)
        "Num_SpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),  # Spiro atoms
        "Num_BridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),  # Bridgehead atoms in fused systems
        "Num_SaturatedHeterocycles": rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        "Num_SaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "Num_AromaticCarbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        "Num_NumAtomStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        "Num_UnspecifiedAtomStereoCenters": rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),

        # Bond-related properties
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),  # Rotatable bonds
        "Num_AmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),  # Number of amide bonds
        "Num_DoubleBonds": sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE),
        # Double bonds
        "Num_TripleBonds": sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.TRIPLE),
        # Triple bonds
        "Num_AliphaticBonds": sum(1 for bond in mol.GetBonds() if not bond.GetIsAromatic()),  # Aliphatic bonds
        "Num_AromaticBonds": sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic()),  # Aromatic bonds

        # Halogen Counts
        "Num_Fluorine": sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "F"),  # Fluorine count
        "Num_Chlorine": sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "Cl"),  # Chlorine count
        "Num_Bromine": sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "Br"),  # Bromine count
        "Num_Iodine": sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "I"),  # Iodine count
    }

