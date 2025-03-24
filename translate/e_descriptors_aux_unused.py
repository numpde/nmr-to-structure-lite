from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdForceFieldHelpers import UFFHasAllMoleculeParams

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from rdkit import Chem


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


if __name__ == "__main__":
    smiles = "N # C C C C c 1 n c c c c 1 N".replace(" ", "")
