from rdkit.Chem import MolFromSmiles, CanonSmiles
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def canon_or_none(smiles: str, use_chiral: bool = True):
    if smiles:
        smiles = smiles.replace(" ", "")
        return CanonSmiles(smiles, useChiral=int(use_chiral)) if MolFromSmiles(smiles) else None


def mol_formula_or_none(smiles: str):
    if smiles := canon_or_none(smiles):
        if mol := MolFromSmiles(smiles):
            return CalcMolFormula(mol)


def is_match_while_not_none(x, y) -> bool:
    return (x is not None) and (y is not None) and (x == y)
