import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def ecfp(inchi_list, radius=2, nBits=2048):
    """Returns a DataFrame of ECFP.

    Args:
        inchi_list (array-like): List of InChi.
        radius (int, optional): Radius of ECFP. Defaults to 2.
        nBits (int, optional): Number of ECFP bit. Defaults to 2048.

    Returns:
        DataFrame: DataFrame of ECFP.
    """
    molecules = [Chem.MolFromInchi(inchi) for inchi in inchi_list]

    fp_list = []
    for mol in molecules:
        fp = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)]
        fp_list.append(fp)
    
    df1 = pd.DataFrame(fp_list)

    return df1
