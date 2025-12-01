from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw

from SSMetrics.toolbox.ecfp import ecfp


def fp_dropna(inchi_list, radius, nBits):
    """Deletes the bits taking 0 in all molecules.

    Args:
        inchi_list (list): List of InChi.
        radius (int): Radius of ECFP.
        nBits (int): Number of ECFP bit.

    Returns:
        DataFrame: DataFrame of ECFP without the bit taking 0 in all molecules.
    """
    df1 = ecfp(inchi_list, radius=radius, nBits=nBits)
    df1.loc[len(df1)] = df1.sum(numeric_only=True)
    df2 = df1.T[df1.T[len(df1) - 1] != 0].T
    
    return df2


def on_bits(inchi_list, radius, nBits):
    """Calculates ECFP OnBits.

    Args:
        inchi_list (list): List of InChi.
        radius (int): Radius of ECFP.
        nBits (int): Number of ECFP bit.

    Returns:
        tuple: ECFP OnBits (int), Bit numbers (index), DataFrame
    """
    df2 = fp_dropna(inchi_list, radius=radius, nBits=nBits)
    bit_list = df2.columns
    ans = len(bit_list)
    print(ans)

    return ans, bit_list, df2


def draw_on_bits(inchi_list, svg, radius=2, nBits=2048):
    """Calculates ECFP OnBits and draws partial structures of bits.

    Args:
        inchi_list (list): List of InChi.
        svg (str): Destination path of svg file.
        radius (int, optional): Radius of ECFP. Defaults to 2.
        nBits (int, optional): Number of ECFP bit. Defaults to 2048.

    Returns:
        int: ECFP OnBits
    """
    ans, bit_list, df2 = on_bits(inchi_list, radius=radius, nBits=nBits)

    draw_list = []
    for i in bit_list:
        for j in range(len(df2) - 1):

            if df2.at[j, i] == 1:
                mol = Chem.MolFromInchi(inchi_list[j])
                bi = {}
                AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=bi)

                draw_list.append((mol, i, bi))
                break

    print(len(draw_list))
    img = Draw.DrawMorganBits(draw_list, molsPerRow=10,
                              legends=[str(x) for x in bit_list], useSVG=True)

    with open(svg, mode="w") as f:
        f.write(img.data)

    return ans


def tanimoto(inchi_list, radius=2, nBits=2048):
    """Calculates Tanimoto distances between molecules.

    Tanimoto distance = 1 - Tanimoto similarity

    Returns
    - Average value
    - Maximum value
    - Minimum value
    - Combination of molecules providing maximum value (InChI)
    - Combination of molecules providing minimum value (InChI)

    Args:
        inchi_list (list): List of InChi.
        radius (int, optional): Radius of ECFP. Defaults to 2.
        nBits (int, optional): Number of ECFP bit. Defaults to 2048.

    Returns:
        tuple: As mentioned above (float, float, float, list, list)
    """
    molecules = [Chem.MolFromInchi(inchi) for inchi in inchi_list]

    fp_list = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        for mol in molecules]
    
    comb = list(combinations([_ for _ in range(len(inchi_list))], 2))

    distances = []
    for i in comb:
        tanimoto = DataStructs.TanimotoSimilarity(fp_list[i[0]], fp_list[i[1]])
        distances.append(1 - tanimoto)

    distance_avg = np.mean(distances)
    distance_max = max(distances)
    distance_min = min(distances)

    max_inchi = []
    for i in [j for j, v in enumerate(distances) if v == distance_max]:
        max_inchi.append([inchi_list[comb[i][0]], inchi_list[comb[i][1]]])

    min_inchi = []
    for i in [j for j, v in enumerate(distances) if v == distance_min]:
        min_inchi.append([inchi_list[comb[i][0]], inchi_list[comb[i][1]]])

    print(f"Average:\t{distance_avg}")
    print(f"Max:\t\t{distance_max}")
    print(f"min:\t\t{distance_min}")

    return distance_avg, distance_max, distance_min, max_inchi, min_inchi
