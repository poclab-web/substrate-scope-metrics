"""
Reads descriptors from DataFrame.
"""

def read_descriptors(df):
    """Reads descriptors from DataFrame.

    Args:
        df (DataFrame): DataFrame containing descriptors.

    Returns:
        DataFrame: DataFrame consisting only of descriptors.
    """
    columns = ["m_1_L", "m_1_B1", "m_1_B5", "p_L", "p_B1", "p_B5",
               "m_2_L", "m_2_B1", "m_2_B5", "o_L", "o_B1", "o_B5",
               "Charge_O", "Charge_next_O", "Charge_o_1", "Charge_m_1",
               "Charge_p", "Charge_m_2", "Charge_o_2",
               "HOMO", "MolLogP", "MolWt",
               "num_N", "num_O", "num_S", "num_F", "num_Cl", "num_Br"]
    
    df1 = df[columns].copy()
    df1.reset_index(inplace=True, drop=True)

    return df1


def read_DataFrame(df, smiles):
    """Reads SMILES and descriptors from DataFrame.

    Args:
        df (DataFrame): DataFrame containing SMILES and descriptors.
        smiles (str): Column name of SMILES.

    Returns:
        DataFrame: DataFrame consisting only of SMILES and descriptors.
    """
    columns = [smiles,
               "m_1_L", "m_1_B1", "m_1_B5", "p_L", "p_B1", "p_B5",
               "m_2_L", "m_2_B1", "m_2_B5", "o_L", "o_B1", "o_B5",
               "Charge_O", "Charge_next_O", "Charge_o_1", "Charge_m_1",
               "Charge_p", "Charge_m_2", "Charge_o_2",
               "HOMO", "MolLogP", "MolWt",
               "num_N", "num_O", "num_S", "num_F", "num_Cl", "num_Br"]
    
    df1 = df[columns].copy()
    df1.reset_index(inplace=True, drop=True)

    return df1
