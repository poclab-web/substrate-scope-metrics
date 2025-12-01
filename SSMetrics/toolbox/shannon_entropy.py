"""
Calculates the Shannon entropy.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import gaussian_kde

from SSMetrics.toolbox.read_descriptors import read_descriptors


def read_bits(df, nBits):
    """Reads ECFP bits from DataFrame.

    Args:
        df (DataFrame): DataFrame containing ECFP bits.
        nBits (int): The number of ECFP bits.

    Returns:
        DataFrame: DataFrame consisting only of ECFP bits.
    """
    columns = list(range(0, nBits))
    
    df1 = df[columns].copy()
    df1.reset_index(inplace=True, drop=True)

    return df1


def entropy_continuous_kde(sr):
    """Calculates the Shannon entropy of continuous data using KDE.

    The integral range of the entropy calculation is from min value -1 to max value +1.

    Args:
        sr (Series): Calculation target (Continuous data)

    Returns:
        tuple: Entropy (numpy.float), Probabilities (Series)
    """
    min_sr = min(sr)
    max_sr = max(sr)

    kde = gaussian_kde(sr)
    x = np.linspace(min_sr - 1, max_sr + 1, 100)
    p = pd.Series(kde(x), index=x)

    entropy = integrate.quad(lambda x: - kde(x) * np.log2(kde(x)),
                             min_sr - 1, max_sr + 1)

    print(f"{entropy[0]} ± {entropy[1]} bit")

    return entropy[0], p


def entropy_discrete(sr):
    """Calculates the Shannon entropy of discrete data.

    Args:
        sr (Series): Calculation target (Discrete data)

    Returns:
        tuple: Entropy (numpy.float), Probabilities (Series)
    """
    p = sr.value_counts() / len(sr)
    
    entropy = 0
    for x in p:
        if x == 0:
            continue
        else:
            entropy -= x * np.log2(x)

    print(f"{entropy} bit")

    return entropy, p


def sum_entropy_28d(df, png):
    """Calculates the sum of Shannon entropy.

    Args:
        df (DataFrame): DataFrame containing descriptors.
        png (str): Destination path of the diagram of the probability distribution.

    Returns:
        tuple: Sum of entropy (numpy.float), Entropy for each descriptor (list)
    """
    df1 = read_descriptors(df)

    li_entropy = []
    fig = plt.figure(figsize=(20, 30))

    i = 0
    for column in df1.columns:
        print(f"{column}:", end="\t")
        ax = fig.add_subplot(6, 5, i+1)
        ax.set_xlabel(column)

        if "num_" in column:
            entropy = entropy_discrete(df1[column])
            ax.bar(entropy[1].index, entropy[1].values)
            
        else:
            entropy = entropy_continuous_kde(df1[column])
            ax.plot(entropy[1].index, entropy[1].values)
        
        li_entropy.append(entropy[0])
        i += 1

    ans = sum(li_entropy)
    print(f"\nSum of Entropy:\t{ans} bit")

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.show()

    return ans, li_entropy


def sum_entropy_bits(df, nBits):
    """Calculates the sum of Shannon entropy.
    Args:
        df (DataFrame): DataFrame containing ECFP bits.
        nBits (int): The number of ECFP bits.

    Returns:
        tuple: Sum of entropy (numpy.float), Entropy for each ECFP bit (list)
    """
    df1 = read_bits(df, nBits)
    p = df1.sum()
    p /= p.sum()

    li_entropy = []
    for x in p:
        if x == 0:
            li_entropy.append(0)
        else:
            li_entropy.append(-(x * np.log2(x) + (1-x) * np.log2(1-x)))

    ans = sum(li_entropy)
    print(f"Sum of Entropy:\t{ans} bit")

    return ans, li_entropy
