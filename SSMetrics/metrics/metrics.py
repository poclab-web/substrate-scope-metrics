"""
Calculate metrics of the substrate scope.
"""

import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import distance

from SSMetrics.toolbox.read_descriptors import read_descriptors
from SSMetrics.toolbox.read_descriptors import read_DataFrame


def metrics_range(df):
    """Calculates the sum of differences between max and min values of each descriptors.

    Returns
    - Sum of differences between max and min values of each descriptors.
    - DataFrame indicating max value, min value, and range of each descriptors.

    Args:
        df (DataFrame): DataFrame consisting descriptors.
    Returns:
        tuple: As mentioned above (float, DataFrame)
    """
    df1 = read_descriptors(df)
    print(f"Substrate count: {len(df1)}")
    df1.loc["min"] = df1.min()
    df1.loc["max"] = df1.max()

    df2 = df1.loc["min":"max"].transpose()
    df2["range"] = df2["max"] - df2["min"]
    range_sum = df2["range"].sum()
    print(f"Range: {range_sum}")
    print(df2)

    return range_sum, df2


def distance_matrix(df, smiles, metric, VI):
    """Calculates distances between substrates and creates distance matrix.

    Args:
        df (DataFrame): DataFrame consisting only of SMILES and descriptors.
        smiles (str): Column name of SMILES.
        metric (str): Argument metric of scipy.spatial.distance.pdist
        VI (array_like): Argument VI of scipy.spatial.distance.mahalanobis

    Returns:
        DataFrame: Distance matrix.
    """
    df_descriptors = df.loc[:, "m_1_L":"num_Br"]

    if metric == "mahalanobis":
        distances = distance.pdist(df_descriptors, metric=metric, VI=VI)
    else:
        distances = distance.pdist(df_descriptors, metric=metric)
        
    df_distance_matrix = pd.DataFrame(distance.squareform(distances))

    return pd.concat([df[smiles], df_distance_matrix], axis=1)


def metrics_distance(df, smiles="SMILES", metric="euclidean", VI=None):
    """Calculates average and max values of distances from average coordinate to each substrate.

    Returns
    - Average coordinate
    - Sum of distances from average coodinate to each substrate
    - Distance_avg
    - SMILES of the substrate nearest to the average coordinate
    - Distance_max
    - SMILES of the substrate farthest to the average coordinate
    - Distance matrix

    Args:
        df (DataFrame): DataFrame consisting SMILES and descriptors.
        smiles (str, optional): Column name of SMILES. Defaults to "SMILES".
        metric (str, optional): Argument metric of scipy.spatial.distance.pdist. Defaults to "euclidean".
        VI (array_like, optional): Argument VI of scipy.spatial.distance.mahalanobis. Defaults to None.

    Returns:
        tuple: As mentioned above (tuple, float, float, str, float, str, DataFrame)
    """
    df1 = read_DataFrame(df, smiles)
    average = df1.mean(numeric_only=True)

    average_coordinate = tuple(average.to_list())
    print(f"Average coordinate: {average_coordinate}")
    
    average["SMILES"] = "Average"
    average_index = len(df1)
    df1.loc[average_index] = average

    df2 = distance_matrix(df1, smiles, metric, VI)
    df2["Sum"] = df2.sum(axis=1, numeric_only=True)

    distance_sum = df2.at[average_index, "Sum"]
    print(f"Sum of distances: {distance_sum}")

    distance_avg = distance_sum / (len(df2) - 1)
    print(f"Distance_avg: {distance_avg}")

    df3 = df2.drop(average_index)

    nearest_smiles = df3.at[df3[average_index].idxmin(), smiles]
    print(f"Nearest SMILES: {nearest_smiles}")

    distance_max = df3[average_index].max()
    print(f"Distance_max: {distance_max}")

    farthest_smiles = df3.at[df3[average_index].idxmax(), smiles]
    print(f"Farthest SMILES: {farthest_smiles}")
    
    return (average_coordinate, distance_sum, distance_avg,
            nearest_smiles, distance_max, farthest_smiles, df2)


def metrics_ConvexHull(array):
    """Calculates the area and vertices of the convex hull.

    Args:
        array (array): 2-dimentional coodinates of the dataset.

    Returns:
        tuple: The area and indices of vertices.
    """
    hull = ConvexHull(array)
    area = hull.area
    vertices = hull.vertices

    return area, vertices
