import os
from colorsys import hls_to_rgb

import numpy as np


def get_data_subfolders(folder):
    # iterate over multiple data folders
    subfolders = []
    for file in os.listdir(folder):
        d = os.path.join(folder, file)
        if os.path.isdir(d):
            subfolders.append(os.path.basename(d))

    return subfolders[::-1]


def get_color(number):
    return "rgb" + str(hls_to_rgb((0.75 - number * 3 / 20) % 1, 0.4, 0.75))


def get_color_state0(number):
    return "rgb" + str(hls_to_rgb((-0.35 - number * 9 / 20) % 1, 0.6, 0.75))


def get_color_state1(number):
    return "rgb" + str(hls_to_rgb((-0.02 - number * 9 / 20) % 1, 0.6, 0.75))


def grouped_by_mean(df, g_column1, m_column1, g_column2=None, m_column2=None):
    r"""
    Given a pandas data frame, the function generates groups of unique pairs of g_column1 and g_column2
    and calculates the mean of the given columns m_column1 and m_column2.

    Args:
        df: Pandas Data Frame
        g_column1 (int): Data Frame column to group data
        g_column2 (int): Data Frame column to group data
        m_column1 (int): Data Frame column to calculate data mean
        m_column2 (int): Data Frame column to calculate data mean

    Returns:
        unique_column1 (array): Array with unique values of m_column1
        unique_column2 (array): Array with unique values of m_column2
        grouped_m_column1_mean (array): Array with the mean of m_column1 values for the differents groups created
        grouped_m_column2_mean (array): Array with the mean values of m_column2 for the differents groups created
    """

    df = df.astype(float)
    data = df.to_numpy()

    # group by 1 column
    if (g_column2 == None) and (m_column2 == None):
        print("grouped by 1 columns")
        # Group by column g_column and calculate mean of m_column
        unique_column, column_indexes = np.unique(
            data[:, g_column1], return_inverse=True
        )
        data_m_column = data[:, m_column1]
        grouped_m_column = np.zeros(len(unique_column))

        for i, param in enumerate(unique_column):
            mask = column_indexes == i
            grouped_m_column[i] = np.mean(data_m_column[mask])

        return unique_column, grouped_m_column

    # group by 2 columns
    else:
        print("grouped by 2 columns")
        # Group by columns g_column1 and g_column2 and calculate mean of m_column1 and m_column2
        unique_column1, column1_indexes = np.unique(
            data[:, g_column1], return_inverse=True
        )
        unique_column2, column2_indexes = np.unique(
            data[:, g_column2], return_inverse=True
        )

        data_m_column1 = data[:, m_column1]
        data_m_column2 = data[:, m_column2]

        grouped_m_column1 = np.zeros((len(unique_column1), len(unique_column2)))
        grouped_m_column2 = np.zeros((len(unique_column1), len(unique_column2)))

        for i, param_c1 in enumerate(unique_column1):
            for j, param_c2 in enumerate(unique_column2):
                mask = (column1_indexes == i) & (column2_indexes == j)
                grouped_m_column1[i, j] = np.mean(data_m_column1[mask])
                grouped_m_column2[i, j] = np.mean(data_m_column2[mask])

        grouped_m_column1_mean = np.transpose(grouped_m_column1)
        grouped_m_column2_mean = np.transpose(grouped_m_column2)

        return (
            unique_column1,
            unique_column2,
            grouped_m_column1_mean,
            grouped_m_column2_mean,
        )
