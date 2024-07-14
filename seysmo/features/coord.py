import pandas as pd
import segyio
import numpy as np


def coord_func(x):
    """
    Function to calculate the coordinate of a central point
    :param x: dataframe
    :return: coordinate of the central point (dataframe)
    """
    return pd.DataFrame(
        [[(np.min(x['REC_X']) + np.max(x['REC_X'])) / 2, (np.min(x['REC_Y']) + np.max(x['REC_Y'])) / 2]],
        columns=["Receiver Midpoint_X", "Receiver Midpoint_Y"])


def segy2df(filename):
    """
    Function to read a segy and return df with coordinates of central points
    :param filename: path to segy file
    :return: dataframe with coordinates of central points
    """
    base = {'FFID': [], 'CHAN': [], "REC_X": [], "REC_Y": []}
    try:
        with segyio.open(filename, "r", ignore_geometry=True) as segyfile:
            headers = segyfile.header
            for header in headers:
                base['FFID'].append(header[segyio.TraceField.FieldRecord])
                base['CHAN'].append(header[segyio.TraceField.TraceNumber])
                base['REC_X'].append(header[segyio.TraceField.GroupX])
                base['REC_Y'].append(header[segyio.TraceField.GroupY])
    except:
        with segyio.open(filename, "r", endian='little') as segyfile:
            headers = segyfile.header
            for header in headers:
                base['FFID'].append(header[segyio.TraceField.FieldRecord])
                base['CHAN'].append(header[segyio.TraceField.TraceNumber])
                base['REC_X'].append(header[segyio.TraceField.GroupX])
                base['REC_Y'].append(header[segyio.TraceField.GroupY])
    df = pd.DataFrame(base)
    return df.groupby('FFID').apply(coord_func)


def merge_df(coord_df, filepath):
    """
    Merge two dataframes (coord and depth) into a single
    :param coord_df: dataframe containing coordinate of central point
    :param filepath: filepath to .txt file with depths
    :return: merged dataframe
    """
    profile_txt = pd.read_csv(filepath, delimiter="\t")
    profile_txt.rename(columns={"Receiver Midpoint": "Receiver Midpoint_X"}, inplace=True)
    coord_df["Receiver Midpoint_X"] = coord_df["Receiver Midpoint_X"] / 10
    coord_df["Receiver Midpoint_Y"] = coord_df["Receiver Midpoint_Y"] / 10
    df_last = pd.merge(coord_df, profile_txt, left_on='Receiver Midpoint_X', right_on='Receiver Midpoint_X')
    df_last['Depth'] = -df_last['Depth']
    df_last.drop(['REC_STAT1', 'REC_STAT2', 'Vs30'], axis=1, inplace=True)
    return df_last


if __name__ == "__main__":
    df = segy2df('./Train_data/SGY/Profile.sgy')
    last_df = merge_df(df, f"./Train_data/TXT/Profile.txt")
    #print(last_df)
    last_df.to_csv("./Train_data/OUTPUT/Profile.txt", sep=",", index=False)
