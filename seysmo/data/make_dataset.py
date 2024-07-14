import pandas as pd
import numpy as np

from obspy import read
import segyio

import os
from tqdm import tqdm


def find_files(root_dir, extension='.sgy'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list


def make_unite_output(root_dir, extension='.txt'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_list.append(pd.read_csv(os.path.join(root, file), delimiter="\t"))
    return pd.concat(file_list, ignore_index=True)


def round_to_nearest_5(number):
    last_digit = number % 10
    if last_digit < 3:
        return number - last_digit
    elif last_digit < 8:
        return number - last_digit + 5
    else:
        return number - last_digit + 10

def make_input_output_dataframe(root_directory):
    inputs = []
    outputs = []
    sgy_files = find_files(root_directory)
    output_df = make_unite_output(root_directory[:-3] + 'TXT')
    for file in tqdm(sgy_files):
        segyfile = read(file)
        input_vector = segyfile[0].data
        inputs.append(input_vector)
        with segyio.open(file, "r", endian='little') as segyfile:
            # Читаем заголовки трасс
            headers = segyfile.header
            x_coord = round_to_nearest_5(headers[0][segyio.TraceField.SourceX])
            y_coord = round_to_nearest_5(headers[0][segyio.TraceField.SourceY])
            output_vector = output_df[(output_df['Receiver_Midpoint_X'] == x_coord/100)]['Velocity'].to_numpy()
            if len(output_vector) > 10:
                output_vector = output_df[(output_df['Receiver_Midpoint_X'] == x_coord/100) & (output_df['Receiver_Midpoint_Y'] == y_coord/100)]['Velocity'].to_numpy()
            if len(output_vector) == 0 or np.isnan(np.sum(output_vector)):
                print(x_coord)
                print(y_coord)
                print(file)
            outputs.append(output_vector)
    input_df = pd.DataFrame(inputs)
    output_df = pd.DataFrame(outputs)
    input_df.to_csv("../../data/processed/max_inputs.csv")
    output_df.to_csv("../../data/processed/max_outputs.csv")
#TODO три незаписанных аутпута

if __name__ == '__main__':
    make_input_output_dataframe("../../data/raw/SGY")
