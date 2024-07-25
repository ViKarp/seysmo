import pandas as pd
import numpy as np
from math import floor

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
    outputs_files = find_files(root_directory)
    seysm_set = dict()
    filename = "../../data/raw/MAX_SGY/For_Masw_Resample.sgy"
    with segyio.open(filename, "r", endian='big', ignore_geometry=True) as segyfile:
        tr_start = 0
        tr_last = 1

        while tr_last != len(segyfile.trace) + 1:
            while segyfile.header[tr_last][segyio.TraceField.FieldRecord] == segyfile.header[tr_start][
                segyio.TraceField.FieldRecord]:
                tr_last += 1
                if tr_last == len(segyfile.trace):
                    break
            seysmogramm = pd.DataFrame(segyfile.trace[tr_start:tr_last]).values
            rec_mid_x = floor(
                ((segyfile.header[tr_start][segyio.TraceField.GroupX] / 100 + segyfile.header[tr_last - 1][
                    segyio.TraceField.GroupX] / 100) / 2))
            rec_mid_y = floor(
                ((segyfile.header[tr_start][segyio.TraceField.GroupY] / 100 + segyfile.header[tr_last - 1][
                    segyio.TraceField.GroupY] / 100) / 2))
            seysm_set[(rec_mid_x, rec_mid_y)] = seysmogramm
            tr_start, tr_last = tr_last, tr_last + 1
            print(tr_start, '/', len(segyfile.trace))

    for files in tqdm(outputs_files):
        with segyio.open(files, "r", endian='big', ignore_geometry=True) as segyfile_output:
            for tr_index in range(len(segyfile_output.header)):
                rec_mid_x = floor(segyfile_output.header[tr_index][segyio.TraceField.CDP_X] / 100)
                rec_mid_y = floor(segyfile_output.header[tr_index][segyio.TraceField.CDP_Y] / 100)
                if (rec_mid_x, rec_mid_y) in seysm_set.keys():
                    inputs.append(seysm_set[(rec_mid_x, rec_mid_y)])
                    outputs.append(segyfile_output.trace[tr_index])
                else:
                    print(rec_mid_x, rec_mid_y)
                    print(files)
                    print(tr_index)

    input_df = np.array(inputs)
    output_df = np.array(outputs)
    print(input_df.shape)
    print(output_df.shape)
    np.save("../../data/processed/max_inputs.npy", input_df)
    np.save("../../data/processed/max_outputs.npy", output_df)


# TODO три незаписанных аутпута

if __name__ == '__main__':
    make_input_output_dataframe("../../data/raw/2023_MSU_MASW/02.Data/03.Result/")
