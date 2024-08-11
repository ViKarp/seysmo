import pandas as pd
import numpy as np
from math import floor
import pickle
import swprocess

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


def create_input_dict(filenames, seysm_dict, only_x=False):
    for filename in filenames:
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
                    ((segyfile.header[tr_start][segyio.TraceField.GroupX] / 10 + segyfile.header[tr_last - 1][
                        segyio.TraceField.GroupX] / 10) / 2))
                rec_mid_y = floor(
                    ((segyfile.header[tr_start][segyio.TraceField.GroupY] / 10 + segyfile.header[tr_last - 1][
                        segyio.TraceField.GroupY] / 10) / 2))
                if only_x:
                    seysm_dict[rec_mid_x] = (seysmogramm, rec_mid_y)
                else:
                    seysm_dict[(rec_mid_x, rec_mid_y)] = seysmogramm
                tr_start, tr_last = tr_last, tr_last + 1
                print(tr_start, '/', len(segyfile.trace))


def create_input_power_dict(filename, seysm_dict):
    fmin, fmax = 1, 10
    vmin, vmax, nvel, vspace = 50, 500, 500, "linear"
    settings = swprocess.Masw.create_settings_dict(fmin=fmin, fmax=fmax,
                                                   vmin=vmin, vmax=vmax, nvel=nvel, vspace=vspace)
    with segyio.open(filename, "r", endian='big',
                     ignore_geometry=True) as segyfile:
        tr_start = 0
        tr_last = 1

        while tr_last != len(segyfile.trace) + 1:
            while segyfile.header[tr_last][segyio.TraceField.FieldRecord] == segyfile.header[tr_start][
                segyio.TraceField.FieldRecord]:
                tr_last += 1
                if tr_last == len(segyfile.trace):
                    break
            spec = segyio.spec()
            spec.ilines = np.arange(1, tr_last - tr_start)
            spec.xlines = np.arange(1)
            spec.samples = segyfile.samples
            spec.format = 1
            with segyio.create("./temp.sgy", spec) as f:
                f.text[0] = segyio.tools.wrap("Created with segyio")
                f.bin = segyfile.bin
                f.trace = segyfile.trace[tr_start:tr_last]
                f.header = segyfile.header[tr_start:tr_last]
            wavefieldtransform = swprocess.Masw.run(fnames="./temp.sgy", settings=settings)
            wavefieldtransform.normalize()
            power_disp = wavefieldtransform.power
            rec_mid_x = floor(
                ((segyfile.header[tr_start][segyio.TraceField.GroupX] / 100 + segyfile.header[tr_last - 1][
                    segyio.TraceField.GroupX] / 100) / 2))
            rec_mid_y = floor(
                ((segyfile.header[tr_start][segyio.TraceField.GroupY] / 100 + segyfile.header[tr_last - 1][
                    segyio.TraceField.GroupY] / 100) / 2))
            seysm_dict[(rec_mid_x, rec_mid_y)] = power_disp
            tr_start, tr_last = tr_last, tr_last + 1
            print(tr_start, '/', len(segyfile.trace))


def make_input_output_dataframe(root_directory, output_filename):
    inputs = []
    outputs = []
    outputs_files = find_files(root_directory)
    seysm_dict = dict()
    filename = "../../data/raw/second_place/Area2_processed_data_for_MASW_part.sgy"
    create_input_dict(filename, seysm_dict)

    for files in tqdm(outputs_files):
        with segyio.open(files, "r", endian='big', ignore_geometry=True) as segyfile_output:
            for tr_index in range(len(segyfile_output.header)):
                rec_mid_x = floor(segyfile_output.header[tr_index][segyio.TraceField.CDP_X] / 100)
                rec_mid_y = floor(segyfile_output.header[tr_index][segyio.TraceField.CDP_Y] / 100)
                if (rec_mid_x, rec_mid_y) in seysm_dict.keys():
                    inputs.append(seysm_dict[(rec_mid_x, rec_mid_y)])
                    outputs.append(segyfile_output.trace[tr_index])
                else:
                    print(rec_mid_x, rec_mid_y)
                    print(files)
                    print(tr_index)

    input_df = np.array(inputs)
    output_df = np.array(outputs)
    print(input_df.shape)
    print(output_df.shape)
    np.save(f"../../data/processed/{output_filename}_inputs.npy", input_df)
    np.save(f"../../data/processed/{output_filename}_outputs.npy", output_df)


def create_dict_with_coord(root_directory, input_format: str = 'seysmo', output_filename: str = 'coord_dict'):
    outputs_files = find_files(root_directory)
    seysm_dict = dict()
    final_dict = dict()
    filename = "../../data/raw/MAX_SGY/For_Masw_Resample.sgy"
    if input_format == 'seysmo':
        create_input_dict(filename, seysm_dict)
    if input_format == 'power':
        create_input_power_dict(filename, seysm_dict)

    for files in tqdm(outputs_files):
        with segyio.open(files, "r", endian='big', ignore_geometry=True) as segyfile_output:
            for tr_index in range(len(segyfile_output.header)):
                rec_mid_x = floor(segyfile_output.header[tr_index][segyio.TraceField.CDP_X] / 100)
                rec_mid_y = floor(segyfile_output.header[tr_index][segyio.TraceField.CDP_Y] / 100)
                if (rec_mid_x, rec_mid_y) in seysm_dict.keys():
                    final_dict[(rec_mid_x, rec_mid_y)] = (
                        seysm_dict[(rec_mid_x, rec_mid_y)], segyfile_output.trace[tr_index])
                else:
                    print(rec_mid_x, rec_mid_y)
                    print(files)
                    print(tr_index)

    with open(f'../../data/processed/{output_filename}.pkl', 'wb') as f:
        pickle.dump(final_dict, f)


def create_dict_with_coord_from_txt(root_directory, output_filename):
    outputs_files = find_files(root_directory, '.txt')
    seysm_dict = dict()
    final_dict = dict()
    filenames = ["../../data/raw/second_place/Area2_processed_data_for_MASW_part.sgy",
                 "../../data/raw/second_place/Area1_processed_data_for_MASW.sgy",
                 "../../data/raw/second_place/Area3_all_processed_data_for_MASW_1.sgy",
                 "../../data/raw/second_place/Area3_processed_data_for_MASW-2.sgy",
                 "../../data/raw/second_place/Area5_processed_data_for_MASW.sgy"]
    create_input_dict(filenames, seysm_dict, only_x=True)

    for file in tqdm(outputs_files):
        df = pd.read_csv(file, delimiter='\t')
        df = df[['Receiver Midpoint', 'Depth', 'Velocity']]
        def replace_commas(value):
            if isinstance(value, str):  # Проверяем, является ли значение строкой
                return value.replace(',', '.')
            return value  # Возвращаем значение без изменений, если оно не строка

        df = df.applymap(replace_commas)
        df = df.astype(float)
        df = df.groupby(['Receiver Midpoint'])
        def process_group(group):
            # Сортировка по Depth
            sorted_group = group.sort_values(by='Depth')
            # Получаем все значения Velocity в виде списка
            velocity_array = sorted_group['Velocity'].tolist()
            return velocity_array

        result = df.apply(process_group)
        result_df = pd.DataFrame(result, columns=['Velocity'])
        for mid_x in result_df.index:
            rec_mid_x = floor(mid_x)
            if rec_mid_x in seysm_dict.keys():
                seysmogramm, rec_mid_y = seysm_dict[rec_mid_x]
                final_dict[(rec_mid_x, rec_mid_y)] = (
                    seysmogramm, result_df.loc[mid_x])
            else:
                print(rec_mid_x)
                print(mid_x)

    with open(f'../../data/processed/{output_filename}.pkl', 'wb') as f:
        pickle.dump(final_dict, f)


if __name__ == '__main__':
    # make_input_output_dataframe("../../data/raw/second_place/Result_TXT/", output_filename="second_place")
    # create_dict_with_coord("../../data/raw/2023_MSU_MASW/02.Data/03.Result/", input_format='power',
    #                        output_filename='coord_power_dict')
    create_dict_with_coord_from_txt("../../data/raw/second_place/Result_TXT/", output_filename="second_place")