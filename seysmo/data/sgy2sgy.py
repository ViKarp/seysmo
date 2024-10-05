import sys
from math import floor

import pandas as pd
import segyio
import numpy as np
import os

def coord_func(x):
    """
    Function to calculate the coordinate of a central point
    :param x: dataframe
    :return: coordinate of the central point (dataframe)
    """
    min_chan = x.loc[x['CHAN'].idxmin()]
    max_chan = x.loc[x['CHAN'].idxmax()]
    midpoint_x = (min_chan['REC_X'] + max_chan['REC_X']) / 2
    midpoint_y = (min_chan['REC_Y'] + max_chan['REC_Y']) / 2
    return pd.Series({'Receiver Midpoint_X': midpoint_x, 'Receiver Midpoint_Y': midpoint_y})


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
            scalar = abs(headers[0][segyio.TraceField.SourceGroupScalar])
            for header in headers:
                base['FFID'].append(header[segyio.TraceField.FieldRecord])
                base['CHAN'].append(header[segyio.TraceField.TraceNumber])
                base['REC_X'].append(header[segyio.TraceField.GroupX]/scalar)
                base['REC_Y'].append(header[segyio.TraceField.GroupY]/scalar)
    except:
        with segyio.open(filename, "r", endian='little') as segyfile:
            headers = segyfile.header
            scalar = abs(headers[0][segyio.TraceField.SourceGroupScalar])
            for header in headers:
                base['FFID'].append(header[segyio.TraceField.FieldRecord])
                base['CHAN'].append(header[segyio.TraceField.TraceNumber])
                base['REC_X'].append(header[segyio.TraceField.GroupX] / scalar)
                base['REC_Y'].append(header[segyio.TraceField.GroupY] / scalar)
    df = pd.DataFrame(base)
    return df.groupby('FFID').apply(coord_func).reset_index()


def merge_df(coord_df, filepath):
    """
    Merge two dataframes (coord and depth) into a single
    :param coord_df: dataframe containing coordinate of central point
    :param filepath: filepath to .txt file with depths
    :return: merged dataframe
    """
    profile_txt = pd.read_csv(filepath, delimiter="\t")
    profile_txt.rename(columns={"Receiver Midpoint": "Receiver Midpoint_X"}, inplace=True)
    coord_df["Receiver Midpoint_X"] = coord_df["Receiver Midpoint_X"].apply(floor)
    coord_df["Receiver Midpoint_Y"] = coord_df["Receiver Midpoint_Y"].apply(floor)
    profile_txt["Receiver Midpoint_X"] = profile_txt["Receiver Midpoint_X"].apply(floor)
    df_last = pd.merge(coord_df, profile_txt, left_on='Receiver Midpoint_X', right_on='Receiver Midpoint_X')
    df_last['Depth'] = -df_last['Depth']
    df_last.drop(['REC_STAT1', 'REC_STAT2', 'Vs30'], axis=1, inplace=True)
    return df_last


def txt2sgy(input_df, output_filepath: str) -> None:
    grouped_df = input_df.groupby(['Receiver Midpoint_X', 'Receiver Midpoint_Y', 'FFID'])
    max_samples = max(group.shape[0] for name, group in grouped_df)
    spec = segyio.spec()
    spec.ilines = np.arange(1, len(grouped_df) + 1)
    spec.xlines = np.arange(1)
    spec.samples = np.arange(max_samples)
    spec.format = 1

    with segyio.create(output_filepath, spec) as f:
        f.text[0] = segyio.tools.wrap("Created with segyio")
        f.bin.update({
            segyio.BinField.Interval: 100,
            segyio.BinField.Samples: max_samples,
            segyio.BinField.Format: 1,
        })

        trace_idx = 0

        for (x, y, ffid), group in grouped_df:
            group = group.sort_values(by='Depth', ascending=False)

            trace_data = np.zeros(max_samples, dtype=np.float32)
            trace_data[:len(group['Velocity'])] = group['Velocity'].values

            f.trace[trace_idx] = trace_data

            f.header[trace_idx].update({
                segyio.TraceField.SourceGroupScalar: -100,
                segyio.TraceField.ElevationScalar: -100,
                segyio.TraceField.CDP_X: int(x * 100),
                segyio.TraceField.CDP_Y: int(y * 100),
                segyio.TraceField.SourceX: int(x * 100),
                segyio.TraceField.SourceY: int(y * 100),
                segyio.TraceField.TRACE_SAMPLE_COUNT: max_samples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: 10000,
                segyio.TraceField.FieldRecord: ffid,
                segyio.TraceField.TraceNumber: trace_idx + 1

            })

            trace_idx += 1

def find_files(root_dir, extension='.sgy'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list


def full_pipeline(input_sgy: str, input_txt: str, output_sgy: str):
    txt2sgy(merge_df(segy2df(input_sgy), input_txt), output_sgy)


if __name__ == "__main__":
    # Нам для запуска необходимо 3 пути:
    # input_sgy - путь до входного профиля для подсчета координат, нахождения FFID
    # input_txt - путь до входной кривой для глубин
    # output_sgy - путь по которому будет записан выходной файл
    # Для того чтобы это запустить полное выполнение для многих файлов необходимо создать 3 массива с каждыми путями,
    # где по индексам будет полное соответствие. То есть первый элемент в каждом массиве будет использован для преобразования
    # Для удобства создания таких файлов я написал тебе функцию find_files, которая возвращает все файлы в директории с определенным
    # форматом. Если у тебя входные файлы имеют одинаковое название - оно автоматически отсортирует оба файла в правильном
    # порядке. Приведу пример директории и для нее использование функций
    # INPUTS:
    #   001.sgy
    #   002.sgy
    #   003.sgy
    #   001.txt
    #   002.txt
    #   003.txt
    input_sgy = find_files('../../data/raw/exp/', '.sgy')
    input_txt = find_files('../../data/raw/exp/', '.txt')
    output_sgy = [f"../../data/interim/OUTPUT/sg{i}.sgy" for i in range(len(input_sgy))]
    for i in range(len(input_sgy)):
        full_pipeline(input_sgy[i], input_txt[i], output_sgy[i])
    # То есть ты первым этапом находишь все .sgy файлы в директории INPUTS, затем все .txt файлы в этой директории.
    # Так как они названы одинаково, то они будут в одинаковом порядке. Затем создаешь названия для выходного файла.
    # Моим кодом они будут иметь такой путь: "OUTPUTS/1.sgy", "OUTPUTS/2.sgy", "OUTPUTS/3.sgy".
    # Далее проходим и выполняем всю последовательность действий по всем файлам.
    # Ты всегда можешь скопировать все функции в Jupyter Notebook и проверить в каком порядке у тебя создаются массивы,
    # что они все в правильном порядке.
    # Надеюсь понятно объяснил) Удачи в использовании!
    #
    #

