import pickle
import random

import segyio
import numpy as np


def create_input_dict_field(filenames, seysm_array):
    all_shape = 0
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
                seysmogramm = np.array([segyfile.trace[tr_start]])
                for ind in range(tr_start + 1, tr_last):
                    seysmogramm = np.concatenate([seysmogramm, np.array([segyfile.trace[ind]])])
                if seysm_array is None:
                    if seysmogramm.shape == (27, 313):
                        seysm_array = np.array([seysmogramm.copy()])
                else:
                    if seysmogramm.shape == (27, 313):
                        seysm_array = np.concatenate([seysm_array, np.array([seysmogramm.copy()])])

                tr_start, tr_last = tr_last, tr_last + 1
                if random.random() > 0.99:
                    print(tr_start, '/', len(segyfile.trace))
                if seysm_array is not None and len(seysm_array) > 3000:
                    with open(f'../../data/processed/{filename.split(".")[0].split("/")[-1]}_{tr_start}.pkl', 'wb') as f:
                        pickle.dump(seysm_array, f)
                    all_shape += seysm_array.shape[0]
                    seysm_array = None
                    print('saved')
    with open(f'../../data/processed/last.pkl', 'wb') as f:
        pickle.dump(seysm_array, f)
    all_shape += seysm_array.shape[0]
    print(all_shape)


seysm_array = None
filenames = [f'D://Vitya/Работа/second_place/New_data/Area{i}_all_processed_data_ReS.sgy' for i in range(6, 7)]
create_input_dict_field(filenames, seysm_array)
