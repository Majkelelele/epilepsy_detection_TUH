import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sci_sig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby
from pyedflib import highlevel, EdfReader
import re
from scipy.signal import resample
import os
from constants import GLOBAL_DATA


slices_path = "sliced_data/train"

def search_walk(info):
    searched_list = []
    root_path = info.get('path')
    extension = info.get('extension')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    if searched_list:
        return searched_list
    else:
        return False

def get_paths(directory, extension=".edf"):
    info = {}
    info["path"] = directory
    info["extension"] = extension
    paths = search_walk(info)
    return paths

def extract_seiz_intervals(labels_file_path):
    seizure_intervals = []
    recording_duration = None
    with open(labels_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Extract the total recording duration
            if line.startswith("# duration"):
                match = re.search(r"(\d+\.\d+)", line)  # Extract the numeric value
                if match:
                    recording_duration = int(float(match.group(1)))

            # Extract seizure intervals
            elif not line.startswith("#") and "seiz" in line:
                parts = line.split(",")
                if len(parts) >= 3:
                    start_time = int(float(parts[1]))
                    stop_time = int(float(parts[2]))

                    seizure_intervals.append((start_time, stop_time))
    return seizure_intervals, recording_duration



def save_slices(path_to_slices, slice_index, slice_data):
    os.makedirs(path_to_slices, exist_ok=True)
    file_name = f"slice_{slice_index}.npy"
    np.savez_compressed(os.path.join(path_to_slices, file_name), slice_data)

def generate_slices(file_name, signals_np, index_eps, index_no_eps, slice_size = 6 * 250, step = 250):
    sample_rate = GLOBAL_DATA['sample_rate']

    labels_file_path = file_name + ".csv_bi"
    seizure_intervals, recording_duration = extract_seiz_intervals(labels_file_path)
    slices_path = "sliced_data/train"


    record_len = signals_np.shape[1]
    
    print(f"slice size = {slice_size}")
    for start_idx in range(0, record_len - slice_size, step):
        end_idx = start_idx + slice_size
        current_slice = signals_np[:, start_idx:end_idx]

        # Check if the slice overlaps with any seizure interval
        is_seizure = False
        for seizure_start, seizure_end in seizure_intervals:
            if start_idx < seizure_end * sample_rate and end_idx > seizure_start * sample_rate:
                is_seizure = True

        label = "eps" if is_seizure else "no_eps"
        if label == "eps":
            index_eps += 1
            save_slices(slices_path + "/" + label, index_eps, current_slice)
        if label == "no_eps":
            index_no_eps += 1
            save_slices(slices_path + "/" + label, index_no_eps, current_slice)  
    print(f"created {index_eps + index_no_eps} slices")
    return index_eps, index_no_eps

def prepare_file(file, index_no_eps=0, index_eps=0):
    sample_rate = GLOBAL_DATA['sample_rate']
    electrodes_num = len(GLOBAL_DATA['labels'])
    file_name = file.split(".")[:-1][0]
    signals, signal_headers, header = highlevel.read_edf(file)

    y_labels = list(set([i['label'].replace("EEG ", "").replace("-REF", "").replace("-LE","") for i in signal_headers]))
    record_len = max([len(i) for i in signals])

    signal_sample_rate = int(max([i['sample_frequency'] for i in signal_headers]))
    assert sample_rate <= signal_sample_rate, "sample_rate is greater than signal_sample_rate"
    # assert all(elem in y_labels for elem in GLOBAL_DATA['labels']), "not all required labels in EEG signal"
    if not all(elem in y_labels for elem in GLOBAL_DATA['labels']):
        return None, signals, signal_headers, signal_sample_rate, electrodes_num, file_name
    
    return record_len, signals, signal_headers, signal_sample_rate, electrodes_num, file_name, y_labels

def generate_training_data_slices_downsampling(file, index_no_eps=0, index_eps=0):
    record_len, signals, signal_headers, signal_sample_rate, electrodes_num, file_name, y_labels = prepare_file(file, index_no_eps, index_eps)
    if record_len is None:
        return index_eps, index_no_eps
    
    target_freq = GLOBAL_DATA["resampled_freq"]
    
    resampled_len = int(record_len * target_freq / signal_sample_rate)

    signals_np = np.zeros((electrodes_num, resampled_len))
    signal_label_list = []
    next_place = 0
    for idx, signal in enumerate(signals):
        label = y_labels[idx]

        if label in GLOBAL_DATA['labels']:
            assert len(signal) == record_len, "different recording lengths in electrodes"
            signals_np[next_place, :] = resample(signal, resampled_len)
            signal_label_list.append(label)
            next_place += 1

    # assert len(signal_label_list) == len(GLOBAL_DATA['labels']), "not all required electrodes in data"
    if len(signal_label_list) != len(GLOBAL_DATA['labels']):
        print("not enough labels")
        return index_eps, index_no_eps
    
    slice_size = GLOBAL_DATA["slice_size_scs"] * target_freq
    # just to reduce memory usage
    signals_np = signals_np.astype(np.float32) 
    
    index_eps, index_no_eps = generate_slices(file_name, signals_np, index_eps, index_no_eps, slice_size, target_freq)
    return index_eps, index_no_eps
   

def generate_training_data_slices_fft(file, index_no_eps=0, index_eps=0):
    # record_len, signals, signal_headers, signal_sample_rate, electrodes_num, file_name, y_labels = prepare_file(file, index_no_eps, index_eps)
    pass

    


def generate_entire_train_set_downsampling():
    paths = get_paths("edf/train")
    index_eps = 0
    index_no_eps = 0
    for p in paths:
        index_eps, index_no_eps = generate_training_data_slices_downsampling(p, index_no_eps, index_eps)
        
def generate_entire_train_set_fft():
    paths = get_paths("edf/train")
    index_eps = 0
    index_no_eps = 0
    # for p in paths:
    #     index_eps, index_no_eps = generate_training_data_slices_downsampling(p, index_no_eps, index_eps)
    print(paths[0])
    generate_training_data_slices_fft(paths[0], index_no_eps, index_eps)
    
    

if __name__ == "__main__":        
    # generate_entire_train_set_downsampling()
    generate_entire_train_set_fft()