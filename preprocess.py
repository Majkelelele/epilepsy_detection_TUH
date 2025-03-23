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




sampling_rate = 250
seiz_path = "edf/train/aaaaaogk/s001_2012/01_tcp_ar/aaaaaogk_s001_t002.edf"

no_seiz_path = "edf/train/aaaaaogk/s001_2012/01_tcp_ar/aaaaaogk_s001_t000.edf"
no_seiz_2 = "edf/train/aaaaanzr/s003_2012/01_tcp_ar/aaaaanzr_s003_t000.edf"
GLOBAL_DATA = {}

slices_path = "sliced_data/train"

def read_and_plot(file_name):

    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    print(f"signals in file = {n}")
    signal_labels = np.array(f.getSignalLabels())
    l = max(f.getNSamples())
    print(f"len = {l}")
    real_labels = signal_labels[f.getNSamples() == l]
    print(f"real labels = {real_labels}")
    n = sum(f.getNSamples() == l)
    print(n)
    sigbufs = np.zeros((n, l))
    for i in np.arange(n):
            sig = f.readSignal(i)
            if len(sig) == l:
                sigbufs[i, :] = sig
            
    start = 1
    end = 10
    seizure = 6
    no_seiz = 7
    no_seiz2 = 8
            
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot seizure signal
    axes[0].plot(sigbufs[seizure, start * sampling_rate : end * sampling_rate], color="blue")
    axes[0].set_title(f"Seizure Signal ({real_labels[seizure]})")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Plot no_seiz signal
    axes[1].plot(sigbufs[no_seiz, start * sampling_rate : end * sampling_rate], color="red")
    axes[1].set_title(f"No Seizure Signal ({real_labels[no_seiz]})")
    axes[1].set_xlabel("Time (samples)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(sigbufs[no_seiz2, start * sampling_rate : end * sampling_rate], color="yellow")
    axes[2].set_title(f"No Seizure Signal ({real_labels[no_seiz2]})")
    axes[2].set_xlabel("Time (samples)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()


import numpy as np
import os
import re
from pyedflib import highlevel


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

def generate_slices(file_name, signals_np, slice_size = 6 * 250, step = 250):
    sample_rate = GLOBAL_DATA['sample_rate']

    labels_file_path = file_name + ".csv_bi"
    seizure_intervals, recording_duration = extract_seiz_intervals(labels_file_path)
    slices_path = "sliced_data/train"
    index_eps = 0
    index_no_eps = 0

    record_len = signals_np.shape[1]
    

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

def generate_training_data_slices(file):
    sample_rate = GLOBAL_DATA['sample_rate']
    electrodes_num = len(GLOBAL_DATA['labels'])
    file_name = file.split(".")[:-1][0]
    signals, signal_headers, header = highlevel.read_edf(file)

    y_labels = list(set([i['label'].replace("EEG ", "").replace("-REF", "") for i in signal_headers]))
    record_len = max([len(i) for i in signals])

    signal_sample_rate = int(max([i['sample_frequency'] for i in signal_headers]))
    assert sample_rate <= signal_sample_rate, "sample_rate is greater than signal_sample_rate"
    assert all(elem in y_labels for elem in GLOBAL_DATA['labels']), "not all required labels in EEG signal"
    target_freq = 100
    
    resampled_len = int(record_len * target_freq / sampling_rate)

    signals_np = np.zeros((electrodes_num, resampled_len))
    signal_label_list = []
    next_place = 0
    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].replace("EEG ", "").replace("-REF", "")

        if label in GLOBAL_DATA['labels']:
            assert len(signal) == record_len, "different recording lengths in electrodes"
            signals_np[next_place, :] = resample(signal, resampled_len)
            signal_label_list.append(label)
            next_place += 1

    assert len(signal_label_list) == len(GLOBAL_DATA['labels']), "not all required electrodes in data"
    slice_size = GLOBAL_DATA["slice_size_scs"] * signal_sample_rate
    # just to reduce memory usage
    signals_np = signals_np.astype(np.float32) 
    
    generate_slices(file_name, signals_np, slice_size)
   


# Example usage
GLOBAL_DATA = {}
GLOBAL_DATA['sample_rate'] = 250
GLOBAL_DATA['labels'] = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'CZ', 'T3', 'T4', 'P3', 'P4', 'O1', 'O2', 'T5', 'T6', 'PZ', 'FZ']
GLOBAL_DATA['slice_size_scs'] = 6




generate_training_data_slices(seiz_path)