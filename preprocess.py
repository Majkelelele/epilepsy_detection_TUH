import os
import re
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
from scipy.signal import resample, stft, windows
from pyedflib import highlevel
from constants import GLOBAL_DATA

slices_path: str = "sliced_data/train"


def search_walk(info: Dict[str, str]) -> Union[List[str], bool]:
    searched_list: List[str] = []
    root_path = info.get('path')
    extension = info.get('extension')

    for path, _, files in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension:
                list_file = os.path.join(path, filename)
                searched_list.append(list_file)

    return searched_list if searched_list else False


def get_paths(directory: str, extension: str = ".edf") -> Union[List[str], bool]:
    info: Dict[str, str] = {"path": directory, "extension": extension}
    return search_walk(info)


def extract_seiz_intervals(labels_file_path: str) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    seizure_intervals: List[Tuple[int, int]] = []
    recording_duration: Optional[int] = None

    with open(labels_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("# duration"):
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    recording_duration = int(float(match.group(1)))
            elif not line.startswith("#") and "seiz" in line:
                parts = line.split(",")
                if len(parts) >= 3:
                    seizure_intervals.append((int(float(parts[1])), int(float(parts[2]))))

    return seizure_intervals, recording_duration


def save_slices(path_to_slices: str, slice_index: int, slice_data: np.ndarray) -> None:
    os.makedirs(path_to_slices, exist_ok=True)
    file_name = f"slice_{slice_index}.npy"
    np.savez_compressed(os.path.join(path_to_slices, file_name), slice_data)


def generate_slices(
    file_name: str,
    signals_np: np.ndarray,
    index_eps: int,
    index_no_eps: int,
    slice_size: int = 6 * 250,
    step: int = 250
) -> Tuple[int, int]:
    sample_rate = GLOBAL_DATA['sample_rate']
    labels_file_path = file_name + ".csv_bi"
    seizure_intervals, _ = extract_seiz_intervals(labels_file_path)

    record_len = signals_np.shape[1]

    print(f"slice size = {slice_size}")
    for start_idx in range(0, record_len - slice_size, step):
        end_idx = start_idx + slice_size
        current_slice = signals_np[:, start_idx:end_idx]

        is_seizure = any(
            start_idx < seizure_end * sample_rate and end_idx > seizure_start * sample_rate
            for seizure_start, seizure_end in seizure_intervals
        )

        label = "eps" if is_seizure else "no_eps"
        if label == "eps":
            index_eps += 1
            save_slices(f"{slices_path}/eps", index_eps, current_slice)
        else:
            index_no_eps += 1
            save_slices(f"{slices_path}/no_eps", index_no_eps, current_slice)

    print(f"created {index_eps + index_no_eps} slices")
    return index_eps, index_no_eps


def prepare_file(
    file: str,
    index_no_eps: int = 0,
    index_eps: int = 0
) -> Union[
    Tuple[int, List[np.ndarray], List[dict], int, int, str, List[str]],
    Tuple[None, List[np.ndarray], List[dict], int, int, str]
]:
    sample_rate = GLOBAL_DATA['sample_rate']
    electrodes_num = len(GLOBAL_DATA['labels'])
    file_name = file.rsplit(".", 1)[0]
    signals, signal_headers, header = highlevel.read_edf(file)

    y_labels = list(set(
        i['label'].replace("EEG ", "").replace("-REF", "").replace("-LE", "")
        for i in signal_headers
    ))
    record_len = max(len(sig) for sig in signals)
    signal_sample_rate = int(max(i['sample_frequency'] for i in signal_headers))

    if not all(label in y_labels for label in GLOBAL_DATA['labels']):
        return None, signals, signal_headers, signal_sample_rate, electrodes_num, file_name

    return record_len, signals, signal_headers, signal_sample_rate, electrodes_num, file_name, y_labels


def generate_training_data_slices_downsampling(
    file: str,
    index_no_eps: int = 0,
    index_eps: int = 0
) -> Tuple[int, int]:
    prepared = prepare_file(file, index_no_eps, index_eps)
    if prepared[0] is None:
        return index_eps, index_no_eps

    record_len, signals, _, signal_sample_rate, electrodes_num, file_name, y_labels = prepared
    target_freq = GLOBAL_DATA["resampled_freq"]
    resampled_len = int(record_len * target_freq / signal_sample_rate)

    signals_np = np.zeros((electrodes_num, resampled_len))
    signal_label_list: List[str] = []
    next_place = 0

    for idx, signal in enumerate(signals):
        label = y_labels[idx]
        if label in GLOBAL_DATA['labels']:
            assert len(signal) == record_len, "Mismatched recording length"
            signals_np[next_place, :] = resample(signal, resampled_len)
            signal_label_list.append(label)
            next_place += 1

    if len(signal_label_list) != len(GLOBAL_DATA['labels']):
        print("not enough labels")
        return index_eps, index_no_eps

    slice_size = GLOBAL_DATA["slice_size_scs"] * target_freq
    signals_np = signals_np.astype(np.float32)

    return generate_slices(file_name, signals_np, index_eps, index_no_eps, slice_size, target_freq)


def generate_entire_train_set_downsampling() -> None:
    paths = get_paths("edf/train")
    index_eps, index_no_eps = 0, 0
    if paths:
        for p in paths:
            index_eps, index_no_eps = generate_training_data_slices_downsampling(p, index_no_eps, index_eps)


def make_spectogram(data: np.ndarray) -> torch.Tensor:
    N = GLOBAL_DATA["slice_size_scs"]
    fs = GLOBAL_DATA["resampled_freq"]
    window_len = int(GLOBAL_DATA["window_len_seconds"] * fs)
    w = windows.hann(window_len)
    shift = int(0.5 * window_len)

    res: List[torch.Tensor] = []

    for i in range(len(GLOBAL_DATA["labels"])):
        _, _, Zxx = stft(
            data[i], fs=fs, window=w,
            nperseg=window_len,
            noverlap=window_len - shift,
            return_onesided=True
        )
        spectrogram = torch.from_numpy(np.abs(Zxx))
        res.append(spectrogram)

    tensor_feature = torch.stack(res)
    return tensor_feature


if __name__ == "__main__":
    generate_entire_train_set_downsampling()
