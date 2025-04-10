import os

GLOBAL_DATA = {}
GLOBAL_DATA['sample_rate'] = 250
GLOBAL_DATA['labels'] = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'CZ', 'T3', 'T4', 'P3', 'P4', 'O1', 'O2', 'T5', 'T6', 'PZ', 'FZ']
GLOBAL_DATA['resampled_freq'] = 100
GLOBAL_DATA["slice_size_scs"] = 6
GLOBAL_DATA['batch_size'] = 64
GLOBAL_DATA["epochs"] = 5
GLOBAL_DATA['epilepsy_path'] = "sliced_data/train/eps"
GLOBAL_DATA['no_epilepsy_path'] = "sliced_data/train/no_eps"
GLOBAL_DATA["saving_models_path"] = "saved_models/"
GLOBAL_DATA["cores_count"] = os.cpu_count()
GLOBAL_DATA["window_len_seconds"] = 1
GLOBAL_DATA["workers"] = os.cpu_count() - 1
EPS = 1
NO_EPS = 0
SEED = 42