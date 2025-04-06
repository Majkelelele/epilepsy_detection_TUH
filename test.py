import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
from constants import GLOBAL_DATA

# Load data
data = torch.tensor(np.load("/Users/michalzmyslony/Projects/TUH/sliced_data/train/eps/slice_36291.npy.npz")["arr_0"], dtype=torch.float32)
x = data[0].numpy()

# Signal settings
N = 6
F = 100
fs = F
t = torch.arange(N * F) / fs
# f1 = 10
# f2 = 25
# x1 = np.sin(t * 2 * np.pi * f1)
# x1[:300] = 0
# x2 = np.sin(t * 2 * np.pi * f2)
# x2[200:400] = 0

# x = x1 + x2

window_len = int(1 * F)
shift = int(0.5 * window_len)

# # Define window
# g_std = 8
# w = windows.gaussian(window_len, std=g_std, sym=True)
w = windows.hann(window_len)


# Perform STFT
# res = torch.zeros((len(GLOBAL_DATA["labels"]), window_len + 1, 13))

# for i in range(len(GLOBAL_DATA["labels"])):
#     x = data[0]
#     f, t_stft, Zxx = stft(x, fs=fs, window=w, nperseg=window_len, noverlap=window_len - shift, nfft=200, return_onesided=True)
#     res[i, :, :] = torch.abs(torch.from_numpy(Zxx))
    
# print(res.shape)

x = data[0]
f, t_stft, Zxx = stft(x, fs=fs, window=w, nperseg=window_len, noverlap=window_len - shift, nfft=200, return_onesided=True)
print(Zxx.shape)

# Plot
# fig1, ax1 = plt.subplots(figsize=(6., 4.))
# # ax1.set_title(f"STFT (Gaussian window, σ={g_std})")
# ax1.set_xlabel(f"Time $t$ in seconds")
# ax1.set_ylabel(f"Frequency $f$ in Hz")

# im1 = ax1.imshow(np.abs(Zxx), origin='lower', aspect='auto',
#                  extent=[t_stft[0], t_stft[-1], f[0], f[-1]],
#                  cmap='viridis')

# fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
# ax1.axvline(0, color='y', linestyle='--', alpha=0.5)
# ax1.axvline(N, color='y', linestyle='--', alpha=0.5)
# fig1.tight_layout()
# plt.show()

