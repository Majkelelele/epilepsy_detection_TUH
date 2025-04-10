import numpy as np
import matplotlib.pyplot as plt

# Load EEG data
data = np.load("/Users/michalzmyslony/Projects/TUH/sliced_data/train/eps/slice_2618.npy.npz")["arr_0"]
# data = np.load("/Users/michalzmyslony/Projects/TUH/sliced_data/train/no_eps/slice_1036698.npy.npz")["arr_0"]

# print("Data shape:", data.shape)  

# Settings
n_channels, n_times = data.shape
time = np.linspace(0, n_times, n_times)  # Assuming uniform sampling
offset = 100  # Vertical spacing between channels

# Plot
plt.figure(figsize=(12, 8))
for i in range(n_channels):
    plt.plot(time, data[i] + i * offset, label=f"Ch {i+1}")

plt.title("EEG Signal (19 channels)")
plt.xlabel("Time")
plt.ylabel("Amplitude (offset by channel)")
plt.yticks(np.arange(n_channels) * offset, [f"Ch {i+1}" for i in range(n_channels)])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
