import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
folder_normal = r"C:\Users\HP\Pictures\Dataset\Test 01_Normal Condition"
fs = 25000  # sampling frequency
gain_factor = 10**(3/20)  # gain factor for +3 dB
f = 42  # frequency of fault sine
start_idx = 1001
end_idx = 1420

# Folders to save the results
output_synthetic_folder = r"C:\Users\HP\Pictures\Synthetic misalignment\Synthetic_Faults"
output_combined_folder = r"C:\Users\HP\Pictures\Synthetic misalignment\Combined_Signals"
os.makedirs(output_synthetic_folder, exist_ok=True)
os.makedirs(output_combined_folder, exist_ok=True)

# Looping through all files
for idx in range(start_idx, end_idx + 1):
    filename_normal = f"normal_{idx}.npy"
    filepath_normal = os.path.join(folder_normal, filename_normal)

    print(f"\nProcessing file {idx - start_idx + 1}/{end_idx - start_idx + 1}: {filename_normal}")

    if not os.path.exists(filepath_normal):
        print(f"File not found: {filepath_normal}")
        continue

    # load the normal signal
    normal_signal = np.load(filepath_normal)  # shape (4, 25000)
    n = normal_signal.shape[1]
    t = np.arange(n) / fs
    freqs = np.fft.rfftfreq(n, d=1/fs)

    # FFT for channel 1
    fft_values_ch1 = np.fft.rfft(normal_signal[0])
    amplitude_ch1 = np.abs(fft_values_ch1) / n
    mask = (freqs >= 2) & (freqs <= 500)
    max_amp_ch1 = amplitude_ch1[mask].max()

    # FFT for channel 3
    fft_values_ch3 = np.fft.rfft(normal_signal[2])
    amplitude_ch3 = np.abs(fft_values_ch3) / n
    max_amp_ch3 = amplitude_ch3[mask].max()

    print(f"Max amplitude Ch1: {max_amp_ch1:.6f}, Ch3: {max_amp_ch3:.6f}")

    # make sine signals for channel 1 and 3
    sinusoidal_ch1 = (max_amp_ch1 * gain_factor) * np.sin(2 * np.pi * f * t)
    sinusoidal_ch3 = (max_amp_ch3 * gain_factor) * np.sin(2 * np.pi * f * t)

    # create empty array and add the synthetic signals
    synthetic_fault_signal = np.zeros((4, n))
    synthetic_fault_signal[0] = sinusoidal_ch1
    synthetic_fault_signal[2] = sinusoidal_ch3

    # save only synthetic signal
    output_file_fault = os.path.join(output_synthetic_folder, f"synthetic_misalignment_{idx}.npy")
    np.save(output_file_fault, synthetic_fault_signal)
    print(f"Synthetic fault saved: {output_file_fault}")

    # add synthetic fault into normal signal
    combined_signal = (0.5 * normal_signal) + (2 * synthetic_fault_signal)
    output_file_combined = os.path.join(output_combined_folder, f"misalignment_{idx}.npy")
    np.save(output_file_combined, combined_signal)
    print(f"Combined signal saved: {output_file_combined}")

print("\nAll signals processed successfully!")
