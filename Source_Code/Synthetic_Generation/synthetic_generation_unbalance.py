import numpy as np
import os

# Parameters
folder_normal = r"C:\Users\HP\Pictures\Dataset\Test 17_Normal Condition"
fs = 25000  # sampling frequency
gain_factor = 10**(3/20)  # +3 dB
f1 = 21  # unbalance low frequency
f2 = 63  # unbalance higher harmonic
start_idx = 5001
end_idx = 5420

# Output folders
output_synthetic_folder = r"C:\Users\HP\Pictures\Synthetic unbalance\Synthetic_Faults"
output_combined_folder = r"C:\Users\HP\Pictures\Synthetic unbalance\Combined_Signals"
os.makedirs(output_synthetic_folder, exist_ok=True)
os.makedirs(output_combined_folder, exist_ok=True)

# Loop through all files
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
    mask = (freqs >= 2) & (freqs <= 500)

    # find max FFT amplitude for all channels
    fft_vals = [np.abs(np.fft.rfft(normal_signal[ch])) / n for ch in range(4)]
    max_amps = [fft_ch[mask].max() for fft_ch in fft_vals]

    print(f"Max amps → Ch1: {max_amps[0]:.6f}, Ch2: {max_amps[1]:.6f}, Ch3: {max_amps[2]:.6f}, Ch4: {max_amps[3]:.6f}")

    # create sine signals for 21 Hz and 63 Hz
    sine21 = [max_amps[ch] * np.sin(2 * np.pi * f1 * t) for ch in range(4)]
    sine63 = [max_amps[ch] * np.sin(2 * np.pi * f2 * t) for ch in range(4)]

    # apply +3 dB gain to Ch1 at 21 Hz
    sine21[0] *= gain_factor

    # make synthetic fault signal
    synthetic_fault = np.zeros_like(normal_signal)
    synthetic_fault[0] = (sine21[0] * 3) + (sine63[0] * 2)
    synthetic_fault[1] = (sine21[1] * 2) + (sine63[1])
    synthetic_fault[2] = (sine21[2] * 3)
    synthetic_fault[3] = (sine21[3] * 2) + (sine63[3])

    # save synthetic fault only
    output_file_fault = os.path.join(output_synthetic_folder, f"synthetic_unbalance_{idx}.npy")
    np.save(output_file_fault, synthetic_fault)
    print(f"Synthetic fault saved: {output_file_fault}")

    # combine normal and fault signals
    combined_signal = normal_signal + synthetic_fault

    # save combined signal
    output_file_combined = os.path.join(output_combined_folder, f"unbalance_{idx}.npy")
    np.save(output_file_combined, combined_signal)
    print(f"Combined signal saved: {output_file_combined}")

print("\nAll unbalance signals processed successfully!")
