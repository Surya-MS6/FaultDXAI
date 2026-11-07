import numpy as np
import os

# Parameters
folder_normal = r"C:\Users\HP\Pictures\Dataset\Test 17_Normal Condition"
output_folder = r"C:\Users\HP\Pictures\New folder\Test 17"

fs = 25000  # sampling frequency
gain_factor = 10**(3/20)  # +3 dB gain factor
rotation_freq = 21  # base frequency
harmonics_N = 5  # number of harmonics

# create output folder
os.makedirs(output_folder, exist_ok=True)

# ------------------- Step 1: Analyze all files -------------------
print("Analyzing all .npy files to find maximum peaks...")

# list all .npy files
all_files = sorted([f for f in os.listdir(folder_normal) if f.endswith('.npy')])
print(f"Found {len(all_files)} .npy files")

# store amplitudes
all_max_amps = []
file_max_amps = {}

for file_count, filename in enumerate(all_files, start=1):
    filepath_normal = os.path.join(folder_normal, filename)

    # load normal signal
    normal_signal = np.load(filepath_normal)  # shape (4, 25000)

    n = normal_signal.shape[1]
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mask = (freqs >= 2) & (freqs <= 200)

    # find max FFT amplitude for each channel
    fft_vals = [np.abs(np.fft.rfft(normal_signal[ch])) / n for ch in range(4)]
    max_amps = [fft_ch[mask].max() for fft_ch in fft_vals]

    # store results
    file_max_amps[filename] = max_amps
    all_max_amps.append(max_amps)

    # print only first 5 files
    if file_count <= 5:
        print(f"Max amps → Ch1: {max_amps[0]:.6f}, Ch2: {max_amps[1]:.6f}, Ch3: {max_amps[2]:.6f}, Ch4: {max_amps[3]:.6f}")

# convert to numpy array
all_max_amps = np.array(all_max_amps)  # shape (420, 4)

# print statistics
print("\nAnalysis results:")
print("=" * 60)
for ch in range(4):
    ch_amps = all_max_amps[:, ch]
    print(f"Channel {ch+1}:")
    print(f"   Min amplitude: {ch_amps.min():.6f}")
    print(f"   Max amplitude: {ch_amps.max():.6f}")
    print(f"   Mean amplitude: {ch_amps.mean():.6f}")
    print(f"   Std amplitude: {ch_amps.std():.6f}")
    print()

# ------------------- Step 2: Generate synthetic looseness -------------------
print("Generating synthetic looseness faults...")

for file_count, filename in enumerate(all_files, start=1):
    filepath_normal = os.path.join(folder_normal, filename)

    print(f"\nProcessing file {file_count}/{len(all_files)}: {filename}")

    # load normal signal again
    normal_signal = np.load(filepath_normal)  # shape (4, 25000)

    n = normal_signal.shape[1]
    t = np.arange(n) / fs

    # get max amplitudes for this file
    max_amps = file_max_amps[filename]

    # create synthetic fault signal
    synthetic_fault = np.zeros_like(normal_signal)

    # function to define harmonics for looseness fault
    def generate_looseness_frequencies(channel, base_freq=21):
        if channel == 0:
            return [base_freq, 3 * base_freq]
        elif channel == 1:
            return [base_freq, 2 * base_freq, 6 * base_freq, 7 * base_freq, 8 * base_freq]
        elif channel == 2:
            return [base_freq, 2 * base_freq]
        elif channel == 3:
            return [base_freq, 2 * base_freq, 3 * base_freq, 4 * base_freq,
                    7 * base_freq, 8 * base_freq, 9 * base_freq]
        return []

    # generate signals for each channel
    for ch in range(4):
        ch_signal = np.zeros(n)
        fault_amplitude = 2 * max_amps[ch] * gain_factor
        frequencies = generate_looseness_frequencies(ch)

        # add sine waves for selected harmonics
        for i, freq in enumerate(frequencies):
            amplitude_factor = 1.0 / (1 + 0.1 * i)  # reduce higher harmonics
            A = fault_amplitude * amplitude_factor
            ch_signal += A * np.sin(2 * np.pi * freq * t)

        synthetic_fault[ch] = ch_signal

    # combine normal and fault signals
    combined_signal = normal_signal + synthetic_fault

    # save the new signal
    output_file = os.path.join(output_folder, filename.replace('normal', 'looseness'))
    np.save(output_file, combined_signal)

    if file_count <= 5:
        print(f"Saved → {output_file}")

print("\nAnalysis complete!")
print(f"Processed all {len(all_files)} files")
print(f"Generated synthetic looseness signals for each file")
print(f"Saved all combined signals to: {output_folder}")
