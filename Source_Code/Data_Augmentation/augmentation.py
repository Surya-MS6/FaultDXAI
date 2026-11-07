import numpy as np
import os
from scipy.signal import resample, periodogram

# Input & Output
input_folder = r"C:\Users\HP\Documents\Normal\Test 17_Normal Condition"
output_base = r"C:\Users\HP\Documents\Normal\Test 17_Normal Condition\Augmented"
os.makedirs(output_base, exist_ok=True)

fs = 25000  # sampling frequency

# ---------- 1. Gaussian Noise ----------
def gaussian_noise(signal):
    rms = np.sqrt(np.mean(signal**2))  # overall RMS
    beta = 0.05  # 5% RMS scaling
    noise = np.random.normal(0, 1, signal.shape)
    return signal + beta * rms * noise

# ---------- 2. Masking Noise ----------
def masking_noise(signal):
    n = signal.size
    alpha_mask = min(0.2, max(0.05, 2500 / n))  # 5–20%
    num_mask = int(alpha_mask * n)

    augmented = signal.copy().flatten()
    mask_indices = np.random.choice(n, num_mask, replace=False)
    augmented[mask_indices] = 0
    return augmented.reshape(signal.shape)

# ---------- 3. Signal Translation ----------
def signal_translation(signal):
    n = signal.shape[1]
    gamma = 0.02  # shift = 2% of length
    max_shift = int(gamma * n)

    shift_amount = np.random.randint(-max_shift, max_shift)
    augmented = np.zeros_like(signal)

    if shift_amount > 0:   # shift right
        augmented[:, shift_amount:] = signal[:, :-shift_amount]
    elif shift_amount < 0: # shift left
        augmented[:, :shift_amount] = signal[:, -shift_amount:]
    else:
        augmented = signal.copy()
    return augmented

# ---------- 4. Amplitude Shifting ----------
def amplitude_shift(signal):
    rms = np.sqrt(np.mean(signal**2))
    variation = 0.2 if rms > 0.1 else 0.1  # stronger signals → bigger scaling
    alpha_scal = np.random.uniform(1 - variation, 1 + variation)
    return alpha_scal * signal

# ---------- 5. Time Stretching ----------
def time_stretch(signal):
    n = signal.shape[1]

    # Estimate dominant frequency from channel 1
    f, Pxx = periodogram(signal[0], fs=fs)
    dominant_freq = f[np.argmax(Pxx)]

    if dominant_freq > 100:
        stretch_range = (0.9, 1.1)   # keep tight if sharp features
    else:
        stretch_range = (0.8, 1.2)   # allow more variation

    factor = np.random.uniform(*stretch_range)
    new_length = int(n * factor)

    stretched = np.zeros((signal.shape[0], n))
    for ch in range(signal.shape[0]):
        stretched_ch = resample(signal[ch], new_length)

        if new_length > n:    # crop
            stretched[ch] = stretched_ch[:n]
        else:                 # pad
            stretched[ch, :new_length] = stretched_ch
    return stretched

# ---------- Main Loop ----------
augmentations = {
    "Gaussian_Noise": gaussian_noise,
    "Masking_Noise": masking_noise,
    "Signal_Translation": signal_translation,
    "Amplitude_Shift": amplitude_shift,
    "Time_Stretch": time_stretch
}

for file in os.listdir(input_folder):
    if file.endswith(".npy"):
        filepath = os.path.join(input_folder, file)
        signal = np.load(filepath)  # (4, 25000)

        for aug_name, aug_func in augmentations.items():
            output_folder = os.path.join(output_base, aug_name)
            os.makedirs(output_folder, exist_ok=True)

            augmented_signal = aug_func(signal)
            out_path = os.path.join(output_folder, f"{aug_name.lower()}_{file}")
            np.save(out_path, augmented_signal)
            print(f"Saved {aug_name}: {out_path}")

print("\n✅ All adaptive augmentations completed successfully!")
