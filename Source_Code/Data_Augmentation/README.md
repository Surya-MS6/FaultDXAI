# 📊 Data Augmentation

Five signal-level augmentation techniques implemented to improve the diversity and robustness of vibration signals used in rotating machinery fault diagnosis.

## ⚙️ Augmentation Techniques

| Technique | Implementation |
|:---|:---|
| Gaussian Noise | Adds random noise scaled to 5% of signal RMS. |
| Masking Noise | Randomly replaces 5–20% of signal samples with zeros. |
| Signal Translation | Shifts signals by up to 2% of their length. |
| Amplitude Shifting | Applies adaptive amplitude scaling based on signal RMS. |
| Time Stretching | Resamples signals using frequency-dependent stretching factors. |

## 📡 Signal Configuration

- Sampling Frequency: 25 kHz
- Signal Length: 25,000 samples
- Accelerometer Channels: 4
- Input Format: NumPy (`.npy`)
- Processing: Individual signal augmentation

## 🧪 Implementation

**`augmentation.py`**

The script applies all five augmentation techniques to the input vibration signals and saves the generated signals into separate output directories.

### Dependencies
- Python
- NumPy
- SciPy

### Output Organisation

```text
Augmented/
├── Gaussian_Noise/
├── Masking_Noise/
├── Signal_Translation/
├── Amplitude_Shift/
└── Time_Stretch/
```
