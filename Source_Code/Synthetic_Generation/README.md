# 🧪 Synthetic Signal Generation

Python implementation of analytical vibration signal generation for rotating machinery fault conditions.

## ⚙️ Selected Generation Methods

| Fault Condition | Python Script |
|---|---|
| Looseness | synthetic_generation_looseness.py |
| Misalignment | synthetic_generation_misalignment.py |
| Unbalance | synthetic_generation_unbalance.py |

## 📡 Signal Configuration

- Sampling Frequency: 25 kHz
- Signal Length: 25,000 samples
- Accelerometer Channels: 4
- Signal Generation: Fault-specific harmonic modelling

## 📊 Signal Analysis

Generated vibration signals are evaluated against real vibration data using frequency-domain comparison techniques.
