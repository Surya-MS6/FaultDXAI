# 🧠 Trained Model – With Attention Mechanism (With AM)

This folder contains the trained 1D CNN model developed with a frequency-position attention mechanism. The model is trained using synthetic vibration signals in the frequency domain and is designed to classify four rotating machinery conditions.

## ⚙️ Model Details

- **Model Architecture:** 1D Convolutional Neural Network (CNN)
- **Attention Mechanism:** Frequency-position attention
- **Input Domain:** Frequency domain
- **Frequency Range:** Up to 500 Hz
- **Sampling Frequency:** 25 kHz
- **Input Channels:** 4 accelerometer channels
- **Normalization:** Z-score normalization
- **Fault Classes:** Normal, Misalignment, Looseness, and Unbalance
- **Training Data:** Synthetic vibration signals

## 📦 Model File

| File | Description |
|---|---|
| `frequency_domain_fault_classifier.keras` | Trained 1D CNN model with frequency-position attention, saved in Keras format. |

## 🔄 Usage

The saved model can be loaded using TensorFlow/Keras and used for fault classification and further evaluation on real vibration signals.

This model is used by the corresponding analysis script in:

`Source_Code/Model_Training/With_Attention_Mechanism/`

## 📝 Note

This model uses frequency-position attention and is maintained separately from the baseline model without attention for performance comparison.
