# 🧠 Trained Model – Without Attention Mechanism (Without AM)

This folder contains the trained 1D CNN model developed without an attention mechanism. The model is trained using synthetic vibration signals in the frequency domain and is designed to classify four rotating machinery conditions.

## ⚙️ Model Details

- **Model Architecture:** 1D Convolutional Neural Network (CNN)
- **Attention Mechanism:** Not used
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
| `frequency_domain_fault_classifier.keras` | Trained 1D CNN model saved in Keras format. |

## 🔄 Usage

The saved model can be loaded using TensorFlow/Keras and used for fault classification and further evaluation on real vibration signals.

This model is used by the corresponding analysis script in:

`Source_Code/Model_Training/Without_Attention_Mechanism/`

## 📝 Note

This is the baseline model without an attention mechanism. The attention-based model is maintained separately for comparison.
