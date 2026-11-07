# 🧠 Model Training With Attention Mechanism (With AM)

This folder contains the implementation of a 1D CNN model trained with an attention mechanism. It includes the training script and the analysis script used to evaluate the model's performance.

## ⚙️ Model Training

The training script, `Frequency AM Model Training.py`, implements a 1D CNN-based fault classification model with a frequency-position attention mechanism using synthetic vibration signals.

### Key Features

- **Input Data:** Synthetic vibration signals from four fault classes.
- **Fault Classes:** Normal, Misalignment, Looseness, and Unbalance.
- **Frequency Transformation:** Fast Fourier Transform (FFT).
- **Frequency Range:** Up to 500 Hz.
- **Normalization:** Z-score normalization.
- **Input Channels:** Four accelerometer channels.
- **Model:** 1D Convolutional Neural Network (CNN) with Frequency Attention.
- **Training Split:** 90% training and 10% validation.
- **Early Stopping:** Used to monitor validation accuracy and prevent unnecessary training.

## 📊 Model Analysis

The analysis script, `Frequency AM Model Analysis.py`, evaluates the trained model using real-world vibration signals to examine how well it generalizes from synthetic training data to real data.

### Analysis Includes

- Overall classification accuracy.
- Classification report with precision, recall, and F1-score.
- Confusion matrix and normalized confusion matrix.
- Per-class accuracy.
- Prediction confidence analysis.
- ROC curves and AUC scores.
- Misclassification analysis.
- Performance visualization and summary plots.

## 📁 Files

| File | Description |
|---|---|
| `Frequency AM Model Training.py` | Trains the 1D CNN model with frequency-position attention using synthetic vibration signals. |
| `Frequency AM Model Analysis.py` | Evaluates the trained model using real vibration signals. |

## 🔄 Workflow

1. Load synthetic vibration signals.
2. Convert signals from the time domain to the frequency domain using FFT.
3. Apply Z-score normalization and retain frequencies up to 500 Hz.
4. Train the 1D CNN model with frequency-position attention.
5. Load the trained model for evaluation.
6. Test using real vibration signals.
7. Generate performance metrics and analysis plots.

**Note:** This implementation uses a frequency-position attention mechanism. The baseline model without attention is maintained separately.
