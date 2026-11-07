# 🧠 Model Architecture

Deep learning architecture for vibration-based fault classification using 1D CNN, attention-based feature weighting and synthetic-to-real domain adaptation.

## ⚙️ Network Architecture

| Component | Function |
|---|---|
| Input Layer | Receives preprocessed multi-channel vibration signals |
| Convolutional Layers | Extracts fault-related features |
| Batch Normalization | Stabilizes network training |
| Max Pooling | Reduces feature dimensions |
| Attention Mechanism | Assigns importance to relevant frequency features |
| Dense Layer | Learns high-level representations |
| Softmax Layer | Predicts four operating conditions |

## 🎯 Attention Mechanism

The attention layer assigns adaptive weights to frequency-domain features, allowing the network to emphasize diagnostically relevant spectral components.

## 🔄 Training Strategy

- Synthetic vibration data used for initial model training.
- Baseline 1D CNN evaluated without attention.
- Attention-based 1D CNN evaluated against the baseline.
- Limited real vibration samples incorporated through domain adaptation.
- Performance evaluated using unseen real vibration signals.

## 📊 Classification Output

Four output classes:

- Normal
- Misalignment
- Looseness
- Unbalance

## 📈 Model Evaluation

Model performance is assessed using:

- Classification Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
