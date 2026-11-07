# 🧠 Trained Models

This folder contains the trained deep learning models developed as part of the FaultDXAI framework for rotating machinery fault diagnosis.

Three model implementations are included to study the effect of attention mechanisms and varying amounts of real-world vibration data on fault classification performance.

## ⚙️ Model Implementations

| Model | Description |
|---|---|
| 1. Without Attention Mechanism | Baseline 1D CNN model for vibration-based fault classification. |
| 2. With Attention Mechanism | 1D CNN integrated with a frequency-position attention mechanism. |
| 3. Hybrid Model With Attention | Attention-based hybrid model evaluated using varying amounts of real-world vibration data. |

## 🧠 Model Architecture

The models use deep learning techniques to extract relevant features from frequency-domain vibration signals.

The main components include:

- **Convolutional Layers:** Extract fault-related features.
- **Batch Normalization:** Stabilizes model training.
- **Max Pooling:** Reduces feature dimensions.
- **Attention Mechanism:** Highlights relevant frequency-position features in attention-based models.
- **Dense Layers:** Learn higher-level feature representations.
- **Softmax Layer:** Classifies the four operating conditions.

## 🎯 Classification Output

All three model implementations classify four operating conditions:

- Normal
- Misalignment
- Looseness
- Unbalance

## 🔄 Training Approaches

The repository contains three training approaches:

- **Baseline Training:** Model training without an attention mechanism.
- **Attention-Based Training:** Model training with a frequency-position attention mechanism.
- **Hybrid Training:** Attention-based model training and evaluation using different quantities of real-world vibration data.

## 📁 Model Organisation

The trained models are organised into the following directories:

- `Without_Attention_Mechanism/` – Baseline trained model.
- `With_Attention_Mechanism/` – Trained model with attention mechanism.
- `Hybrid_Model_Attention_Mechanism/` – Trained models and checkpoints for different real-data percentages.

## 📊 Model Evaluation

The trained models are evaluated using different performance measures, including:

- Classification Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Class-wise Performance

Detailed evaluation results are available in the `Results/` directory.

## 🎯 Purpose

The purpose of these model implementations is to investigate the effectiveness of attention mechanisms and hybrid training strategies for improving rotating machinery fault diagnosis using vibration signals.
