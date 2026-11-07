# 🔍 Explainable AI (XAI)

This folder contains the Explainable AI implementations developed to interpret the predictions of FaultDXAI models. Grad-CAM is used to identify important learned features and frequency regions that influence rotating machinery fault classification.

## 🧠 Grad-CAM Analysis

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to analyse the internal feature representations of trained deep learning models and highlight the regions contributing to their classification decisions.

The framework includes two XAI implementations:

- **Without Attention Mechanism:** Grad-CAM analysis of the baseline 1D CNN model.
- **With Attention Mechanism:** Grad-CAM analysis of the attention-based fault classification model.

## ⚙️ Interpretation Workflow

1. Load the trained fault classification model.
2. Input frequency-domain vibration signals.
3. Extract relevant features through the model.
4. Apply gradient-based activation analysis.
5. Generate Grad-CAM visualizations.
6. Interpret the features influencing fault classification.

## 📊 Visualization Categories

| Analysis | Purpose |
|---|---|
| Grad-CAM Comparison | Compare model activation patterns |
| Class-wise Analysis | Examine feature responses for different fault classes |
| Channel-wise Analysis | Analyse feature responses across vibration channels |
| Important Frequency Regions | Identify frequency regions contributing to predictions |
| XAI Summary | Present the overall interpretability analysis |

## 📁 XAI Organisation

The implementations are organised into two directories:

- `Without_Attention_Mechanism/` – Grad-CAM implementation using the baseline model.
- `With_Attention_Mechanism/` – Grad-CAM implementation using the attention-based model.

Each directory contains its corresponding Python implementation, trained model or checkpoint, and generated visualization results.

## 🎯 Purpose

The main purpose of the Explainable AI component is to improve the transparency of FaultDXAI by understanding which learned features and frequency regions influence the model's fault classification decisions.
