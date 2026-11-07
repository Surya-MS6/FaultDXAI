# 🧠 Explainable AI – Without Attention Mechanism

## 📌 Overview
This folder contains the Explainable AI (XAI) implementation for the baseline fault diagnosis model without an attention mechanism. Grad-CAM is used to understand how the trained model identifies and classifies different mechanical faults from frequency-domain vibration signals.

## ⚙️ Grad-CAM Implementation

The implementation uses Grad-CAM to highlight important frequency regions that influence the model's fault classification decisions.

### Key Features
- Grad-CAM-based model interpretation.
- Analysis of important frequency regions.
- Class-wise visualization of model predictions.
- Comparison of Grad-CAM outputs.
- Identification of important frequency bands.

## 📊 XAI Visualizations

The generated visualizations include:

- `xai_gradcam_comparison` – Grad-CAM comparison.
- `xai_gradcam_per_class` – Class-wise Grad-CAM visualization.
- `xai_gradcam_vs_conv` – Comparison of Grad-CAM and convolution outputs.
- `xai_important_bands` – Important frequency band analysis.
- `xai_summary` – Overall XAI visualization summary.

## 📁 Files

- `XAI without AM.py` – Grad-CAM implementation and analysis.
- `frequency_domain_fault_classifier.keras` – Trained baseline model used for XAI analysis.
- Visualization files – Generated Grad-CAM analysis results.

## 🔄 Workflow

1. Load the trained baseline fault classification model.
2. Load frequency-domain vibration signals.
3. Apply Grad-CAM to analyse model predictions.
4. Identify important frequency regions.
5. Generate visualizations for different fault classes.
6. Analyse and interpret the model's classification behaviour.

## 🎯 Purpose

The main purpose of this implementation is to improve the interpretability of the baseline fault diagnosis model by identifying the frequency regions that contribute to its classification decisions.
