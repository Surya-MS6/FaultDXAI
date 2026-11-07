# 🧠 Explainable AI – With Attention Mechanism

## 📌 Overview
This folder contains the Explainable AI (XAI) implementation for the fault diagnosis model integrated with an attention mechanism. Grad-CAM is used to interpret the model's classification decisions and identify important frequency regions in vibration signals.

## ⚙️ Grad-CAM Implementation

The implementation uses Grad-CAM to analyse the model's predictions and highlight the frequency regions that contribute to fault classification.

### Key Features
- Grad-CAM-based model interpretation.
- Analysis of important frequency regions.
- Class-wise visualization of model predictions.
- Attention-based feature analysis.
- Visualization of important frequency bands.

## 📊 XAI Visualizations

The generated visualizations are used to analyse:
- Grad-CAM comparisons.
- Class-wise Grad-CAM outputs.
- Important frequency regions.
- Attention-related feature contributions.
- Overall XAI analysis.

## 📁 Files

- Python implementation – Grad-CAM analysis for the attention-based model.
- Trained attention model – Model used for XAI analysis.
- Visualization files – Generated Grad-CAM and attention analysis results.

## 🔄 Workflow

1. Load the trained fault classification model with attention.
2. Load frequency-domain vibration signals.
3. Apply Grad-CAM to analyse model predictions.
4. Identify important frequency regions.
5. Generate class-wise visualizations.
6. Interpret the model's classification behaviour.

## 🎯 Purpose

The main purpose of this implementation is to improve the interpretability of the attention-based fault diagnosis model by analysing the important frequency regions that influence its classification decisions.
