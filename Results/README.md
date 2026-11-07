# 📊 Experimental Results

This folder contains the experimental evaluation results of the FaultDXAI framework. It includes performance analysis, comparison plots and visualizations generated from the three model implementations.

## 📈 Model Configurations

The experimental results are organised into three categories:

| Model Configuration | Description |
|---|---|
| Without Attention Mechanism | Evaluation results of the baseline 1D CNN model. |
| With Attention Mechanism | Evaluation results of the 1D CNN model integrated with frequency-position attention. |
| Hybrid Model With Attention | Evaluation results obtained using varying amounts of real-world vibration data with the attention-based hybrid model. |

## 🔍 Evaluation Metrics

Model performance is analysed using the following metrics:

- Classification Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Per-Class Accuracy
- ROC Curves
- Prediction Confidence

## 📉 Result Visualizations

The repository includes different visualizations to analyse model performance, including:

- Confusion matrices and normalized confusion matrices.
- Training and validation performance plots.
- Per-class accuracy and classification metrics.
- ROC curves.
- Prediction confidence distributions.
- Model performance comparisons.
- Frequency-domain stability analysis.
- Performance summaries.

## 📁 Results Organisation

The experimental results are maintained in the following directories:

- `Without_Attention_Mechanism/` – Baseline model evaluation results.
- `With_Attention_Mechanism/` – Attention-based model evaluation results.
- `Hybrid_Model_Attention_Mechanism/` – Hybrid model evaluation results using varying amounts of real-world data.

## 🎯 Purpose

The purpose of this section is to evaluate and compare the performance of different FaultDXAI model configurations and understand the effect of attention mechanisms and varying real-world data availability on rotating machinery fault diagnosis.
