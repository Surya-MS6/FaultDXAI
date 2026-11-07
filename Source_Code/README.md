# 💻 Source Code

This folder contains the Python implementation of the FaultDXAI framework, including synthetic vibration signal generation, data augmentation and deep learning-based fault diagnosis.

## ⚙️ Implementation Modules

| Module | Description |
|---|---|
| 🧪 Synthetic Signal Generation | Generation of fault-specific synthetic vibration signals |
| 📊 Data Augmentation | Application of five signal augmentation techniques |
| 🧠 Model Training | Training and analysis of fault classification models |
| 🎯 Attention Mechanism | Integration of frequency-position attention into the classification model |
| 🔄 Hybrid Model | Training and evaluation using varying amounts of real-world vibration data |

## 🛠️ Tech Stack

- Python
- TensorFlow and Keras
- NumPy
- SciPy
- Matplotlib
- Scikit-learn

## 📂 Code Organisation

The source code is organised into the following directories:

### 1. Synthetic Generation
Contains scripts for generating synthetic vibration signals representing:
- Misalignment
- Looseness
- Unbalance

### 2. Data Augmentation
Contains the implementation of five signal augmentation techniques:
- Gaussian Noise
- Masking Noise
- Signal Translation
- Amplitude Shifting
- Time Stretching

### 3. Model Training
Contains three model implementations:

- **Without Attention Mechanism:** Baseline 1D CNN model training and analysis.
- **With Attention Mechanism:** 1D CNN integrated with frequency-position attention.
- **Hybrid Model:** Attention-based hybrid model training and analysis using varying amounts of real-world data.

## 🔄 Overall Workflow

1. Generate synthetic vibration signals.
2. Apply data augmentation techniques.
3. Prepare and process vibration signals for model training.
4. Train the fault classification models.
5. Evaluate model performance using real-world vibration data.
6. Analyse and compare the different model implementations.

## 🎯 Purpose

The main purpose of this source code is to implement the different stages of FaultDXAI, from synthetic signal generation to model training and evaluation, for interpretable rotating machinery fault diagnosis.
