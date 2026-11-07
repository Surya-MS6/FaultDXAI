# 📊 Dataset – Rotating Machinery Fault Diagnosis

## 📌 Overview

FaultDXAI uses vibration signals from a rotating machinery dataset to develop and evaluate an intelligent fault diagnosis system.

The dataset contains vibration measurements collected under four operating conditions: Normal, Misalignment, Looseness and Unbalance. These signals are used for fault classification, synthetic signal generation, model training and performance evaluation.

## ⚙️ Dataset Source

The original dataset is publicly available on Mendeley Data.

**Dataset Name:** Mechanical faults in rotating machinery dataset (normal, unbalance, misalignment, looseness)

**Download Link:** https://data.mendeley.com/datasets/zx8pfhdtnb/3

**DOI:** 10.17632/zx8pfhdtnb.3

## 🔍 Fault Categories

The dataset contains four operating conditions:

- **Normal:** Vibration signals recorded under healthy operating conditions.
- **Misalignment:** Signals representing shaft or coupling misalignment.
- **Looseness:** Signals representing mechanical looseness.
- **Unbalance:** Signals representing rotor unbalance.

## 🏭 Experimental Setup

The vibration signals were collected using a rotating machinery test bench consisting of:

- Three-phase induction motor.
- Frequency inverter.
- Bearings and bearing housing.
- Two pulleys connected through a belt.
- Rotor disc assembly.
- Four accelerometers for vibration measurement.

The dataset contains 20 experimental tests, with five tests for each operating condition.

## 📈 Dataset Specifications

| Parameter | Description |
|---|---|
| Fault Classes | 4 |
| Experimental Tests | 20 |
| Tests per Class | 5 |
| Accelerometer Channels | 4 |
| Sampling Frequency | 25 kHz |
| Samples per Signal | 25,000 |
| File Format | NumPy (.npy) |
| Data per File | (4, 25000) |
| Signals per Accelerometer | 8,400 |

Each signal file contains four accelerometer channels, with 25,000 samples in each channel.

## 📁 Dataset Structure

The original dataset is organised into 20 experimental test folders. Each test contains vibration signal files corresponding to its operating condition.

The downloaded data can be organised into four fault categories for use in the FaultDXAI training and evaluation pipeline:

- Normal
- Misalignment
- Looseness
- Unbalance

## 🔄 Dataset Usage in FaultDXAI

The dataset is used for:

1. Analysing real-world vibration signals.
2. Developing and comparing synthetic vibration signals.
3. Applying signal augmentation techniques.
4. Training and evaluating deep learning fault classification models.
5. Studying model performance with different amounts of real-world data.
6. Evaluating model predictions using Explainable AI techniques.

## 📥 Download Instructions

1. Open the official Mendeley Data link provided above.
2. Download the dataset ZIP files.
3. Extract the downloaded files.
4. Organise the signals according to their fault categories.
5. Use the extracted vibration signals with the FaultDXAI source code.

**Note:** The original dataset files are not stored in this GitHub repository. Users can download them directly from Mendeley Data.

## 📚 Dataset Citation

Brito, L. C., Susto, G. A., Brito, J. N., & Duarte, M. A. V. (2023). *Mechanical faults in rotating machinery dataset (normal, unbalance, misalignment, looseness).* Mendeley Data, Version 3.

DOI: 10.17632/zx8pfhdtnb.3

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
