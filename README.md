<div align="center">

# ⚙️ FaultDXAI

### Intelligent & Interpretable Fault Diagnosis for Rotating Machinery

**Synthetic Intelligence · Attention-Based Learning · Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Development-yellow)](#)

</div>

---

## 🎯 Motivation

Rotating machinery plays a critical role in industrial operations. Unexpected mechanical faults can lead to equipment damage, production downtime and increased maintenance costs.

**Key challenges:**
- 📉 Limited availability of real-world faulty vibration data.
- 💰 High cost of data acquisition and expert labelling.
- 🧠 Limited interpretability of conventional deep learning models.
- 🔄 Difficulty in generalizing from synthetic signals to real-world conditions.

**FaultDXAI** addresses these challenges through synthetic signal generation, attention-guided classification, domain adaptation and explainable AI.

## 🚀 Key Contributions

- 🧪 **Synthetic Signal Generation** – Analytical waveform modelling using fault-specific frequencies and harmonics.
- 📊 **Signal Augmentation** – Five augmentation techniques for improved signal diversity.
- 🧠 **Attention-Based Learning** – Frequency-selective attention integrated with 1D CNN.
- 🔄 **Domain Adaptation** – Knowledge transfer from synthetic signals to real vibration data.
- 🔍 **Explainable Diagnosis** – Grad-CAM-based visualization of diagnostically relevant features.

## ⚙️ Faults Classified

| Fault Condition | Characteristic Signature |
|:---|:---|
| 🟢 Normal | Healthy operating condition |
| ⚖️ Unbalance | Dominant 1× rotational frequency |
| 🔧 Misalignment | 1×, 2× and 3× harmonics |
| ⚙️ Mechanical Looseness | Multiple harmonics and sub-harmonics |

## 🔬 Methodology

The FaultDXAI framework follows a multi-stage signal processing and deep learning pipeline.

**Diagnosis Pipeline**

```text
  Vibration Signal Acquisition
              │
              ▼
   Synthetic Signal Generation
              │
              ▼
      Signal Augmentation
              │
              ▼
   Frequency-Domain Processing
              │
              ▼
       1D CNN Feature
          Extraction
              │
              ▼
     Attention Mechanism
              │
              ▼
    Synthetic-to-Real
      Domain Adaptation
              │
              ▼
      Fault Classification
              │
              ▼
       Grad-CAM Analysis
```

**Signal Augmentation Techniques**

- Gaussian Noise Addition
- Masking Noise
- Signal Translation
- Amplitude Scaling
- Time Stretching

## 🛠️ Tech Stack

<div align="center">

| Technology | Application |
|:---|:---|
| 🐍 Python | Core programming |
| 🧠 TensorFlow / Keras | Deep learning |
| 🔢 NumPy | Numerical computation |
| 📡 SciPy | Signal processing |
| 📈 Matplotlib | Data visualization |
| 🔍 Grad-CAM | Model interpretability |
| 💻 Google Colab / PyCharm | Development environments |

</div>

## 📊 Performance & Results

Experimental evaluation demonstrates the contribution of attention mechanisms and synthetic-to-real domain adaptation.

| Model Configuration | Accuracy |
|:---|---:|
| Baseline 1D CNN | **84.32%** |
| Attention-Based 1D CNN | **88.89%** |
| Attention + 30 Real Samples | **91.06%** |
| Attention + 70 Real Samples | **92.31%** |

> 💡 **Key Finding:** 91.06% accuracy was achieved using only 30 real training samples. This was identified as the optimal trade-off between classification performance and real-data requirements.

## 📁 Repository Structure

```text
FaultDXAI/
│
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📂 Dataset/
├── 📂 Source_Code/
├── 📂 Model/
├── 📂 Results/
├── 📂 Explainable_AI/
```

---

<div align="center">

### ⚙️ FaultDXAI

**Intelligent Fault Diagnosis | Attention-Based Learning | Explainable AI**

*Building interpretable and data-efficient diagnostic intelligence for rotating machinery.*

</div>
