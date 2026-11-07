<div align="center">

# ⚙️ FaultDXAI

### Intelligent & Interpretable Fault Diagnosis for Rotating Machinery

**Synthetic Signal Generation · Attention-Based Learning · Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Development-yellow)](#)

</div>

---

## 🎯 Project Overview

Rotating machinery plays an important role in industrial operations. Mechanical faults can cause equipment damage, unexpected downtime and increased maintenance costs.

However, developing reliable fault diagnosis systems involves several challenges:

- Limited availability of real-world faulty vibration data.
- High cost of data acquisition and expert labelling.
- Difficulty in generalizing models trained using limited real-world data.
- Limited interpretability of conventional deep learning models.

**FaultDXAI** is an intelligent and interpretable fault diagnosis framework developed to address these challenges through synthetic vibration signal generation, data augmentation, deep learning, attention mechanisms and Explainable AI.

## 🚀 Key Contributions

- 🧪 **Synthetic Signal Generation** – Generation of fault-specific vibration signals using analytical frequency components and harmonics.
- 📊 **Signal Augmentation** – Application of five augmentation techniques to increase signal diversity.
- 🧠 **Deep Learning Classification** – Development of 1D CNN-based fault classification models.
- 🎯 **Attention-Based Learning** – Integration of frequency-position attention into the classification architecture.
- 🔄 **Hybrid Model Training** – Evaluation using varying amounts of real-world vibration data.
- 🔍 **Explainable AI** – Grad-CAM-based visualization to understand features influencing model predictions.

## ⚙️ Faults Classified

FaultDXAI focuses on four rotating machinery operating conditions:

| Fault Condition | Description |
|:---|:---|
| 🟢 Normal | Healthy operating condition |
| ⚖️ Unbalance | Rotor mass distribution imbalance |
| 🔧 Misalignment | Shaft or coupling misalignment |
| ⚙️ Looseness | Mechanical looseness condition |

## 🔬 Dataset Specifications

The framework uses a publicly available rotating machinery vibration dataset.

| Parameter | Specification |
|:---|:---|
| Fault Classes | 4 |
| Sampling Frequency | 25 kHz |
| Samples per Signal | 25,000 |
| Accelerometer Channels | 4 |
| File Format | NumPy (.npy) |
| Signal Array Shape | (4, 25000) |

**Dataset Source:** [Mendeley Data – Mechanical Faults in Rotating Machinery](https://data.mendeley.com/datasets/zx8pfhdtnb/3)

The dataset documentation and download instructions are available in the `Dataset/` directory.

## 🔬 Methodology

The FaultDXAI framework follows a sequence of signal generation, preprocessing, model development and interpretation stages.

### Diagnosis Workflow

```text
       Vibration Dataset
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
       1D CNN Architecture
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
    Without AM      With AM
        │             │
        │             ▼
        │       Attention Layer
        │             │
        └──────┬──────┘
               │
               ▼
       Hybrid Model Training
       with Varying Real Data
               │
               ▼
       Fault Classification
               │
               ▼
        Grad-CAM Analysis
```

### Signal Augmentation Techniques

Five augmentation methods are implemented:

1. Gaussian Noise Addition
2. Masking Noise
3. Signal Translation
4. Amplitude Shifting
5. Time Stretching

### Model Development

Three model implementations are included:

- **Without Attention Mechanism:** Baseline 1D CNN for fault classification.
- **With Attention Mechanism:** 1D CNN integrated with frequency-position attention.
- **Hybrid Model With Attention:** Attention-based hybrid model evaluated using varying amounts of real-world vibration data.

## 🛠️ Technology Stack

| Technology | Application |
|:---|:---|
| Python | Core programming |
| TensorFlow / Keras | Deep learning |
| NumPy | Numerical computation |
| SciPy | Signal processing |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| Grad-CAM | Model interpretability |
| PyCharm / Google Colab | Development environments |

## 📊 Experimental Evaluation

FaultDXAI includes three experimental evaluation categories:

| Model Configuration | Evaluation |
|:---|:---|
| Without Attention Mechanism | Baseline model performance |
| With Attention Mechanism | Performance of frequency-position attention-based model |
| Hybrid Model With Attention | Performance across varying real-data levels |

Evaluation includes classification accuracy, precision, recall, F1-score, confusion matrices, ROC curves, class-wise performance and prediction confidence.

Detailed experimental plots and analysis are available in the `Results/` directory.

## 🔍 Explainable AI

Grad-CAM is integrated to improve the interpretability of the trained fault classification models.

Two implementations are provided:

- **Without Attention Mechanism:** Interpretation of baseline model predictions.
- **With Attention Mechanism:** Interpretation of attention-based model predictions.

The generated visualizations help identify important learned features and frequency regions contributing to fault classification.

## 📁 Repository Structure

```text
FaultDXAI/
│
├── Dataset/
│   └── README.md
│
├── Source_Code/
│   ├── Synthetic_Generation/
│   ├── Data_Augmentation/
│   └── Model_Training/
│       ├── Without_Attention_Mechanism/
│       ├── With_Attention_Mechanism/
│       └── Hybrid_Model_With_Attention/
│
├── Model/
│   ├── Without_Attention_Mechanism/
│   ├── With_Attention_Mechanism/
│   └── Hybrid_Model_Attention_Mechanism/
│
├── Results/
│   ├── Without_Attention_Mechanism/
│   ├── With_Attention_Mechanism/
│   └── Hybrid_Model_Attention_Mechanism/
│
└── Explainable_AI/
    ├── Without_Attention_Mechanism/
    └── With_Attention_Mechanism/
```

## 📥 Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/Surya-MS6/FaultDXAI.git
   ```

2. Download the vibration dataset using the link provided in `Dataset/README.md`.

3. Install the required Python libraries used by the implementation.

4. Explore the source code, trained models and experimental results.

5. Run the respective training, analysis and Explainable AI implementations.

**Note:** Large trained model files are maintained using Git Large File Storage (Git LFS).

---

<div align="center">

### ⚙️ FaultDXAI

**Intelligent Fault Diagnosis | Attention-Based Learning | Explainable AI**

*Developing interpretable and data-efficient diagnostic intelligence for rotating machinery.*

</div>
