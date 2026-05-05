<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/License-MIT-000000?style=for-the-badge" alt="License">
</p>

<h1 align="center">🔍 High-Density Object Segmentation System</h1>

<p align="center">
  <strong>A Multi-Phase Hybrid Neuro-Symbolic Framework for Retail Analytics</strong><br/>
  <em>Optimized for Dense Scene Inference and Occlusion Robustness</em>
</p>

<p align="center">
  <a href="#-core-features">Key Features</a> •
  <a href="#-technical-architecture">Architecture</a> •
  <a href="#-performance-benchmarks">Benchmarks</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-academic-report">Report</a>
</p>

---

## 🎯 Project Overview

This repository hosts a state-of-the-art framework for **High-Density Object Segmentation**, specifically engineered for retail environments where occlusion and object density (up to 700+ items per frame) render standard detectors ineffective. 

Our approach integrates **YOLACT-based Deep Learning** with a **Neuro-Symbolic Spatial Reasoning Engine**, achieving superior localization and confidence recalibration.

### 🌟 Key Features
- 🚀 **Real-time Inference**: Optimized ONNX pipeline delivering ~8.3 FPS on standard CPUs.
- 🧠 **Hybrid Reasoning**: Integrated GMM-KDE spatial prior for shelf-aware detection.
- 🛡️ **Soft-NMS Integration**: Gaussian-decayed suppression to preserve overlapping detections.
- 📊 **Comprehensive Analytics**: Built-in EDA and ablation study frameworks.
- 🌐 **Web Integration**: Full-stack Next.js dashboard for model comparison and custom uploads.

---

## 🏛️ Technical Architecture

### 1. Detection Backbone (Phase 2)
The system utilizes a **MobileNetV3-Large** backbone coupled with a **Feature Pyramid Network (FPN)** and **CBAM (Convolutional Block Attention Module)** for efficient feature extraction and multi-scale detection.

### 2. Neuro-Symbolic Fusion (Phase 3)
A learnable **Gated Spatial Attention** mechanism recalibrates detection confidence based on environmental priors (Shelf Row Structure via GMM & Point Density via KDE).

---

## 📈 Performance Benchmarks

### Detection Accuracy (SKU-110K Dataset)
| Model Variant | Val Loss | mAP@0.5 | Efficiency |
|:--- |:---:|:---:|:---:|
| **Classical (HOG+SVM)** | — | 2.10% | 0.2 FPS |
| **YOLACT (Baseline)** | 3.145 | 2.73% | 3.1 FPS |
| **Hybrid (Proposed)** | **3.097** | **3.03%** | 3.0 FPS |

---

## 🚀 Getting Started

### 📦 Installation

```bash
# Clone and enter directory
git clone https://github.com/Harshkesharwani789/AmlDlProject.git
cd AmlDlProject

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 🛠️ Execution Pipeline

1. **Data Prep**: `bash scripts/download_data.sh`
2. **Exploration**: `python scripts/run_eda.py`
3. **Training**: `python scripts/train.py --config configs/default.yaml`
4. **Evaluation**: `python scripts/evaluate.py`
5. **Web Demo**: `cd web && npm run dev`

---

## 📂 Repository Structure

- `src/models/`: Core neural architectures and spatial reasoning logic.
- `scripts/`: Production-ready entry points for training and evaluation.
- `web/`: Next.js frontend for interactive model demonstration.
- `report/`: Academic documentation and project findings in LaTeX.
- `configs/`: Hyperparameter and environment specifications.

---

## 👥 Contributors

- **Siddhartha Shukla** - [GitHub](https://github.com/SiddharthaShukla8)
- **Harsh Gupta** - [GitHub](https://github.com/Harshkesharwani789)

---

## 📜 License & Citation

Distributed under the MIT License. See `LICENSE` for more information.

```bibtex
@article{shukla2026high,
  title={High-Density Object Segmentation in Retail Scenes},
  author={Shukla, Siddhartha and Gupta, Harsh},
  journal={B.Tech Final Year Project - CS},
  year={2026}
}
```

---
<p align="center">
  Developed at the Department of Computer Science<br/>
  <strong>Applied Machine Learning & Deep Learning Project 2026</strong>
</p>
