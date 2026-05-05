# 🔍 High-Density Object Segmentation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/Docker-Supported-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

## 📖 Project Overview
Detecting and segmenting objects in **high-density retail environments** (e.g., supermarket shelves) presents unique computer vision challenges: severe inter-object occlusion, homogeneous object appearances, and extreme variation in counts (ranging from 10 to 700+ per scene).

This project implements a **Three-Phase Hybrid Neuro-Symbolic Framework** that progressively evolves from classical feature engineering to a sophisticated deep-learning pipeline augmented by probabilistic spatial reasoning.

---

## 🏗️ Technical Architecture

### 1. Phase 1: Classical Foundation (HOG + SVM)
Established a baseline using **Histogram of Oriented Gradients (HOG)** descriptors and a linear **SVM** sliding-window classifier. This phase quantified the limitations of manual feature engineering in overlapping dense scenes, motivating the transition to deep feature learning.

### 2. Phase 2: Deep Instance Segmentation (YOLACT)
Developed a real-time segmentation core based on the **YOLACT** architecture:
- **Backbone**: MobileNetV3-Large for optimal parameter efficiency (~5.4M params).
- **Neck**: Feature Pyramid Network (FPN) with integrated **CBAM (Convolutional Block Attention Module)** for density-aware feature refinement.
- **Heads**: Parallel branches for mask prototypes (ProtoNet) and classification/box/mask-coefficient predictions.
- **Inference**: Optimized via **Soft-NMS** (Gaussian decay), which is critical for preserving detections in high-density clusters.

### 3. Phase 3: Hybrid Neuro-Symbolic Fusion
Augmented the neural pipeline with a **Probabilistic Spatial Engine**:
- **Gaussian Mixture Models (GMM)**: Detects underlying shelf-row structures via BIC-optimized component selection.
- **Kernel Density Estimation (KDE)**: Learns a 2D spatial prior indicating "expected" object hotspots.
- **Gated Attention**: A learnable fusion layer that dynamically recalibrates neural confidence scores based on spatial feasibility.

---

## 📊 Experimental Results & Metrics

### Training Performance
| Metric | Baseline (YOLACT) | Hybrid Model |
| :--- | :---: | :---: |
| **Best Val Loss** | 3.145 | **3.097** |
| **Training Duration** | ~137 min (H100) | +35 min (Fusion) |
| **Model Size** | 38.2 MB | 38.3 MB |

### Deployment Efficiency (Inference Latency)
| Environment | Device | Latency | FPS |
| :--- | :---: | :---: | :---: |
| **PyTorch (Native)** | GPU/MPS | 318 ms | 3.1 |
| **ONNX Optimized** | **CPU** | **120 ms** | **8.3** |

> [!IMPORTANT]
> **Ablation Insight**: The switch from Soft-NMS to standard NMS results in a **98.4% mAP degradation**. In dense scenes, standard suppression is "catastrophic" because valid objects inherently overlap significantly.

---

## ⚡ Quick Start & Deployment

### Environment Setup
```bash
# Clone and install
git clone https://github.com/Harshkesharwani789/AmlDlProject.git
cd AmlDlProject
pip install -r requirements.txt && pip install -e .
```

### Data Pipeline
```bash
# Download SKU-110K dataset assets
bash scripts/download_data.sh
```

### Execution
```bash
# Run full training pipeline
python scripts/train.py --config configs/default.yaml
python scripts/train_hybrid.py --config configs/hybrid.yaml

# Evaluate and Benchmark
python scripts/evaluate.py --benchmark
```

### Interactive Web Demo
The system includes a production-ready **Next.js** dashboard for real-time inference visualization.
```bash
cd web
npm install && npm run dev
# Access at http://localhost:3000
```

---

## 📂 Repository Structure
- `src/` - Modular implementation of Backbones, FPNs, and Spatial Engines.
- `configs/` - YAML definitions for all model variants and hyperparameters.
- `scripts/` - Automated pipelines for training, evaluation, and ONNX export.
- `web/` - Full-stack React/Next.js dashboard with API integration.
- `report/` - Formal technical report and LaTeX documentation.

---

## 🛡️ Security & Compliance
This project is designed with privacy-first principles:
- **No Personal Data**: All annotations are strictly object-centric bounding boxes.
- **Anonymized Assets**: Retail images do not contain identifiable person or sensitive metadata.
- **Standard Licensing**: Distributed under the MIT Open Source License.

---

## 📜 Authors & Acknowledgments
**Siddhartha Shukla** & **Harsh Gupta**  
*Advanced Machine Learning & Deep Learning Project*  
*Department of Computer Science*

Built with ❤️ using PyTorch, ONNX, and Next.js.
