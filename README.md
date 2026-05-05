# 🔍 High-Density Object Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js">
</p>

A state-of-the-art **Hybrid Neuro-Symbolic Framework** designed for detecting and segmenting objects in extremely dense retail environments. This project addresses the challenges of severe occlusion and near-identical appearances in high-density scenes (100+ objects per image).

---

## 🌟 Key Features
- **🚀 Real-time Performance**: Optimized ONNX inference (8.3 FPS on CPU).
- **🧠 Hybrid Intelligence**: Combines Deep Learning (YOLACT) with Probabilistic Graphical Models (GMM/KDE).
- **🔬 Advanced Refinement**: Implements **Soft-NMS** and **CBAM Attention** for superior mask precision.
- **🌐 Web Dashboard**: Interactive Next.js interface for model visualization and real-time testing.

---

## 🏗️ Methodology: The Three-Phase Evolution
The project follows a structured progression from classical methods to a sophisticated hybrid system:

| Phase | Paradigm | Methodology | Primary Goal |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Classical | HOG + SVM + Sliding Window | Establishing spatial feature baselines. |
| **Phase 2** | Deep Learning | YOLACT + MobileNetV3 + Soft-NMS | Real-time instance segmentation core. |
| **Phase 3** | Hybrid | Neuro-Symbolic Fusion | Refining detections via spatial priors. |

> [!TIP]
> **Soft-NMS** is critical: Switching to standard Hard-NMS results in a catastrophic **98.4% drop in mAP** due to dense object overlapping.

---

## 📈 Performance at a Glance
| Model Configuration | Val Loss | Inference (CPU) | Model Size |
| :--- | :--- :| :--- :| :--- |
| **YOLACT (Baseline)** | 3.145 | 120ms | 38.2 MB |
| **Hybrid (Refined)** | **3.097** | 128ms | 38.3 MB |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Harshkesharwani789/AmlDlProject.git
cd AmlDlProject
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Full Pipeline
```bash
# Initialize data
bash scripts/download_data.sh

# Train & Evaluate
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py
```

### 3. Launch Web Demo
```bash
cd web
npm install && npm run dev
```

---

## 📂 Project Structure
- `src/models/` - Core architectures (YOLACT, Hybrid, Spatial Engines).
- `scripts/` - Entry points for training, evaluation, and export.
- `web/` - Full-stack Next.js web application.
- `report/` - Comprehensive technical documentation and LaTeX sources.

---

## 📜 License & Authors
Distributed under the **MIT License**.

**Siddhartha Shukla** & **Harsh Gupta**  
*Applied Machine Learning & Deep Learning Project*
