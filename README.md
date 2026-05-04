<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🔍 High-Density Object Segmentation</h1>

<p align="center">
  <strong>A Three-Phase Hybrid Neuro-Symbolic Framework for Dense Retail Scene Analysis</strong>
</p>

<p align="center">
  <em>B.Tech Final Year Project — Applied Machine Learning & Deep Learning</em><br/>
  <strong>Siddhartha Shukla</strong> · <strong>Harsh Gupta</strong><br/>
  Department of Computer Science
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-web-demo">Web Demo</a> •
  <a href="#-deployment">Deployment</a>
</p>

---

## 🎯 Overview

Detecting and segmenting objects in **high-density retail environments** presents significant challenges due to severe inter-object occlusion, near-identical appearance, and extreme variation in object counts (10–700+ per image). This project implements a **three-phase hybrid framework** that progressively builds from classical machine learning to a neuro-symbolic system.

> 📊 **Dataset:** [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) — 11,762 images · 1.73M annotations · 147 avg objects/image

### 🏆 Phase Summary

| Phase | Approach | Method | Key Result |
|:---:|:---:|:---:|:---:|
| **Phase 1** | Classical ML | HOG + SVM + Sliding Window | 86.4% precision, 2.1% recall |
| **Phase 2** | Deep Learning | YOLACT + MobileNetV3 + Soft-NMS | Best val loss: **3.145**, 8.3 FPS |
| **Phase 3** | Hybrid Fusion | YOLACT + GMM + KDE Spatial Reasoning | Best val loss: **3.097**, 98.4% mAP drop w/o Soft-NMS |

> 💡 **Key Finding:** Soft-NMS is the single most critical component — replacing it with Hard-NMS causes a **98.4% relative mAP drop**.

---

## 🏗️ Architecture

### Phase 2: YOLACT (Core Detector)

```
Input Image (3 × 550 × 550)
         │
  MobileNetV3-Large (ImageNet pretrained, 5.4M params)
    │           │           │
  C3 (40ch)  C4 (112ch)  C5 (960ch)
    │           │           │
  Feature Pyramid Network (256ch, 3 levels) + CBAM Attention
    │     │     │
   P3    P4    P5
    │     │     │
    └─────┴─────┘
    │             │
 ProtoNet    Prediction Head (shared)
 (32 masks)   cls │ box │ mask coeffs
    │             │
  Assembly: masks = σ(proto @ coeffsᵀ)
                │
         Soft-NMS (Gaussian, σ=0.5)
                │
         Final Detections
```

### Phase 3: Hybrid Spatial Fusion

```
YOLACT Detections
       │
  ┌────┴────┐
  │         │
  ▼         ▼
 GMM       KDE
(7 row    (5K point
 comps.)   density)
  │         │
  └────┬────┘
       │
  8-dim Spatial Feature Vector
       │
  ┌────┴────┐
  │         │
Gated     Confidence
Spatial   Recalibrator
Attention (17.8K params)
(g=0.408)
  │         │
  └────┬────┘
       │
  Refined Detections
```

### 📐 Parameter Breakdown

| Component | Parameters | % of Total |
|:---------:|:----------:|:----------:|
| MobileNetV3-Large (backbone) | ~5.4M | 54% |
| FPN + CBAM Attention | ~3.3M | 33% |
| ProtoNet (32 prototypes) | ~1.0M | 10% |
| Prediction Head | ~0.3M | 3% |
| **YOLACT Total** | **~10.0M** | — |
| Spatial Attention + Recalibrator | ~37K | **0.4%** |

---

## 📊 Results

### Training Convergence (20 Epochs, H100 GPU)

| Epoch | Train Loss | Val Loss | Classification | Box | Mask |
|:-----:|:----------:|:--------:|:--------------:|:---:|:----:|
| 1 | 8.620 | — | .283 | 3.479 | .509 |
| 10 | 4.216 | 3.374 | .081 | 1.137 | .259 |
| 20 | 3.808 | **3.145** | .075 | 1.040 | .247 |

> ✅ **No overfitting:** Validation loss decreases monotonically · **73% classification loss reduction**

### Ablation Study — 8 Variants on 588 Validation Images

| Variant | mAP@0.5 | Δ mAP | Impact |
|:-------:|:-------:|:-----:|:------:|
| Full Hybrid | 2.73% | baseline | — |
| DL Only (no spatial) | 3.03% | +0.30 | 🟡 Marginal |
| **Hard NMS** | **0.04%** | **−2.68** | 🔴 **Catastrophic** |
| No CBAM | 2.73% | 0.00 | 🟢 Neutral |

### 🚀 Deployment Performance

| Backend | Device | Latency | FPS | Model Size |
|:-------:|:------:|:-------:|:---:|:----------:|
| PyTorch FP32 | MPS (GPU) | 318 ms | 3.1 | 38.2 MB |
| **ONNX FP32** | **CPU** | **120 ms** | **8.3** | **~38 MB** |

---

## ⚡ Quick Start

### Prerequisites

- Python ≥ 3.8
- NVIDIA GPU with CUDA (recommended) or Apple Silicon Mac (MPS) or CPU
- ~10 GB disk space (dataset + checkpoints)
- Node.js ≥ 18 (for web demo only)

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/Harshkesharwani789/AmlDlProject.git
cd AmlDlProject

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

Or use **Docker**:

```bash
docker build -t sku110k-detector .
docker run -it --gpus all sku110k-detector bash
```

### Step 2: Download Dataset

```bash
bash scripts/download_data.sh
```

> Downloads and extracts the SKU-110K dataset (~2.4 GB) to `data/`

### Step 3: Run Full Pipeline

```bash
# Phase 1: Exploratory Data Analysis
python scripts/run_eda.py

# Phase 1: Classical ML Baseline (HOG+SVM)
python scripts/run_baseline.py

# Phase 2: Train YOLACT (20 epochs)
python scripts/train.py --config configs/default.yaml

# Phase 2: Evaluate + generate visualizations
python scripts/evaluate.py

# Phase 2: Export to ONNX
python scripts/export.py

# Phase 3: Train Hybrid Detector
python scripts/train_hybrid.py --config configs/hybrid.yaml

# Phase 3: Run Ablation Study (8 variants)
python scripts/run_ablation.py --config configs/hybrid.yaml
```

### ⏱️ Expected Training Times (NVIDIA H100)

| Stage | Duration |
|:-----:|:--------:|
| Phase 2 — YOLACT (20 epochs) | ~137 min |
| Phase 3 — Fit GMM+KDE | ~3 min |
| Phase 3 — Fusion (7 epochs) | ~3 hours |
| Ablation Study | ~35 min |

---

## 🌐 Web Demo

An interactive **Next.js** web application for real-time model comparison:

```bash
cd web
npm install
npm run dev
# Open http://localhost:3000
```

**Features:**
- 🔄 Switch between **3 models**: YOLACT, Hybrid, and HOG+SVM
- 📸 Upload custom retail shelf images
- 🎚️ Real-time confidence threshold adjustment
- 🔍 Fullscreen zoom & pan viewer
- 📥 Download annotated results

---

## 📁 Project Structure

```
AmlDlProject/
│
├── configs/                    # Training configuration files
│   ├── default.yaml            # Phase 2 training config
│   └── hybrid.yaml             # Phase 3 hybrid config
│
├── src/                        # Core source code
│   ├── models/                 # Neural network architecture
│   │   ├── yolact.py           # YOLACT model assembly
│   │   ├── hybrid.py           # Hybrid detector (Phase 3)
│   │   ├── spatial_reasoning.py# GMM + KDE spatial engine
│   │   ├── backbone.py         # MobileNetV3-Large
│   │   ├── fpn.py              # Feature Pyramid Network + CBAM
│   │   └── detection.py        # Post-processing (Soft-NMS)
│   ├── training/               # Training pipelines
│   ├── evaluation/             # Metrics & ablation framework
│   ├── data/                   # Dataset loader & augmentations
│   └── utils/                  # Soft-NMS, helpers, visualization
│
├── scripts/                    # Entry-point scripts
│   ├── train.py                # Phase 2 training
│   ├── train_hybrid.py         # Phase 3 training
│   ├── evaluate.py             # COCO-style evaluation
│   ├── run_ablation.py         # Ablation study
│   └── export.py               # ONNX export + quantization
│
├── web/                        # Next.js web demo
│   └── src/app/
│       ├── page.tsx            # Landing page
│       ├── demo/page.tsx       # Interactive inference demo
│       └── api/                # Model inference endpoints
│
├── report/                     # IEEE-format LaTeX report
│   └── phase3.tex              # Full academic paper
│
├── results/                    # Generated outputs
│   ├── training/checkpoints/   # Model checkpoints (.pth)
│   ├── hybrid/                 # Hybrid model outputs
│   ├── ablation/               # Ablation results
│   └── deployment/             # ONNX models + benchmarks
│
├── Dockerfile                  # Reproducible environment
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── README.md                   # You are here
```

---

## 🔬 Technical Highlights

### Soft-NMS — The Critical Component

Standard NMS hard-suppresses all overlapping detections, causing missed detections in dense scenes. Soft-NMS applies **Gaussian decay**:

```
score_i = score_i × exp(−IoU(M, b_i)² / σ)
```

Our ablation proves this is the **single most critical component**: Hard-NMS causes **98.4% mAP drop** (2.73% → 0.04%).

### Spatial Reasoning Engine

| Component | Description |
|:---------:|:-----------:|
| **GMM** (7 components) | Detects shelf-row structure via BIC model selection |
| **KDE** (5,000 points) | Generates 2D density field as spatial prior |
| **Gated Attention** | Learnable gate (g=0.408) controls spatial influence |

### Device Support

| Backend | Training | Inference | AMP |
|:-------:|:--------:|:---------:|:---:|
| CUDA | ✅ | ✅ | ✅ |
| MPS (Apple Silicon) | ✅ | ✅ | ❌ |
| CPU | ✅ | ✅ | ❌ |

---

## 🗂️ Pre-trained Checkpoints

| File | Size | Description |
|:----:|:----:|:-----------:|
| `results/training/checkpoints/best_model.pth` | 76 MB | YOLACT (20 epochs, val_loss=3.145) |
| `results/hybrid/checkpoints/hybrid_best.pth` | 76 MB | Hybrid detector (val_loss=3.097) |
| `results/hybrid/spatial_models/spatial_engine.pkl` | 127 KB | Fitted GMM + KDE models |

---

## 📖 References

1. Goldman et al., "Precise Detection in Densely Packed Scenes," CVPR 2019
2. Bolya et al., "YOLACT: Real-time Instance Segmentation," ICCV 2019
3. Howard et al., "Searching for MobileNetV3," ICCV 2019
4. He et al., "Mask R-CNN," ICCV 2017
5. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
6. Bodla et al., "Soft-NMS — Improving Object Detection With One Line of Code," ICCV 2017
7. Lin et al., "Feature Pyramid Networks for Object Detection," CVPR 2017
8. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
9. Dalal and Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR 2005
10. Zhang et al., "mixup: Beyond Empirical Risk Minimization," ICLR 2018

---

## 📄 License

```
MIT License — Copyright (c) 2026 Siddhartha Shukla, Harsh Gupta
```

---

<p align="center">
  <strong>Built with ❤️ by Siddhartha Shukla & Harsh Gupta</strong><br/>
  <em>Department of Computer Science · Applied ML & Deep Learning</em>
</p>
 
