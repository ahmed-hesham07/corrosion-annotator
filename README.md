# Corrosion Segmentation and Classification Projects

This repository contains three deep learning projects for detecting, segmenting, and classifying corrosion in industrial assets. The models leverage state-of-the-art computer vision architectures to support predictive maintenance, reduce manual inspection time, and enhance structural safety.

---

## Project Structure

```
C:\Users\Ahmed Hesham/
└── corrosion-annotator/
    ├── .gitignore
    ├── Datasets/
    │   ├── Datasets Extractor.exe
    │   ├── Datasets.part02.rar
    │   ├── Datasets.part03.rar
    │   ├── Datasets.part04.rar
    │   ├── Datasets.part05.rar
    │   ├── Datasets.part06.rar
    │   ├── Datasets.part07.rar
    │   ├── Datasets.part08.rar
    │   ├── Datasets.part09.rar
    │   ├── Datasets.part10.rar
    │   ├── Datasets.part11.rar
    │   ├── Datasets.part12.rar
    │   ├── Datasets.part13.rar
    │   ├── Datasets.part14.rar
    │   ├── Datasets.part15.rar
    │   ├── Datasets.part16.rar
    │   └── Datasets.part17.rar
    ├── main.py
    ├── README.md
    ├── requirements.txt
    ├── src/
    │   └── __pycache__/
    │       ├── __init__.cpython-310.pyc
    │       └── corrosion_annotation.cpython-310.pyc
    └── train_model.py
```

---

## Projects Overview

### 1. **Corrosion-Annotator**
A segmentation model designed to localize corroded areas on metallic surfaces using deep learning. Outputs include:
- Original image
- Segmentation mask
- Overlay image

**Technologies**: TensorFlow, OpenCV  
**Output**: Pixel-level corrosion mask  
**Limitation**: No severity classification

---

### 2. **SpotRust (HRNet + Bayesian Uncertainty)**
High-resolution corrosion segmentation using HRNet with Monte Carlo dropout for uncertainty estimation.

**Outputs**:
- Binary segmentation map
- Epistemic uncertainty map
- Aleatoric uncertainty map

**Technologies**: PyTorch, HRNet, Bayesian Inference  
**Strength**: Fine-grained segmentation with uncertainty estimation  
**Note**: Post-processing is needed for real-time deployment

---

### 3. **Corrosion Condition State Classification (DeepLabV3+)**
Semantic segmentation model that classifies corrosion severity into four classes:
- Good
- Fair
- Poor
- Severe

**Model**: DeepLabV3+  
**Output**: Condition severity classification  
**F1-score**: 86.67%  
**Use case**: Predictive maintenance for bridges and industrial assets

---

## Dataset Setup

To extract the datasets for the three projects, **you must run the dataset extractor tool**:

```bash
Run: Datasets/Datasets Extractor.exe
```

This will unpack the required high-resolution annotated corrosion images into the proper directories for training and evaluation.

---

## System Requirements

- **GPU**: NVIDIA GTX 1080 or better (8 GB VRAM minimum)
- **RAM**: 16 GB or more
- **Python**: 3.8+
- **Frameworks**: TensorFlow 2.x, PyTorch, OpenCV, NumPy, Matplotlib
- **Additional**: CUDA Toolkit, cuDNN (for GPU acceleration)