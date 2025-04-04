# Corrosion Segmentation and Classification Projects

This repository contains three deep learning projects for detecting, segmenting, and classifying corrosion in industrial assets. The models leverage state-of-the-art computer vision architectures to support predictive maintenance, reduce manual inspection time, and enhance structural safety.

---

## Project Structure

```
C:\Users\Ahmed Hesham/
└── corrosion-annotator/
    ├── .gitignore
    ├── Datasets/
    │   ├── Datasets.part01.exe
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
    │   ├── Datasets.part17.rar
    │   ├── Datasets.part18.rar
    │   ├── Datasets.part19.rar
    │   ├── Datasets.part20.rar
    │   ├── Datasets.part21.rar
    │   ├── Datasets.part22.rar
    │   ├── Datasets.part23.rar
    │   ├── Datasets.part24.rar
    │   ├── Datasets.part25.rar
    │   ├── Datasets.part26.rar
    │   ├── Datasets.part27.rar
    │   ├── Datasets.part28.rar
    │   ├── Datasets.part29.rar
    │   ├── Datasets.part30.rar
    │   ├── Datasets.part31.rar
    │   ├── Datasets.part32.rar
    │   ├── Datasets.part33.rar
    │   ├── Datasets.part34.rar
    │   ├── Datasets.part35.rar
    │   ├── Datasets.part36.rar
    │   ├── Datasets.part37.rar
    │   ├── Datasets.part38.rar
    │   ├── Datasets.part39.rar
    │   ├── Datasets.part40.rar
    │   ├── Datasets.part41.rar
    │   ├── Datasets.part42.rar
    │   ├── Datasets.part43.rar
    │   ├── Datasets.part44.rar
    │   ├── Datasets.part45.rar
    │   ├── Datasets.part46.rar
    │   ├── Datasets.part47.rar
    │   ├── Datasets.part48.rar
    │   ├── Datasets.part49.rar
    │   ├── Datasets.part50.rar
    │   ├── Datasets.part51.rar
    │   ├── Datasets.part52.rar
    │   ├── Datasets.part53.rar
    │   ├── Datasets.part54.rar
    │   ├── Datasets.part55.rar
    │   ├── Datasets.part56.rar
    │   ├── Datasets.part57.rar
    │   ├── Datasets.part58.rar
    │   ├── Datasets.part59.rar
    │   ├── Datasets.part60.rar
    │   └── Datasets.part61.rar
    ├── main.py
    ├── README.md
    ├── requirements.txt
    ├── src/
    │   ├── __init__.py
    │   └── corrosion_annotation.py
    ├── structure.txt
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
Run: Datasets/Datasets.part01.exe
```

This will unpack the required high-resolution annotated corrosion images into the proper directories for training and evaluation.

---

## System Requirements

- **GPU**: NVIDIA GTX 1080 or better (8 GB VRAM minimum)
- **RAM**: 16 GB or more
- **Python**: 3.8+
- **Frameworks**: TensorFlow 2.x, PyTorch, OpenCV, NumPy, Matplotlib
- **Additional**: CUDA Toolkit, cuDNN (for GPU acceleration)