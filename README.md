# 🔬 Microstructure Analysis System

A complete deep learning system for analyzing steel microstructures using:

- Classification (UHCS microstructures)
- Defect detection (NEU dataset)
- Severity estimation
- Explainability (Grad-CAM)
- Interactive UI

---

## 🚀 Features

### 1. UHCS Classification
- Classifies microstructures into:
  - spheroidite
  - pearlite
  - network
  - mixed structures
- Provides **Top-3 predictions** with confidence
- Includes **Grad-CAM visualization** to show model focus

---

### 2. NEU Defect Detection
- Uses YOLOv8 for localization
- Detects:
  - crazing
  - inclusion
  - patches
  - pitted surface
  - rolled-in scale
  - scratches
- Outputs bounding boxes with confidence

---

### 3. Severity Estimation
- Estimates defect severity based on:
  - number of detected regions
  - average confidence
- Levels:
  - LOW
  - MEDIUM
  - HIGH

---

### 4. Integrated Pipeline
Single system that:
- Takes an image
- Runs appropriate model (UHCS or NEU)
- Returns:
  - prediction
  - confidence
  - visualization (Grad-CAM or bounding boxes)
  - severity (for defects)

---

### 5. Interactive UI
Built with Streamlit:
- Upload image
- Select type (UHCS / NEU)
- View results instantly

---

## 🧠 Key Insight

Some UHCS classes have high visual similarity (e.g., spheroidite vs spheroidite+widmanstatten).  
To address this, the system provides:

- Top-3 predictions instead of single output  
- Grad-CAM visual explanations  

This improves interpretability and trust in model predictions.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit

---

## 📁 Project Structure

microstructure_project/
│
├── app/                              ← UI
│   └── app.py
│
├── data/
│   └── NEU_DET/                      ← YOLO dataset
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── data.yaml
│   │
│   │
│   ├── processed/                     ← UHCS dataset
│       └── UHCS/
│           ├── train/
│           ├── val/
│           └── test/  
│
├── models/                           ← trained models
│   ├── uhcs_model.pth
│   └── neu_yolo_best.pt
│
├── output/                          
│   ├── figures/                      ← detection images
│   ├── gradcam/                      ← heatmaps
│   ├── logs/
│   └── predictions/                  ← generated automatically
│
├── src/                              ← ALL CODE
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── severity.py
│   └── system.py
│
│── .gitignore
├── README.md
├── requirements.txt




---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py