# config.py

import torch
import os
import random
import numpy as np
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Mode — switch between pipelines ───────────────────────────
MODE = "UHCS"   # change to "NEU" to switch pipeline

# ── Base Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "processed"
COLAB_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/microstructure_project/processed")

if COLAB_DATA_DIR.exists():
    BASE_DIR = str(COLAB_DATA_DIR)
else:
    BASE_DIR = str(LOCAL_DATA_DIR)

OUTPUT_DIR = str(PROJECT_ROOT / "output")
MODEL_DIR  = str(PROJECT_ROOT / "output" / "models")

PREDICTIONS_DIR = f"{OUTPUT_DIR}/predictions"
FIGURES_DIR     = f"{OUTPUT_DIR}/figures"
LOGS_DIR        = f"{OUTPUT_DIR}/logs"
GRADCAM_DIR     = f"{OUTPUT_DIR}/gradcam"

# Create all output folders automatically
for folder in [MODEL_DIR, PREDICTIONS_DIR,
               FIGURES_DIR, LOGS_DIR, GRADCAM_DIR]:
    os.makedirs(folder, exist_ok=True)

# ── UHCS Settings ─────────────────────────────────────────────
UHCS_TRAIN = f"{BASE_DIR}/UHCS/train"
UHCS_VAL   = f"{BASE_DIR}/UHCS/val"
UHCS_TEST  = f"{BASE_DIR}/UHCS/test"

UHCS_CLASSES = [
    "spheroidite",
    "network",
    "spheroidite+widmanstatten",
    "pearlite+spheroidite",
    "pearlite",
    "pearlite+widmanstatten"
]
UHCS_NUM_CLASSES = len(UHCS_CLASSES)

UHCS_MODEL_PATH = "/content/drive/MyDrive/microstructure_project/models/uhcs_model.pth"

# ── NEU Settings ──────────────────────────────────────────────
NEU_TRAIN       = f"{BASE_DIR}/NEU/train/images"
NEU_VAL         = f"{BASE_DIR}/NEU/val/images"
NEU_TEST        = f"{BASE_DIR}/NEU/test/images"
NEU_ANNOTATIONS = f"{BASE_DIR}/NEU/annotations.csv"

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]
NEU_NUM_CLASSES = len(NEU_CLASSES)

NEU_CLASSIFIER_PATH = f"{MODEL_DIR}/neu_classifier.pth"
NEU_DETECTOR_PATH   = "/content/drive/MyDrive/microstructure_project/models/neu_yolo_best.pt"

# ── Image Settings ────────────────────────────────────────────
IMG_SIZE       = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ── Training Settings ─────────────────────────────────────────
BATCH_SIZE   = 32
LR           = 0.0001       # correct for fine tuning pretrained
EPOCHS       = 50
WEIGHT_DECAY = 1e-4         # L2 regularization for UHCS imbalance
NUM_WORKERS  = 2

# ── Model Settings ────────────────────────────────────────────
BACKBONE   = "resnet50"     # change to "efficientnet" for second run
PRETRAINED = True
DROPOUT    = 0.5            # for uncertainty quantification

# ── Uncertainty Settings ──────────────────────────────────────
MC_SAMPLES = 30             # Monte Carlo dropout forward passes
