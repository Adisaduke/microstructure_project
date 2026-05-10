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
MODE = "UHCS"  

# ── Base Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LOCAL PATH
LOCAL_DATA_DIR = PROJECT_ROOT / "data"

# GOOGLE DRIVE PATH (COLAB)
DRIVE_ROOT = Path("/content/drive/MyDrive//Colab Notebooks/microstructure_project")

DRIVE_DATA_DIR   = DRIVE_ROOT
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "output"
DRIVE_MODEL_DIR  = DRIVE_ROOT / "models"

# AUTO SWITCH
if DRIVE_ROOT.exists():
    print("Running in Google Colab with Drive")

    BASE_DIR   = DRIVE_DATA_DIR
    OUTPUT_DIR = str(DRIVE_OUTPUT_DIR)
    MODEL_DIR  = str(DRIVE_MODEL_DIR)

else:
    print("Running on Local System")

    BASE_DIR   = LOCAL_DATA_DIR
    OUTPUT_DIR = str(PROJECT_ROOT / "output")
    MODEL_DIR  = str(PROJECT_ROOT / "models")


# OUTPUT SUBFOLDERS
PREDICTIONS_DIR = f"{OUTPUT_DIR}/predictions"
FIGURES_DIR     = f"{OUTPUT_DIR}/figures"
LOGS_DIR        = f"{OUTPUT_DIR}/logs"
GRADCAM_DIR     = f"{OUTPUT_DIR}/gradcam"

# CREATE FOLDERS
for folder in [
    MODEL_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    GRADCAM_DIR
]:
    os.makedirs(folder, exist_ok=True)

# ── UHCS Settings ─────────────────────────────────────────────
UHCS_TRAIN = str(BASE_DIR / "processed" / "UHCS" / "train")
UHCS_VAL   = str(BASE_DIR / "processed" / "UHCS" / "val")
UHCS_TEST  = str(BASE_DIR / "processed" / "UHCS" / "test")

UHCS_CLASSES = [
    "spheroidite",
    "network",
    "spheroidite+widmanstatten",
    "pearlite+spheroidite",
    "pearlite",
    "pearlite+widmanstatten"
]
UHCS_NUM_CLASSES = len(UHCS_CLASSES)

UHCS_MODEL_PATH = str(PROJECT_ROOT / "models/uhcs_model.pth")

# ── NEU Settings ──────────────────────────────────────────────
NEU_TRAIN       = str(BASE_DIR / "NEU_DET" / "train" / "images")
NEU_VAL         = str(BASE_DIR / "NEU_DET" / "valid" / "images")  
NEU_TEST        = str(BASE_DIR / "NEU_DET" / "test" / "images")
NEU_ANNOTATIONS = str(BASE_DIR / "NEU_DET" / "annotations.csv")

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
NEU_DETECTOR_PATH   = str(PROJECT_ROOT / "models/neu_yolo_best.pt")

# ── Image Settings ────────────────────────────────────────────
IMG_SIZE       = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ── Training Settings ─────────────────────────────────────────
BATCH_SIZE   = 32
LR           = 0.0001
EPOCHS       = 50
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2

# ── Model Settings ────────────────────────────────────────────
BACKBONE   = "efficientnet"
PRETRAINED = True
DROPOUT    = 0.5

# ── Uncertainty Settings ──────────────────────────────────────
MC_SAMPLES = 30