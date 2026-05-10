# config.py

import torch
import os
import random
import numpy as np
from pathlib import Path

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODE
# ============================================================

MODE = "NEU"

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# LOCAL PATHS
# ============================================================

LOCAL_DATA_DIR   = PROJECT_ROOT / "data"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "output"
LOCAL_MODEL_DIR  = PROJECT_ROOT / "models"

# ============================================================
# GOOGLE DRIVE PATHS (COLAB)
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/microstructure_project")

DRIVE_DATA_DIR   = DRIVE_ROOT
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "output"
DRIVE_MODEL_DIR  = DRIVE_ROOT / "models"

# ============================================================
# AUTO SWITCH: COLAB OR LOCAL
# ============================================================

IN_COLAB = DRIVE_ROOT.exists()

if IN_COLAB:
    print("Running in Google Colab with Drive")

    BASE_DIR   = DRIVE_DATA_DIR
    OUTPUT_DIR = DRIVE_OUTPUT_DIR
    MODEL_DIR  = DRIVE_MODEL_DIR

else:
    print("Running on Local System")

    BASE_DIR   = LOCAL_DATA_DIR
    OUTPUT_DIR = LOCAL_OUTPUT_DIR
    MODEL_DIR  = LOCAL_MODEL_DIR

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FIGURES_DIR     = OUTPUT_DIR / "figures"
LOGS_DIR        = OUTPUT_DIR / "logs"
GRADCAM_DIR     = OUTPUT_DIR / "gradcam"

for folder in [
    MODEL_DIR,
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    GRADCAM_DIR
]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# UHCS DATASET
# ============================================================

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

# ============================================================
# MODEL SAVE PATHS
# ============================================================

# CHANGE THIS WHEN SWITCHING MODELS
BACKBONE = "efficientnet"

if BACKBONE == "resnet50":
    UHCS_MODEL_PATH = str(MODEL_DIR / "resnet50_model999.pth")

elif BACKBONE == "efficientnet":
    UHCS_MODEL_PATH = str(MODEL_DIR / "efficientnet_model999.pth")

# ============================================================
# NEU DATASET
# ============================================================

NEU_TRAIN       = str(BASE_DIR / "data" / "NEU_DET" / "train" / "images")
NEU_VAL         = str(BASE_DIR / "data" / "NEU_DET" / "valid" / "images")
NEU_TEST        = str(BASE_DIR / "data" / "NEU_DET" / "test" / "images")

NEU_ANNOTATIONS = str(BASE_DIR / "data" / "NEU_DET" / "annotations.csv")

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

NEU_NUM_CLASSES = len(NEU_CLASSES)

NEU_CLASSIFIER_PATH = str(MODEL_DIR / "neu_classifier.pth")
NEU_DETECTOR_PATH   = str(MODEL_DIR / "neu_yolo_best.pt")

# ============================================================
# IMAGE SETTINGS
# ============================================================

IMG_SIZE = 224

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE   = 32
LR           = 0.0001
EPOCHS       = 50
WEIGHT_DECAY = 1e-4

# IMPORTANT:
# Windows multiprocessing issue fix
if IN_COLAB:
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 0

# ============================================================
# MODEL SETTINGS
# ============================================================

PRETRAINED = True
DROPOUT    = 0.5

# ============================================================
# UNCERTAINTY SETTINGS
# ============================================================

MC_SAMPLES = 30