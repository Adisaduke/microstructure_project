import torch
import os
import random
import numpy as np
from pathlib import Path

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Mode - switch between pipelines
# ----------------------------
MODE = "UHCS"   # change to "NEU" if needed

# ----------------------------
# Project root
# ----------------------------
# This file is inside src/, so project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------
# Candidate data locations
# ----------------------------
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "processed"
COLAB_DRIVE_DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/microstructure_project/processed")

if COLAB_DRIVE_DATA_DIR.exists():
    BASE_DIR = str(COLAB_DRIVE_DATA_DIR)
elif LOCAL_DATA_DIR.exists():
    BASE_DIR = str(LOCAL_DATA_DIR)
else:
    raise FileNotFoundError(
        f"Could not find processed data folder.\n"
        f"Checked:\n"
        f" - {COLAB_DRIVE_DATA_DIR}\n"
        f" - {LOCAL_DATA_DIR}"
    )

# ----------------------------
# Output paths
# ----------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

for folder in [MODEL_DIR, PREDICTIONS_DIR, FIGURES_DIR, LOGS_DIR, GRADCAM_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ----------------------------
# UHCS Settings
# ----------------------------
UHCS_TRAIN = os.path.join(BASE_DIR, "UHCS", "train")
UHCS_VAL   = os.path.join(BASE_DIR, "UHCS", "val")
UHCS_TEST  = os.path.join(BASE_DIR, "UHCS", "test")

UHCS_CLASSES = [
    "spheroidite",
    "network",
    "spheroidite+widmanstatten",
    "pearlite+spheroidite",
    "pearlite",
    "pearlite+widmanstatten"
]
UHCS_NUM_CLASSES = len(UHCS_CLASSES)

UHCS_MODEL_PATH = str(MODEL_DIR / "uhcs_model.pth")

# ----------------------------
# NEU Settings
# ----------------------------
NEU_TRAIN = os.path.join(BASE_DIR, "NEU", "train")
NEU_VAL   = os.path.join(BASE_DIR, "NEU", "val")
NEU_TEST  = os.path.join(BASE_DIR, "NEU", "test")

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]
NEU_NUM_CLASSES = len(NEU_CLASSES)

NEU_MODEL_PATH = str(MODEL_DIR / "neu_model.pth")