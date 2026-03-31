# evaluate.py

import os
import torch
import config
from dataset import get_uhcs_loaders, get_neu_loaders
from model import get_model

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report
)
import numpy as np


# ═════════════════════════════════════════════════════════════
# SETUP PIPELINE FROM MODE
# ═════════════════════════════════════════════════════════════
if config.MODE == "UHCS":
    train_loader, val_loader, test_loader, class_names = get_uhcs_loaders()
    model = get_model(config.UHCS_NUM_CLASSES).to(config.DEVICE)
    model_path = config.UHCS_MODEL_PATH

elif config.MODE == "NEU":
    train_loader, val_loader, test_loader, class_names = get_neu_loaders()
    model = get_model(config.NEU_NUM_CLASSES).to(config.DEVICE)
    model_path = config.NEU_CLASSIFIER_PATH

else:
    raise ValueError("MODE must be 'UHCS' or 'NEU'")

print(f"Mode    : {config.MODE}")
print(f"Classes : {class_names}")
print(f"Device  : {config.DEVICE}")
print(f"Loading : {model_path}\n")


# ═════════════════════════════════════════════════════════════
# LOAD SAVED CHECKPOINT
# ═════════════════════════════════════════════════════════════
checkpoint = torch.load(model_path, map_location=config.DEVICE)
model.load_state_dict(checkpoint["model"])
model.to(config.DEVICE)
model.eval()

print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}\n")


# ═════════════════════════════════════════════════════════════
# COLLECT PREDICTIONS AND LABELS
# ═════════════════════════════════════════════════════════════
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# ═════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════
overall_acc = (all_preds == all_labels).mean()

f1_macro = f1_score(all_labels, all_preds, average="macro")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")

cm = confusion_matrix(all_labels, all_preds)

print("═════════════════════════════════════════════════════════════")
print("TEST METRICS")
print("═════════════════════════════════════════════════════════════")
print(f"Overall Accuracy : {overall_acc:.4f}")
print(f"F1 Macro         : {f1_macro:.4f}")
print(f"F1 Weighted      : {f1_weighted:.4f}\n")


# ═════════════════════════════════════════════════════════════
# PER-CLASS ACCURACY
# ═════════════════════════════════════════════════════════════
print("═════════════════════════════════════════════════════════════")
print("PER-CLASS ACCURACY")
print("═════════════════════════════════════════════════════════════")

for i, class_name in enumerate(class_names):
    total_class = cm[i].sum()
    correct_class = cm[i, i]

    if total_class == 0:
        class_acc = 0.0
    else:
        class_acc = correct_class / total_class

    print(f"{class_name:30s}: {class_acc:.4f} ({correct_class}/{total_class})")


# ═════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════
print("\n═════════════════════════════════════════════════════════════")
print("CONFUSION MATRIX")
print("═════════════════════════════════════════════════════════════")
print(cm)


# ═════════════════════════════════════════════════════════════
# FULL CLASSIFICATION REPORT
# ═════════════════════════════════════════════════════════════
print("\n═════════════════════════════════════════════════════════════")
print("CLASSIFICATION REPORT")
print("═════════════════════════════════════════════════════════════")
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
))
