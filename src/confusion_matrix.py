import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



import config
from model import get_model


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    model = get_model(config.UHCS_NUM_CLASSES)

    checkpoint = torch.load(config.UHCS_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(config.DEVICE)
    model.eval()

    class_names = checkpoint["class_names"]
    return model, class_names


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
])

dataset = datasets.ImageFolder(config.UHCS_TEST, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
model, class_names = load_model()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(config.DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (UHCS Classification)")

# SAVE
save_path = f"{config.FIGURES_DIR}/confusion_matrix.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved → {save_path}")



# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n=== CLASSIFICATION METRICS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")