
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from dataset import get_uhcs_loaders


EPOCHS = 20 

# ═════════════════════════════════════════════════════════════
# SIMPLE CNN — NO PRETRAINED WEIGHTS
# ═════════════════════════════════════════════════════════════
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # global average pooling
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


# ═════════════════════════════════════════════════════════════
# SETUP
# ═════════════════════════════════════════════════════════════
torch.manual_seed(42)
np.random.seed(42)

print(f"Device : {config.DEVICE}")
print("Model  : SimpleCNN (no pretrained weights)")
print("Task   : UHCS Baseline\n")

train_loader, val_loader, test_loader, class_names = get_uhcs_loaders()

model     = SimpleCNN(num_classes=config.UHCS_NUM_CLASSES).to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR,
                       weight_decay=config.WEIGHT_DECAY)

save_path = os.path.join(config.MODEL_DIR, "baseline_cnn.pth")
os.makedirs(config.MODEL_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════
# TRAIN FUNCTION
# ═════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds      = torch.max(outputs, dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


# ═════════════════════════════════════════════════════════════
# VALIDATE FUNCTION
# ═════════════════════════════════════════════════════════════
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds      = torch.max(outputs, dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


# ═════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════
best_val_acc = 0.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, config.DEVICE
    )
    val_loss, val_acc = validate(
        model, val_loader, criterion, config.DEVICE
    )

    print(f"Epoch [{epoch+1:3d}/{config.EPOCHS}] "
          f"Train Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f} "
          f"Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"  → Best model saved (Val Acc: {val_acc:.4f})")


# ═════════════════════════════════════════════════════════════
# TEST EVALUATION
# ═════════════════════════════════════════════════════════════
print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
print("\nEvaluating on test set...")

model.load_state_dict(torch.load(save_path, map_location=config.DEVICE))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(config.DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy  = accuracy_score(all_labels, all_preds) * 100
precision = precision_score(all_labels, all_preds,
                            average="weighted",
                            zero_division=0) * 100
recall    = recall_score(all_labels, all_preds,
                         average="weighted") * 100
f1        = f1_score(all_labels, all_preds,
                     average="weighted") * 100

print("\n" + "=" * 50)
print("BASELINE CNN — TEST RESULTS")
print("=" * 50)
print(f"Accuracy  : {accuracy:.2f}%")
print(f"Precision : {precision:.2f}%")
print(f"Recall    : {recall:.2f}%")
print(f"F1-Score  : {f1:.2f}%")
print("=" * 50)

print("\nComparison:")
print(f"{'Model':<20} {'Accuracy':>10} {'F1-Score':>10}")
print("-" * 42)
print(f"{'Baseline CNN':<20} {accuracy:>9.2f}% {f1:>9.2f}%")
print(f"{'ResNet50':<20} {'88.52':>9}% {'88.59':>9}%")
print(f"{'EfficientNetB3':<20} {'87.04':>9}% {'86.67':>9}%")
print("-" * 42)
print("\nBaseline complete.")