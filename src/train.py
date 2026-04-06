# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_uhcs_loaders, get_neu_loaders
from model import get_model


# ═════════════════════════════════════════════════════════════
# SETUP PIPELINE FROM MODE
# ═════════════════════════════════════════════════════════════
if config.MODE == "UHCS":
    train_loader, val_loader, test_loader, class_names = get_uhcs_loaders()
    model           = get_model(config.UHCS_NUM_CLASSES).to(config.DEVICE)
    model_save_path = config.UHCS_MODEL_PATH

elif config.MODE == "NEU":
    train_loader, val_loader, test_loader, class_names = get_neu_loaders()
    model           = get_model(config.NEU_NUM_CLASSES).to(config.DEVICE)
    model_save_path = config.NEU_CLASSIFIER_PATH

else:
    raise ValueError("MODE must be 'UHCS' or 'NEU'")

print(f"Mode    : {config.MODE}")
print(f"Classes : {class_names}")
print(f"Device  : {config.DEVICE}")
print(f"Backbone: {config.BACKBONE}")
print(f"Save to : {model_save_path}\n")


# ═════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═════════════════════════════════════════════════════════════
if config.MODE == "UHCS":
    # Weighted loss — fixes UHCS class imbalance
    class_counts = []
    for cls in class_names:
        cls_path = os.path.join(config.UHCS_TRAIN, cls)
        class_counts.append(len(os.listdir(cls_path)))

    total   = sum(class_counts)
    weights = [total / (len(class_counts) * count)
               for count in class_counts]
    weights   = torch.tensor(weights,
                             dtype=torch.float32).to(config.DEVICE)
    
    if config.MODE == "UHCS":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif config.MODE == "NEU":
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    print(f"Class weights : {[round(w, 2) for w in weights.tolist()]}")

else:
    # NEU is balanced — normal loss
    criterion = nn.CrossEntropyLoss()


# ═════════════════════════════════════════════════════════════
# OPTIMIZER
# ═════════════════════════════════════════════════════════════
if config.MODE == "UHCS":
    lr = 0.0003
else:
    lr = config.LR

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr           = lr,
    weight_decay = config.WEIGHT_DECAY
)


# ═════════════════════════════════════════════════════════════
# TRAIN ONE EPOCH
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
# VALIDATE ONE EPOCH
# ═════════════════════════════════════════════════════════════
def validate_one_epoch(model, loader, criterion, device):
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

for epoch in range(config.EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, config.DEVICE
    )
    val_loss, val_acc = validate_one_epoch(
        model, val_loader, criterion, config.DEVICE
    )

    print(f"Epoch [{epoch+1:3d}/{config.EPOCHS}] "
          f"Train Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f} "
          f"Val Acc: {val_acc:.4f}")

    # Save best model — full checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            "epoch"      : epoch + 1,
            "model"      : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "val_acc"    : val_acc,
            "class_names": class_names,
            "mode"       : config.MODE,
            "backbone"   : config.BACKBONE
        }, model_save_path)
        print(f"  → Best model saved (Val Acc: {val_acc:.4f})")


# ═════════════════════════════════════════════════════════════
# TEST EVALUATION
# ═════════════════════════════════════════════════════════════
print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
print("\nLoading best model for test evaluation...")

checkpoint = torch.load(model_save_path, map_location=config.DEVICE)
model.load_state_dict(checkpoint["model"])
model.to(config.DEVICE)

test_loss, test_acc = validate_one_epoch(
    model, test_loader, criterion, config.DEVICE
)

print(f"Test Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_acc:.4f}")
print(f"\nModel saved to: {model_save_path}")
