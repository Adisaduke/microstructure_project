import torch.nn as nn
from torchvision import models
import config


# ── Shared Model Builder ──────────────────────────────────────
def get_model(num_classes):
    """
    Builds and returns a model based on config.BACKBONE
    Works for both UHCS and NEU pipelines
    num_classes is passed as argument:
        UHCS → config.UHCS_NUM_CLASSES
        NEU  → config.NEU_NUM_CLASSES
    """

    if config.BACKBONE == "resnet50":
        model = _build_resnet50(num_classes)

    elif config.BACKBONE == "efficientnet":
        model = _build_efficientnet(num_classes)

    else:
        raise ValueError(f"Unknown backbone: {config.BACKBONE}. "
                         f"Use 'resnet50' or 'efficientnet'")

    return model


# ── ResNet50 ──────────────────────────────────────────────────
def _build_resnet50(num_classes):
    model = models.resnet50(pretrained=config.PRETRAINED)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block for fine tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final FC layer
    # Add Dropout before linear — needed for uncertainty
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=config.DROPOUT),
        nn.Linear(num_features, num_classes)
    )

    return model


# ── EfficientNetB3 ────────────────────────────────────────────
def _build_efficientnet(num_classes):
    model = models.efficientnet_b3(pretrained=config.PRETRAINED)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block for fine tuning
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Replace final classifier layer
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT),
        nn.Linear(num_features, num_classes)
    )

    return model


# ── Quick Test ────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print(f"Testing backbone: {config.BACKBONE}")
    print(f"Device          : {config.DEVICE}")
    # Test UHCS model
    print("\nBuilding UHCS model...")
    uhcs_model = get_model(num_classes=config.UHCS_NUM_CLASSES)
    uhcs_model = uhcs_model.to(config.DEVICE)
    dummy_input = torch.randn(2, 3, config.IMG_SIZE,
                              config.IMG_SIZE).to(config.DEVICE)
    output = uhcs_model(dummy_input)
    print(f"UHCS output shape : {output.shape}")
    print(f"Expected          : torch.Size([2, {config.UHCS_NUM_CLASSES}])")

    # Test NEU model
    print("\nBuilding NEU model...")
    neu_model = get_model(num_classes=config.NEU_NUM_CLASSES)
    neu_model = neu_model.to(config.DEVICE)

    output = neu_model(dummy_input)
    print(f"NEU output shape  : {output.shape}")
    print(f"Expected          : torch.Size([2, {config.NEU_NUM_CLASSES}])")

    # Count trainable parameters
    uhcs_params = sum(p.numel() for p in uhcs_model.parameters()
                      if p.requires_grad)
    print(f"\nTrainable parameters : {uhcs_params:,}")

    print("\nmodel.py verified successfully.")
