# gradcam.py

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import config
from model import get_model


# ═════════════════════════════════════════════════════════════
# LOAD MODEL
# ═════════════════════════════════════════════════════════════
def load_model():
    if config.MODE == "UHCS":
        model = get_model(config.UHCS_NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model"])
        model_path = config.UHCS_MODEL_PATH

    elif config.MODE == "NEU":
        model = get_model(config.NEU_NUM_CLASSES)
        model_path = config.NEU_CLASSIFIER_PATH

    else:
        raise ValueError("Invalid MODE")

    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(config.DEVICE)
    model.eval()

    class_names = checkpoint["class_names"]

    print(f"Loaded model from: {model_path}")
    return model, class_names


# ═════════════════════════════════════════════════════════════
# GRADCAM CLASS
# ═════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        loss = output[:, class_idx]

        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients[0]        # [C, H, W]
        activations = self.activations[0]    # [C, H, W]

        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(config.DEVICE)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().detach().numpy()


# ═════════════════════════════════════════════════════════════
# IMAGE TRANSFORM
# ═════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ═════════════════════════════════════════════════════════════
# APPLY GRADCAM
# ═════════════════════════════════════════════════════════════
def apply_gradcam(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_class = class_names[pred.item()]
    confidence = conf.item()

    # Initialize GradCAM
    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(input_tensor)

    # Resize CAM to image size
    cam = cv2.resize(cam, (224, 224))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert original image
    image_np = np.array(image.resize((224, 224)))

    # Overlay heatmap
    overlay = heatmap * 0.4 + image_np

    return image_np, heatmap, overlay, pred_class, confidence


# ═════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, class_names = load_model()

    image_path = "/content/drive/MyDrive/Colab Notebooks/microstructure_project/processed/UHCS/test/spheroidite+widmanstatten/Croppedmicrograph717.png"

    image, heatmap, overlay, pred_class, confidence = apply_gradcam(
        image_path, model, class_names
    )

    print(f"Prediction : {pred_class}")
    print(f"Confidence : {confidence:.4f}")

    # Show results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay.astype(np.uint8))
    plt.axis("off")

    plt.show()
