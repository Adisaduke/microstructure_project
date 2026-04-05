# predict.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import os

import config
from model import get_model


# ═════════════════════════════════════════════════════════════
# LOAD MODEL
# ═════════════════════════════════════════════════════════════
def load_model():
    if config.MODE == "UHCS":
        model = get_model(config.UHCS_NUM_CLASSES)
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
    print(f"Classes: {class_names}")

    return model, class_names


# ═════════════════════════════════════════════════════════════
# IMAGE TRANSFORM (must match training)
# ═════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ═════════════════════════════════════════════════════════════
# PREDICT FUNCTION
# ═════════════════════════════════════════════════════════════
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension
    image = image.to(config.DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, dim=1)

    pred_class = class_names[pred.item()]
    confidence = conf.item()

    return pred_class, confidence


# ═════════════════════════════════════════════════════════════
# RUN PREDICTION
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, class_names = load_model()

    # CHANGE THIS PATH
    image_path = "/content/microstructure_project/data/processed/test/spheroidite/micrograph196.png"

    if not os.path.exists(image_path):
        print("Image not found. Check path.")
    else:
        pred_class, confidence = predict_image(
            image_path, model, class_names
        )

        print("\n=== RESULT ===")
        print(f"Image      : {image_path}")
        print(f"Prediction : {pred_class}")
        print(f"Confidence : {confidence:.4f}")
