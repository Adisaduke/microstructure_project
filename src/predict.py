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
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=config.NORMALIZE_MEAN,
        std=config.NORMALIZE_STD
    )
])


# ═════════════════════════════════════════════════════════════
# PREDICT FUNCTION
# ═════════════════════════════════════════════════════════════
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

        # Top-K predictions
        topk = 3
        top_probs, top_idxs = torch.topk(probs, topk)

    top_probs = top_probs.squeeze().cpu().numpy()
    top_idxs  = top_idxs.squeeze().cpu().numpy()

    print("\nTop Predictions:")
    for i in range(topk):
        class_name = class_names[top_idxs[i]]
        confidence = top_probs[i]
        print(f"{i+1}. {class_name} → {confidence:.4f}")

    # Return top-1 for compatibility
    return class_names[top_idxs[0]], top_probs[0]

# ═════════════════════════════════════════════════════════════
# RUN PREDICTION
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, class_names = load_model()

    # CHANGE THIS PATH
    image_path = "/content/drive/MyDrive/Colab Notebooks/microstructure_project/processed/UHCS/test/pearlite+widmanstatten/Croppedmicrograph1078.png"
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
