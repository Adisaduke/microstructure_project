# system.py

import config
from severity import load_detector, detect_and_analyze
from gradcam import apply_gradcam
from model import get_model
import torch
import cv2
import os


# ─────────────────────────────────────────────
# LOAD UHCS CLASSIFIER
# ─────────────────────────────────────────────
def load_classifier():
    model = get_model(config.UHCS_NUM_CLASSES)

    checkpoint = torch.load(config.UHCS_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(config.DEVICE)
    model.eval()

    class_names = checkpoint["class_names"]

    print(f"Loaded UHCS model from: {config.UHCS_MODEL_PATH}")

    return model, class_names


# ─────────────────────────────────────────────
# MAIN SYSTEM
# ─────────────────────────────────────────────
def run_system(image_path, mode="UHCS"):

    print("\n==============================")
    print(f"Running system on: {image_path}")
    print(f"Mode: {mode}")
    print("==============================")

    # ───────────── NEU PIPELINE ─────────────
    if mode == "NEU":

        detector = load_detector()
        results = detect_and_analyze(image_path, detector)

        return results


    # ───────────── UHCS PIPELINE ─────────────
    elif mode == "UHCS":

        # Fix GradCAM mode dependency
        config.MODE = "UHCS"

        model, class_names = load_classifier()

        image, heatmap, overlay, pred_class, confidence = apply_gradcam(
            image_path, model, class_names
        )

        print("\n=== CLASSIFICATION RESULT ===")
        print(f"Class: {pred_class}")
        print(f"Confidence: {confidence:.4f}")

        # Save GradCAM image
        save_path = os.path.join(config.GRADCAM_DIR, os.path.basename(image_path))
        cv2.imwrite(save_path, overlay)

        print(f"GradCAM saved → {save_path}")

        return {
            "class": pred_class,
            "confidence": confidence,
            "gradcam": save_path
        }

    else:
        raise ValueError("Mode must be 'UHCS' or 'NEU'")


# ─────────────────────────────────────────────
# TEST RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 🔹 CHANGE THIS FOR TESTING

    # NEU TEST
    run_system(
        "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/patches_32_jpg.rf.d49d033b6294470ccf79c15b686b04db.jpg",
        mode="NEU"
    )

    # UHCS TEST
    run_system(
        "/content/drive/MyDrive/Colab Notebooks/microstructure_project/processed/UHCS/test/spheroidite/Croppedmicrograph465.png",
        mode="UHCS"
    )