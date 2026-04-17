# system.py

import config
from predict import load_model, predict_image
from gradcam import generate_gradcam
from severity import load_detector, detect_and_analyze

import os


# ─────────────────────────────────────────────
# MAIN SYSTEM FUNCTION
# ─────────────────────────────────────────────
def run_system(image_path, mode="UHCS"):
    print("\n==============================")
    print("RUNNING MICROSTRUCTURE SYSTEM")
    print("==============================")

    if not os.path.exists(image_path):
        print("Image not found.")
        return

    # ─────────────────────────
    # UHCS PIPELINE
    # ─────────────────────────
    if mode == "UHCS":
        print("\n=== UHCS MODE ===")

        model, class_names = load_model()

        pred_class, confidence = predict_image(
            image_path, model, class_names
        )

        print("\n=== RESULT ===")
        print(f"Type        : UHCS")
        print(f"Prediction  : {pred_class}")
        print(f"Confidence  : {confidence:.4f}")

        # Grad-CAM
        print("\nGenerating Grad-CAM...")
        generate_gradcam(image_path, model, class_names)

    # ─────────────────────────
    # NEU PIPELINE
    # ─────────────────────────
    elif mode == "NEU":
        print("\n=== NEU MODE ===")

        detector = load_detector()

        results = detect_and_analyze(image_path, detector)

        print("\n=== FINAL RESULT ===")
        print("Type: NEU")

        if results is None:
            print("No defects found.")
            return

        for res in results:
            print(f"\nDefect       : {res['defect']}")
            print(f"Boxes        : {res['boxes']}")
            print(f"Avg Conf     : {res['avg_conf']:.4f}")
            print(f"Severity     : {res['severity']}")

    else:
        print("Invalid mode. Choose UHCS or NEU.")


# ─────────────────────────────────────────────
# RUN EXAMPLE
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # CHANGE IMAGE HERE
    image_path = "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/patches_32_jpg.rf.d49d033b6294470ccf79c15b686b04db.jpg"

    # SELECT MODE
    run_system(image_path, mode="NEU")
    # run_system(image_path, mode="UHCS")