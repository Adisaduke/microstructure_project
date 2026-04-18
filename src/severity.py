# severity.py

from ultralytics import YOLO
import config
import numpy as np
import cv2
import os
from pathlib import Path


# ─────────────────────────────────────────────
# LOAD YOLO MODEL
# ─────────────────────────────────────────────
def load_detector():
    model = YOLO(config.NEU_DETECTOR_PATH)
    print(f"Loaded YOLO model from: {config.NEU_DETECTOR_PATH}")
    return model


# ─────────────────────────────────────────────
# SEVERITY RULE
# ─────────────────────────────────────────────
def compute_severity(avg_conf, num_boxes):
    if avg_conf > 0.6 or num_boxes > 10:
        return "HIGH"
    elif avg_conf > 0.3 or num_boxes > 5:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────
# DETECT + ANALYZE
# ─────────────────────────────────────────────
def detect_and_analyze(image_path, model):

    results = model(image_path, conf=0.05)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("\nNo defects detected.")
        return None

    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    # Group detections by class
    defect_dict = {}

    for cls, conf in zip(class_ids, confidences):
        cls = int(cls)
        if cls not in defect_dict:
            defect_dict[cls] = []
        defect_dict[cls].append(conf)

    final_results = []

    print("\n=== SEVERITY RESULT ===")

    for cls, confs in defect_dict.items():
        defect_name = config.NEU_CLASSES[cls]
        num_boxes = len(confs)
        avg_conf = np.mean(confs)
        severity = compute_severity(avg_conf, num_boxes)

        print(f"\nDefect: {defect_name}")
        print(f"Boxes: {num_boxes}")
        print(f"Avg Confidence: {avg_conf:.4f}")
        print(f"Severity: {severity}")

        final_results.append({
            "defect": defect_name,
            "boxes": num_boxes,
            "avg_conf": avg_conf,
            "severity": severity
        })

    # Save annotated image
    save_path = os.path.join(config.PREDICTIONS_DIR, os.path.basename(image_path))
    annotated = result.plot()
    cv2.imwrite(save_path, annotated)

    print(f"\nSaved image → {save_path}")

    return final_results


# ─────────────────────────────────────────────
# TEST RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    model = load_detector()

    image_paths = [
         str(Path(__file__).resolve().parent.parent / "data" / "NEU_DET" / "test" / "images" / "crazing_9_jpg.rf.a1d9b959edabd458da7e8bf46ccd4beb.jpg"),
         str(Path(__file__).resolve().parent.parent / "data" / "NEU_DET" / "test" / "images" / "inclusion_7_jpg.rf.e84f7d387b8ce1c9c9923d633b12fc03.jpg"),
         str(Path(__file__).resolve().parent.parent / "data" / "NEU_DET" / "test" / "images" / "patches_32_jpg.rf.d49d033b6294470ccf79c15b686b04db.jpg"),
    ]

    for img in image_paths:
        print(f"\nTesting: {img}")
        detect_and_analyze(img, model)