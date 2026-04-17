# severity.py

from ultralytics import YOLO
import config
import numpy as np
import cv2
import os


# ─────────────────────────────────────────────
# LOAD YOLO MODEL
# ─────────────────────────────────────────────
def load_detector():
    model = YOLO(config.NEU_DETECTOR_PATH)
    print(f"Loaded YOLO model from: {config.NEU_DETECTOR_PATH}")
    return model


# ─────────────────────────────────────────────
# COMPUTE SEVERITY
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
    results = model(image_path)

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("\nNo defects detected.")
        return None

    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    # Group by class
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

    # Save image with boxes
    save_path = os.path.join(config.PREDICTIONS_DIR, os.path.basename(image_path))
    annotated = result.plot()
    cv2.imwrite(save_path, annotated)

    print(f"\nSaved detection image to: {save_path}")

    return final_results