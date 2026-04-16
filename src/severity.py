# severity.py

from ultralytics import YOLO
import config
import cv2
import numpy as np


# ═════════════════════════════════════════════════════════════
# LOAD YOLO MODEL
# ═════════════════════════════════════════════════════════════
def load_detector():
    model = YOLO(config.NEU_DETECTOR_PATH)
    print(f"Loaded YOLO model from: {config.NEU_DETECTOR_PATH}")
    return model


# ═════════════════════════════════════════════════════════════
# COMPUTE SEVERITY
# ═════════════════════════════════════════════════════════════
def compute_severity(box, image_shape):
    """
    box: [x1, y1, x2, y2]
    image_shape: (H, W, C)
    """

    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)

    image_area = image_shape[0] * image_shape[1]

    ratio = box_area / image_area

    # Severity rules
    if ratio < 0.02:
        severity = "LOW"
    elif ratio < 0.10:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return severity, ratio


# ═════════════════════════════════════════════════════════════
# RUN SEVERITY ANALYSIS
# ═════════════════════════════════════════════════════════════
def analyze_image(image_path, model):
    results = model(image_path)

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    print("\n=== DETECTIONS ===")

    for r in results:
        boxes = r.boxes

        if boxes is None:
            print("No defects detected.")
            return

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls]

            severity, ratio = compute_severity(xyxy, img.shape)

            print(f"\nDefect     : {class_name}")
            print(f"Confidence : {conf:.4f}")
            print(f"Severity   : {severity}")
            print(f"Area Ratio : {ratio:.4f}")


# ═════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = load_detector()

    image_path = "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/crazing_9_jpg.rf.a1d9b959edabd458da7e8bf46ccd4beb.jpg"

    analyze_image(image_path, model)