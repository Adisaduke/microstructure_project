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

    # Load model
    model = YOLO(config.NEU_DETECTOR_PATH)
    print(f"Loaded YOLO model from: {config.NEU_DETECTOR_PATH}")

    # Test multiple images
    image_paths = [
        "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/crazing_9_jpg.rf.a1d9b959edabd458da7e8bf46ccd4beb.jpg",
        "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/inclusion_7_jpg.rf.e84f7d387b8ce1c9c9923d633b12fc03.jpg",
        "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/patches_32_jpg.rf.d49d033b6294470ccf79c15b686b04db.jpg",
    ]

    for image_path in image_paths:
        print(f"\nTesting: {image_path}")

        results = model(image_path, conf=0.05)  # 🔥 LOWER CONFIDENCE

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            print("No detections")
        else:
            print("Detections found:")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                print(f"Class: {cls_id}, Confidence: {conf:.4f}")