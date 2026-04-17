# severity.py

from ultralytics import YOLO
import numpy as np
import config


# ═════════════════════════════════════════════════════════════
# LOAD MODEL
# ═════════════════════════════════════════════════════════════
def load_model():
    model = YOLO(config.NEU_DETECTOR_PATH)
    print(f"Loaded YOLO model from: {config.NEU_DETECTOR_PATH}")
    return model


# ═════════════════════════════════════════════════════════════
# SEVERITY LOGIC
# ═════════════════════════════════════════════════════════════
def compute_severity(boxes, names):
    if boxes is None or len(boxes) == 0:
        return "No Defect", 0, 0.0, "NONE"

    confidences = []
    classes = []

    for box in boxes:
        conf = float(box.conf[0])
        cls  = int(box.cls[0])

        confidences.append(conf)
        classes.append(cls)

    avg_conf = np.mean(confidences)
    num_boxes = len(boxes)

    # Majority class
    defect_class = max(set(classes), key=classes.count)
    defect_name = names[defect_class]

    # ── Severity Rules ──
    if avg_conf < 0.1:
        severity = "LOW"
    elif avg_conf < 0.3:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return defect_name, num_boxes, avg_conf, severity


# ═════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = load_model()

    # TEST IMAGE
    image_paths = [
        "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/crazing_9_jpg.rf.a1d9b959edabd458da7e8bf46ccd4beb.jpg",
        "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/inclusion_7_jpg.rf.e84f7d387b8ce1c9c9923d633b12fc03.jpg",
        "/content/drive/MyDrive/microstructure_project/data/NEU_DET/test/images/patches_32_jpg.rf.d49d033b6294470ccf79c15b686b04db.jpg",
    ]

    for image_path in image_paths:
        results = model(image_path, conf=0.05)

        boxes = results[0].boxes
        names = results[0].names

        defect, count, avg_conf, severity = compute_severity(boxes, names)

        print("\n=== SEVERITY RESULT ===")
        print(f"Image           : {image_path}")
        print(f"Defect Type     : {defect}")
        print(f"Boxes Detected  : {count}")
        print(f"Avg Confidence  : {avg_conf:.4f}")
        print(f"Severity Level  : {severity}")