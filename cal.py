import numpy as np

print("=" * 60)
print("MEAN AND STANDARD DEVIATION ACROSS 3 RUNS")
print("=" * 60)

# ── ResNet50 ──────────────────────────────────────────────────
resnet_acc  = [87.78, 90.00, 87.78]
resnet_prec = [87.77, 90.62, 88.47]
resnet_rec  = [87.78, 90.00, 87.78]
resnet_f1   = [87.70, 90.13, 87.95]

print("\nResNet50:")
print(f"  Accuracy  : {np.mean(resnet_acc):.2f} ± {np.std(resnet_acc):.2f}%")
print(f"  Precision : {np.mean(resnet_prec):.2f} ± {np.std(resnet_prec):.2f}%")
print(f"  Recall    : {np.mean(resnet_rec):.2f} ± {np.std(resnet_rec):.2f}%")
print(f"  F1-Score  : {np.mean(resnet_f1):.2f} ± {np.std(resnet_f1):.2f}%")

# ── EfficientNetB3 ────────────────────────────────────────────
eff_acc  = [87.78, 87.78, 85.56]
eff_prec = [87.28, 87.28, 85.81]
eff_rec  = [87.78, 87.78, 85.56]
eff_f1   = [87.35, 87.35, 85.32]

print("\nEfficientNetB3:")
print(f"  Accuracy  : {np.mean(eff_acc):.2f} ± {np.std(eff_acc):.2f}%")
print(f"  Precision : {np.mean(eff_prec):.2f} ± {np.std(eff_prec):.2f}%")
print(f"  Recall    : {np.mean(eff_rec):.2f} ± {np.std(eff_rec):.2f}%")
print(f"  F1-Score  : {np.mean(eff_f1):.2f} ± {np.std(eff_f1):.2f}%")

# ── YOLO ──────────────────────────────────────────────────────
yolo_prec  = [0.622, 0.676, 0.662]
yolo_rec   = [0.712, 0.698, 0.703]
yolo_map50 = [0.712, 0.723, 0.711]
yolo_map95 = [0.401, 0.398, 0.403]

print("\nYOLOv8 Detection:")
print(f"  Precision    : {np.mean(yolo_prec):.3f} ± {np.std(yolo_prec):.3f}")
print(f"  Recall       : {np.mean(yolo_rec):.3f} ± {np.std(yolo_rec):.3f}")
print(f"  mAP@0.5      : {np.mean(yolo_map50):.3f} ± {np.std(yolo_map50):.3f}")
print(f"  mAP@0.5:0.95 : {np.mean(yolo_map95):.3f} ± {np.std(yolo_map95):.3f}")

# ── Summary Table ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY TABLE FOR PAPER")
print("=" * 60)
print(f"\n{'Model':<20} {'Accuracy':>15} {'Precision':>15} {'Recall':>15} {'F1-Score':>15}")
print("-" * 80)
print(f"{'ResNet50':<20} "
      f"{np.mean(resnet_acc):.2f}±{np.std(resnet_acc):.2f}%".rjust(15) + " "
      f"{np.mean(resnet_prec):.2f}±{np.std(resnet_prec):.2f}%".rjust(15) + " "
      f"{np.mean(resnet_rec):.2f}±{np.std(resnet_rec):.2f}%".rjust(15) + " "
      f"{np.mean(resnet_f1):.2f}±{np.std(resnet_f1):.2f}%".rjust(15))
print(f"{'EfficientNetB3':<20} "
      f"{np.mean(eff_acc):.2f}±{np.std(eff_acc):.2f}%".rjust(15) + " "
      f"{np.mean(eff_prec):.2f}±{np.std(eff_prec):.2f}%".rjust(15) + " "
      f"{np.mean(eff_rec):.2f}±{np.std(eff_rec):.2f}%".rjust(15) + " "
      f"{np.mean(eff_f1):.2f}±{np.std(eff_f1):.2f}%".rjust(15))

print(f"\n{'Model':<20} {'Precision':>15} {'Recall':>15} {'mAP@0.5':>15} {'mAP@0.5:0.95':>15}")
print("-" * 80)
print(f"{'YOLOv8':<20} "
      f"{np.mean(yolo_prec):.3f}±{np.std(yolo_prec):.3f}".rjust(15) + " "
      f"{np.mean(yolo_rec):.3f}±{np.std(yolo_rec):.3f}".rjust(15) + " "
      f"{np.mean(yolo_map50):.3f}±{np.std(yolo_map50):.3f}".rjust(15) + " "
      f"{np.mean(yolo_map95):.3f}±{np.std(yolo_map95):.3f}".rjust(15))

print("\nDone. Copy these numbers into your paper.")