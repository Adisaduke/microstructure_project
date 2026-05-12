# gradcam_batch.py
# Runs Grad-CAM on multiple images per class
# Saves heatmaps and quantification results

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd

import config
from model import get_model


# ═════════════════════════════════════════════════════════════
# LOAD MODEL
# ═════════════════════════════════════════════════════════════
def load_model():
    if config.MODE == "UHCS":
        model      = get_model(config.UHCS_NUM_CLASSES)
        model_path = config.UHCS_MODEL_PATH
    elif config.MODE == "NEU":
        model      = get_model(config.NEU_NUM_CLASSES)
        model_path = config.NEU_CLASSIFIER_PATH
    else:
        raise ValueError("Invalid MODE")

    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(config.DEVICE)
    model.eval()
    class_names = checkpoint["class_names"]
    print(f"Model loaded from : {model_path}")
    return model, class_names


# ═════════════════════════════════════════════════════════════
# GRADCAM CLASS
# ═════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        gradients  = self.gradients[0]
        activations = self.activations[0]
        weights    = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:],
                          dtype=torch.float32).to(config.DEVICE)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().detach().numpy(), class_idx


# ═════════════════════════════════════════════════════════════
# TRANSFORMS
# ═════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORMALIZE_MEAN,
                         std =config.NORMALIZE_STD)
])


# ═════════════════════════════════════════════════════════════
# PROCESS ONE IMAGE
# ═════════════════════════════════════════════════════════════
def process_image(image_path, model, class_names, gradcam):
    image        = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Get prediction and confidence
    with torch.no_grad():
        outputs = model(input_tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_class = class_names[pred.item()]
    confidence = conf.item()

    # Generate CAM — need grad so no no_grad here
    cam, pred_idx = gradcam.generate(input_tensor)
    cam_resized   = cv2.resize(cam, (config.IMG_SIZE, config.IMG_SIZE))

    # ── Quantification ────────────────────────────────────────
    # High activation = top 30% of cam values
    threshold      = np.percentile(cam_resized, 70)
    high_act_mask  = cam_resized >= threshold

    # Center region = middle 60% of image (where microstructure is)
    h, w = cam_resized.shape
    margin_h = int(h * 0.2)
    margin_w = int(w * 0.2)
    center_mask = np.zeros_like(cam_resized, dtype=bool)
    center_mask[margin_h:h-margin_h, margin_w:w-margin_w] = True

    # What percentage of high activation falls in center
    high_in_center  = np.sum(high_act_mask & center_mask)
    total_high      = np.sum(high_act_mask)
    center_focus    = high_in_center / (total_high + 1e-8)

    # Average activation score
    avg_activation  = float(np.mean(cam_resized))

    return {
        "image"         : np.array(image.resize((config.IMG_SIZE,
                                                  config.IMG_SIZE))),
        "cam"           : cam_resized,
        "pred_class"    : pred_class,
        "confidence"    : confidence,
        "center_focus"  : center_focus,
        "avg_activation": avg_activation
    }


# ═════════════════════════════════════════════════════════════
# BATCH PROCESSING — ALL CLASSES
# ═════════════════════════════════════════════════════════════
def run_batch(model, class_names, test_dir, output_dir,
              images_per_class=3):

    os.makedirs(output_dir, exist_ok=True)
    target_layer = model.layer4
    gradcam      = GradCAM(model, target_layer)

    all_records  = []

    for cls in sorted(os.listdir(test_dir)):
        cls_path = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path)
                  if f.endswith((".png", ".jpg", ".jpeg", ".bmp"))]

        if len(images) == 0:
            print(f"  No images found in {cls}")
            continue

        # Pick up to images_per_class
        selected = images[:images_per_class]
        print(f"\nProcessing class: {cls} ({len(selected)} images)")

        class_results = []

        for img_name in selected:
            img_path = os.path.join(cls_path, img_name)

            result = process_image(img_path, model,
                                   class_names, gradcam)

            class_results.append(result)

            all_records.append({
                "class"         : cls,
                "image"         : img_name,
                "predicted"     : result["pred_class"],
                "correct"       : cls == result["pred_class"],
                "confidence"    : round(result["confidence"], 4),
                "center_focus"  : round(result["center_focus"], 4),
                "avg_activation": round(result["avg_activation"], 4)
            })

            print(f"  {img_name[:30]:<30} "
                  f"Pred: {result['pred_class']:<30} "
                  f"Conf: {result['confidence']:.3f} "
                  f"CenterFocus: {result['center_focus']:.3f}")

        # ── Save figure for this class ─────────────────────────
        n = len(class_results)
        fig, axes = plt.subplots(n, 3,
                                 figsize=(12, 4 * n))

        if n == 1:
            axes = [axes]

        for i, res in enumerate(class_results):
            # Original
            axes[i][0].imshow(res["image"])
            axes[i][0].set_title(f"Original\nTrue: {cls}",
                                  fontsize=9)
            axes[i][0].axis("off")

            # Heatmap
            heatmap = cv2.applyColorMap(
                np.uint8(255 * res["cam"]),
                cv2.COLORMAP_JET
            )
            axes[i][1].imshow(
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            )
            axes[i][1].set_title(
                f"Grad-CAM Heatmap\nPred: {res['pred_class']}",
                fontsize=9
            )
            axes[i][1].axis("off")

            # Overlay
            image_np = res["image"].astype(np.float32)
            heat_np  = heatmap.astype(np.float32)
            overlay  = cv2.addWeighted(
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                0.4,
                res["image"],
                0.6,
                0
            )
            axes[i][2].imshow(overlay)
            axes[i][2].set_title(
                f"Overlay\nConf: {res['confidence']:.3f} "
                f"Focus: {res['center_focus']:.3f}",
                fontsize=9
            )
            axes[i][2].axis("off")

        plt.suptitle(f"Grad-CAM — Class: {cls}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        save_name = cls.replace("+", "_").replace(" ", "_")
        fig_path  = os.path.join(output_dir,
                                  f"gradcam_{save_name}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    return all_records


# ═════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════
def print_summary(records, output_dir):
    df = pd.DataFrame(records)

    print("\n" + "=" * 65)
    print("GRAD-CAM QUANTIFICATION SUMMARY")
    print("=" * 65)

    # Per class summary
    summary = df.groupby("class").agg(
        Images       = ("image", "count"),
        Correct      = ("correct", "sum"),
        Avg_Conf     = ("confidence", "mean"),
        Avg_Focus    = ("center_focus", "mean"),
        Avg_Act      = ("avg_activation", "mean")
    ).reset_index()

    print(f"\n{'Class':<35} {'Imgs':>5} {'Correct':>8} "
          f"{'AvgConf':>9} {'CenterFocus':>12} {'AvgAct':>8}")
    print("-" * 80)

    for _, row in summary.iterrows():
        print(f"{row['class']:<35} "
              f"{int(row['Images']):>5} "
              f"{int(row['Correct']):>8} "
              f"{row['Avg_Conf']:>9.3f} "
              f"{row['Avg_Focus']:>12.3f} "
              f"{row['Avg_Act']:>8.3f}")

    print("-" * 80)
    print(f"\nOverall Average Center Focus : "
          f"{df['center_focus'].mean():.3f}")
    print(f"Overall Average Confidence   : "
          f"{df['confidence'].mean():.3f}")
    print(f"Overall Accuracy             : "
          f"{df['correct'].mean()*100:.2f}%")

    # Save to CSV
    csv_path = os.path.join(output_dir, "gradcam_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    summary_path = os.path.join(output_dir,
                                 "gradcam_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to          : {summary_path}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Change MODE in config.py to switch between UHCS and NEU
    print(f"Mode    : {config.MODE}")
    print(f"Backbone: {config.BACKBONE}")
    print(f"Device  : {config.DEVICE}")

    model, class_names = load_model()

    if config.MODE == "UHCS":
        test_dir   = config.UHCS_TEST
        output_dir = os.path.join(config.GRADCAM_DIR, "UHCS")
    else:
        test_dir   = config.NEU_TEST
        output_dir = os.path.join(config.GRADCAM_DIR, "NEU")

    records = run_batch(
        model            = model,
        class_names      = class_names,
        test_dir         = test_dir,
        output_dir       = output_dir,
        images_per_class = 3        # change to 5 for more coverage
    )

    print_summary(records, output_dir)
    print("\nGrad-CAM batch complete.")