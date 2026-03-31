# dataset.py

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from config import (
    UHCS_TRAIN,
    UHCS_VAL,
    UHCS_TEST,
    UHCS_CLASSES,
    NEU_NUM_CLASSES,
    NEU_TRAIN,
    NEU_VAL,
    NEU_TEST,
    NEU_ANNOTATIONS,
    NEU_CLASSES,
    NEU_NUM_CLASSES,
    IMG_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)


# =========================================================
# TRANSFORMS
# =========================================================
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


# =========================================================
# UHCS DATASET
# Folder structure:
# data/processed/UHCS/train/class_name/image.png
# data/processed/UHCS/val/class_name/image.png
# data/processed/UHCS/test/class_name/image.png
# =========================================================
class UHCSDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform

        # Use config class order if provided, else read from folders
        if class_names is None:
            self.class_names = sorted(
                [d for d in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, d))]
            )
        else:
            self.class_names = class_names

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.idx_to_class = {
            idx: class_name for class_name, idx in self.class_to_idx.items()
        }

        self.samples = []

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        for class_name in self.class_names:
            class_folder = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_folder):
                continue

            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith(valid_exts):
                    image_path = os.path.join(class_folder, file_name)
                    label_idx = self.class_to_idx[class_name]
                    self.samples.append((image_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# NEU CLASSIFICATION DATASET
# Folder structure:
# data/processed/NEU/train/images/class_name/image.jpg
# data/processed/NEU/val/images/class_name/image.jpg
# data/processed/NEU/test/images/class_name/image.jpg
#
# This is for classification first.
# =========================================================
class NEUClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform

        if class_names is None:
            self.class_names = sorted(
                [d for d in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, d))]
            )
        else:
            self.class_names = class_names

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.idx_to_class = {
            idx: class_name for class_name, idx in self.class_to_idx.items()
        }

        self.samples = []

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        for class_name in self.class_names:
            class_folder = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_folder):
                continue

            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith(valid_exts):
                    image_path = os.path.join(class_folder, file_name)
                    label_idx = self.class_to_idx[class_name]
                    self.samples.append((image_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# NEU DETECTION / LOCALIZATION DATASET
# Uses annotations.csv
#
# Expected columns in annotations.csv:
# filename, label, xmin, ymin, xmax, ymax, split
#
# Returns:
# image, target
#
# target contains:
# - boxes
# - labels
# - image_id
# - area
# - iscrowd
# =========================================================
class NEUDetectionDataset(Dataset):
    def __init__(self, images_root, annotations_csv, split, transform=None,
                 class_names=None):
        self.images_root = images_root
        self.annotations_csv = annotations_csv
        self.split = split
        self.transform = transform

        self.df = pd.read_csv(annotations_csv)
        self.df = self.df[self.df["split"] == split].copy().reset_index(drop=True)

        if class_names is None:
            self.class_names = sorted(self.df["label"].unique().tolist())
        else:
            self.class_names = class_names

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        # Unique images only
        self.image_records = self.df[["filename", "label"]].drop_duplicates().reset_index(drop=True)

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, idx):
        row = self.image_records.iloc[idx]
        filename = row["filename"]
        label_name = row["label"]

        image_path = os.path.join(self.images_root, label_name, filename)
        image = Image.open(image_path).convert("RGB")

        # All annotation rows for this image
        ann_rows = self.df[
            (self.df["filename"] == filename) &
            (self.df["label"] == label_name)
        ]

        boxes = []
        labels = []
        areas = []

        for _, ann in ann_rows.iterrows():
            xmin = float(ann["xmin"])
            ymin = float(ann["ymin"])
            xmax = float(ann["xmax"])
            ymax = float(ann["ymax"])

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[ann["label"]])

            area = (xmax - xmin) * (ymax - ymin)
            areas.append(area)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
            "filename": filename,
        }

        if self.transform:
            image = self.transform(image)

        return image, target


# =========================================================
# HELPERS TO BUILD DATASETS FAST
# =========================================================
def build_uhcs_datasets():
    train_dataset = UHCSDataset(
        root_dir=UHCS_TRAIN,
        transform=get_train_transforms(),
        class_names=UHCS_CLASSES
    )

    val_dataset = UHCSDataset(
        root_dir=UHCS_VAL,
        transform=get_val_test_transforms(),
        class_names=UHCS_CLASSES
    )

    test_dataset = UHCSDataset(
        root_dir=UHCS_TEST,
        transform=get_val_test_transforms(),
        class_names=UHCS_CLASSES
    )

    return train_dataset, val_dataset, test_dataset


def build_neu_classification_datasets():
    train_dataset = NEUClassificationDataset(
        root_dir=NEU_TRAIN,
        transform=get_train_transforms(),
        class_names=NEU_CLASSES
    )

    val_dataset = NEUClassificationDataset(
        root_dir=NEU_VAL,
        transform=get_val_test_transforms(),
        class_names=NEU_CLASSES
    )

    test_dataset = NEUClassificationDataset(
        root_dir=NEU_TEST,
        transform=get_val_test_transforms(),
        class_names=NEU_CLASSES
    )

    return train_dataset, val_dataset, test_dataset


def build_neu_detection_datasets():
    train_dataset = NEUDetectionDataset(
        images_root=NEU_TRAIN,
        annotations_csv=NEU_ANNOTATIONS,
        split="train",
        transform=get_val_test_transforms(),
        class_names=NEU_CLASSES
    )

    val_dataset = NEUDetectionDataset(
        images_root=NEU_VAL,
        annotations_csv=NEU_ANNOTATIONS,
        split="val",
        transform=get_val_test_transforms(),
        class_names=NEU_CLASSES
    )

    test_dataset = NEUDetectionDataset(
        images_root=NEU_TEST,
        annotations_csv=NEU_ANNOTATIONS,
        split="test",
        transform=get_val_test_transforms(),
        class_names=NEU_CLASSES
    )

    return train_dataset, val_dataset, test_dataset


# =========================================================
# QUICK TEST
# Run this file directly to verify datasets
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING UHCS DATASET")
    print("=" * 60)
    uhcs_train, uhcs_val, uhcs_test = build_uhcs_datasets()

    print("UHCS Train size:", len(uhcs_train))
    print("UHCS Val size  :", len(uhcs_val))
    print("UHCS Test size :", len(uhcs_test))

    image, label = uhcs_train[0]
    print("UHCS sample image shape:", image.shape)
    print("UHCS sample label index:", label)
    print("UHCS sample label name :", UHCS_CLASSES[label])

    print("\n" + "=" * 60)
    print("TESTING NEU CLASSIFICATION DATASET")
    print("=" * 60)
    neu_train, neu_val, neu_test = build_neu_classification_datasets()

    print("NEU Train size:", len(neu_train))
    print("NEU Val size  :", len(neu_val))
    print("NEU Test size :", len(neu_test))

    image, label = neu_train[0]
    print("NEU sample image shape:", image.shape)
    print("NEU sample label index:", label)
    print("NEU sample label name :", NEU_CLASSES[label])

    print("\n" + "=" * 60)
    print("TESTING NEU DETECTION DATASET")
    print("=" * 60)
    neu_det_train, neu_det_val, neu_det_test = build_neu_detection_datasets()

    print("NEU Detection Train size:", len(neu_det_train))
    print("NEU Detection Val size  :", len(neu_det_val))
    print("NEU Detection Test size :", len(neu_det_test))

    image, target = neu_det_train[0]
    print("Detection sample image shape:", image.shape)
    print("Detection sample target keys:", target.keys())
    print("Boxes:", target["boxes"])
    print("Labels:", target["labels"])
