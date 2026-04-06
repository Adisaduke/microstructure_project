# dataset.py

import config
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np

# ── UHCS Transforms ───────────────────────────────────────────
# Microstructure specific — grayscale, contrast, full rotation
uhcs_train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=180),
    transforms.RandomAutocontrast(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORMALIZE_MEAN,
                         std =config.NORMALIZE_STD)
])

uhcs_val_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORMALIZE_MEAN,
                         std =config.NORMALIZE_STD)
])

# ── NEU Transforms ────────────────────────────────────────────
# Surface defect images — RGB, standard augmentation
neu_train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORMALIZE_MEAN,
                         std =config.NORMALIZE_STD)
])

neu_val_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORMALIZE_MEAN,
                         std =config.NORMALIZE_STD)
])

# ── UHCS DataLoaders ──────────────────────────────────────────


def get_uhcs_loaders():

    # Get labels
    targets = [label for _, label in train_dataset.samples]

# Count class frequency
    class_count = np.bincount(targets)

# Compute class weights
    class_weights = 1. / class_count

# Assign weight to each sample
    sample_weights = [class_weights[label] for label in targets]

# Create sampler
    sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True)


    train_dataset = datasets.ImageFolder(
        root      = config.UHCS_TRAIN,
        transform = uhcs_train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root      = config.UHCS_VAL,
        transform = uhcs_val_transforms
    )
    test_dataset = datasets.ImageFolder(
        root      = config.UHCS_TEST,
        transform = uhcs_val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        sampler     = sampler ,
        num_workers = config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS
    )

    print(f"UHCS Classes    : {train_dataset.classes}")
    print(f"UHCS Train size : {len(train_dataset)}")
    print(f"UHCS Val size   : {len(val_dataset)}")
    print(f"UHCS Test size  : {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset.classes


# ── NEU DataLoaders ───────────────────────────────────────────
def get_neu_loaders():
    train_dataset = datasets.ImageFolder(
        root      = config.NEU_TRAIN,
        transform = neu_train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root      = config.NEU_VAL,
        transform = neu_val_transforms
    )
    test_dataset = datasets.ImageFolder(
        root      = config.NEU_TEST,
        transform = neu_val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS
    )

    print(f"NEU Classes    : {train_dataset.classes}")
    print(f"NEU Train size : {len(train_dataset)}")
    print(f"NEU Val size   : {len(val_dataset)}")
    print(f"NEU Test size  : {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset.classes


# ── Quick Test — run this file directly to verify ─────────────
if __name__ == "__main__":
    print("Testing UHCS loaders...")
    train_loader, val_loader, test_loader, classes = get_uhcs_loaders()
    images, labels = next(iter(train_loader))
    print(f"UHCS batch shape : {images.shape}")
    print(f"UHCS label shape : {labels.shape}")

    print("\nTesting NEU loaders...")
    train_loader, val_loader, test_loader, classes = get_neu_loaders()
    images, labels = next(iter(train_loader))
    print(f"NEU batch shape  : {images.shape}")
    print(f"NEU label shape  : {labels.shape}")

    print("\ndataset.py verified successfully.")
