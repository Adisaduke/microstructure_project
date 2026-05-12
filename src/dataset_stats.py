# dataset_stats.py
import os

print("=" * 55)
print("UHCS DATASET STATISTICS")
print("=" * 55)

UHCS_BASE = "data/processed/UHCS"
total_uhcs = 0

for split in ["train", "val", "test"]:
    split_path = os.path.join(UHCS_BASE, split)
    split_total = 0
    print(f"\n{split.upper()} SET:")
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            split_total += count
            print(f"  {cls:<35} : {count}")
    print(f"  {'TOTAL':<35} : {split_total}")
    total_uhcs += split_total

print(f"\nUHCS GRAND TOTAL : {total_uhcs}")

print("\n" + "=" * 55)
print("NEU DATASET STATISTICS")
print("=" * 55)

NEU_BASE  = "data/processed/NEU"
total_neu = 0

for split in ["train", "val", "test"]:
    split_path = os.path.join(NEU_BASE, split, "images")
    if not os.path.exists(split_path):
        continue
    split_total = 0
    print(f"\n{split.upper()} SET:")
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            split_total += count
            print(f"  {cls:<35} : {count}")
    print(f"  {'TOTAL':<35} : {split_total}")
    total_neu += split_total

print(f"\nNEU GRAND TOTAL : {total_neu}")
print("\nDataset stats complete.")