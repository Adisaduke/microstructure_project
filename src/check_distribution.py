import os

BASE_PATH = "data/processed/UHCS"

splits = ["train", "val", "test"]

print("=" * 50)
print("UHCS CLASS DISTRIBUTION")
print("=" * 50)

for split in splits:
    print(f"\n{split.upper()} SET:")
    print("-" * 30)

    split_path = os.path.join(BASE_PATH, split)

    total = 0
    class_counts = {}

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            class_counts[cls] = count
            total += count

    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls:30} : {count}")

    print(f"\nTotal: {total}")

