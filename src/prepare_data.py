import os
import re
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from glob import glob

#Paths
BASE_DIR        = "data/raw"

UHCS_IMAGES     = f"{BASE_DIR}/microstructure/images"
UHCS_EXCEL      = f"{BASE_DIR}/microstructure/meta-data.csv"
UHCS_OUT        = "data/processed/UHCS"





NEU_TRAIN       = f"{BASE_DIR}/NEU-DET/train"
NEU_VAL         = f"{BASE_DIR}/NEU-DET/validation"
NEU_OUT         = "data/processed/NEU"


IMAGE_COL       = "path"
LABEL_COL       = "primary_microconstituent"

# ════════════════════════════════════════════════════════════
# PART 1 — UHCS MICROSTRUCTURE DATASET
# ════════════════════════════════════════════════════════════
print("=" * 50)
print("PART 1 — Preparing UHCS Dataset")
print("=" * 50)

df = pd.read_csv(UHCS_EXCEL)

uhcs_file_map = {}
for img_path in glob(os.path.join(UHCS_IMAGES, "**", "*"), recursive=True):
    if os.path.isfile(img_path):
        fname = os.path.basename(img_path).strip().lower()
        uhcs_file_map[fname] = img_path

print(f"Indexed {len(uhcs_file_map)} UHCS image files")
print(f"Columns found     : {df.columns.tolist()}")
print(f"Total images      : {len(df)}")
print(f"Unique classes    : {df[LABEL_COL].dropna().unique()}")

# Clean rows
df = df[[IMAGE_COL, LABEL_COL]].copy()
df = df.dropna(subset=[IMAGE_COL, LABEL_COL])
df[IMAGE_COL] = df[IMAGE_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

# Split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df[LABEL_COL]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df[LABEL_COL]
)

print("\nUHCS Split:")
print(f"  Train : {len(train_df)}")
print(f"  Val   : {len(val_df)}")
print(f"  Test  : {len(test_df)}")


def copy_uhcs(dataframe, split_name):
    copied = 0
    missing = 0
    missing_counts = {}

    for _, row in dataframe.iterrows():
        # Convert Excel path to just filename, then lowercase for matching
        filename = os.path.basename(str(row[IMAGE_COL]).strip()).lower()
        label = str(row[LABEL_COL]).strip().lower()

        if label not in missing_counts:
            missing_counts[label] = 0

        dest_folder = os.path.join(UHCS_OUT, split_name, label)
        os.makedirs(dest_folder, exist_ok=True)

        # Look up real source path from indexed files
        def extract_number(name):
            nums = re.findall(r'\d+', name)
            return nums[0] if nums else None
        file_num = extract_number(filename)

        src = None
        for k, v in uhcs_file_map.items():
            if extract_number(k) == file_num:
                src = v
                break



        dest = os.path.join(dest_folder, filename)

        if src is not None and os.path.exists(src):
            shutil.copy(src, dest)
            copied += 1
        else:
            print(f"Missing: {filename} | label: {label}")
            missing += 1
            missing_counts[label] += 1

    print(f"\n{split_name}: {copied} copied, {missing} missing")
    print("Missing per class:")
    for k, v in missing_counts.items():
        print(f"  {k}: {v}")


copy_uhcs(train_df, "train")
copy_uhcs(val_df, "val")
copy_uhcs(test_df, "test")

print("\nUHCS preparation complete.")




# ════════════════════════════════════════════════════════════
# PART 2 — NEU DEFECT DATASET
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("PART 2 — Preparing NEU Dataset")
print("=" * 50)

def parse_xml(xml_path):
    tree     = ET.parse(xml_path)
    root     = tree.getroot()
    filename = root.find("filename").text
    obj      = root.find("object")
    name     = obj.find("name").text
    bndbox   = obj.find("bndbox")
    xmin     = int(bndbox.find("xmin").text)
    ymin     = int(bndbox.find("ymin").text)
    xmax     = int(bndbox.find("xmax").text)
    ymax     = int(bndbox.find("ymax").text)
    return filename, name, xmin, ymin, xmax, ymax

def process_neu_split(images_dir, annot_dir, out_dir,
                      split_name):
    records = []
    copied  = 0
    missing = 0

    xml_files = glob(os.path.join(annot_dir, "*.xml"))
    if len(xml_files) == 0:
        xml_files = glob(os.path.join(annot_dir, "*.html"))

    print(f"  {split_name}: {len(xml_files)} annotation files found")

    for xml_path in xml_files:
        try:
            filename, label, xmin, ymin, xmax, ymax = parse_xml(xml_path)

            src = os.path.join(images_dir, label, filename)
            if not os.path.exists(src):
                src = os.path.join(images_dir, filename)

            dest_folder = os.path.join(out_dir, split_name,
                                       "images", label)
            os.makedirs(dest_folder, exist_ok=True)
            dest = os.path.join(dest_folder, filename)

            if os.path.exists(src):
                shutil.copy(src, dest)
                copied += 1
                records.append({
                    "filename" : filename,
                    "label"    : label,
                    "xmin"     : xmin,
                    "ymin"     : ymin,
                    "xmax"     : xmax,
                    "ymax"     : ymax,
                    "split"    : split_name
                })
            else:
                missing += 1

        except Exception as e:
            print(f"  Error: {xml_path} — {e}")

    print(f"  {split_name}: {copied} copied, {missing} missing")
    return records

# Process validation set
val_records = process_neu_split(
    images_dir = f"{NEU_VAL}/images",
    annot_dir  = f"{NEU_VAL}/annotations",
    out_dir    = NEU_OUT,
    split_name = "val"
)

# Process train set fully first
train_records = process_neu_split(
    images_dir = f"{NEU_TRAIN}/images",
    annot_dir  = f"{NEU_TRAIN}/annotations",
    out_dir    = NEU_OUT,
    split_name = "train_full"
)

# Split train into 85% train and 15% test
train_df_neu = pd.DataFrame(train_records)

neu_train, neu_test = train_test_split(
    train_df_neu,
    test_size    = 0.15,
    random_state = 42,
    stratify     = train_df_neu["label"]
)

# Copy train split into final train folder
for _, row in neu_train.iterrows():
    src = os.path.join(NEU_OUT, "train_full",
                       "images", row["label"], row["filename"])
    dest_folder = os.path.join(NEU_OUT, "train",
                               "images", row["label"])
    os.makedirs(dest_folder, exist_ok=True)
    dest = os.path.join(dest_folder, row["filename"])
    if os.path.exists(src):
        shutil.copy(src, dest)

# Copy test split into final test folder
for _, row in neu_test.iterrows():
    src = os.path.join(NEU_OUT, "train_full",
                       "images", row["label"], row["filename"])
    dest_folder = os.path.join(NEU_OUT, "test",
                               "images", row["label"])
    os.makedirs(dest_folder, exist_ok=True)
    dest = os.path.join(dest_folder, row["filename"])
    if os.path.exists(src):
        shutil.copy(src, dest)

# Save all annotations to CSV
all_records = (
    val_records +
    neu_train.assign(split="train").to_dict("records") +
    neu_test.assign(split="test").to_dict("records")
)
pd.DataFrame(all_records).to_csv(
    f"{NEU_OUT}/annotations.csv", index=False
)

print(f"\nNEU Split:")
print(f"  Train : {len(neu_train)}")
print(f"  Val   : {len(val_records)}")
print(f"  Test  : {len(neu_test)}")





# ════════════════════════════════════════════════════════════
# PART 3 — VERIFICATION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("PART 3 — Verification")
print("=" * 50)

print("\nUHCS:")
for split in ["train", "val", "test"]:
    path = os.path.join(UHCS_OUT, split)
    if os.path.exists(path):
        for cls in sorted(os.listdir(path)):
            cls_path = os.path.join(path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f"  {split}/{cls}: {count} images")

print("\nNEU:")
for split in ["train", "val", "test"]:
    path = os.path.join(NEU_OUT, split, "images")
    if os.path.exists(path):
        for cls in sorted(os.listdir(path)):
            cls_path = os.path.join(path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f"  {split}/images/{cls}: {count} images")

print("\nAnnotations CSV:", f"{NEU_OUT}/annotations.csv")
print("\nDay 1 Complete.")
