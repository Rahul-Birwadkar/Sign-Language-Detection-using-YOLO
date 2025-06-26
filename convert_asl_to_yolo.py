"""
Convert ASL alphabet classification dataset into YOLO object detection format.
Only uses 5,000 images (randomly sampled) for faster training.
"""

import os
import cv2
import shutil
import random
from sklearn.model_selection import train_test_split

# pylint: disable=no-member

# === PATHS ===
INPUT_PATH = r"D:\Project\sign_language_yolo\Dataset\asl_alphabet_train\asl_alphabet_train"
OUTPUT_PATH = r"D:\Project\sign_language_yolo\Dataset\dataset"

IMAGE_OUTPUT_TRAIN = os.path.join(OUTPUT_PATH, 'images', 'train')
IMAGE_OUTPUT_VAL = os.path.join(OUTPUT_PATH, 'images', 'val')
LABEL_OUTPUT_TRAIN = os.path.join(OUTPUT_PATH, 'labels', 'train')
LABEL_OUTPUT_VAL = os.path.join(OUTPUT_PATH, 'labels', 'val')

# === Create output folders ===
for folder in [IMAGE_OUTPUT_TRAIN, IMAGE_OUTPUT_VAL, LABEL_OUTPUT_TRAIN, LABEL_OUTPUT_VAL]:
    os.makedirs(folder, exist_ok=True)

# === Create class mapping ===
class_names = sorted(os.listdir(INPUT_PATH))
class_to_id = {cls_name: idx for idx, cls_name in enumerate(class_names)}

# === Gather image paths and class labels ===
all_data = []
for class_name in class_names:
    class_dir = os.path.join(INPUT_PATH, class_name)
    for file in os.listdir(class_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(class_dir, file)
            all_data.append((full_path, class_to_id[class_name]))

# === Limit to 5,000 images ===
random.seed(42)
random.shuffle(all_data)
all_data = all_data[:5000]  # Use only 5,000 images total

# === Split into train/val ===
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

# === Save YOLO labels and copy images ===
def save_yolo_format(data_list, image_dir, label_dir):
    """Save images and YOLO-format labels."""
    for img_path, class_id in data_list:
        img = cv2.imread(img_path)
        if img is None:
            continue

        filename = os.path.basename(img_path)
        label_name = os.path.splitext(filename)[0] + ".txt"

        # Save image
        shutil.copy(img_path, os.path.join(image_dir, filename))

        # Save label (entire image as a single bounding box)
        label_path = os.path.join(label_dir, label_name)
        with open(label_path, "w", encoding="utf-8") as label_file:
            label_file.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Convert data
save_yolo_format(train_data, IMAGE_OUTPUT_TRAIN, LABEL_OUTPUT_TRAIN)
save_yolo_format(val_data, IMAGE_OUTPUT_VAL, LABEL_OUTPUT_VAL)

# === Create dataset.yaml ===
yaml_path = os.path.join(OUTPUT_PATH, 'dataset.yaml')
with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
    yaml_file.write(f"path: {OUTPUT_PATH}\n")
    yaml_file.write("train: images/train\n")
    yaml_file.write("val: images/val\n")
    yaml_file.write("names:\n")
    for class_id, class_name in enumerate(class_names):
        yaml_file.write(f"  {class_id}: {class_name}\n")

print(" Dataset successfully converted to YOLO format (5,000 images).")
