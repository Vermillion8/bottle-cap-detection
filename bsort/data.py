"""Data utilities for relabeling and splitting bottle-cap datasets."""

import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# --- Configuration ---
# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define your new class names
CLASS_NAMES = {0: "light_blue", 1: "dark_blue", 2: "other"}

HSV_RANGES = {
    "light_blue": (75, 99),
    "dark_blue": (100, 120),
}
# --- End Configuration ---


def get_dominant_color_hsv(image: np.ndarray) -> float:
    """
    Finds the dominant Hue value in a cropped image.

    Args:
        image: A BGR image crop (from cv2) containing one bottle cap.

    Returns:
        The most frequent (modal) Hue value.
    """
    # Convert the BGR crop to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the Hue channel
    # hsv_image is (height, width, 3) -> Hue is the 0th channel
    hue_channel = hsv_image[:, :, 0]

    # Calculate a histogram for the Hue channel
    # 180 bins because Hue in OpenCV ranges from 0-179
    hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])

    # Find the Hue value with the highest frequency (the "mode")
    dominant_hue = np.argmax(hist)

    return float(dominant_hue)


def classify_hue(hue: float) -> int:
    """
    Classifies a Hue value into one of the new classes.

    Args:
        hue: The Hue value (0-179)

    Returns:
        The new class ID (0: light_blue, 1: dark_blue, 2: other)
    """
    if HSV_RANGES["light_blue"][0] <= hue <= HSV_RANGES["light_blue"][1]:
        return 0  # light_blue
    if HSV_RANGES["dark_blue"][0] <= hue <= HSV_RANGES["dark_blue"][1]:
        return 1  # dark_blue
    return 2  # other


def process_data(image_dir: Path, label_dir: Path, output_label_dir: Path) -> None:
    """
    Processes all images and labels, analyzes cap colors, and writes
    new, corrected label files.

    Args:
        image_dir: Path to the directory containing images.
        label_dir: Path to the directory with original YOLO labels.
        output_label_dir: Path to save the new, corrected labels.
    """
    logging.info("Starting data processing...")
    logging.info(f"Image source: {image_dir}")
    logging.info(f"Label source: {label_dir}")
    logging.info(f"Outputting new labels to: {output_label_dir}")

    # Ensure output directory exists
    output_label_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    # Iterate over all original label files
    for label_file in label_dir.glob("*.txt"):
        image_file = image_dir / f"{label_file.stem}.jpg"  # Assumes .jpg

        if not image_file.exists():
            logging.warning(f"No matching image for {label_file.name}, skipping.")
            skipped_count += 1
            continue

        # Load the image
        image = cv2.imread(str(image_file))
        if image is None:
            logging.error(f"Failed to read image {image_file}, skipping.")
            skipped_count += 1
            continue

        img_height, img_width, _ = image.shape
        new_labels = []

        # Read the original bounding boxes
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                try:
                    # Original class is '0', we ignore it
                    _, x_center, y_center, width, height = map(
                        float, line.strip().split()
                    )
                except ValueError:
                    logging.warning(f"Skipping malformed line in {label_file.name}")
                    continue

                # 1. Convert YOLO coords back to pixel (x, y) coords
                # Top-left corner (x1, y1)
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                # Bottom-right corner (x2, y2)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                # 2. Crop the image to the bounding box
                # Add padding to avoid cutting off edges, but clip at image bounds
                pad = 5  # 5-pixel padding
                y1_pad = max(0, y1 - pad)
                y2_pad = min(img_height, y2 + pad)
                x1_pad = max(0, x1 - pad)
                x2_pad = min(img_width, x2 + pad)

                cap_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

                if cap_crop.size == 0:
                    logging.warning(f"Empty crop in {label_file.name}, skipping box.")
                    continue

                # 3. Analyze the color and get new class ID
                dominant_hue = get_dominant_color_hsv(cap_crop)
                new_class_id = classify_hue(dominant_hue)

                # 4. Store the new label line
                new_labels.append(
                    f"{new_class_id} {x_center} {y_center} {width} {height}"
                )

        # 5. Write the new label file
        if new_labels:
            output_file_path = output_label_dir / label_file.name
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_labels))
            processed_count += 1
        else:
            logging.warning(f"No valid objects found in {label_file.name}")
            skipped_count += 1

    logging.info("--- Processing Complete ---")
    logging.info(f"Processed and wrote {processed_count} new label files.")
    logging.info(f"Skipped {skipped_count} files (missing images/errors).")


def split_dataset(
    image_dir: Path, label_dir: Path, output_dir: Path, train_ratio: float = 0.8
):
    """
    Splits data into train/val sets and organizes them for YOLO.
    """
    # 1. Setup Directories
    # defined structure: output/images/train, output/labels/val, etc.
    dirs = {
        "images_train": output_dir / "images" / "train",
        "images_val": output_dir / "images" / "val",
        "labels_train": output_dir / "labels" / "train",
        "labels_val": output_dir / "labels" / "val",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # 2. Get all pairs
    # We assume if image exists, label might exist (we verify)
    image_files = list(image_dir.glob("*.jpg"))
    random.shuffle(image_files)  # Shuffle to ensure random distribution

    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Splitting: {len(train_files)} Train / {len(val_files)} Val")

    def copy_pair(files, kind):
        for img_src in files:
            # Determine destination
            img_dest = dirs[f"images_{kind}"] / img_src.name
            label_src = label_dir / f"{img_src.stem}.txt"
            label_dest = dirs[f"labels_{kind}"] / label_src.name

            # Copy Image
            shutil.copy2(img_src, img_dest)

            # Copy Label (if it exists)
            if label_src.exists():
                shutil.copy2(label_src, label_dest)
            else:
                print(f"Warning: Missing label for {img_src.name}")

    # 3. Execute Copy
    copy_pair(train_files, "train")
    copy_pair(val_files, "val")

    print(f"Dataset ready at: {output_dir}")
