"""Image augmentation utilities for the bottle-cap dataset."""

from pathlib import Path

import cv2


def augment_data(image_dir: Path, label_dir: Path, output_dir: Path) -> None:
    """
    Generates new training data by rotating and flipping existing images.
    Turns 1 image into 4 (Original, Rot90, Rot180, Flip).
    """
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    print(f"Augmenting data from {image_dir}...")

    count = 0
    for img_path in image_dir.glob("*.jpg"):
        stem = img_path.stem
        # 1. Read Original
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        label_path = label_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]

        # --- Helper to save ---
        def save_variant(base_name: str, suffix: str, image, new_labels) -> None:
            # Save Image
            new_name = f"{base_name}_{suffix}"
            cv2.imwrite(str(output_images / f"{new_name}.jpg"), image)

            # Save Label
            with open(
                output_labels / f"{new_name}.txt",
                "w",
                encoding="utf-8",
            ) as file_handle:
                for l in new_labels:
                    file_handle.write(
                        f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n"
                    )

        # --- VARIANT 1: Original (Just Copy) ---
        save_variant(stem, "orig", img, labels)

        # --- VARIANT 2: Flip Horizontal ---
        # Math: New X = 1 - Old X
        img_flip = cv2.flip(img, 1)
        labels_flip = []
        for cls, x, y, bw, bh in labels:
            labels_flip.append([cls, 1.0 - x, y, bw, bh])
        save_variant(stem, "flip", img_flip, labels_flip)

        # --- VARIANT 3: Rotate 180 ---
        # Math: New X = 1 - Old X, New Y = 1 - Old Y
        img_180 = cv2.rotate(img, cv2.ROTATE_180)
        labels_180 = []
        for cls, x, y, bw, bh in labels:
            labels_180.append([cls, 1.0 - x, 1.0 - y, bw, bh])
        save_variant(stem, "rot180", img_180, labels_180)

        # --- VARIANT 4: Rotate 90 Clockwise ---
        # Math: New X = 1 - Old Y, New Y = Old X, Swap W/H
        img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        labels_90 = []
        for cls, x, y, bw, bh in labels:
            labels_90.append([cls, 1.0 - y, x, bh, bw])  # Note bw/bh swap
        save_variant(stem, "rot90", img_90, labels_90)

        count += 1

    print(f"Augmentation complete. Turned {count} images into {count * 4} images.")
