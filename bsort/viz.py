"""Visualization helpers for inspecting YOLO labels."""

from pathlib import Path

import cv2

# Define colors for your classes (B, G, R format)
COLORS = {
    0: (255, 255, 0),  # Cyan/Light Blue for Class 0
    1: (139, 0, 0),  # Dark Blue for Class 1
    2: (0, 0, 255),  # Red for Class 2 (The "Error" / Other class)
}

CLASS_NAMES = {0: "Light Blue", 1: "Dark Blue", 2: "Other"}


def view_labels(image_dir: Path, label_dir: Path):
    """
    Iterates through images and labels, drawing bounding boxes
    to verify correctness.
    """
    print(f"Viewing labels from: {label_dir}")
    print("controls: [Space/n] Next Image | [q] Quit")

    image_files = list(image_dir.glob("*.jpg"))

    if not image_files:
        print("No images found.")
        return

    for img_path in image_files:
        # Find corresponding label file
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape

        # Read the label file
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            cx, cy, bw, bh = parts[1], parts[2], parts[3], parts[4]

            # YOLO to Pixel Conversion
            # x1 = (center_x - width/2) * image_width
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Get color and name
            color = COLORS.get(cls_id, (255, 255, 255))
            label_text = CLASS_NAMES.get(cls_id, str(cls_id))

            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw Label Background and Text
            cv2.rectangle(img, (x1, y1 - 20), (x1 + 100, y1), color, -1)
            cv2.putText(
                img,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Resize for easier viewing if 4k
        display_img = cv2.resize(img, (1024, 768)) if w > 1500 else img

        cv2.imshow("Label Viewer (Press 'n' for next, 'q' to quit)", display_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
