"""Interactive HSV tuner for determining color thresholds."""

from pathlib import Path

import cv2
import numpy as np


def nothing(_: int) -> None:
    """Dummy callback for OpenCV trackbars."""
    return None


def run_tuner_ui(image_path: Path):
    """
    Opens an interactive window to tune HSV values.
    """
    # Convert Path object to string for OpenCV
    str_path = str(image_path)
    img = cv2.imread(str_path)

    if img is None:
        print(f"Could not load image at {str_path}")
        return

    # Resize for easier viewing if image is huge
    img = cv2.resize(img, (640, 640))

    # Create a window
    window_name = "HSV Tuner (Press q to quit)"
    cv2.namedWindow(window_name)

    # Create trackbars
    # Range: Hue (0-179), Sat (0-255), Val (0-255)
    cv2.createTrackbar("H Min", window_name, 0, 179, nothing)
    cv2.createTrackbar("S Min", window_name, 0, 255, nothing)
    cv2.createTrackbar("V Min", window_name, 0, 255, nothing)
    cv2.createTrackbar("H Max", window_name, 179, 179, nothing)
    cv2.createTrackbar("S Max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V Max", window_name, 255, 255, nothing)

    print(f"Tuning on: {image_path.name}")
    print("Adjust sliders until the bottle cap is WHITE and background is BLACK.")
    print("Press 'q' to close the window.")

    while True:
        # 1. Get current positions of all trackbars
        h_min = cv2.getTrackbarPos("H Min", window_name)
        s_min = cv2.getTrackbarPos("S Min", window_name)
        v_min = cv2.getTrackbarPos("V Min", window_name)
        h_max = cv2.getTrackbarPos("H Max", window_name)
        s_max = cv2.getTrackbarPos("S Max", window_name)
        v_max = cv2.getTrackbarPos("V Max", window_name)

        # 2. Create the HSV mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # 3. Create a preview (Original Image AND the result combined)
        # Apply mask to original image to see the "cut out" effect
        result = cv2.bitwise_and(img, img, mask=mask)

        # Convert mask to 3 channels so we can stack it side-by-side with color images
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Stack: Original | Mask (Black/White) | Result (Color cut-out)
        # We scale them down slightly to fit 3 in a row
        stacked = np.hstack((img, mask_3ch, result))

        # Show the result
        cv2.imshow(window_name, stacked)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nFinal values found:")
            print(f"   Lower Limit: ({h_min}, {s_min}, {v_min})")
            print(f"   Upper Limit: ({h_max}, {s_max}, {v_max})")
            print("Copy these Hue values (first number) to your configuration!")
            break

    cv2.destroyAllWindows()
