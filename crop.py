import cv2
import numpy as np
from typing import Tuple

def auto_detect_roi(image: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
    """
    Automatically detects the largest rectangular ROI with content above a pixel threshold.
    Useful for removing black borders or flat overlays.
    
    Args:
        image (np.ndarray): Input image (grayscale or RGB).
        threshold (int): Minimum intensity for considering a pixel as part of the ROI.

    Returns:
        (x_min, y_min, x_max, y_max): Bounding box coordinates of detected ROI.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # Threshold the grayscale image to find non-black pixels
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0, 0, image.shape[1], image.shape[0])  # return full image if no content
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # Add a small margin for safety
    margin = 5
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)
    return (x_min, y_min, x_max, y_max)

def crop_image(image: np.ndarray, roi: Tuple[int, int, int, int] = None) -> np.ndarray:
    """
    Crops the image using the provided ROI or automatically detected ROI.

    Args:
        image (np.ndarray): Input image.
        roi (tuple, optional): (x_min, y_min, x_max, y_max). If None, auto-detect.

    Returns:
        np.ndarray: Cropped image.
    """
    if roi is None:
        roi = auto_detect_roi(image)
    x_min, y_min, x_max, y_max = roi
    return image[y_min:y_max, x_min:x_max]

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # Example batch process
    images_dir = "data/raw/"
    cropped_dir = "data/preprocessed/"

    os.makedirs(cropped_dir, exist_ok=True)

    for fname in os.listdir(images_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(images_dir, fname))
            cropped = crop_image(img)
            cv2.imwrite(os.path.join(cropped_dir, fname), cropped)

            # Visualization: original vs cropped
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original")
            axes[0].axis('off')
            axes[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Cropped")
            axes[1].axis('off')
            plt.show()
