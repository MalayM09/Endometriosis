import numpy as np

def percentile_clip(image: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5) -> np.ndarray:
    """
    Clips the input image intensities to the specified percentiles.

    Args:
        image (np.ndarray): Input 2D or 3D array representing an image.
        lower_percentile (float): Lower bound percentile for intensity clipping.
        upper_percentile (float): Upper bound percentile for intensity clipping.

    Returns:
        np.ndarray: Clipped image.
    """
    p_low = np.percentile(image, lower_percentile)
    p_high = np.percentile(image, upper_percentile)
    clipped = np.clip(image, p_low, p_high)
    return clipped

def linear_rescale(image: np.ndarray, out_min: float = 0.0, out_max: float = 255.0) -> np.ndarray:
    """
    Linearly rescales intensities of an image to a specified range.

    Args:
        image (np.ndarray): Input array.
        out_min (float): Minimum output value.
        out_max (float): Maximum output value.

    Returns:
        np.ndarray: Rescaled image.
    """
    in_min = image.min()
    in_max = image.max()
    if in_max - in_min == 0:
        return np.full_like(image, out_min, dtype=np.float32)
    rescaled = (image - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    return rescaled.astype(np.float32)

def normalize_ultrasound_intensity(image: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5) -> np.ndarray:
    """
    Standardizes ultrasound image intensity using percentile clipping and linear rescaling.

    Args:
        image (np.ndarray): Input ultrasound image (2D or 3D numpy array).
        lower_percentile (float): Lower percentile for intensity clipping.
        upper_percentile (float): Upper percentile for intensity clipping.

    Returns:
        np.ndarray: Normalized image in range [0, 255], dtype float32.
    """
    clipped = percentile_clip(image, lower_percentile, upper_percentile)
    normalized = linear_rescale(clipped, 0.0, 255.0)
    return normalized

if __name__ == "__main__":
    import cv2
    import os

    input_dir = "data/preprocessed/"
    output_dir = "data/preprocessed_normalized/"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
            norm_img = normalize_ultrasound_intensity(img)
            norm_img_uint8 = norm_img.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, fname), norm_img_uint8)
