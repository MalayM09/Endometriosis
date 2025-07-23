import numpy as np

def pad_to_square(image: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """
    Pads the input image to a square of the given target_size.
    The image is centered on a black canvas; if the image is larger than the target_size in any dimension,
    it will be cropped in the center.

    Args:
        image (np.ndarray): Input 2D or 3D image array.
        target_size (int): Desired output size for both width and height.

    Returns:
        np.ndarray: Padded (and/or center-cropped) square image.
    """
    # If the image is bigger than target, crop the center
    input_height, input_width = image.shape[:2]

    # Crop if necessary
    if input_height > target_size or input_width > target_size:
        start_y = max(0, (input_height - target_size) // 2)
        start_x = max(0, (input_width - target_size) // 2)
        end_y = start_y + min(input_height, target_size)
        end_x = start_x + min(input_width, target_size)
        image = image[start_y:end_y, start_x:end_x]

    pad_height = target_size - image.shape[0]
    pad_width = target_size - image.shape[1]

    # Compute padding placement
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    pad_widths = (
        (pad_top, pad_bottom),
        (pad_left, pad_right)
    )
    if image.ndim == 3:
        pad_widths += ((0,0),)

    padded_image = np.pad(
        image,
        pad_widths,
        mode='constant',
        constant_values=0
    )
    return padded_image

if __name__ == "__main__":
    import cv2
    import os

    input_dir = "data/preprocessed_normalized/"
    output_dir = "data/padded/"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_UNCHANGED)
            padded = pad_to_square(img, target_size=1024)
            # Ensure output dtype
            padded = padded.astype(img.dtype)
            cv2.imwrite(os.path.join(output_dir, fname), padded)
