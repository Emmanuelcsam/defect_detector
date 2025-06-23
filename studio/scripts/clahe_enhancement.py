"""Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
import cv2
import numpy as np

def process_image(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE for local contrast enhancement.
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)
