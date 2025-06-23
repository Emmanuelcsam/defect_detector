"""Apply binary threshold to the image"""
import cv2
import numpy as np

def process_image(image: np.ndarray, threshold: int = 127, max_value: int = 255) -> np.ndarray:
    """
    Apply binary threshold to the image.
    
    Args:
        image: Input image
        threshold: Threshold value
        max_value: Maximum value to use with THRESH_BINARY
    
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
    return binary
