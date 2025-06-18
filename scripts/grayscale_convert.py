"""Convert image to grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale using OpenCV's color conversion.
    
    Args:
        image: Input image
    
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
