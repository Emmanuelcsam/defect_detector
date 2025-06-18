"""Apply morphological closing to fill gaps"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological closing to fill small gaps.
    
    Args:
        image: Input image
        kernel_size: Size of the structuring element
    
    Returns:
        Processed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
