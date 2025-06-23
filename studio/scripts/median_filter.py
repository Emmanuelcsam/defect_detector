"""Apply median filter for noise reduction"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter to reduce noise while preserving edges.
    
    Args:
        image: Input image
        kernel_size: Size of the median filter kernel
    
    Returns:
        Filtered image
    """
    return cv2.medianBlur(image, kernel_size)
