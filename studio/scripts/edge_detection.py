"""Detect edges using Canny edge detection"""
import cv2
import numpy as np

def process_image(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges using Canny edge detection.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
    
    Returns:
        Edge map
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur first to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Canny edge detection
    return cv2.Canny(blurred, low_threshold, high_threshold)
