"""Processed from complete_do2mr.py - Detected operations: gaussian_blur, median_blur, threshold"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 7, sigma: float = 1) -> np.ndarray:
    """
    Processed from complete_do2mr.py - Detected operations: gaussian_blur, median_blur, threshold
    
    Args:
        image: Input image
        kernel_size: Kernel size
        sigma: Sigma
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        # Convert to grayscale if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 1)
        
        # Apply median blur
        result = cv2.medianBlur(result, 7)
        
        # Apply threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
