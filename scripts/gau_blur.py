"""Processed from gau_blur.py - Detected operations: gaussian_blur, canny_edge, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 7, sigma: float = 0) -> np.ndarray:
    """
    Processed from gau_blur.py - Detected operations: gaussian_blur, canny_edge, grayscale
    
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
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
        # Apply Canny edge detection
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 50, 150)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
