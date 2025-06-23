"""Processed from heatmap.py - Detected operations: gaussian_blur, canny_edge, sobel_edge"""
import cv2
import numpy as np
import matplotlib
import skimage

def process_image(image: np.ndarray, kernel_size: float = 5, threshold: float = 0, sigma: float = 1.4) -> np.ndarray:
    """
    Processed from heatmap.py - Detected operations: gaussian_blur, canny_edge, sobel_edge
    
    Args:
        image: Input image
        kernel_size: Kernel size
        threshold: Threshold
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
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 1.4)
        
        # Apply Canny edge detection
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 0, 0)
        
        # Apply threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
