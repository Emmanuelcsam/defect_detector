"""Processed from threshold_script.py - Detected operations: threshold, adaptive_threshold, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Processed from threshold_script.py - Detected operations: threshold, adaptive_threshold, grayscale
    
    Args:
        image: Input image
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        # Convert to grayscale if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
