"""Processed from thresh_binary_mask.py - Detected operations: canny_edge, threshold, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray, threshold: float = 128) -> np.ndarray:
    """
    Processed from thresh_binary_mask.py - Detected operations: canny_edge, threshold, grayscale
    
    Args:
        image: Input image
        threshold: Threshold
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        # Convert to grayscale if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 128, 384)
        
        # Apply threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
