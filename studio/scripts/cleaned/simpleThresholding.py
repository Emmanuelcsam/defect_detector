"""Processed from simpleThresholding.py - Detected operations: threshold, grayscale"""
import cv2
import numpy as np
import matplotlib

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Processed from simpleThresholding.py - Detected operations: threshold, grayscale
    
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
        
        # Apply threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
