"""Processed from max_filter.py - Detected operations: morphology, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 7) -> np.ndarray:
    """
    Processed from max_filter.py - Detected operations: morphology, grayscale
    
    Args:
        image: Input image
        kernel_size: Kernel size
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        # Convert to grayscale if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
