"""Processed from inten_abv100.py - Detected operations: grayscale, mask"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Processed from inten_abv100.py - Detected operations: grayscale, mask
    
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
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
