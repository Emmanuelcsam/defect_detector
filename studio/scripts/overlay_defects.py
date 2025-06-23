"""Processed from overlay_defects.py - Detected operations: grayscale, mask"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 150) -> np.ndarray:
    """
    Processed from overlay_defects.py - Detected operations: grayscale, mask
    
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
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
