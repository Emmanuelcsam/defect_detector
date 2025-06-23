"""Processed from save mask.py - Detected operations: mask"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Processed from save mask.py - Detected operations: mask
    
    Args:
        image: Input image
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
