"""Processed from zone_generator.py - Detected operations: mask"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 200) -> np.ndarray:
    """
    Processed from zone_generator.py - Detected operations: mask
    
    Args:
        image: Input image
        kernel_size: Kernel size
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
