"""Processed from overlay_heatmap.py - Detected operations: canny_edge, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Processed from overlay_heatmap.py - Detected operations: canny_edge, grayscale
    
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
        
        # Apply Canny edge detection
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 50, 150)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
