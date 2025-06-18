"""
Auto-generated wrapper for split
Detected operations: edge_detection, circle_detection, median_filter, sobel_edge, clahe
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using split logic"""
    try:
        # Default implementation - modify based on original script
        result = image.copy()
        
        # Add your processing here based on the original script
        result = cv2.Canny(result, 50, 150)
        
        return result
    except Exception as e:
        print(f"Error in split: {e}")
        return image
