"""
Auto-generated wrapper for mask_seperation
Detected operations: threshold, morphology, grayscale
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using mask_seperation logic"""
    try:
        # Default implementation - modify based on original script
        result = image.copy()
        
        # Add your processing here based on the original script
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        return result
    except Exception as e:
        print(f"Error in mask_seperation: {e}")
        return image
