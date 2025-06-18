"""
Auto-generated wrapper for concen_split2
Detected operations: gaussian_blur, threshold, grayscale
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using concen_split2 logic"""
    try:
        # Default implementation - modify based on original script
        result = image.copy()
        
        # Add your processing here based on the original script
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.GaussianBlur(result, (5, 5), 0)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        return result
    except Exception as e:
        print(f"Error in concen_split2: {e}")
        return image
