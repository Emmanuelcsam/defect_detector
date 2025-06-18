"""Processed from scratch_strength.py - Detected operations: histogram, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Processed from scratch_strength.py - Detected operations: histogram, grayscale
    
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
        
        # Apply histogram equalization
        if len(result.shape) == 3:
            ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            result = cv2.equalizeHist(result)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
