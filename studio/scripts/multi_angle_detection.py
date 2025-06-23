"""Processed from multi_angle_detection.py - Detected operations: threshold, histogram, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 3) -> np.ndarray:
    """
    Processed from multi_angle_detection.py - Detected operations: threshold, histogram, grayscale
    
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
        
        # Apply threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
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
