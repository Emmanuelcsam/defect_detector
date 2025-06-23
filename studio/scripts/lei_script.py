"""Processed from lei_script.py - Detected operations: threshold, adaptive_threshold, morphology"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 8, clip_limit: float = 3) -> np.ndarray:
    """
    Processed from lei_script.py - Detected operations: threshold, adaptive_threshold, morphology
    
    Args:
        image: Input image
        kernel_size: Kernel size
        clip_limit: Clip limit
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        # Convert to grayscale if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
        if len(result.shape) == 3:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = clahe.apply(result)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
