"""Processed from clahe_script.py - Detected operations: clahe, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 8, clip_limit: float = 2) -> np.ndarray:
    """
    Processed from clahe_script.py - Detected operations: clahe, grayscale
    
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
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
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
