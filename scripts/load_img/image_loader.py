"""Processed from image_loader.py - Detected operations: gaussian_blur, morphology, clahe"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 8, sigma: float = 0) -> np.ndarray:
    """
    Processed from image_loader.py - Detected operations: gaussian_blur, morphology, clahe
    
    Args:
        image: Input image
        kernel_size: Kernel size
        sigma: Sigma
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        # Convert to grayscale if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
        # Apply morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
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
