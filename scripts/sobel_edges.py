"""Apply Sobel edge detection"""
import cv2
import numpy as np

def process_image(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Sobel edge detection to find gradients.
    
    Args:
        image: Input image
        ksize: Size of the Sobel kernel
    
    Returns:
        Edge magnitude image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Calculate magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude
