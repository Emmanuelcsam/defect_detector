"""Apply histogram equalization for contrast enhancement"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.
    
    Args:
        image: Input image
    
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # For color images, convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        # For grayscale images
        return cv2.equalizeHist(image)
