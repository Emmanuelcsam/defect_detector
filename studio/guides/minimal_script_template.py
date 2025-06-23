"""
Minimal Script Template - Copy this to start a new script!
Replace this description with what your script does
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Brief description of what this function does.
    
    Args:
        image: Input image
        
    Returns:
        Processed image
    """
    # Create a copy to avoid modifying the original
    result = image.copy()
    
    # ==================================
    # YOUR PROCESSING CODE GOES HERE
    # ==================================
    
    # Example: Simple brightness adjustment
    result = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
    
    # ==================================
    # END OF YOUR PROCESSING CODE
    # ==================================
    
    return result
