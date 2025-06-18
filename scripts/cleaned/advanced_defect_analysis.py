"""
Auto-generated wrapper for advanced_defect_analysis
Detected operations: morphology, grayscale
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using advanced_defect_analysis logic"""
    try:
        # Default implementation - modify based on original script
        result = image.copy()
        
        # Add your processing here based on the original script
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        return result
    except Exception as e:
        print(f"Error in advanced_defect_analysis: {e}")
        return image
