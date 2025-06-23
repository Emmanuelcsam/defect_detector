"""Processed from analyze_core_script.py - Detected operations: gaussian_blur, circle_detection, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 5, sigma: float = 0) -> np.ndarray:
    """
    Processed from analyze_core_script.py - Detected operations: gaussian_blur, circle_detection, grayscale
    
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
        
        # Detect circles
        if len(result.shape) == 2:
            display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display = result.copy()
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY) if len(display.shape) == 3 else display
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display, (i[0], i[1]), i[2], (0, 255, 0), 2)
        result = display
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
