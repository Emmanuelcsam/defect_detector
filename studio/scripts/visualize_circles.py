"""Processed from visualize_circles.py - Detected operations: circle_detection"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Processed from visualize_circles.py - Detected operations: circle_detection
    
    Args:
        image: Input image
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
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
