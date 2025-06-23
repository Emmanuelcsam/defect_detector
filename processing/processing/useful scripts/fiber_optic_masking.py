"""Processed from fiber_optic_masking.py - Detected operations: gaussian_blur, canny_edge, threshold"""
import cv2
import numpy as np
import matplotlib

def process_image(image: np.ndarray, kernel_size: float = 5, sigma: float = 0) -> np.ndarray:
    """
    Processed from fiber_optic_masking.py - Detected operations: gaussian_blur, canny_edge, threshold
    
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
        
        # Apply Canny edge detection
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 50, 150)
        
        # Apply threshold
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
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
