"""Detect and highlight circles in the image"""
import cv2
import numpy as np

def process_image(image: np.ndarray, min_radius: int = 10, max_radius: int = 0) -> np.ndarray:
    """
    Detect circles using Hough Circle Transform.
    
    Args:
        image: Input image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius (0 for no limit)
    
    Returns:
        Image with detected circles highlighted
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = image.copy()
    else:
        gray = image
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )
    
    # Draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw circle outline
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center point
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return output
