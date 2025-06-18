"""
Detects the fiber optic cladding from a grayscale image.
This script applies contrast enhancement, edge detection, and a Hough Circle Transform
to identify and visualize the main cladding circle.
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  clip_limit: float = 2.0,
                  canny_low: int = 100,
                  canny_high: int = 200,
                  blur_ksize: int = 5,
                  hough_param1: int = 50,
                  hough_param2: int = 40,
                  min_radius: int = 100,
                  max_radius: int = 500) -> np.ndarray:
    """
    Analyzes an image to find and draw the fiber optic cladding.

    Args:
        image: Input image (color or grayscale).
        clip_limit: Contrast limit for CLAHE.
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.
        blur_ksize: Kernel size for Gaussian blur (must be odd).
        hough_param1: Upper threshold for the internal Canny edge detector in HoughCircles.
        hough_param2: Accumulator threshold for circle detection.
        min_radius: Minimum circle radius to detect.
        max_radius: Maximum circle radius to detect.

    Returns:
        A color image with the detected cladding circle drawn on it.
    """
    # 1. Prepare Image
    # Ensure we have a grayscale image for processing and a color image for output
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_image = image.copy()
    else:
        gray = image
        output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. Pre-processing
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # Canny edge detection
    canny_image = cv2.Canny(clahe_image, canny_low, canny_high)

    # Gaussian blur to reduce noise and improve circle detection
    # Ensure kernel size is positive and odd
    blur_ksize = max(1, abs(blur_ksize))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred_image = cv2.GaussianBlur(canny_image, (blur_ksize, blur_ksize), 0)

    # 3. Detect Cladding Circle
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=image.shape[0],
                               param1=hough_param1, param2=hough_param2,
                               minRadius=min_radius, maxRadius=max_radius)

    # 4. Draw Result
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Get the first detected circle
        cladding = circles[0, 0]
        center = (cladding[0], cladding[1])
        radius = cladding[2]

        # Draw the cladding circle
        cv2.circle(output_image, center, radius, (255, 255, 0), 2)  # Cyan
        # Draw the center of the circle
        cv2.circle(output_image, center, 2, (0, 0, 255), 3) # Red

        cv2.putText(output_image, "Cladding Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return output_image