"""
Detects the outer cladding of a fiber optic cable using Hough Circle Transform,
draws the detected circle, and optionally extracts the circular region.
"""
import cv2
import numpy as np

def process_image(image: np.ndarray, 
                  blur_ksize: int = 5,
                  hough_dp: float = 1.0,
                  hough_minDist: float = 20.0,
                  hough_param1: float = 50.0,
                  hough_param2: float = 40.0,
                  min_radius: int = 100,
                  max_radius: int = 500,
                  draw_circle: bool = True,
                  extract_circle: bool = True) -> np.ndarray:
    """
    Applies Median Blur, detects the largest circle (cladding), and visualizes it.
    
    Args:
        image: Input image (color or grayscale).
        blur_ksize: Kernel size for the Median Blur filter. Must be odd.
        hough_dp: Inverse ratio of the accumulator resolution to the image resolution.
        hough_minDist: Minimum distance between the centers of the detected circles.
        hough_param1: Upper threshold for the internal Canny edge detector.
        hough_param2: Threshold for center detection.
        min_radius: Minimum circle radius to detect.
        max_radius: Maximum circle radius to detect.
        draw_circle: If True, draws the detected circle and its center.
        extract_circle: If True, blacks out the area outside the detected circle.
        
    Returns:
        The processed image with the cladding circle drawn and/or extracted.
    """
    # 1. PREPARE THE IMAGE
    # Ensure we have a grayscale image for circle detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Create a color image for drawing the output
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. APPLY PRE-PROCESSING
    # Ensure kernel size is positive and odd for median blur
    blur_ksize = max(1, abs(blur_ksize))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    
    blurred_gray = cv2.medianBlur(gray, blur_ksize)

    # 3. DETECT CIRCLES
    circles = cv2.HoughCircles(
        blurred_gray, 
        cv2.HOUGH_GRADIENT,
        dp=hough_dp, 
        minDist=hough_minDist,
        param1=hough_param1, 
        param2=hough_param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    # 4. PROCESS AND VISUALIZE THE DETECTED CIRCLE
    if circles is not None:
        # Find the largest circle from the detected ones (often the cladding)
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda c: c[2])
        
        center_x, center_y, radius = largest_circle
        center = (center_x, center_y)

        # Draw the outer circle and its center
        if draw_circle:
            cv2.circle(result, center, radius, (0, 255, 0), 2)  # Green outer circle
            cv2.circle(result, center, 2, (0, 0, 255), 3)      # Red center dot

        # Extract the circular area using a mask (efficient method)
        if extract_circle:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            result = cv2.bitwise_and(result, result, mask=mask)
            
    return result

# Optional: Add a standalone test block
if __name__ == '__main__':
    # This code only runs when you execute the script directly
    # It will not run when the script is imported by the GUI
    try:
        # Load a test image (replace 'endface.jpg' with a valid path)
        test_image = cv2.imread('endface.jpg')
        if test_image is None:
            raise FileNotFoundError("Could not find 'endface.jpg'. Please place it in the same directory.")

        # Process the image with default parameters
        processed = process_image(test_image)

        # Display the result
        cv2.imshow('Cladding Detector Test', processed)
        print("Test complete. Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")