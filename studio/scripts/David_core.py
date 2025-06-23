"""
Detects both the fiber optic core and cladding.
This script first identifies the cladding, isolates it, and then runs a second
detection pass on the isolated region to find the core.
"""
import cv2
import numpy as np

def _extract_circle_region(image: np.ndarray, x0: int, y0: int, radius: int) -> np.ndarray:
    """Helper function to extract a circular region of interest from an image."""
    mask = np.zeros_like(image)
    cv2.circle(mask, (x0, y0), radius, 255, -1)
    result = cv2.bitwise_and(image, mask)
    return result

def process_image(image: np.ndarray,
                  clip_limit: float = 2.0,
                  clad_canny_high: int = 200,
                  core_canny_high: int = 200,
                  clad_hough_param2: int = 40,
                  core_hough_param2: int = 40,
                  clad_min_radius: int = 100,
                  core_min_radius: int = 20,
                  core_max_radius: int = 80) -> np.ndarray:
    """
    Analyzes an image to find and draw both the cladding and the core.

    Args:
        image: Input image (color or grayscale).
        clip_limit: Contrast limit for CLAHE.
        clad_canny_high: Canny upper threshold for cladding detection.
        core_canny_high: Canny upper threshold for core detection.
        clad_hough_param2: Accumulator threshold for cladding detection.
        core_hough_param2: Accumulator threshold for core detection.
        clad_min_radius: Minimum radius for cladding circle.
        core_min_radius: Minimum radius for core circle.
        core_max_radius: Maximum radius for core circle.

    Returns:
        A color image with both detected circles drawn on it.
    """
    # 1. Prepare Image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_image = image.copy()
    else:
        gray = image
        output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. === Find Cladding ===
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    canny_clad = cv2.Canny(clahe_image, clad_canny_high // 2, clad_canny_high)
    blurred_clad = cv2.GaussianBlur(canny_clad, (5, 5), 0)

    cladding_circles = cv2.HoughCircles(blurred_clad, cv2.HOUGH_GRADIENT, dp=1, minDist=image.shape[0],
                                        param1=50, param2=clad_hough_param2,
                                        minRadius=clad_min_radius, maxRadius=image.shape[0] // 2)

    if cladding_circles is None:
        cv2.putText(output_image, "Cladding not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return output_image

    # 3. === Isolate Cladding and Find Core ===
    cladding = np.uint16(np.around(cladding_circles[0, 0]))
    clad_x, clad_y, clad_r = cladding[0], cladding[1], cladding[2]

    # Create a new image containing only the cladding
    cladding_isolate = _extract_circle_region(gray, clad_x, clad_y, clad_r)

    # Process the isolated cladding to find the core
    clahe_core = clahe.apply(cladding_isolate)
    canny_core = cv2.Canny(clahe_core, core_canny_high // 2, core_canny_high)
    blurred_core = cv2.GaussianBlur(canny_core, (7, 7), 0)

    core_circles = cv2.HoughCircles(blurred_core, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                    param1=50, param2=core_hough_param2,
                                    minRadius=core_min_radius, maxRadius=core_max_radius)

    # 4. === Draw Results ===
    # Draw cladding circle
    cv2.circle(output_image, (clad_x, clad_y), clad_r, (255, 255, 0), 2)  # Cyan
    cv2.circle(output_image, (clad_x, clad_y), 2, (0, 0, 255), 3) # Red Center

    if core_circles is not None:
        core = np.uint16(np.around(core_circles[0, 0]))
        core_x, core_y, core_r = core[0], core[1], core[2]

        # Draw core circle
        cv2.circle(output_image, (core_x, core_y), core_r, (0, 255, 0), 2)  # Green
        cv2.circle(output_image, (core_x, core_y), 2, (0, 0, 255), 3) # Red Center

        cv2.putText(output_image, "Core & Cladding Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(output_image, "Cladding found, Core NOT Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # Yellow warning

    return output_image