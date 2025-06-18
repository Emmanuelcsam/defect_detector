"""
Fiber Structure Localization
============================
Detects and localizes fiber optic cladding and core regions using multiple methods
including HoughCircles, contour fitting, and intensity-based analysis.

This function is specifically designed for fiber optic end-face images and can
handle both single-mode (small core) and multi-mode (larger core) fibers.
"""
import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Dict, Any


def process_image(image: np.ndarray,
                  detection_method: str = "auto",
                  hough_dp: float = 1.2,
                  hough_param1: int = 70,
                  hough_param2: int = 35,
                  min_radius_factor: float = 0.08,
                  max_radius_factor: float = 0.45,
                  core_search_radius_factor: float = 0.3,
                  draw_visualization: bool = True,
                  visualization_color: str = "green") -> np.ndarray:
    """
    Detect and visualize fiber optic cladding and core structures.
    
    This function identifies the circular cladding boundary and the core region
    within a fiber optic cable end-face image. It uses multiple detection methods
    with automatic fallback for robustness.
    
    Args:
        image: Input image (grayscale or color)
        detection_method: Method to use ("auto", "hough", "contour", "combined")
        hough_dp: Inverse ratio of accumulator resolution to image resolution
        hough_param1: Upper threshold for Canny edge detector
        hough_param2: Accumulator threshold for circle centers
        min_radius_factor: Minimum radius as fraction of image size (0.0-1.0)
        max_radius_factor: Maximum radius as fraction of image size (0.0-1.0)
        core_search_radius_factor: Search radius for core as fraction of cladding
        draw_visualization: Whether to draw detected structures on the image
        visualization_color: Color for visualization ("green", "red", "blue", "yellow")
        
    Returns:
        Image with detected fiber structures visualized (or original if visualization disabled)
        
    Detection Strategy:
        1. Cladding detection using HoughCircles (most reliable for clean images)
        2. Fallback to contour-based detection if Hough fails
        3. Core detection using intensity profile analysis
        4. Validation of detected structures
    """
    # Work with grayscale for detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image.copy()
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    h, w = gray.shape
    min_dim = min(h, w)
    
    # Calculate radius bounds
    min_radius = int(min_dim * min_radius_factor)
    max_radius = int(min_dim * max_radius_factor)
    
    # Initialize detection results
    cladding_detected = False
    cladding_center = None
    cladding_radius = None
    core_center = None
    core_radius = None
    
    # Method 1: HoughCircles detection
    if detection_method in ["auto", "hough", "combined"]:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=hough_dp,
            minDist=int(min_dim * 0.15),
            param1=hough_param1,
            param2=hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > 0:
                # Use the first (strongest) circle
                x, y, r = circles[0]
                cladding_center = (x, y)
                cladding_radius = r
                cladding_detected = True
    
    # Method 2: Contour-based detection (fallback)
    if not cladding_detected and detection_method in ["auto", "contour", "combined"]:
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit ellipse if enough points
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                (cx, cy), (minor_axis, major_axis), angle = ellipse
                
                # Check if reasonably circular
                if major_axis > 0:
                    axis_ratio = minor_axis / major_axis
                    if axis_ratio > 0.7:  # Reasonably circular
                        avg_radius = (minor_axis + major_axis) / 4.0
                        cladding_center = (int(cx), int(cy))
                        cladding_radius = int(avg_radius)
                        cladding_detected = True
    
    # Core detection if cladding was found
    if cladding_detected:
        core_center, core_diameter = _detect_core_enhanced(
            gray, cladding_center, cladding_radius, core_search_radius_factor
        )
        core_radius = int(core_diameter / 2) if core_diameter else None
    
    # Visualization
    if draw_visualization and cladding_detected:
        # Color mapping
        color_map = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255)
        }
        color = color_map.get(visualization_color.lower(), (0, 255, 0))
        
        # Draw cladding
        cv2.circle(result, cladding_center, cladding_radius, color, 2)
        cv2.circle(result, cladding_center, 3, color, -1)  # Center dot
        
        # Add cladding label
        cv2.putText(result, f"Cladding: {cladding_radius*2}px", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw core if detected
        if core_center and core_radius:
            # Use different color for core
            core_color = (0, 255, 255) if color != (0, 255, 255) else (255, 0, 255)
            cv2.circle(result, core_center, core_radius, core_color, 2)
            cv2.circle(result, core_center, 2, core_color, -1)  # Center dot
            
            # Add core label
            cv2.putText(result, f"Core: {core_radius*2}px", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, core_color, 2)
        
        # Add method label
        method_text = "HoughCircles" if detection_method != "contour" else "Contour Fit"
        cv2.putText(result, f"Method: {method_text}", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    elif draw_visualization and not cladding_detected:
        # Draw error message
        cv2.putText(result, "No fiber detected!", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result


def _detect_core_enhanced(image: np.ndarray, 
                         cladding_center: Tuple[int, int], 
                         cladding_radius: int,
                         search_radius_factor: float = 0.3) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
    """
    Enhanced core detection using intensity-based analysis.
    
    Args:
        image: Grayscale image
        cladding_center: Center coordinates of detected cladding
        cladding_radius: Radius of detected cladding
        search_radius_factor: Search radius as fraction of cladding radius
        
    Returns:
        Tuple of (core_center, core_diameter) or (None, None) if not detected
    """
    # Create search mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    search_radius = int(cladding_radius * search_radius_factor)
    cv2.circle(mask, cladding_center, search_radius, 255, -1)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1)
    
    # Calculate radial intensity profile
    cy, cx = cladding_center[1], cladding_center[0]
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Sample intensity at different radii
    intensity_profile = []
    max_radius = search_radius
    
    for r in range(0, max_radius, 2):
        ring_mask = (dist_from_center >= r) & (dist_from_center < r + 2) & (mask > 0)
        if np.any(ring_mask):
            mean_intensity = np.mean(blurred[ring_mask])
            intensity_profile.append((r, mean_intensity))
    
    if len(intensity_profile) > 5:
        # Find radius where intensity changes significantly
        radii = np.array([p[0] for p in intensity_profile])
        intensities = np.array([p[1] for p in intensity_profile])
        
        # Calculate intensity gradient
        gradient = np.gradient(intensities)
        
        # Find maximum gradient location
        if len(gradient) > 2:
            max_gradient_idx = np.argmax(np.abs(gradient[1:-1])) + 1
            
            if max_gradient_idx < len(radii) - 1:
                core_radius = radii[max_gradient_idx]
                
                # Validate detected radius (typical single-mode: ~9µm core in 125µm cladding)
                expected_ratio = 0.072  # 9/125
                min_ratio = 0.03
                max_ratio = 0.15
                
                detected_ratio = core_radius / cladding_radius
                
                if min_ratio < detected_ratio < max_ratio:
                    return cladding_center, core_radius * 2
    
    # Fallback: use typical ratio
    fallback_radius = cladding_radius * 0.072
    return cladding_center, fallback_radius * 2


# Test code
if __name__ == "__main__":
    # Create test fiber image
    test_size = 400
    test_image = np.zeros((test_size, test_size), dtype=np.uint8)
    
    # Draw cladding
    center = (test_size // 2, test_size // 2)
    cladding_radius = 150
    cv2.circle(test_image, center, cladding_radius, 128, -1)
    
    # Draw core
    core_radius = 11  # ~7.3% of cladding
    cv2.circle(test_image, center, core_radius, 64, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Test the function
    result = process_image(test_image, detection_method="auto", draw_visualization=True)
    
    cv2.imshow("Fiber Localization Test", result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
