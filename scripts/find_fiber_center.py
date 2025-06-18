"""
Find Fiber Center - Locate the center of optical fiber using Hough Circle Transform
This function detects circular structures in fiber optic images and identifies the fiber center
"""
import cv2
import numpy as np

def process_image(image: np.ndarray, 
                  dp: float = 1.0,
                  min_dist_ratio: float = 0.5,
                  param1: int = 100,
                  param2: int = 30,
                  min_radius_ratio: float = 0.1,
                  max_radius_ratio: float = 0.9,
                  blur_kernel: int = 5,
                  draw_circles: bool = True,
                  mark_center: bool = True,
                  output_mode: str = "overlay") -> np.ndarray:
    """
    Find the center of optical fiber using Hough Circle Transform.
    
    This function is specifically designed for fiber optic images where the fiber
    appears as a circular structure. It detects the outer boundary of the cladding
    layer and marks the center point.
    
    Args:
        image: Input fiber optic image
        dp: Inverse ratio of accumulator resolution to image resolution
        min_dist_ratio: Minimum distance between detected centers as ratio of image width
        param1: Higher threshold for Canny edge detector
        param2: Accumulator threshold for circle detection (lower = more circles)
        min_radius_ratio: Minimum radius as ratio of image size
        max_radius_ratio: Maximum radius as ratio of image size
        blur_kernel: Gaussian blur kernel size for preprocessing
        draw_circles: Whether to draw detected circles
        mark_center: Whether to mark the center point
        output_mode: "overlay", "mask", or "info" for different visualizations
        
    Returns:
        Processed image with detected fiber center marked
    """
    # Ensure we have the right image format
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image.copy()
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    h, w = gray.shape
    
    # Preprocessing - enhance circular features
    # Apply Gaussian blur to reduce noise
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Calculate radius limits based on image size
    min_size = min(h, w)
    min_radius = int(min_size * min_radius_ratio)
    max_radius = int(min_size * max_radius_ratio)
    min_dist = int(w * min_dist_ratio)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Initialize center coordinates and info
    center_x, center_y, radius = w//2, h//2, min_size//3  # Default values
    fiber_found = False
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # If multiple circles detected, choose the one closest to image center
        if len(circles[0]) > 1:
            image_center = np.array([w/2, h/2])
            min_dist_to_center = float('inf')
            best_circle_idx = 0
            
            for idx, (x, y, r) in enumerate(circles[0]):
                dist = np.linalg.norm(np.array([x, y]) - image_center)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_circle_idx = idx
            
            center_x, center_y, radius = circles[0][best_circle_idx]
        else:
            center_x, center_y, radius = circles[0][0]
        
        fiber_found = True
    
    # Generate output based on mode
    if output_mode == "mask":
        # Create binary mask showing fiber region
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
    elif output_mode == "info":
        # Create information overlay
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Draw grid lines through center
        cv2.line(result, (center_x, 0), (center_x, h), (0, 255, 255), 1)
        cv2.line(result, (0, center_y), (w, center_y), (0, 255, 255), 1)
        
        # Add concentric circles for reference
        for i in range(1, 4):
            r = radius * i // 3
            cv2.circle(result, (center_x, center_y), r, (255, 255, 0), 1)
        
    else:  # overlay mode
        # Draw detected circle and center on original image
        if draw_circles and fiber_found:
            # Draw outer circle (cladding boundary)
            cv2.circle(result, (center_x, center_y), radius, (0, 255, 0), 2)
            
            # Draw inner circles for core estimation (typical core/cladding ratio)
            core_radius = int(radius * 0.08)  # Typical single-mode fiber ratio
            cv2.circle(result, (center_x, center_y), core_radius, (255, 255, 0), 2)
            
        if mark_center:
            # Draw center crosshair
            cross_size = 10
            cv2.line(result, (center_x - cross_size, center_y), 
                    (center_x + cross_size, center_y), (0, 0, 255), 2)
            cv2.line(result, (center_x, center_y - cross_size), 
                    (center_x, center_y + cross_size), (0, 0, 255), 2)
            
            # Draw center point
            cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Add text annotations
    info_color = (0, 255, 0) if fiber_found else (0, 0, 255)
    status_text = "Fiber Detected" if fiber_found else "No Fiber Found"
    cv2.putText(result, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
    
    if fiber_found:
        cv2.putText(result, f"Center: ({center_x}, {center_y})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result, f"Radius: {radius}px", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result, f"Diameter: {radius*2}px", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Store center information in image metadata (for pipeline use)
    # This allows subsequent functions to access the detected center
    result.center_x = center_x
    result.center_y = center_y
    result.fiber_radius = radius
    result.fiber_found = fiber_found
    
    return result