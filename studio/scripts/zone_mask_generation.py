"""
Zone Mask Generation for Fiber Optic Analysis
=============================================
Creates binary masks for different zones of a fiber optic cable end-face,
specifically the Core and Cladding regions. These masks are essential for
zone-specific defect detection and analysis.

The function can work with manually specified fiber dimensions or use
automatic detection to create the masks.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def process_image(image: np.ndarray,
                  zone_type: str = "both",
                  cladding_diameter_px: Optional[int] = None,
                  core_diameter_px: Optional[int] = None,
                  center_x: Optional[int] = None,
                  center_y: Optional[int] = None,
                  auto_detect: bool = True,
                  visualization_mode: str = "overlay",
                  core_color: Tuple[int, int, int] = (255, 0, 0),
                  cladding_color: Tuple[int, int, int] = (0, 255, 0),
                  overlay_alpha: float = 0.3) -> np.ndarray:
    """
    Generate zone masks for fiber optic core and cladding regions.
    
    Creates precise binary masks that separate the fiber into distinct zones
    for targeted analysis. Can either use manual parameters or auto-detect
    the fiber structure.
    
    Args:
        image: Input image (grayscale or color)
        zone_type: Which zones to create ("core", "cladding", "both")
        cladding_diameter_px: Cladding diameter in pixels (None for auto-detect)
        core_diameter_px: Core diameter in pixels (None for auto-detect)
        center_x: X coordinate of fiber center (None for auto-detect)
        center_y: Y coordinate of fiber center (None for auto-detect)
        auto_detect: Whether to auto-detect fiber if parameters not provided
        visualization_mode: How to visualize ("overlay", "mask", "contour", "combined")
        core_color: Color for core visualization (B, G, R)
        cladding_color: Color for cladding visualization (B, G, R)
        overlay_alpha: Transparency for overlay mode (0.0-1.0)
        
    Returns:
        Visualization of the zone masks based on selected mode
        
    Visualization Modes:
        - "overlay": Transparent colored overlay on original image
        - "mask": Binary mask image (white=zone, black=background)
        - "contour": Original image with zone boundaries drawn
        - "combined": Side-by-side view of original and overlay
    """
    # Validate parameters
    overlay_alpha = max(0.0, min(1.0, overlay_alpha))
    
    # Convert to grayscale for detection if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray = image.copy()
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    h, w = gray.shape
    
    # Auto-detect fiber structure if needed
    if auto_detect and (center_x is None or center_y is None or 
                       cladding_diameter_px is None or core_diameter_px is None):
        detected = _auto_detect_fiber(gray)
        if detected:
            if center_x is None or center_y is None:
                center_x, center_y = detected['center']
            if cladding_diameter_px is None:
                cladding_diameter_px = detected['cladding_diameter']
            if core_diameter_px is None:
                core_diameter_px = detected['core_diameter']
    
    # Use image center as fallback
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
    
    # Use default sizes if not specified (typical single-mode fiber ratios)
    if cladding_diameter_px is None:
        cladding_diameter_px = int(min(h, w) * 0.6)  # 60% of image size
    if core_diameter_px is None:
        core_diameter_px = int(cladding_diameter_px * 0.072)  # 9µm/125µm ratio
    
    # Create distance map from center
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create masks
    core_radius = core_diameter_px / 2
    cladding_radius = cladding_diameter_px / 2
    
    # Core mask: pixels within core radius
    core_mask = (dist_from_center <= core_radius).astype(np.uint8) * 255
    
    # Cladding mask: pixels within cladding radius but outside core
    cladding_full = (dist_from_center <= cladding_radius).astype(np.uint8)
    core_area = (dist_from_center <= core_radius).astype(np.uint8)
    cladding_mask = (cladding_full - core_area) * 255
    
    # Create visualization based on mode
    if visualization_mode == "mask":
        # Return binary mask
        if zone_type == "core":
            result = core_mask
        elif zone_type == "cladding":
            result = cladding_mask
        else:  # both
            result = np.maximum(core_mask, cladding_mask)
            
        # Convert to 3-channel for consistency
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
    elif visualization_mode == "overlay":
        # Create colored overlay
        overlay = np.zeros_like(color_image)
        
        if zone_type in ["core", "both"]:
            overlay[core_mask > 0] = core_color
            
        if zone_type in ["cladding", "both"]:
            overlay[cladding_mask > 0] = cladding_color
        
        # Blend with original
        result = cv2.addWeighted(color_image, 1 - overlay_alpha, overlay, overlay_alpha, 0)
        
    elif visualization_mode == "contour":
        # Draw zone boundaries
        result = color_image.copy()
        
        # Draw circles
        if zone_type in ["cladding", "both"]:
            cv2.circle(result, (center_x, center_y), int(cladding_radius), cladding_color, 2)
            
        if zone_type in ["core", "both"]:
            cv2.circle(result, (center_x, center_y), int(core_radius), core_color, 2)
        
        # Draw center cross
        cv2.drawMarker(result, (center_x, center_y), (255, 255, 0), 
                      cv2.MARKER_CROSS, 20, 1)
        
        # Add labels
        cv2.putText(result, f"Core: {core_diameter_px}px", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, core_color, 2)
        cv2.putText(result, f"Cladding: {cladding_diameter_px}px", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cladding_color, 2)
        
    elif visualization_mode == "combined":
        # Side-by-side view
        overlay = np.zeros_like(color_image)
        
        if zone_type in ["core", "both"]:
            overlay[core_mask > 0] = core_color
            
        if zone_type in ["cladding", "both"]:
            overlay[cladding_mask > 0] = cladding_color
        
        blended = cv2.addWeighted(color_image, 1 - overlay_alpha, overlay, overlay_alpha, 0)
        
        # Add zone boundaries to blended image
        if zone_type in ["cladding", "both"]:
            cv2.circle(blended, (center_x, center_y), int(cladding_radius), (255, 255, 255), 1)
            
        if zone_type in ["core", "both"]:
            cv2.circle(blended, (center_x, center_y), int(core_radius), (255, 255, 255), 1)
        
        # Create side-by-side
        result = np.hstack([color_image, blended])
        
    else:
        result = color_image
    
    return result


def _auto_detect_fiber(gray_image: np.ndarray) -> Optional[dict]:
    """
    Auto-detect fiber structure in the image.
    
    Args:
        gray_image: Grayscale image
        
    Returns:
        Dictionary with detected parameters or None if detection failed
    """
    h, w = gray_image.shape
    min_dim = min(h, w)
    
    # Try HoughCircles for cladding detection
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_dim * 0.15),
        param1=70,
        param2=35,
        minRadius=int(min_dim * 0.08),
        maxRadius=int(min_dim * 0.45)
    )
    
    if circles is not None and len(circles[0]) > 0:
        # Use the first (strongest) circle
        x, y, r = circles[0][0]
        
        # Estimate core size (typical ratio for single-mode fiber)
        core_r = r * 0.072
        
        return {
            'center': (int(x), int(y)),
            'cladding_diameter': int(r * 2),
            'core_diameter': int(core_r * 2)
        }
    
    return None


# Test code
if __name__ == "__main__":
    # Create test fiber image
    test_size = 400
    test_image = np.zeros((test_size, test_size, 3), dtype=np.uint8)
    
    # Create realistic fiber pattern
    center = (test_size // 2, test_size // 2)
    Y, X = np.ogrid[:test_size, :test_size]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Cladding region (bright)
    cladding_mask = dist <= 150
    test_image[cladding_mask] = (200, 200, 200)
    
    # Core region (slightly darker)
    core_mask = dist <= 11
    test_image[core_mask] = (150, 150, 150)
    
    # Add some texture
    noise = np.random.normal(0, 20, test_image.shape)
    test_image = np.clip(test_image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Test different visualization modes
    modes = ["overlay", "mask", "contour", "combined"]
    
    for mode in modes:
        result = process_image(
            test_image, 
            zone_type="both",
            visualization_mode=mode,
            auto_detect=True
        )
        
        window_name = f"Zone Mask - {mode}"
        cv2.imshow(window_name, result)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
