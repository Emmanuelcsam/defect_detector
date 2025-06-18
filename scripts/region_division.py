"""
Region Division - Divide fiber optic image into distinct zones (core, cladding, coating)
Based on industry standards for fiber optic structure analysis
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  core_diameter_um: float = 8.2,
                  cladding_diameter_um: float = 125.0,
                  coating_diameter_um: float = 250.0,
                  pixels_per_um: float = 2.0,
                  auto_detect_center: bool = True,
                  center_x: int = -1,
                  center_y: int = -1,
                  visualization_mode: str = "colored",
                  show_boundaries: bool = True,
                  show_labels: bool = True,
                  opacity: float = 0.5) -> np.ndarray:
    """
    Divide fiber optic image into distinct zones based on industry standards.
    
    This function segments the fiber into core, cladding, and coating regions
    based on standard fiber optic dimensions. It can either auto-detect the
    center or use manually specified coordinates.
    
    Standard single-mode fiber dimensions:
    - Core: ~8-10 μm diameter
    - Cladding: 125 μm diameter  
    - Coating: 250 μm diameter
    
    Args:
        image: Input fiber optic image
        core_diameter_um: Core diameter in micrometers
        cladding_diameter_um: Cladding diameter in micrometers
        coating_diameter_um: Coating diameter in micrometers
        pixels_per_um: Image scale (pixels per micrometer)
        auto_detect_center: Whether to auto-detect fiber center
        center_x: Manual center X coordinate (-1 for auto)
        center_y: Manual center Y coordinate (-1 for auto)
        visualization_mode: "colored", "mask", "boundaries", or "analysis"
        show_boundaries: Whether to show region boundaries
        show_labels: Whether to show region labels
        opacity: Overlay opacity for colored mode
        
    Returns:
        Image with regions marked according to visualization mode
    """
    # Prepare image
    if len(image.shape) == 2:
        gray = image.copy()
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    
    h, w = gray.shape
    
    # Detect or use provided center
    if auto_detect_center or center_x < 0 or center_y < 0:
        # Try to get center from image metadata first
        if hasattr(image, 'center_x') and hasattr(image, 'center_y'):
            cx, cy = image.center_x, image.center_y
        else:
            # Auto-detect using simple method
            # Apply blur and find brightest region
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Find moments of the bright region
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            M = cv2.moments(thresh)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = w // 2, h // 2
    else:
        cx, cy = center_x, center_y
    
    # Calculate radii in pixels
    core_radius_px = int(core_diameter_um * pixels_per_um / 2)
    cladding_radius_px = int(cladding_diameter_um * pixels_per_um / 2)
    coating_radius_px = int(coating_diameter_um * pixels_per_um / 2)
    
    # Create region masks
    core_mask = np.zeros(gray.shape, dtype=np.uint8)
    cladding_mask = np.zeros(gray.shape, dtype=np.uint8)
    coating_mask = np.zeros(gray.shape, dtype=np.uint8)
    outside_mask = np.ones(gray.shape, dtype=np.uint8) * 255
    
    # Draw filled circles to create masks
    cv2.circle(core_mask, (cx, cy), core_radius_px, 255, -1)
    cv2.circle(cladding_mask, (cx, cy), cladding_radius_px, 255, -1)
    cv2.circle(coating_mask, (cx, cy), coating_radius_px, 255, -1)
    cv2.circle(outside_mask, (cx, cy), coating_radius_px, 0, -1)
    
    # Create exclusive masks for each region
    cladding_only = cv2.bitwise_and(cladding_mask, cv2.bitwise_not(core_mask))
    coating_only = cv2.bitwise_and(coating_mask, cv2.bitwise_not(cladding_mask))
    
    # Generate visualization based on mode
    if visualization_mode == "mask":
        # Create labeled mask (different gray levels for each region)
        mask_result = np.zeros(gray.shape, dtype=np.uint8)
        mask_result[core_mask > 0] = 255  # Core: white
        mask_result[cladding_only > 0] = 170  # Cladding: light gray
        mask_result[coating_only > 0] = 85   # Coating: dark gray
        mask_result[outside_mask > 0] = 0    # Outside: black
        
        result = cv2.cvtColor(mask_result, cv2.COLOR_GRAY2BGR)
        
    elif visualization_mode == "boundaries":
        # Show only the boundaries between regions
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(result, (cx, cy), core_radius_px, (0, 255, 255), 2)
        cv2.circle(result, (cx, cy), cladding_radius_px, (0, 255, 0), 2)
        cv2.circle(result, (cx, cy), coating_radius_px, (255, 0, 0), 2)
        
    elif visualization_mode == "analysis":
        # Create analysis view with measurements
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw radial lines for measurements
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles:
            x_end = int(cx + coating_radius_px * np.cos(angle))
            y_end = int(cy + coating_radius_px * np.sin(angle))
            cv2.line(result, (cx, cy), (x_end, y_end), (100, 100, 100), 1)
        
        # Draw circles with measurements
        cv2.circle(result, (cx, cy), core_radius_px, (0, 255, 255), 1)
        cv2.circle(result, (cx, cy), cladding_radius_px, (0, 255, 0), 1)
        cv2.circle(result, (cx, cy), coating_radius_px, (255, 0, 0), 1)
        
        # Add measurement annotations
        cv2.putText(result, f"{core_diameter_um:.1f}um", 
                   (cx + core_radius_px + 5, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(result, f"{cladding_diameter_um:.1f}um", 
                   (cx + cladding_radius_px + 5, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(result, f"{coating_diameter_um:.1f}um", 
                   (cx + coating_radius_px + 5, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
    else:  # colored mode
        # Create colored overlay
        overlay = result.copy()
        
        # Define colors for each region (BGR format)
        core_color = (255, 255, 0)      # Cyan
        cladding_color = (0, 255, 0)    # Green
        coating_color = (0, 0, 255)     # Red
        
        # Apply colors to regions
        overlay[core_mask > 0] = core_color
        overlay[cladding_only > 0] = cladding_color
        overlay[coating_only > 0] = coating_color
        
        # Blend with original
        result = cv2.addWeighted(result, 1-opacity, overlay, opacity, 0)
        
        # Add boundaries if requested
        if show_boundaries:
            cv2.circle(result, (cx, cy), core_radius_px, (0, 0, 0), 2)
            cv2.circle(result, (cx, cy), cladding_radius_px, (0, 0, 0), 2)
            cv2.circle(result, (cx, cy), coating_radius_px, (0, 0, 0), 2)
    
    # Add center marker
    cv2.drawMarker(result, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    
    # Add labels if requested
    if show_labels:
        # Title
        cv2.putText(result, "Fiber Region Analysis", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Legend
        legend_y = h - 100
        cv2.rectangle(result, (10, legend_y), (150, h-10), (0, 0, 0), -1)
        cv2.rectangle(result, (10, legend_y), (150, h-10), (255, 255, 255), 1)
        
        # Legend items
        cv2.circle(result, (25, legend_y + 15), 5, (255, 255, 0), -1)
        cv2.putText(result, "Core", (40, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.circle(result, (25, legend_y + 35), 5, (0, 255, 0), -1)
        cv2.putText(result, "Cladding", (40, legend_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.circle(result, (25, legend_y + 55), 5, (0, 0, 255), -1)
        cv2.putText(result, "Coating", (40, legend_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Scale info
        cv2.putText(result, f"Scale: {pixels_per_um:.1f} px/um", (10, legend_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Store region information in metadata for pipeline use
    result.fiber_center = (cx, cy)
    result.core_radius_px = core_radius_px
    result.cladding_radius_px = cladding_radius_px
    result.coating_radius_px = coating_radius_px
    result.core_mask = core_mask
    result.cladding_mask = cladding_only
    result.coating_mask = coating_only
    
    return result