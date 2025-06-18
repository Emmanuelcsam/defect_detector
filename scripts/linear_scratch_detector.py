"""
Custom Linear Scratch Detector - Multi-orientation analysis for scratch detection
Calculates scratch strength by comparing central line to parallel adjacent lines
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  angle_step: int = 15,
                  line_length: int = 21,
                  line_width: int = 3,
                  parallel_distance: int = 5,
                  contrast_threshold: float = 0.2,
                  response_mode: str = "max",
                  normalize_response: bool = True,
                  show_orientations: bool = False,
                  highlight_scratches: bool = True) -> np.ndarray:
    """
    Detect linear scratches using multi-orientation analysis.
    
    This function implements a custom linear detector that calculates scratch
    strength for each pixel by analyzing multiple orientations. For each
    orientation, it compares the average intensity along a central line
    (potential scratch) to parallel adjacent lines (background).
    
    Args:
        image: Input image (preferably contrast-enhanced)
        angle_step: Angle increment in degrees (e.g., test every 15°)
        line_length: Length of the analysis line in pixels
        line_width: Width of the line for averaging
        parallel_distance: Distance to parallel comparison lines
        contrast_threshold: Minimum contrast ratio for detection
        response_mode: "max", "mean", or "all" for combining orientations
        normalize_response: Normalize response maps to 0-255
        show_orientations: Display individual orientation responses
        highlight_scratches: Overlay detected scratches on original
        
    Returns:
        Scratch response map or visualization
    """
    # Work with grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # Preprocess: apply slight Gaussian blur to reduce noise
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0.5)
    
    # Generate angles to test
    angles = np.arange(0, 180, angle_step)
    num_angles = len(angles)
    
    # Initialize response maps for each orientation
    response_maps = np.zeros((h, w, num_angles), dtype=np.float32)
    
    # Precompute line offsets for each angle
    half_length = line_length // 2
    
    print(f"Analyzing {num_angles} orientations...")
    
    for angle_idx, angle in enumerate(angles):
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Direction vectors
        dx = np.cos(theta)
        dy = np.sin(theta)
        
        # Perpendicular direction (for parallel lines)
        pdx = -np.sin(theta)
        pdy = np.cos(theta)
        
        # Create response map for this orientation
        response = np.zeros((h, w), dtype=np.float32)
        
        # Process each pixel
        for y in range(half_length, h - half_length):
            for x in range(half_length, w - half_length):
                # Sample points along the central line
                central_vals = []
                left_vals = []
                right_vals = []
                
                for t in range(-half_length, half_length + 1):
                    # Central line point
                    cx = int(x + t * dx)
                    cy = int(y + t * dy)
                    
                    # Left parallel line point
                    lx = int(cx + parallel_distance * pdx)
                    ly = int(cy + parallel_distance * pdy)
                    
                    # Right parallel line point
                    rx = int(cx - parallel_distance * pdx)
                    ry = int(cy - parallel_distance * pdy)
                    
                    # Check bounds and sample
                    if (0 <= cx < w and 0 <= cy < h):
                        central_vals.append(gray_smooth[cy, cx])
                        
                    if (0 <= lx < w and 0 <= ly < h):
                        left_vals.append(gray_smooth[ly, lx])
                        
                    if (0 <= rx < w and 0 <= ry < h):
                        right_vals.append(gray_smooth[ry, rx])
                
                # Calculate scratch strength
                if central_vals and left_vals and right_vals:
                    central_mean = np.mean(central_vals)
                    side_mean = np.mean(left_vals + right_vals)
                    
                    # Scratch response: how much darker is the center than sides
                    if side_mean > 0:
                        # Dark scratch on bright background
                        dark_response = (side_mean - central_mean) / side_mean
                        
                        # Bright scratch on dark background
                        bright_response = (central_mean - side_mean) / (255 - side_mean + 1)
                        
                        # Take maximum response
                        scratch_response = max(dark_response, bright_response)
                        
                        # Apply threshold
                        if scratch_response > contrast_threshold:
                            response[y, x] = scratch_response
        
        # Store response map
        response_maps[:, :, angle_idx] = response
    
    # Combine responses based on mode
    if response_mode == "max":
        # Maximum response across all orientations
        combined_response = np.max(response_maps, axis=2)
        best_angle_map = np.argmax(response_maps, axis=2)
        
    elif response_mode == "mean":
        # Average response across all orientations
        combined_response = np.mean(response_maps, axis=2)
        best_angle_map = None
        
    else:  # all
        # Keep all orientation responses
        combined_response = response_maps
        best_angle_map = np.argmax(response_maps, axis=2)
    
    # Normalize response
    if normalize_response and response_mode != "all":
        combined_response = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX)
        combined_response = combined_response.astype(np.uint8)
    
    # Generate visualization
    if show_orientations and response_mode != "all":
        # Create grid of orientation responses
        grid_size = int(np.ceil(np.sqrt(num_angles)))
        grid_h = grid_size * h // 4
        grid_w = grid_size * w // 4
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, angle in enumerate(angles):
            row = i // grid_size
            col = i % grid_size
            
            # Get response for this angle
            angle_response = response_maps[:, :, i]
            angle_response = cv2.normalize(angle_response, None, 0, 255, cv2.NORM_MINMAX)
            angle_response = angle_response.astype(np.uint8)
            
            # Resize and colorize
            small = cv2.resize(angle_response, (w // 4, h // 4))
            colored = cv2.applyColorMap(small, cv2.COLORMAP_JET)
            
            # Add angle label
            cv2.putText(colored, f"{angle}°", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Place in grid
            y1 = row * (h // 4)
            y2 = y1 + h // 4
            x1 = col * (w // 4)
            x2 = x1 + w // 4
            grid[y1:y2, x1:x2] = colored
        
        result = grid
        
    elif highlight_scratches and response_mode != "all":
        # Overlay scratches on original
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Threshold the response to get binary scratch mask
        _, scratch_mask = cv2.threshold(combined_response, 50, 255, cv2.THRESH_BINARY)
        
        # Colorize based on best angle
        if best_angle_map is not None:
            # Create HSV image where H is based on angle
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[:, :, 0] = (best_angle_map * 180 // num_angles).astype(np.uint8)  # Hue
            hsv[:, :, 1] = 255  # Saturation
            hsv[:, :, 2] = scratch_mask  # Value
            
            # Convert to BGR
            scratch_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Overlay
            result = cv2.addWeighted(result, 0.7, scratch_colored, 0.3, 0)
        else:
            # Simple red overlay
            scratch_overlay = np.zeros_like(result)
            scratch_overlay[:, :, 2] = scratch_mask
            result = cv2.addWeighted(result, 0.7, scratch_overlay, 0.3, 0)
        
        # Draw scratch contours
        contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
        
        # Add title
        cv2.putText(result, f"Linear Scratches ({len(contours)} detected)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    else:
        # Return response map
        if response_mode == "all":
            # Return maximum response for display
            result = cv2.normalize(np.max(response_maps, axis=2), None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)
            result = cv2.applyColorMap(result, cv2.COLORMAP_HOT)
        else:
            result = cv2.applyColorMap(combined_response, cv2.COLORMAP_HOT)
        
        cv2.putText(result, "Scratch Response Map", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add processing parameters
    info_text = f"Angles: {num_angles}, Length: {line_length}px"
    cv2.putText(result, info_text, (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Store scratch data in metadata
    result.scratch_response = combined_response
    result.response_maps = response_maps
    result.angles_tested = angles
    result.best_angles = best_angle_map if best_angle_map is not None else None
    
    return result