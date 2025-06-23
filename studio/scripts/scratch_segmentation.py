"""
Scratch Segmentation - Threshold scratch response maps to create binary masks
Applies adaptive thresholding to each orientation's response map
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  use_stored_responses: bool = True,
                  threshold_method: str = "adaptive",
                  global_threshold: float = 0.3,
                  adaptive_block_size: int = 51,
                  adaptive_constant: float = -10.0,
                  min_scratch_length: int = 20,
                  max_scratch_width: int = 10,
                  connect_broken_scratches: bool = True,
                  connection_distance: int = 10) -> np.ndarray:
    """
    Segment scratches by thresholding response maps from linear detection.
    
    This function applies thresholding to scratch response maps to create
    binary masks. It can process multiple orientation response maps separately
    and includes options for connecting broken scratch segments.
    
    Args:
        image: Input image with scratch response metadata
        use_stored_responses: Use response maps from previous step
        threshold_method: "global", "adaptive", or "otsu"
        global_threshold: Threshold value for global method (0-1)
        adaptive_block_size: Block size for adaptive thresholding
        adaptive_constant: Constant subtracted from mean in adaptive
        min_scratch_length: Minimum length to consider as scratch
        max_scratch_width: Maximum width to filter out blobs
        connect_broken_scratches: Whether to connect nearby segments
        connection_distance: Maximum gap to connect
        
    Returns:
        Binary mask of segmented scratches
    """
    # Get response data
    if use_stored_responses and hasattr(image, 'scratch_response'):
        if hasattr(image, 'response_maps'):
            # Multiple orientation maps available
            response_maps = image.response_maps
            angles = image.angles_tested if hasattr(image, 'angles_tested') else None
            multi_orientation = True
        else:
            # Single combined response
            response_maps = image.scratch_response
            if len(response_maps.shape) == 2:
                response_maps = response_maps[:, :, np.newaxis]
            multi_orientation = False
            angles = None
    else:
        # Use input as response map
        if len(image.shape) == 3:
            response_maps = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            response_maps = image.copy()
        response_maps = response_maps[:, :, np.newaxis]
        multi_orientation = False
        angles = None
    
    h, w = response_maps.shape[:2]
    num_orientations = response_maps.shape[2] if len(response_maps.shape) > 2 else 1
    
    # Initialize binary masks for each orientation
    binary_masks = np.zeros((h, w, num_orientations), dtype=np.uint8)
    
    # Process each orientation
    for i in range(num_orientations):
        response = response_maps[:, :, i]
        
        # Normalize response to 0-255 if needed
        if response.dtype != np.uint8:
            response_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
            response_norm = response_norm.astype(np.uint8)
        else:
            response_norm = response
        
        # Apply thresholding based on method
        if threshold_method == "adaptive":
            # Ensure odd block size
            block_size = max(3, adaptive_block_size)
            if block_size % 2 == 0:
                block_size += 1
                
            binary = cv2.adaptiveThreshold(
                response_norm, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                adaptive_constant
            )
            
        elif threshold_method == "otsu":
            _, binary = cv2.threshold(response_norm, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        else:  # global
            threshold_value = int(global_threshold * 255)
            _, binary = cv2.threshold(response_norm, threshold_value, 255, 
                                    cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Filter by scratch characteristics
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        
        filtered_mask = np.zeros_like(binary)
        
        for j in range(1, num_labels):
            # Get component properties
            x, y, w_comp, h_comp, area = stats[j]
            
            # Calculate aspect ratio
            aspect_ratio = max(w_comp, h_comp) / (min(w_comp, h_comp) + 1)
            
            # Calculate length (diagonal of bounding box)
            length = np.sqrt(w_comp**2 + h_comp**2)
            
            # Estimate width (area / length)
            width = area / (length + 1)
            
            # Keep if it looks like a scratch
            if (length >= min_scratch_length and 
                width <= max_scratch_width and
                aspect_ratio > 2.0):  # Scratches are elongated
                filtered_mask[labels == j] = 255
        
        binary_masks[:, :, i] = filtered_mask
    
    # Combine masks from all orientations
    if multi_orientation:
        combined_mask = np.max(binary_masks, axis=2)
    else:
        combined_mask = binary_masks[:, :, 0]
    
    # Connect broken scratches if requested
    if connect_broken_scratches:
        # Use morphological closing with directional kernels
        connected_mask = combined_mask.copy()
        
        # Try different orientations for connection
        for angle in [0, 45, 90, 135]:
            # Create oriented kernel
            kernel_length = connection_distance
            if angle == 0:  # Horizontal
                kernel = np.ones((1, kernel_length), dtype=np.uint8)
            elif angle == 90:  # Vertical
                kernel = np.ones((kernel_length, 1), dtype=np.uint8)
            else:  # Diagonal
                kernel = cv2.getRotationMatrix2D((kernel_length//2, kernel_length//2), 
                                               angle, 1)
                kernel = cv2.warpAffine(np.ones((1, kernel_length), dtype=np.uint8),
                                      kernel, (kernel_length, kernel_length))
                kernel = (kernel > 0.5).astype(np.uint8)
            
            # Apply closing
            temp = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            connected_mask = cv2.bitwise_or(connected_mask, temp)
        
        # Clean up any thick regions created by connection
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_OPEN, kernel_open)
        
        final_mask = connected_mask
    else:
        final_mask = combined_mask
    
    # Create visualization
    result = cv2.cvtColor(response_maps[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Overlay segmented scratches
    scratch_overlay = np.zeros_like(result)
    scratch_overlay[:, :, 2] = final_mask  # Red channel
    result = cv2.addWeighted(result, 0.5, scratch_overlay, 0.5, 0)
    
    # Draw contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    
    # Analyze and label scratches
    scratch_info = []
    for i, contour in enumerate(contours):
        # Fit line to contour
        if len(contour) >= 5:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate angle
            angle = np.degrees(np.arctan2(vy, vx))
            if angle < 0:
                angle += 180
                
            # Get length
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            length = max(width, height)
            
            scratch_info.append({
                'id': i,
                'angle': angle,
                'length': length,
                'center': (int(x), int(y))
            })
            
            # Draw angle info
            cv2.putText(result, f"{angle:.0f}°", (int(x), int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Add summary information
    cv2.putText(result, f"Segmented Scratches: {len(contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if scratch_info:
        avg_angle = np.mean([s['angle'] for s in scratch_info])
        avg_length = np.mean([s['length'] for s in scratch_info])
        
        cv2.putText(result, f"Avg Angle: {avg_angle:.1f}°", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Avg Length: {avg_length:.1f}px", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add method info
    cv2.putText(result, f"Method: {threshold_method}", (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Store segmentation data
    result.scratch_mask = final_mask
    result.scratch_contours = contours
    result.scratch_info = scratch_info
    result.num_scratches = len(contours)
    
    return result