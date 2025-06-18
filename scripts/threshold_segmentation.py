"""
Threshold Segmentation - Convert residual map to binary defect mask using sigma-based thresholding
Isolates defects based on statistical deviation from normal regions
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  use_stored_residual: bool = True,
                  gamma: float = 2.0,
                  reference_region: str = "auto",
                  roi_size_percent: float = 30.0,
                  threshold_method: str = "sigma",
                  otsu_weight: float = 0.5,
                  min_defect_size: int = 5,
                  max_defect_size: int = 5000,
                  morphology_kernel: int = 3,
                  show_statistics: bool = True) -> np.ndarray:
    """
    Apply sigma-based thresholding to segment defects from residual map.
    
    This function converts the residual map into a binary image where defects
    are white (255) and background is black (0). The threshold is calculated
    as: threshold = mean + gamma * std_dev, where statistics are computed from
    defect-free reference regions.
    
    Args:
        image: Input image (may contain residual map metadata)
        use_stored_residual: Use residual map from previous step if available
        gamma: Sensitivity parameter (higher = less sensitive)
        reference_region: "auto", "center", "corners", or "manual"
        roi_size_percent: Size of reference region as percentage
        threshold_method: "sigma", "otsu", or "adaptive"
        otsu_weight: Weight for Otsu when using mixed method
        min_defect_size: Minimum defect area in pixels
        max_defect_size: Maximum defect area in pixels
        morphology_kernel: Kernel size for morphological operations
        show_statistics: Display threshold statistics
        
    Returns:
        Binary mask with detected defects
    """
    # Get residual map
    if use_stored_residual and hasattr(image, 'residual_map'):
        residual = image.residual_map
        
        # Get original for display
        if len(image.shape) == 3:
            original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            original = image
    else:
        # Use input as residual map
        if len(image.shape) == 3:
            residual = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            original = residual
        else:
            residual = image.copy()
            original = residual
    
    h, w = residual.shape
    
    # Define reference region for statistics
    if reference_region == "center":
        # Use center region (typically defect-free in fibers)
        roi_size = int(min(h, w) * roi_size_percent / 100)
        cy, cx = h // 2, w // 2
        roi = residual[cy-roi_size//2:cy+roi_size//2, 
                      cx-roi_size//2:cx+roi_size//2]
        
    elif reference_region == "corners":
        # Use corners (often background in fiber images)
        corner_size = int(min(h, w) * roi_size_percent / 100)
        roi_tl = residual[:corner_size, :corner_size]
        roi_tr = residual[:corner_size, -corner_size:]
        roi_bl = residual[-corner_size:, :corner_size]
        roi_br = residual[-corner_size:, -corner_size:]
        roi = np.concatenate([roi_tl.flatten(), roi_tr.flatten(), 
                             roi_bl.flatten(), roi_br.flatten()])
        
    else:  # auto
        # Use lower intensity regions (likely defect-free)
        threshold_low = np.percentile(residual, 30)
        roi = residual[residual < threshold_low]
    
    # Calculate statistics from reference region
    mean_ref = np.mean(roi)
    std_ref = np.std(roi)
    
    # Calculate threshold based on method
    if threshold_method == "sigma":
        # Sigma-based threshold
        threshold = mean_ref + gamma * std_ref
        
    elif threshold_method == "otsu":
        # Otsu's method
        _, otsu_thresh = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = _
        
    else:  # adaptive
        # Mixed method - combine sigma and Otsu
        sigma_threshold = mean_ref + gamma * std_ref
        otsu_threshold, _ = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = (1 - otsu_weight) * sigma_threshold + otsu_weight * otsu_threshold
    
    # Apply threshold
    _, binary_mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up
    if morphology_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (morphology_kernel, morphology_kernel))
        # Opening to remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        # Closing to fill small gaps
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Filter by size
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
    
    # Create filtered mask
    filtered_mask = np.zeros_like(binary_mask)
    defect_count = 0
    total_defect_area = 0
    
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_defect_size <= area <= max_defect_size:
            filtered_mask[labels == i] = 255
            defect_count += 1
            total_defect_area += area
    
    # Create visualization
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Overlay defects in red
    defect_overlay = np.zeros_like(result)
    defect_overlay[:, :, 2] = filtered_mask  # Red channel
    result = cv2.addWeighted(result, 0.7, defect_overlay, 0.3, 0)
    
    # Draw contours around defects
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    
    # Add defect markers and numbers
    for i, (cx, cy) in enumerate(centroids[1:defect_count+1]):
        cv2.circle(result, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(result, str(i+1), (int(cx)+5, int(cy)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Add information overlay
    cv2.putText(result, f"Defects Found: {defect_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f"Total Area: {total_defect_area} px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if show_statistics:
        # Add threshold information
        info_y = result.shape[0] - 120
        cv2.rectangle(result, (10, info_y), (250, result.shape[0]-10), (0, 0, 0), -1)
        cv2.rectangle(result, (10, info_y), (250, result.shape[0]-10), (255, 255, 255), 1)
        
        cv2.putText(result, "Threshold Statistics:", (15, info_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Mean (ref): {mean_ref:.2f}", (15, info_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, f"Std (ref): {std_ref:.2f}", (15, info_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, f"Gamma: {gamma:.2f}", (15, info_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, f"Threshold: {threshold:.2f}", (15, info_y + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(result, f"Method: {threshold_method}", (15, info_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Store segmentation data in metadata
    result.binary_mask = filtered_mask
    result.defect_count = defect_count
    result.defect_area = total_defect_area
    result.threshold_value = threshold
    result.threshold_stats = {
        'mean_ref': mean_ref,
        'std_ref': std_ref,
        'gamma': gamma,
        'method': threshold_method
    }
    
    return result