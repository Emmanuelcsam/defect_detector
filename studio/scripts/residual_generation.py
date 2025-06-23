"""
Residual Generation - Create residual map by subtracting min from max filtered images
Highlights areas of sharp change corresponding to defects
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  use_stored_filters: bool = True,
                  kernel_size: int = 5,
                  enhancement_factor: float = 1.5,
                  clip_outliers: bool = True,
                  outlier_percentile: float = 99.5,
                  apply_clahe: bool = True,
                  clahe_clip_limit: float = 2.0,
                  visualization_mode: str = "residual") -> np.ndarray:
    """
    Generate residual map for defect detection (I_r = I_max - I_min).
    
    This function creates a residual map that highlights sharp intensity changes
    which typically correspond to defects. It can use pre-computed max/min filters
    from previous pipeline steps or compute them if needed.
    
    Args:
        image: Input image (may contain max/min filter metadata)
        use_stored_filters: Use max/min filters from previous step if available
        kernel_size: Kernel size if computing filters (ignored if using stored)
        enhancement_factor: Factor to enhance residual contrast
        clip_outliers: Whether to clip extreme outlier values
        outlier_percentile: Percentile for outlier clipping
        apply_clahe: Apply CLAHE to enhance local contrast
        clahe_clip_limit: CLAHE clip limit parameter
        visualization_mode: "residual", "enhanced", "heatmap", or "analysis"
        
    Returns:
        Residual map highlighting potential defects
    """
    # Check if we have stored filter results
    has_stored = (hasattr(image, 'max_filtered') and 
                  hasattr(image, 'min_filtered') and 
                  use_stored_filters)
    
    if has_stored:
        # Use stored results from previous min-max filtering step
        max_filtered = image.max_filtered
        min_filtered = image.min_filtered
        
        # Get original for reference
        if len(image.shape) == 3:
            original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            original = image.copy()
            
    else:
        # Compute min-max filters if not available
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        original = gray.copy()
        
        # Ensure odd kernel size
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Apply filters
        max_filtered = cv2.dilate(gray, kernel)
        min_filtered = cv2.erode(gray, kernel)
    
    # Generate residual map
    residual = cv2.subtract(max_filtered.astype(np.float32), 
                            min_filtered.astype(np.float32))
    
    # Enhancement step
    if enhancement_factor != 1.0:
        residual = residual * enhancement_factor
    
    # Clip outliers if requested
    if clip_outliers:
        threshold = np.percentile(residual, outlier_percentile)
        residual = np.clip(residual, 0, threshold)
    
    # Normalize to 8-bit range
    residual_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE if requested
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8,8))
        residual_enhanced = clahe.apply(residual_norm)
    else:
        residual_enhanced = residual_norm
    
    # Generate visualization based on mode
    if visualization_mode == "enhanced":
        # Show enhanced residual
        result = cv2.cvtColor(residual_enhanced, cv2.COLOR_GRAY2BGR)
        cv2.putText(result, "Enhanced Residual Map", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    elif visualization_mode == "heatmap":
        # Apply heatmap colorization
        result = cv2.applyColorMap(residual_enhanced, cv2.COLORMAP_HOT)
        cv2.putText(result, "Residual Heatmap", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    elif visualization_mode == "analysis":
        # Create comprehensive analysis view
        h, w = original.shape
        
        # Create panels
        panel1 = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        panel2 = cv2.cvtColor(residual_norm, cv2.COLOR_GRAY2BGR)
        panel3 = cv2.applyColorMap(residual_enhanced, cv2.COLORMAP_JET)
        
        # Calculate statistics
        mean_residual = np.mean(residual)
        std_residual = np.std(residual)
        max_residual = np.max(residual)
        
        # Create histogram of residual values
        hist_panel = np.ones((h, w, 3), dtype=np.uint8) * 255
        hist = cv2.calcHist([residual_norm], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, None, 0, h-40, cv2.NORM_MINMAX)
        
        # Draw histogram
        for i in range(256):
            x = int(i * w / 256)
            y = h - int(hist[i])
            cv2.line(hist_panel, (x, h-20), (x, y), (0, 0, 0), 1)
        
        # Add labels
        cv2.putText(panel1, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(panel2, "Residual", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(panel3, "Enhanced", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(hist_panel, "Histogram", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add statistics to histogram panel
        stats_y = 60
        cv2.putText(hist_panel, f"Mean: {mean_residual:.2f}", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(hist_panel, f"Std: {std_residual:.2f}", (10, stats_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(hist_panel, f"Max: {max_residual:.2f}", (10, stats_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Combine panels
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, hist_panel])
        result = np.vstack([top_row, bottom_row])
        
    else:  # Default residual mode
        result = cv2.cvtColor(residual_norm, cv2.COLOR_GRAY2BGR)
        cv2.putText(result, "Residual Map (Max - Min)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add processing information
    info_text = f"Enhancement: {enhancement_factor:.1f}x"
    if apply_clahe:
        info_text += f", CLAHE: {clahe_clip_limit:.1f}"
    cv2.putText(result, info_text, (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Store residual data in metadata
    result.residual_map = residual_enhanced
    result.residual_raw = residual
    result.residual_stats = {
        'mean': np.mean(residual),
        'std': np.std(residual),
        'max': np.max(residual),
        'min': np.min(residual)
    }
    
    return result