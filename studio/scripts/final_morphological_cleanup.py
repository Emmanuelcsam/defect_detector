"""
Final Morphological Cleanup - Final post-processing to refine synthesized defect masks
Apply morphological operations to produce clean, final defect detection results
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  use_synthesized_mask: bool = True,
                  cleanup_sequence: str = "standard",
                  opening_size: int = 3,
                  closing_size: int = 5,
                  remove_thin_structures: bool = True,
                  thin_threshold: int = 2,
                  smooth_boundaries: bool = True,
                  min_defect_area: int = 20,
                  max_defect_area: int = 10000,
                  export_statistics: bool = True) -> np.ndarray:
    """
    Apply final morphological cleanup to synthesized defect detection results.
    
    This is the final step in the processing pipeline, producing clean,
    production-ready defect masks with smooth boundaries and filtered results.
    
    Args:
        image: Input image with synthesized mask metadata
        use_synthesized_mask: Use final mask from synthesis step
        cleanup_sequence: "standard", "aggressive", or "minimal"
        opening_size: Kernel size for opening operation
        closing_size: Kernel size for closing operation
        remove_thin_structures: Remove very thin features (likely noise)
        thin_threshold: Maximum width for thin structure removal
        smooth_boundaries: Apply boundary smoothing
        min_defect_area: Minimum area to keep
        max_defect_area: Maximum area to keep
        export_statistics: Generate detailed statistics
        
    Returns:
        Final cleaned defect mask with professional visualization
    """
    # Get mask to process
    if use_synthesized_mask and hasattr(image, 'final_mask'):
        mask = image.final_mask.copy()
        
        # Get original for display
        if len(image.shape) == 3:
            original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            display_image = image.copy()
        else:
            original = image
            display_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        # Use input as mask
        if len(image.shape) == 3:
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            display_image = image.copy()
        else:
            mask = image.copy()
            display_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
        # Ensure binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        original = mask
    
    h, w = mask.shape
    
    # Store original for comparison
    original_mask = mask.copy()
    
    # Apply cleanup sequence
    if cleanup_sequence == "aggressive":
        # Aggressive cleanup for noisy images
        # 1. Strong opening to remove small objects
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (opening_size + 2, opening_size + 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 2. Closing to fill gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (closing_size, closing_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 3. Final opening to clean up
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (opening_size, opening_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_final)
        
    elif cleanup_sequence == "minimal":
        # Minimal cleanup to preserve details
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
    else:  # standard
        # Standard cleanup sequence
        # 1. Opening to remove noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (opening_size, opening_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 2. Closing to connect nearby regions
        if closing_size > 0:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (closing_size, closing_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove thin structures if requested
    if remove_thin_structures:
        # Use skeleton and distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Find skeleton
        skeleton = np.zeros_like(mask)
        temp = mask.copy()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(temp) > 0:
            eroded = cv2.erode(temp, kernel)
            temp_open = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            subset = eroded - temp_open
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
        
        # Remove parts where distance to edge is small
        thin_parts = cv2.bitwise_and(skeleton, 
                                    (dist_transform <= thin_threshold).astype(np.uint8) * 255)
        
        # Remove thin parts from mask
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thin_parts_dilated = cv2.dilate(thin_parts, kernel_dilate)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(thin_parts_dilated))
    
    # Smooth boundaries if requested
    if smooth_boundaries:
        # Apply Gaussian blur and re-threshold
        blurred = cv2.GaussianBlur(mask.astype(float), (5, 5), 1.0)
        mask = (blurred > 127).astype(np.uint8) * 255
    
    # Filter by size
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    
    final_mask = np.zeros_like(mask)
    kept_defects = []
    removed_defects = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_defect_area <= area <= max_defect_area:
            final_mask[labels == i] = 255
            kept_defects.append({
                'id': len(kept_defects) + 1,
                'area': area,
                'center': (int(centroids[i][0]), int(centroids[i][1])),
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            })
        else:
            removed_defects.append({
                'area': area,
                'reason': 'too small' if area < min_defect_area else 'too large'
            })
    
    # Create professional visualization
    result = display_image.copy()
    
    # Create smooth overlay
    overlay = np.zeros_like(result)
    
    # Apply distance transform for gradient effect
    if cv2.countNonZero(final_mask) > 0:
        dist = cv2.distanceTransform(final_mask, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create gradient overlay
        overlay[:, :, 2] = (final_mask > 0).astype(np.uint8) * (128 + dist.astype(np.uint8) // 2)
        overlay[:, :, 1] = (final_mask > 0).astype(np.uint8) * (dist.astype(np.uint8) // 4)
    
    result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
    
    # Draw refined contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Smooth contours
    smooth_contours = []
    for contour in contours:
        if len(contour) > 5:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            smooth_contours.append(approx)
        else:
            smooth_contours.append(contour)
    
    cv2.drawContours(result, smooth_contours, -1, (0, 255, 0), 2)
    
    # Add defect annotations
    for defect in kept_defects:
        # Draw bounding box
        x, y, w, h = defect['bbox']
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 1)
        
        # Add defect number
        cv2.putText(result, str(defect['id']), 
                   (defect['center'][0] - 10, defect['center'][1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result, str(defect['id']), 
                   (defect['center'][0] - 10, defect['center'][1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add title
    cv2.putText(result, f"Final Defect Detection: {len(kept_defects)} defects", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(result, f"Final Defect Detection: {len(kept_defects)} defects", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add statistics panel if requested
    if export_statistics:
        # Create statistics panel
        stats_h = 200
        stats_panel = np.ones((stats_h, w, 3), dtype=np.uint8) * 30
        
        # Add statistics
        cv2.putText(stats_panel, "FINAL STATISTICS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Processing summary
        cv2.putText(stats_panel, f"Total Defects Found: {len(kept_defects)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(stats_panel, f"Defects Removed: {len(removed_defects)}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Size statistics
        if kept_defects:
            areas = [d['area'] for d in kept_defects]
            cv2.putText(stats_panel, f"Average Size: {np.mean(areas):.1f} px", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(stats_panel, f"Size Range: {min(areas)} - {max(areas)} px", (10, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Coverage
        total_defect_area = sum(d['area'] for d in kept_defects)
        coverage = (total_defect_area / (h * w)) * 100
        cv2.putText(stats_panel, f"Defect Coverage: {coverage:.3f}%", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Quality assessment
        quality_score = 100 - min(coverage * 10, 100)  # Simple quality metric
        color = (0, 255, 0) if quality_score > 90 else (0, 255, 255) if quality_score > 70 else (0, 0, 255)
        cv2.putText(stats_panel, f"Quality Score: {quality_score:.0f}/100", (w - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Combine with result
        result = np.vstack([result, stats_panel])
    
    # Store final results
    result.final_cleaned_mask = final_mask
    result.defects = kept_defects
    result.quality_metrics = {
        'defect_count': len(kept_defects),
        'total_area': sum(d['area'] for d in kept_defects),
        'coverage_percent': (sum(d['area'] for d in kept_defects) / (h * w)) * 100,
        'removed_count': len(removed_defects)
    }
    
    return result