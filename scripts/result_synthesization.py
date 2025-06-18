"""
Result Synthesization - Combine multiple binary masks using logical OR operation
Merges defect detections from different methods or orientations into final result
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  use_pipeline_masks: bool = True,
                  combine_method: str = "or",
                  weight_do2mr: float = 1.0,
                  weight_scratch: float = 1.0,
                  consensus_threshold: int = 1,
                  merge_nearby_defects: bool = True,
                  merge_distance: int = 5,
                  classify_defects: bool = True,
                  generate_report: bool = True) -> np.ndarray:
    """
    Synthesize final defect mask by combining results from different detection methods.
    
    This function combines binary masks from DO2MR (spot defects) and linear
    scratch detection into a comprehensive defect map. It can use OR, AND, or
    weighted voting to merge results.
    
    Args:
        image: Input image with stored detection masks
        use_pipeline_masks: Use masks from previous pipeline steps
        combine_method: "or", "and", "weighted", or "consensus"
        weight_do2mr: Weight for DO2MR detections (spot defects)
        weight_scratch: Weight for scratch detections
        consensus_threshold: Minimum detections for consensus method
        merge_nearby_defects: Merge defects that are close together
        merge_distance: Distance for merging nearby defects
        classify_defects: Classify defects by type
        generate_report: Generate detailed defect report
        
    Returns:
        Final synthesized defect mask with visualization
    """
    # Collect available masks
    masks = {}
    mask_count = 0
    
    # Get original image for display
    if len(image.shape) == 3:
        original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_image = image.copy()
    else:
        original = image.copy()
        display_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    h, w = original.shape
    
    if use_pipeline_masks:
        # Check for DO2MR mask (spot defects)
        if hasattr(image, 'binary_mask'):
            masks['do2mr'] = image.binary_mask
            mask_count += 1
        elif hasattr(image, 'processed_mask'):
            masks['do2mr'] = image.processed_mask
            mask_count += 1
            
        # Check for scratch detection mask
        if hasattr(image, 'scratch_mask'):
            masks['scratch'] = image.scratch_mask
            mask_count += 1
            
        # Check for any other defect masks
        if hasattr(image, 'defect_mask'):
            masks['other'] = image.defect_mask
            mask_count += 1
    
    # If no masks found, try to use input as mask
    if mask_count == 0:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Threshold to get binary mask
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        masks['input'] = mask
        mask_count = 1
    
    # Combine masks based on method
    if mask_count == 1:
        # Only one mask available
        combined_mask = list(masks.values())[0]
        
    elif combine_method == "or":
        # Logical OR - any detection counts
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
    elif combine_method == "and":
        # Logical AND - require agreement
        combined_mask = np.ones((h, w), dtype=np.uint8) * 255
        for mask in masks.values():
            combined_mask = cv2.bitwise_and(combined_mask, mask)
            
    elif combine_method == "weighted":
        # Weighted combination
        weighted_sum = np.zeros((h, w), dtype=np.float32)
        
        if 'do2mr' in masks:
            weighted_sum += masks['do2mr'].astype(float) * weight_do2mr / 255
        if 'scratch' in masks:
            weighted_sum += masks['scratch'].astype(float) * weight_scratch / 255
        if 'other' in masks:
            weighted_sum += masks['other'].astype(float) / 255
            
        # Normalize and threshold
        if mask_count > 0:
            weighted_sum /= mask_count
        
        combined_mask = (weighted_sum > 0.5).astype(np.uint8) * 255
        
    else:  # consensus
        # Require multiple detections
        vote_count = np.zeros((h, w), dtype=np.uint8)
        for mask in masks.values():
            vote_count += (mask > 0).astype(np.uint8)
            
        combined_mask = (vote_count >= consensus_threshold).astype(np.uint8) * 255
    
    # Merge nearby defects if requested
    if merge_nearby_defects and merge_distance > 0:
        # Use morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (merge_distance, merge_distance))
        merged_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Clean up
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel_small)
        
        final_mask = merged_mask
    else:
        final_mask = combined_mask
    
    # Classify defects if requested
    defect_types = {}
    if classify_defects and mask_count > 1:
        # Analyze each connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, 8)
        
        for i in range(1, num_labels):
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Check which original masks contributed
            sources = []
            if 'do2mr' in masks and cv2.countNonZero(cv2.bitwise_and(component_mask, masks['do2mr'])) > 0:
                sources.append('spot')
            if 'scratch' in masks and cv2.countNonZero(cv2.bitwise_and(component_mask, masks['scratch'])) > 0:
                sources.append('scratch')
                
            # Classify based on shape and source
            x, y, w_comp, h_comp, area = stats[i]
            aspect_ratio = max(w_comp, h_comp) / (min(w_comp, h_comp) + 1)
            
            if 'scratch' in sources or aspect_ratio > 3:
                defect_type = 'scratch'
                color = (0, 255, 255)  # Yellow
            elif area < 50:
                defect_type = 'point'
                color = (255, 0, 0)    # Blue
            else:
                defect_type = 'spot'
                color = (0, 0, 255)    # Red
                
            defect_types[i] = {
                'type': defect_type,
                'sources': sources,
                'area': area,
                'center': (int(centroids[i][0]), int(centroids[i][1])),
                'color': color
            }
    
    # Create visualization
    result = display_image.copy()
    
    # Create colored overlay based on defect types
    if classify_defects and defect_types:
        # Color each defect by type
        overlay = np.zeros_like(result)
        
        for defect_id, defect_info in defect_types.items():
            defect_mask = (labels == defect_id)
            overlay[defect_mask] = defect_info['color']
            
        result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
        
        # Add defect markers
        for defect_id, defect_info in defect_types.items():
            cv2.circle(result, defect_info['center'], 3, (255, 255, 255), -1)
            cv2.putText(result, str(defect_id), 
                       (defect_info['center'][0] + 5, defect_info['center'][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        # Simple red overlay
        overlay = np.zeros_like(result)
        overlay[:, :, 2] = final_mask
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    
    # Draw contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    
    # Add title and statistics
    cv2.putText(result, f"Synthesized Defects: {len(contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f"Method: {combine_method.upper()}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Sources: {', '.join(masks.keys())}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Generate report if requested
    if generate_report and classify_defects and defect_types:
        # Count defects by type
        type_counts = {'scratch': 0, 'spot': 0, 'point': 0}
        total_area = 0
        
        for defect_info in defect_types.values():
            type_counts[defect_info['type']] += 1
            total_area += defect_info['area']
        
        # Create report panel
        report_h = 150
        report_w = result.shape[1]
        report_panel = np.ones((report_h, report_w, 3), dtype=np.uint8) * 20
        
        # Add report title
        cv2.putText(report_panel, "DEFECT ANALYSIS REPORT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add statistics
        y_pos = 60
        for defect_type, count in type_counts.items():
            if count > 0:
                cv2.putText(report_panel, f"{defect_type.capitalize()}: {count}", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 25
        
        # Add total area
        coverage = (total_area / (h * w)) * 100
        cv2.putText(report_panel, f"Total Defect Area: {total_area} px ({coverage:.2f}%)", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Add color legend
        legend_x = report_w - 200
        cv2.putText(report_panel, "Legend:", (legend_x, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(report_panel, (legend_x, 50), (legend_x + 15, 65), (0, 255, 255), -1)
        cv2.putText(report_panel, "Scratch", (legend_x + 20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(report_panel, (legend_x, 70), (legend_x + 15, 85), (0, 0, 255), -1)
        cv2.putText(report_panel, "Spot", (legend_x + 20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(report_panel, (legend_x, 90), (legend_x + 15, 105), (255, 0, 0), -1)
        cv2.putText(report_panel, "Point", (legend_x + 20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine with result
        result = np.vstack([result, report_panel])
    
    # Store final data
    result.final_mask = final_mask
    result.defect_count = len(contours)
    result.defect_types = defect_types if classify_defects else None
    result.synthesis_method = combine_method
    result.source_masks = list(masks.keys())
    
    return result