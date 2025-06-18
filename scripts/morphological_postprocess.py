"""
Morphological Post-Processing - Remove small noise and refine defect detection
Apply morphological opening (erosion + dilation) to clean up binary masks
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  use_stored_mask: bool = True,
                  operation: str = "opening",
                  kernel_size: int = 3,
                  kernel_shape: str = "ellipse",
                  iterations: int = 1,
                  remove_border_objects: bool = True,
                  fill_holes: bool = True,
                  min_object_size: int = 10,
                  show_comparison: bool = True) -> np.ndarray:
    """
    Apply morphological operations to refine defect detection results.
    
    Morphological opening (erosion followed by dilation) removes small isolated
    noise points while preserving the shape of larger defects. This is the final
    cleanup step in the defect detection pipeline.
    
    Args:
        image: Input binary mask or image with mask metadata
        use_stored_mask: Use binary mask from previous step if available
        operation: "opening", "closing", "gradient", or "tophat"
        kernel_size: Size of the morphological kernel
        kernel_shape: "ellipse", "rectangle", or "cross"
        iterations: Number of times to apply the operation
        remove_border_objects: Remove objects touching image borders
        fill_holes: Fill holes inside detected objects
        min_object_size: Minimum size to keep after processing
        show_comparison: Show before/after comparison
        
    Returns:
        Cleaned binary mask with refined defects
    """
    # Get binary mask
    if use_stored_mask and hasattr(image, 'binary_mask'):
        binary_mask = image.binary_mask.copy()
        
        # Get original for display
        if len(image.shape) == 3:
            original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            original = image
    else:
        # Convert to binary if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Threshold if not already binary
        if gray.dtype != np.uint8 or gray.max() > 1:
            _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            binary_mask = gray.copy()
            
        original = gray
    
    # Store original mask for comparison
    original_mask = binary_mask.copy()
    
    # Create morphological kernel
    if kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == "rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    else:  # cross
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply morphological operation
    if operation == "opening":
        # Opening: erosion followed by dilation
        processed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
    elif operation == "closing":
        # Closing: dilation followed by erosion
        processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
    elif operation == "gradient":
        # Morphological gradient: difference between dilation and erosion
        processed = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
        
    else:  # tophat
        # Top hat: difference between input and opening
        processed = cv2.morphologyEx(binary_mask, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    
    # Fill holes if requested
    if fill_holes:
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill contours
        processed_filled = np.zeros_like(processed)
        cv2.drawContours(processed_filled, contours, -1, 255, -1)
        processed = processed_filled
    
    # Remove border objects if requested
    if remove_border_objects:
        h, w = processed.shape
        border_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create border mask
        border_mask[0, :] = 1
        border_mask[-1, :] = 1
        border_mask[:, 0] = 1
        border_mask[:, -1] = 1
        
        # Label connected components
        num_labels, labels = cv2.connectedComponents(processed)
        
        # Find labels touching borders
        border_labels = set(labels[border_mask == 1])
        
        # Remove border objects
        for label in border_labels:
            if label != 0:  # Don't remove background
                processed[labels == label] = 0
    
    # Filter by minimum size
    if min_object_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, 8)
        
        # Create filtered mask
        filtered = np.zeros_like(processed)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_object_size:
                filtered[labels == i] = 255
                
        processed = filtered
    
    # Count objects before and after
    num_before = cv2.connectedComponents(original_mask)[0] - 1
    num_after = cv2.connectedComponents(processed)[0] - 1
    
    # Generate visualization
    if show_comparison:
        # Create comparison view
        h, w = original_mask.shape
        
        # Convert masks to BGR for visualization
        before_bgr = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR)
        after_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Create difference visualization
        removed = cv2.bitwise_and(original_mask, cv2.bitwise_not(processed))
        added = cv2.bitwise_and(cv2.bitwise_not(original_mask), processed)
        
        diff_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        diff_bgr[:, :, 2] = removed  # Removed in red
        diff_bgr[:, :, 1] = added    # Added in green
        diff_bgr[processed > 0] = [255, 255, 255]  # Final result in white
        
        # Add labels
        cv2.putText(before_bgr, f"Before ({num_before} objects)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(after_bgr, f"After ({num_after} objects)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(diff_bgr, "Changes", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add legend to diff panel
        cv2.putText(diff_bgr, "Red: Removed", (10, h-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(diff_bgr, "White: Final", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine panels
        result = np.hstack([before_bgr, after_bgr, diff_bgr])
        
        # Add divider lines
        cv2.line(result, (w, 0), (w, h), (255, 255, 255), 2)
        cv2.line(result, (2*w, 0), (2*w, h), (255, 255, 255), 2)
        
    else:
        # Single result view
        result = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Overlay on original if available
        if hasattr(image, 'fiber_found') or len(original.shape) == 2:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            
            # Create overlay
            overlay = np.zeros_like(original_bgr)
            overlay[:, :, 2] = processed  # Red channel
            
            result = cv2.addWeighted(original_bgr, 0.7, overlay, 0.3, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
        
        cv2.putText(result, f"Morphological {operation.capitalize()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Objects: {num_after}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add processing information
    info_text = f"Kernel: {kernel_size}x{kernel_size} {kernel_shape}"
    if iterations > 1:
        info_text += f", {iterations} iterations"
    cv2.putText(result, info_text, (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Store processed mask in metadata
    result.processed_mask = processed
    result.objects_before = num_before
    result.objects_after = num_after
    result.morphology_operation = operation
    
    return result