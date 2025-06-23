"""
Min-Max Filtering - Apply maximum and minimum filters for defect detection preparation
Parallel processing of max/min filters to highlight intensity variations
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  kernel_size: int = 5,
                  kernel_shape: str = "square",
                  iterations: int = 1,
                  output_mode: str = "both",
                  show_original: bool = False,
                  normalize_output: bool = True) -> np.ndarray:
    """
    Apply minimum and maximum filters to prepare for residual-based defect detection.
    
    This function implements the min-max filtering approach used in the DO2MR method.
    Maximum filter replaces each pixel with the maximum in its neighborhood,
    while minimum filter uses the minimum value. The difference between these
    highlights areas of sharp intensity change (defects).
    
    Args:
        image: Input preprocessed fiber optic image
        kernel_size: Size of the filter kernel
        kernel_shape: Shape of kernel - "square", "cross", or "circle"
        iterations: Number of times to apply the filters
        output_mode: "both", "max", "min", or "difference"
        show_original: Include original image in output
        normalize_output: Whether to normalize the output values
        
    Returns:
        Filtered image(s) based on output mode
    """
    # Ensure odd kernel size
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Work with grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # Create kernel based on shape
    if kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    elif kernel_shape == "circle":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    else:  # square
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply maximum filter (dilation in morphological terms)
    max_filtered = gray.copy()
    for _ in range(iterations):
        max_filtered = cv2.dilate(max_filtered, kernel)
    
    # Apply minimum filter (erosion in morphological terms)
    min_filtered = gray.copy()
    for _ in range(iterations):
        min_filtered = cv2.erode(min_filtered, kernel)
    
    # Calculate difference (residual)
    difference = cv2.subtract(max_filtered, min_filtered)
    
    # Normalize if requested
    if normalize_output:
        max_filtered = cv2.normalize(max_filtered, None, 0, 255, cv2.NORM_MINMAX)
        min_filtered = cv2.normalize(min_filtered, None, 0, 255, cv2.NORM_MINMAX)
        difference = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
    
    # Generate output based on mode
    if output_mode == "max":
        result = cv2.cvtColor(max_filtered, cv2.COLOR_GRAY2BGR)
        cv2.putText(result, "Maximum Filter", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    elif output_mode == "min":
        result = cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
        cv2.putText(result, "Minimum Filter", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    elif output_mode == "difference":
        # Apply colormap to difference for better visualization
        result = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
        cv2.putText(result, "Max-Min Difference", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    else:  # both mode - show all results
        # Create a grid display
        # Convert to BGR for display
        original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        max_bgr = cv2.cvtColor(max_filtered, cv2.COLOR_GRAY2BGR)
        min_bgr = cv2.cvtColor(min_filtered, cv2.COLOR_GRAY2BGR)
        diff_bgr = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
        
        # Add labels
        cv2.putText(original_bgr, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(max_bgr, "Maximum Filter", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(min_bgr, "Minimum Filter", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(diff_bgr, "Difference (Residual)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create 2x2 grid
        top_row = np.hstack([original_bgr, max_bgr])
        bottom_row = np.hstack([min_bgr, diff_bgr])
        result = np.vstack([top_row, bottom_row])
        
        # Add grid lines
        cv2.line(result, (w, 0), (w, 2*h), (255, 255, 255), 2)
        cv2.line(result, (0, h), (2*w, h), (255, 255, 255), 2)
    
    # Add filter information
    info_text = f"Kernel: {kernel_size}x{kernel_size} {kernel_shape}"
    if iterations > 1:
        info_text += f", {iterations} iterations"
    
    cv2.putText(result, info_text, (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Store filtered images in metadata for pipeline use
    result.max_filtered = max_filtered
    result.min_filtered = min_filtered
    result.difference = difference
    result.kernel_size = kernel_size
    result.kernel_shape = kernel_shape
    
    return result