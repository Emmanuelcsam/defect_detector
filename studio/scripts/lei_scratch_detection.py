"""
LEI (Linear Element Imaging) Scratch Detection
==============================================
Advanced scratch and linear defect detection algorithm using multi-scale
and multi-directional filtering. Optimized for detecting scratches, cracks,
and other linear defects on fiber optic end-faces.

This implementation includes enhanced directional filtering with Gaussian-weighted
kernels and multi-scale analysis for robust detection of linear features.
"""
import cv2
import numpy as np
from typing import List, Tuple


def process_image(image: np.ndarray,
                  kernel_lengths: str = "7,11,15,21,31",
                  angle_step: int = 10,
                  detection_scales: str = "1.0,0.75,1.25",
                  enhance_contrast: bool = True,
                  use_tophat: bool = True,
                  min_scratch_length: int = 10,
                  min_aspect_ratio: float = 2.5,
                  threshold_method: str = "combined",
                  visualization_mode: str = "overlay") -> np.ndarray:
    """
    Detect scratches and linear defects using Linear Element Imaging (LEI).
    
    LEI uses directional filtering at multiple orientations and scales to
    detect linear features. It's particularly effective for scratches that
    may be faint or have varying orientations.
    
    Args:
        image: Input image (grayscale or color)
        kernel_lengths: Comma-separated list of kernel lengths to use
        angle_step: Angular resolution in degrees (5-30)
        detection_scales: Comma-separated scale factors for multi-scale analysis
        enhance_contrast: Apply CLAHE for better scratch visibility
        use_tophat: Apply morphological top-hat before detection
        min_scratch_length: Minimum length for valid scratches (pixels)
        min_aspect_ratio: Minimum aspect ratio (length/width) for scratches
        threshold_method: Thresholding method ("otsu", "adaptive", "combined")
        visualization_mode: Output mode ("overlay", "mask", "enhanced", "directional")
        
    Returns:
        Visualization of detected scratches based on selected mode
        
    Technical Details:
        - Multi-directional filtering captures scratches at any angle
        - Gaussian-weighted kernels provide better response to linear features
        - Multi-scale processing detects both fine and wide scratches
        - Non-maximum suppression removes duplicate detections
    """
    # Parse parameters
    try:
        kernel_list = [int(k.strip()) for k in kernel_lengths.split(',')]
        scale_list = [float(s.strip()) for s in detection_scales.split(',')]
    except:
        kernel_list = [7, 11, 15, 21, 31]
        scale_list = [1.0, 0.75, 1.25]
    
    # Validate parameters
    angle_step = max(5, min(30, angle_step))
    min_aspect_ratio = max(1.5, min(10.0, min_aspect_ratio))
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray = image.copy()
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Initialize result storage
    all_scratch_maps = []
    directional_responses = {}
    
    # Multi-scale processing
    for scale in scale_list:
        # Resize image if needed
        if scale != 1.0:
            scaled_h = int(gray.shape[0] * scale)
            scaled_w = int(gray.shape[1] * scale)
            scaled_image = cv2.resize(gray, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = gray.copy()
        
        # Enhance contrast if requested
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(scaled_image)
        else:
            enhanced = scaled_image
        
        # Apply morphological top-hat to enhance linear structures
        if use_tophat:
            kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_tophat)
            # Also compute bottom-hat for dark scratches
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_tophat)
            # Combine both
            enhanced = cv2.add(tophat, blackhat)
        
        # Initialize scratch map for this scale
        scratch_map = np.zeros_like(enhanced, dtype=np.float32)
        
        # Process each orientation
        for angle in range(0, 180, angle_step):
            angle_rad = np.deg2rad(angle)
            
            # Response for this angle
            angle_response = np.zeros_like(enhanced, dtype=np.float32)
            
            for kernel_length in kernel_list:
                # Create enhanced Gaussian-weighted linear kernel
                kernel = _create_gaussian_line_kernel(kernel_length, angle)
                
                # Apply directional filter
                response = cv2.filter2D(enhanced.astype(np.float32), cv2.CV_32F, kernel)
                
                # Non-maximum suppression in perpendicular direction
                nms_response = _directional_nms(response, angle + 90)
                
                # Accumulate response
                angle_response = np.maximum(angle_response, nms_response)
            
            # Update scratch map with maximum response
            scratch_map = np.maximum(scratch_map, angle_response)
            
            # Store directional response for visualization
            if scale == 1.0:  # Only store for original scale
                directional_responses[angle] = angle_response
        
        # Resize back to original size if needed
        if scale != 1.0:
            scratch_map = cv2.resize(scratch_map, (gray.shape[1], gray.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        
        all_scratch_maps.append(scratch_map)
    
    # Combine multi-scale results
    combined_scratch_map = np.mean(all_scratch_maps, axis=0)
    
    # Normalize to 0-255 range
    combined_scratch_map = cv2.normalize(combined_scratch_map, None, 0, 255, 
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply thresholding
    if threshold_method == "otsu":
        _, binary = cv2.threshold(combined_scratch_map, 0, 255, 
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(combined_scratch_map, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, -2)
    elif threshold_method == "combined":
        # Use both Otsu and adaptive, combine results
        _, otsu = cv2.threshold(combined_scratch_map, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(combined_scratch_map, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, -2)
        binary = cv2.bitwise_or(otsu, adaptive)
    else:
        # Simple threshold
        _, binary = cv2.threshold(combined_scratch_map, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to connect scratch fragments
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_connect)
    
    # Filter by linear characteristics
    cleaned_mask = _filter_linear_components(binary, min_scratch_length, min_aspect_ratio)
    
    # Generate visualization
    if visualization_mode == "mask":
        # Binary mask
        result = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        
    elif visualization_mode == "enhanced":
        # Enhanced scratch visibility
        enhanced_vis = cv2.normalize(combined_scratch_map, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_colored = cv2.applyColorMap(enhanced_vis, cv2.COLORMAP_HOT)
        result = enhanced_colored
        
    elif visualization_mode == "directional":
        # Directional response visualization
        result = _create_directional_visualization(directional_responses, gray.shape[:2])
        
    elif visualization_mode == "overlay":
        # Overlay on original
        result = color_image.copy()
        
        # Create colored overlay for scratches
        scratch_overlay = np.zeros_like(result)
        scratch_overlay[cleaned_mask > 0] = (0, 255, 255)  # Yellow for scratches
        
        # Find individual scratches for labeling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8
        )
        
        # Draw each scratch with bounding box
        for i in range(1, num_labels):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # Blend overlay
        result = cv2.addWeighted(result, 0.7, scratch_overlay, 0.3, 0)
        
        # Add scratch count
        cv2.putText(result, f"Scratches: {num_labels - 1}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    else:
        result = color_image
    
    return result


def _create_gaussian_line_kernel(length: int, angle: float, sigma_ratio: float = 6.0) -> np.ndarray:
    """
    Create a Gaussian-weighted linear kernel at specified angle.
    
    Args:
        length: Length of the line kernel
        angle: Angle in degrees
        sigma_ratio: Ratio for Gaussian sigma (length/sigma_ratio)
        
    Returns:
        Rotated Gaussian line kernel
    """
    # Create kernel with padding for rotation
    kernel_size = length + 4
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    # Create Gaussian-weighted line
    sigma = length / sigma_ratio
    for i in range(length):
        pos = i - length // 2
        weight = np.exp(-pos**2 / (2 * sigma**2))
        y = center + pos
        if 0 <= y < kernel_size:
            kernel[y, center] = weight
    
    # Rotate kernel
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    
    # Normalize
    kernel_sum = np.sum(rotated_kernel)
    if kernel_sum > 0:
        rotated_kernel = rotated_kernel / kernel_sum
    
    return rotated_kernel


def _directional_nms(response: np.ndarray, angle: float, kernel_size: int = 5) -> np.ndarray:
    """
    Apply non-maximum suppression in specified direction.
    
    Args:
        response: Filter response map
        angle: Direction for NMS (perpendicular to features)
        kernel_size: Size of NMS kernel
        
    Returns:
        Response after NMS
    """
    # Create directional kernel for NMS
    nms_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    nms_kernel[center, :] = 1.0
    
    # Rotate for specified angle
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    nms_kernel_rot = cv2.warpAffine(nms_kernel, M, (kernel_size, kernel_size))
    
    # Apply NMS
    local_max = cv2.dilate(response, nms_kernel_rot)
    nms_result = np.where(response >= local_max * 0.95, response, 0)
    
    return nms_result


def _filter_linear_components(binary: np.ndarray, min_length: int, min_aspect_ratio: float) -> np.ndarray:
    """
    Filter connected components by linear characteristics.
    
    Args:
        binary: Binary mask
        min_length: Minimum length for valid components
        min_aspect_ratio: Minimum aspect ratio
        
    Returns:
        Filtered binary mask
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create filtered mask
    filtered_mask = np.zeros_like(binary)
    
    for i in range(1, num_labels):
        # Get component properties
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calculate aspect ratio
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            length = max(width, height)
            
            # Keep if linear enough and long enough
            if aspect_ratio >= min_aspect_ratio and length >= min_length:
                filtered_mask[labels == i] = 255
    
    return filtered_mask


def _create_directional_visualization(directional_responses: dict, shape: Tuple[int, int]) -> np.ndarray:
    """
    Create visualization showing directional responses.
    
    Args:
        directional_responses: Dictionary of angle -> response map
        shape: Output image shape
        
    Returns:
        Color-coded directional visualization
    """
    # Create HSV image for directional encoding
    h, w = shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Combine all responses
    max_response = np.zeros((h, w), dtype=np.float32)
    max_angle = np.zeros((h, w), dtype=np.float32)
    
    for angle, response in directional_responses.items():
        mask = response > max_response
        max_response[mask] = response[mask]
        max_angle[mask] = angle
    
    # Normalize response
    if np.max(max_response) > 0:
        max_response = max_response / np.max(max_response)
    
    # Encode angle as hue, response as value
    hsv[:, :, 0] = (max_angle * 255 / 180).astype(np.uint8)  # Hue
    hsv[:, :, 1] = 255  # Full saturation
    hsv[:, :, 2] = (max_response * 255).astype(np.uint8)  # Value
    
    # Convert to BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add color wheel legend
    legend_size = 60
    center = (w - legend_size - 10, legend_size + 10)
    for angle in range(0, 360, 10):
        color_hsv = np.array([[[angle // 2, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        end_x = int(center[0] + legend_size//2 * np.cos(np.radians(angle)))
        end_y = int(center[1] + legend_size//2 * np.sin(np.radians(angle)))
        cv2.line(result, center, (end_x, end_y), color_bgr.tolist(), 2)
    
    return result


# Test code
if __name__ == "__main__":
    # Create test image with synthetic scratches
    test_size = 400
    test_image = np.ones((test_size, test_size), dtype=np.uint8) * 200
    
    # Add scratches at various angles
    scratch_params = [
        (100, 50, 300, 150, 2),    # Diagonal scratch
        (50, 200, 350, 200, 1),    # Horizontal scratch
        (200, 50, 200, 350, 3),    # Vertical scratch
        (50, 300, 150, 350, 1),    # Short diagonal
    ]
    
    for x1, y1, x2, y2, thickness in scratch_params:
        cv2.line(test_image, (x1, y1), (x2, y2), 100, thickness)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Test different visualization modes
    modes = ["overlay", "mask", "enhanced", "directional"]
    
    for mode in modes:
        result = process_image(
            test_image,
            kernel_lengths="7,11,15,21",
            angle_step=15,
            visualization_mode=mode
        )
        
        cv2.imshow(f"LEI Scratch Detection - {mode}", result)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
