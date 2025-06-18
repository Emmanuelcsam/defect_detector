"""
DO2MR (Directional Opening with 2-D Median Residue) Defect Detection
====================================================================
Advanced morphological defect detection algorithm specifically designed for
fiber optic end-face inspection. Uses multi-scale analysis to detect various
defect types including pits, particles, and contamination.

This implementation includes adaptive thresholding and zone-specific sensitivity
adjustments for optimal defect detection in different fiber regions.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


def process_image(image: np.ndarray,
                  kernel_sizes: str = "3,5,7,9",
                  gamma: float = 1.5,
                  adaptive_sensitivity: float = 1.0,
                  enable_multi_scale: bool = True,
                  use_guided_filter: bool = True,
                  min_defect_area: int = 5,
                  apply_hysteresis: bool = True,
                  hysteresis_ratio: float = 0.6,
                  visualization_mode: str = "overlay") -> np.ndarray:
    """
    Apply DO2MR defect detection algorithm with enhanced multi-scale analysis.
    
    DO2MR (Directional Opening with 2-D Median Residue) is a morphological
    algorithm that detects defects by analyzing the residual between maximum
    and minimum filtered versions of the image at multiple scales.
    
    Args:
        image: Input image (grayscale or color)
        kernel_sizes: Comma-separated kernel sizes for multi-scale analysis
        gamma: Sensitivity parameter (0.5-3.0, higher = more sensitive)
        adaptive_sensitivity: Additional sensitivity adjustment (0.5-2.0)
        enable_multi_scale: Use multiple kernel sizes for better coverage
        use_guided_filter: Apply guided filter for edge-preserving smoothing
        min_defect_area: Minimum defect size in pixels to keep
        apply_hysteresis: Use hysteresis thresholding for better connectivity
        hysteresis_ratio: Ratio for low threshold (0.0-1.0)
        visualization_mode: Output mode ("overlay", "mask", "heatmap", "combined")
        
    Returns:
        Visualization of detected defects based on selected mode
        
    Technical Details:
        - Multi-scale morphological analysis captures defects of various sizes
        - Adaptive thresholding based on local statistics (median and MAD)
        - Hysteresis thresholding connects weak but adjacent defect regions
        - Guided filtering preserves edges while smoothing noise
    """
    # Parse kernel sizes
    try:
        kernel_list = [int(k.strip()) for k in kernel_sizes.split(',')]
        # Ensure odd kernel sizes
        kernel_list = [k if k % 2 == 1 else k + 1 for k in kernel_list]
    except:
        kernel_list = [3, 5, 7, 9]
    
    if not enable_multi_scale:
        kernel_list = kernel_list[:1]  # Use only first kernel size
    
    # Validate parameters
    gamma = max(0.5, min(3.0, gamma))
    adaptive_sensitivity = max(0.5, min(2.0, adaptive_sensitivity))
    hysteresis_ratio = max(0.0, min(1.0, hysteresis_ratio))
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray = image.copy()
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Pre-filter to reduce noise
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # Multi-scale DO2MR processing
    combined_result = np.zeros_like(gray, dtype=np.float32)
    confidence_maps = []
    
    for kernel_size in kernel_list:
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Max and min filtering (morphological operations)
        max_filtered = cv2.dilate(denoised, kernel)
        min_filtered = cv2.erode(denoised, kernel)
        
        # Calculate residual (difference between max and min)
        residual = cv2.subtract(max_filtered, min_filtered)
        
        # Apply guided filter if enabled
        if use_guided_filter:
            try:
                # Simple guided filter implementation
                residual_filtered = _guided_filter(denoised, residual, radius=3, eps=10)
            except:
                # Fallback to median blur if guided filter fails
                residual_filtered = cv2.medianBlur(residual, 3)
        else:
            residual_filtered = cv2.medianBlur(residual, 3)
        
        # Calculate adaptive threshold using robust statistics
        # Use Median Absolute Deviation (MAD) for robustness
        median_val = np.median(residual_filtered)
        mad = np.median(np.abs(residual_filtered - median_val))
        std_robust = 1.4826 * mad  # Conversion factor for normal distribution
        
        # Calculate adaptive gamma based on local contrast
        local_mean = np.mean(denoised)
        local_std = np.std(denoised)
        local_contrast = local_std / (local_mean + 1e-6)
        adaptive_gamma = gamma * adaptive_sensitivity * (1 + 0.5 * np.clip(local_contrast, 0, 1))
        
        if apply_hysteresis:
            # High and low thresholds for hysteresis
            threshold_high = median_val + adaptive_gamma * std_robust
            threshold_low = median_val + (adaptive_gamma * hysteresis_ratio) * std_robust
            
            # Apply thresholds
            _, high_mask = cv2.threshold(residual_filtered, threshold_high, 255, cv2.THRESH_BINARY)
            _, low_mask = cv2.threshold(residual_filtered, threshold_low, 255, cv2.THRESH_BINARY)
            
            # Convert to uint8
            high_mask = high_mask.astype(np.uint8)
            low_mask = low_mask.astype(np.uint8)
            
            # Morphological reconstruction
            kernel_recon = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            marker = cv2.erode(high_mask, kernel_recon)
            
            # Reconstruct from markers within low threshold regions
            reconstructed = cv2.dilate(marker, kernel_recon, iterations=2)
            reconstructed = cv2.bitwise_and(reconstructed, low_mask)
            
            # Include all high confidence regions
            result_for_scale = cv2.bitwise_or(reconstructed, high_mask)
        else:
            # Single threshold
            threshold = median_val + adaptive_gamma * std_robust
            _, result_for_scale = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
            result_for_scale = result_for_scale.astype(np.uint8)
        
        # Weight by kernel size (smaller kernels get higher weight for small defects)
        weight = 1.0 / (1 + 0.5 * np.log(kernel_size))
        combined_result += result_for_scale.astype(np.float32) * weight
        
        # Store confidence map
        confidence = residual_filtered.astype(np.float32) / 255.0
        confidence_maps.append(confidence * weight)
    
    # Normalize combined result
    if kernel_list:
        total_weight = sum(1.0 / (1 + 0.5 * np.log(k)) for k in kernel_list)
        combined_result = combined_result / total_weight
    
    # Final thresholding
    _, defect_mask = cv2.threshold(combined_result, 127, 255, cv2.THRESH_BINARY)
    defect_mask = defect_mask.astype(np.uint8)
    
    # Morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    cleaned_mask = np.zeros_like(defect_mask)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_defect_area:
            cleaned_mask[labels == i] = 255
    
    # Create confidence heatmap
    if confidence_maps:
        confidence_combined = np.sum(confidence_maps, axis=0) / len(confidence_maps)
    else:
        confidence_combined = np.zeros_like(gray, dtype=np.float32)
    
    # Generate visualization
    if visualization_mode == "mask":
        # Binary mask output
        result = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        
    elif visualization_mode == "heatmap":
        # Confidence heatmap
        heatmap = (confidence_combined * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        result = heatmap_colored
        
    elif visualization_mode == "overlay":
        # Overlay defects on original
        result = color_image.copy()
        defect_overlay = np.zeros_like(result)
        defect_overlay[cleaned_mask > 0] = (0, 0, 255)  # Red for defects
        result = cv2.addWeighted(result, 0.7, defect_overlay, 0.3, 0)
        
        # Add defect count
        num_defects = num_labels - 1  # Exclude background
        cv2.putText(result, f"Defects: {num_defects}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    elif visualization_mode == "combined":
        # Combined view: original, mask, heatmap, overlay
        mask_colored = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        
        heatmap = (confidence_combined * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        overlay = color_image.copy()
        defect_overlay = np.zeros_like(overlay)
        defect_overlay[cleaned_mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.7, defect_overlay, 0.3, 0)
        
        # Resize images for 2x2 grid
        h, w = color_image.shape[:2]
        half_h, half_w = h // 2, w // 2
        
        # Create grid
        top_row = np.hstack([
            cv2.resize(color_image, (half_w, half_h)),
            cv2.resize(mask_colored, (half_w, half_h))
        ])
        bottom_row = np.hstack([
            cv2.resize(heatmap_colored, (half_w, half_h)),
            cv2.resize(overlay, (half_w, half_h))
        ])
        result = np.vstack([top_row, bottom_row])
        
        # Add labels
        label_color = (255, 255, 255)
        cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        cv2.putText(result, "Mask", (half_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        cv2.putText(result, "Confidence", (10, half_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        cv2.putText(result, "Overlay", (half_w + 10, half_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
    
    else:
        result = color_image
    
    return result


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 3, eps: float = 10) -> np.ndarray:
    """
    Simple guided filter implementation for edge-preserving smoothing.
    
    Args:
        guide: Guide image
        src: Source image to filter
        radius: Filter radius
        eps: Regularization parameter
        
    Returns:
        Filtered image
    """
    # Convert to float
    guide = guide.astype(np.float32) / 255.0
    src = src.astype(np.float32) / 255.0
    
    # Box filter kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*radius+1, 2*radius+1))
    kernel = kernel.astype(np.float32) / np.sum(kernel)
    
    # Compute local statistics
    mean_guide = cv2.filter2D(guide, -1, kernel)
    mean_src = cv2.filter2D(src, -1, kernel)
    corr_guide_src = cv2.filter2D(guide * src, -1, kernel)
    corr_guide = cv2.filter2D(guide * guide, -1, kernel)
    
    # Compute coefficients
    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src
    
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide
    
    # Apply filter
    mean_a = cv2.filter2D(a, -1, kernel)
    mean_b = cv2.filter2D(b, -1, kernel)
    
    output = mean_a * guide + mean_b
    output = (output * 255).astype(np.uint8)
    
    return output


# Test code
if __name__ == "__main__":
    # Create test image with synthetic defects
    test_size = 300
    test_image = np.ones((test_size, test_size), dtype=np.uint8) * 200
    
    # Add various defects
    # Small pits
    for _ in range(10):
        x, y = np.random.randint(50, test_size-50, 2)
        radius = np.random.randint(2, 5)
        cv2.circle(test_image, (x, y), radius, 100, -1)
    
    # Larger contamination
    for _ in range(3):
        x, y = np.random.randint(50, test_size-50, 2)
        radius = np.random.randint(8, 15)
        cv2.circle(test_image, (x, y), radius, 120, -1)
    
    # Add noise
    noise = np.random.normal(0, 5, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Test different visualization modes
    modes = ["overlay", "mask", "heatmap", "combined"]
    
    for i, mode in enumerate(modes):
        result = process_image(
            test_image,
            kernel_sizes="3,5,7",
            gamma=1.5,
            visualization_mode=mode
        )
        
        cv2.imshow(f"DO2MR - {mode}", result)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
