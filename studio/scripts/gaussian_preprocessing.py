"""
Gaussian Filter Preprocessing - Smooth image to reduce noise influence on defect detection
Essential preprocessing step for fiber optic defect analysis
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  kernel_size: int = 5,
                  sigma_x: float = 0.0,
                  sigma_y: float = 0.0,
                  adaptive_sigma: bool = True,
                  edge_preserve_mode: str = "none",
                  bilateral_d: int = 9,
                  bilateral_sigma_color: float = 75.0,
                  bilateral_sigma_space: float = 75.0,
                  show_difference: bool = False) -> np.ndarray:
    """
    Apply Gaussian filtering as preprocessing for defect detection.
    
    This function implements various Gaussian-based filtering techniques optimized
    for fiber optic images. It reduces noise while attempting to preserve important
    features like defects and fiber boundaries.
    
    Args:
        image: Input fiber optic image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma_x: Gaussian kernel standard deviation in X direction (0 = auto)
        sigma_y: Gaussian kernel standard deviation in Y direction (0 = auto)
        adaptive_sigma: Whether to adapt sigma based on image content
        edge_preserve_mode: "none", "bilateral", or "guided" for edge preservation
        bilateral_d: Diameter for bilateral filter
        bilateral_sigma_color: Bilateral filter sigma in color space
        bilateral_sigma_space: Bilateral filter sigma in coordinate space
        show_difference: Show the difference between original and filtered
        
    Returns:
        Filtered image ready for defect detection
    """
    # Ensure odd kernel size
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Work with appropriate format
    if len(image.shape) == 2:
        working_image = image.copy()
        is_grayscale = True
    else:
        # Convert to grayscale for processing but keep color for output
        working_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_grayscale = False
    
    h, w = working_image.shape
    
    # Calculate adaptive sigma if requested
    if adaptive_sigma and (sigma_x == 0 or sigma_y == 0):
        # Analyze image characteristics
        # Calculate local standard deviation as a measure of detail
        kernel_std = np.ones((5, 5)) / 25
        mean = cv2.filter2D(working_image.astype(float), -1, kernel_std)
        mean_sq = cv2.filter2D(working_image.astype(float)**2, -1, kernel_std)
        variance = mean_sq - mean**2
        std_dev = np.sqrt(np.maximum(variance, 0))
        
        # Use median of local std dev to set sigma
        median_std = np.median(std_dev)
        
        # Adaptive sigma: higher std dev = more detail = lower sigma
        if sigma_x == 0:
            sigma_x = max(0.5, min(5.0, 30.0 / (median_std + 10)))
        if sigma_y == 0:
            sigma_y = sigma_x
    else:
        # Use automatic sigma calculation if zero
        if sigma_x == 0:
            sigma_x = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        if sigma_y == 0:
            sigma_y = sigma_x
    
    # Apply filtering based on edge preservation mode
    if edge_preserve_mode == "bilateral":
        # Bilateral filter preserves edges better
        filtered = cv2.bilateralFilter(
            working_image, 
            bilateral_d, 
            bilateral_sigma_color, 
            bilateral_sigma_space
        )
        
    elif edge_preserve_mode == "guided":
        # Guided filter (implemented using edge-aware smoothing)
        # First apply regular Gaussian
        temp = cv2.GaussianBlur(working_image, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
        
        # Detect edges
        edges = cv2.Canny(working_image, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Blend original and filtered based on edges
        edge_mask = (255 - edges).astype(float) / 255.0
        filtered = (working_image * (1 - edge_mask) + temp * edge_mask).astype(np.uint8)
        
    else:  # Standard Gaussian
        filtered = cv2.GaussianBlur(working_image, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
    
    # Prepare output
    if show_difference:
        # Show the difference map
        diff = cv2.absdiff(working_image, filtered)
        
        # Enhance difference for visualization
        diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create a three-panel view
        panel1 = cv2.cvtColor(working_image, cv2.COLOR_GRAY2BGR)
        panel2 = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        panel3 = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_HOT)
        
        # Add labels
        cv2.putText(panel1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(panel2, "Filtered", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(panel3, "Difference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine panels
        if w > h:
            result = np.vstack([panel1, panel2, panel3])
        else:
            result = np.hstack([panel1, panel2, panel3])
            
    else:
        # Return filtered result
        if is_grayscale:
            result = filtered
        else:
            # Apply same filtering to each channel for color images
            result = image.copy()
            for i in range(3):
                if edge_preserve_mode == "bilateral":
                    result[:, :, i] = cv2.bilateralFilter(
                        image[:, :, i], 
                        bilateral_d, 
                        bilateral_sigma_color, 
                        bilateral_sigma_space
                    )
                else:
                    result[:, :, i] = cv2.GaussianBlur(
                        image[:, :, i], 
                        (kernel_size, kernel_size), 
                        sigma_x, 
                        sigmaY=sigma_y
                    )
    
    # Add processing information
    info_text = f"Gaussian Filter: {kernel_size}x{kernel_size}, sigma=({sigma_x:.2f}, {sigma_y:.2f})"
    if edge_preserve_mode != "none":
        info_text += f", {edge_preserve_mode}"
    
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(result, info_text, (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Store filter parameters in metadata
    result.gaussian_kernel_size = kernel_size
    result.gaussian_sigma_x = sigma_x
    result.gaussian_sigma_y = sigma_y
    result.filter_mode = edge_preserve_mode
    
    return result