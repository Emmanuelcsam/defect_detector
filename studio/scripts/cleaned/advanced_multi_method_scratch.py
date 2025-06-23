"""
Advanced Multi-Method Scratch Detection
=======================================
Combines multiple advanced scratch detection techniques including gradient-based,
Gabor filters, Hessian matrix analysis, morphological operations, and frequency
domain analysis. Each method captures different aspects of scratch characteristics.

This comprehensive approach ensures robust detection of various scratch types
including fine scratches, deep grooves, and surface marks.
"""
import cv2
import numpy as np
from scipy import ndimage
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def process_image(image: np.ndarray,
                  detection_methods: str = "gradient,gabor,hessian",
                  fusion_mode: str = "weighted",
                  gradient_weight: float = 1.0,
                  gabor_weight: float = 1.0,
                  hessian_weight: float = 1.0,
                  morphological_weight: float = 0.8,
                  frequency_weight: float = 0.7,
                  min_scratch_length: int = 20,
                  visualization_mode: str = "overlay") -> np.ndarray:
    """
    Detect scratches using multiple advanced methods with intelligent fusion.
    
    Combines different detection algorithms that excel at finding specific
    scratch characteristics:
    - Gradient: Edge-based detection for sharp scratches
    - Gabor: Orientation-specific texture analysis
    - Hessian: Ridge/valley detection for continuous scratches
    - Morphological: Shape-based detection
    - Frequency: Periodic pattern detection
    
    Args:
        image: Input image (grayscale or color)
        detection_methods: Comma-separated list of methods to use
        fusion_mode: How to combine results ("weighted", "voting", "maximum")
        gradient_weight: Weight for gradient method (0.0-2.0)
        gabor_weight: Weight for Gabor method (0.0-2.0)
        hessian_weight: Weight for Hessian method (0.0-2.0)
        morphological_weight: Weight for morphological method (0.0-2.0)
        frequency_weight: Weight for frequency method (0.0-2.0)
        min_scratch_length: Minimum length for valid scratches
        visualization_mode: Output mode ("overlay", "individual", "combined", "confidence")
        
    Returns:
        Visualization of detected scratches based on selected mode
    """
    # Parse methods
    available_methods = {
        'gradient': (_gradient_based_detection, gradient_weight),
        'gabor': (_gabor_based_detection, gabor_weight),
        'hessian': (_hessian_based_detection, hessian_weight),
        'morphological': (_morphological_detection, morphological_weight),
        'frequency': (_frequency_based_detection, frequency_weight)
    }
    
    method_list = [m.strip().lower() for m in detection_methods.split(',')]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray = image.copy()
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Run selected detection methods
    individual_results = {}
    weighted_sum = np.zeros_like(gray, dtype=np.float32)
    total_weight = 0
    
    for method_name in method_list:
        if method_name in available_methods:
            detect_func, weight = available_methods[method_name]
            try:
                result = detect_func(gray)
                individual_results[method_name] = result
                
                # Add to weighted sum
                weighted_sum += result.astype(np.float32) * weight
                total_weight += weight
                
            except Exception as e:
                print(f"Warning: {method_name} detection failed: {e}")
                individual_results[method_name] = np.zeros_like(gray, dtype=np.uint8)
    
    # Combine results based on fusion mode
    if fusion_mode == "weighted" and total_weight > 0:
        # Weighted average
        combined = (weighted_sum / total_weight).astype(np.uint8)
        
    elif fusion_mode == "voting":
        # Voting: pixel is scratch if majority of methods agree
        vote_map = np.zeros_like(gray, dtype=np.float32)
        for result in individual_results.values():
            vote_map += (result > 0).astype(np.float32)
        threshold = len(individual_results) / 2
        combined = (vote_map > threshold).astype(np.uint8) * 255
        
    elif fusion_mode == "maximum":
        # Maximum: pixel is scratch if any method detects it
        combined = np.zeros_like(gray, dtype=np.uint8)
        for result in individual_results.values():
            combined = np.maximum(combined, result)
    else:
        combined = weighted_sum.astype(np.uint8)
    
    # Post-process combined result
    final_mask = _postprocess_scratch_mask(combined, min_scratch_length)
    
    # Generate visualization
    if visualization_mode == "individual":
        # Show all methods side by side
        result = _create_individual_visualization(individual_results, gray.shape)
        
    elif visualization_mode == "confidence":
        # Confidence map showing agreement between methods
        confidence = (weighted_sum / (total_weight * 255) * 255).astype(np.uint8)
        confidence_colored = cv2.applyColorMap(confidence, cv2.COLORMAP_JET)
        result = confidence_colored
        
    elif visualization_mode == "combined":
        # Combined view with original, final mask, and individual results
        result = _create_combined_visualization(
            color_image, final_mask, individual_results
        )
        
    elif visualization_mode == "overlay":
        # Overlay on original
        result = color_image.copy()
        
        # Create colored overlay
        scratch_overlay = np.zeros_like(result)
        scratch_overlay[final_mask > 0] = (0, 255, 255)  # Yellow
        
        # Find individual scratches
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            final_mask, connectivity=8
        )
        
        # Draw bounding boxes
        for i in range(1, num_labels):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # Blend
        result = cv2.addWeighted(result, 0.7, scratch_overlay, 0.3, 0)
        
        # Add info
        cv2.putText(result, f"Methods: {len(individual_results)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Scratches: {num_labels - 1}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    else:
        result = color_image
    
    return result


def _gradient_based_detection(gray: np.ndarray) -> np.ndarray:
    """Multi-scale gradient analysis for scratch detection."""
    scales = [1, 2, 3]
    gradient_responses = []
    
    for scale in scales:
        # Gaussian smoothing
        sigma = scale * 0.5
        smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # Compute gradients
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Gradient direction
        grad_dir = np.arctan2(grad_y, grad_x)
        
        # Directional non-maximum suppression
        nms_result = _directional_nms_gradient(grad_mag, grad_dir)
        
        gradient_responses.append(nms_result)
    
    # Combine multi-scale responses
    combined = np.max(gradient_responses, axis=0)
    
    # Normalize and threshold
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def _gabor_based_detection(gray: np.ndarray) -> np.ndarray:
    """Gabor filter based scratch detection."""
    # Gabor parameters
    ksize = 21
    sigma = 3.0
    lambd = 10.0
    gamma = 0.5
    psi = 0
    
    # Multiple orientations
    orientations = np.arange(0, np.pi, np.pi / 8)
    gabor_responses = []
    
    for theta in orientations:
        # Create Gabor kernel
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
        )
        
        # Apply filter
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        response = np.abs(filtered)
        gabor_responses.append(response)
    
    # Max response across orientations
    max_response = np.max(gabor_responses, axis=0)
    
    # Enhance linear structures
    enhanced = np.zeros_like(max_response)
    
    for theta in orientations:
        # Create oriented kernel
        kernel_length = 15
        kernel = _create_oriented_kernel(kernel_length, theta, width=3)
        
        # Morphological closing
        max_response_uint8 = cv2.normalize(
            max_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        closed = cv2.morphologyEx(max_response_uint8, cv2.MORPH_CLOSE, kernel)
        enhanced = np.maximum(enhanced, closed)
    
    # Normalize and threshold
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def _hessian_based_detection(gray: np.ndarray) -> np.ndarray:
    """Hessian matrix based ridge detection."""
    try:
        from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    except ImportError:
        # Fallback implementation
        return _simple_ridge_detection(gray)
    
    scales = [1.0, 1.5, 2.0]
    hessian_responses = []
    
    for sigma in scales:
        # Compute Hessian matrix
        Hxx, Hxy, Hyy = hessian_matrix(gray, sigma=sigma, order='xy')
        lambda1, lambda2 = hessian_matrix_eigvals((Hxx, Hxy, Hyy))
        
        # Frangi vesselness filter parameters
        beta = 0.5
        c = 15
        
        # Ensure correct eigenvalue ordering
        abs_lambda1 = np.abs(lambda1)
        abs_lambda2 = np.abs(lambda2)
        idx = abs_lambda1 > abs_lambda2
        lambda1[idx], lambda2[idx] = lambda2[idx], lambda1[idx]
        
        # Ridge measure
        Rb_sq = (lambda1 / (lambda2 + 1e-10))**2
        S_sq = lambda1**2 + lambda2**2
        
        # Vesselness
        vesselness = np.exp(-Rb_sq / (2 * beta**2)) * (1 - np.exp(-S_sq / (2 * c**2)))
        vesselness[lambda2 < 0] = 0
        
        hessian_responses.append(vesselness)
    
    # Combine scales
    combined = np.max(hessian_responses, axis=0) if hessian_responses else np.zeros_like(gray)
    
    # Normalize and threshold
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)
    
    return binary


def _simple_ridge_detection(gray: np.ndarray) -> np.ndarray:
    """Simple ridge detection fallback."""
    # Use Laplacian of Gaussian for ridge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Find zero crossings (ridges)
    laplacian_sign = np.sign(laplacian)
    zero_crossing = np.logical_or(
        np.logical_and(laplacian_sign[:-1, :] > 0, laplacian_sign[1:, :] < 0),
        np.logical_and(laplacian_sign[:-1, :] < 0, laplacian_sign[1:, :] > 0)
    )
    
    # Convert to uint8
    result = np.zeros_like(gray)
    result[:-1, :][zero_crossing] = 255
    
    return result


def _morphological_detection(gray: np.ndarray) -> np.ndarray:
    """Morphological approach for scratch detection."""
    orientations = np.arange(0, 180, 15)
    tophat_responses = []
    
    for angle in orientations:
        # Create oriented structuring element
        kernel_length = 21
        kernel = _create_oriented_kernel(kernel_length, np.deg2rad(angle), width=3)
        
        # Black top-hat for dark scratches
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # White top-hat for bright scratches
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Combine both
        combined = cv2.add(blackhat, tophat)
        tophat_responses.append(combined)
    
    # Max response across orientations
    max_response = np.max(tophat_responses, axis=0)
    
    # Normalize and threshold
    enhanced = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return binary


def _frequency_based_detection(gray: np.ndarray) -> np.ndarray:
    """Frequency domain analysis for periodic scratches."""
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create directional filters
    orientations = np.arange(0, 180, 15)
    filtered_images = []
    
    for angle in orientations:
        # Create oriented band-pass filter
        mask = _create_oriented_bandpass(rows, cols, angle, width=10)
        
        # Apply filter
        filtered_fshift = f_shift * mask
        
        # Inverse FFT
        filtered_ishift = np.fft.ifftshift(filtered_fshift)
        filtered_image = np.fft.ifft2(filtered_ishift)
        filtered_image = np.abs(filtered_image)
        
        filtered_images.append(filtered_image)
    
    # Combine filtered images
    combined = np.max(filtered_images, axis=0)
    
    # Normalize and threshold
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def _create_oriented_kernel(length: int, angle: float, width: int = 3) -> np.ndarray:
    """Create an oriented linear kernel."""
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    
    # Draw horizontal line
    kernel[center-width//2:center+width//2+1, :] = 1
    
    # Rotate kernel
    M = cv2.getRotationMatrix2D((center, center), np.degrees(angle), 1)
    rotated = cv2.warpAffine(kernel, M, (length, length))
    
    return rotated


def _create_oriented_bandpass(rows: int, cols: int, angle: float, width: int) -> np.ndarray:
    """Create oriented band-pass filter in frequency domain."""
    u = np.arange(cols) - cols // 2
    v = np.arange(rows) - rows // 2
    U, V = np.meshgrid(u, v)
    
    # Rotate coordinates
    angle_rad = np.deg2rad(angle)
    U_rot = U * np.cos(angle_rad) + V * np.sin(angle_rad)
    V_rot = -U * np.sin(angle_rad) + V * np.cos(angle_rad)
    
    # Create band-pass filter
    mask = np.exp(-(V_rot**2) / (2 * width**2))
    
    # Suppress DC component
    center_mask = np.sqrt(U**2 + V**2) > 5
    mask = mask * center_mask
    
    return mask


def _directional_nms_gradient(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Directional non-maximum suppression for gradient method."""
    rows, cols = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255
            
            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
    
    return suppressed


def _postprocess_scratch_mask(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Post-process scratch detection results."""
    if np.sum(mask) == 0:
        return mask
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        # Get component properties
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Check if it looks like a scratch
        if min(width, height) > 0:
            aspect_ratio = max(width, height) / min(width, height)
            
            # Keep if elongated and large enough
            if aspect_ratio > 3 and max(width, height) > min_length:
                cleaned_mask[labels == i] = 255
    
    # Apply thinning to get centerlines
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    return cleaned_mask


def _create_individual_visualization(results: Dict[str, np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Create visualization showing individual method results."""
    h, w = shape
    n_methods = len(results)
    
    # Calculate grid layout
    grid_cols = min(3, n_methods)
    grid_rows = (n_methods + grid_cols - 1) // grid_cols
    
    # Create output image
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    output = np.zeros((cell_h * grid_rows, cell_w * grid_cols, 3), dtype=np.uint8)
    
    # Place each result
    for idx, (method_name, result) in enumerate(results.items()):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Resize result to cell size
        resized = cv2.resize(result, (cell_w, cell_h))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        # Place in grid
        y1 = row * cell_h
        y2 = (row + 1) * cell_h
        x1 = col * cell_w
        x2 = (col + 1) * cell_w
        output[y1:y2, x1:x2] = resized
        
        # Add label
        cv2.putText(output, method_name, (x1 + 5, y1 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return output


def _create_combined_visualization(original: np.ndarray, final_mask: np.ndarray, 
                                 individual_results: Dict[str, np.ndarray]) -> np.ndarray:
    """Create comprehensive combined visualization."""
    h, w = original.shape[:2]
    
    # Create panels
    # Top: Original and final result
    final_colored = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    top_row = np.hstack([original, final_colored])
    
    # Bottom: Individual method results
    method_vis = _create_individual_visualization(individual_results, (h, w*2))
    
    # Combine
    if method_vis.shape[1] == top_row.shape[1]:
        result = np.vstack([top_row, method_vis])
    else:
        # Resize to match
        method_vis = cv2.resize(method_vis, (top_row.shape[1], method_vis.shape[0]))
        result = np.vstack([top_row, method_vis])
    
    return result


# Test code
if __name__ == "__main__":
    # Create test image with various scratch types
    test_size = 400
    test_image = np.ones((test_size, test_size), dtype=np.uint8) * 200
    
    # Add different types of scratches
    # Fine scratch
    cv2.line(test_image, (50, 50), (200, 100), 150, 1)
    
    # Deep scratch
    cv2.line(test_image, (100, 200), (300, 250), 100, 3)
    
    # Curved scratch
    pts = np.array([[50, 300], [150, 320], [250, 310], [350, 330]], np.int32)
    cv2.polylines(test_image, [pts], False, 120, 2)
    
    # Add noise
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Test different visualization modes
    modes = ["overlay", "individual", "confidence", "combined"]
    
    for mode in modes:
        result = process_image(
            test_image,
            detection_methods="gradient,gabor,hessian,morphological",
            fusion_mode="weighted",
            visualization_mode=mode
        )
        
        cv2.imshow(f"Advanced Scratch Detection - {mode}", result)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
