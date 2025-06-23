"""
Matrix Variance Anomaly Detection
=================================
Divides the image into a grid of segments and detects anomalies based on
local pixel variance analysis. Effective for finding localized defects that
deviate significantly from their local neighborhood.

This method is particularly useful for detecting contamination, particles,
and other anomalies that create local statistical variations.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def process_image(image: np.ndarray,
                  grid_size: int = 3,
                  variance_threshold: float = 15.0,
                  z_score_threshold: float = 2.0,
                  local_window_size: int = 3,
                  use_adaptive_threshold: bool = True,
                  apply_zone_mask: bool = False,
                  zone_center_x: Optional[int] = None,
                  zone_center_y: Optional[int] = None,
                  zone_radius: Optional[int] = None,
                  visualization_mode: str = "overlay") -> np.ndarray:
    """
    Detect anomalies using matrix-based variance analysis.
    
    Divides the image into segments and analyzes each pixel's deviation from
    its local neighborhood statistics. Pixels with high variance or significant
    deviation from local mean are marked as anomalies.
    
    Args:
        image: Input image (grayscale or color)
        grid_size: Number of segments per dimension (2-10, creates grid_size x grid_size segments)
        variance_threshold: Absolute difference threshold for anomaly detection
        z_score_threshold: Statistical threshold (number of standard deviations)
        local_window_size: Size of local neighborhood for statistics (must be odd)
        use_adaptive_threshold: Adapt thresholds based on segment statistics
        apply_zone_mask: Only process within a circular zone
        zone_center_x: X coordinate of zone center (None = image center)
        zone_center_y: Y coordinate of zone center (None = image center)
        zone_radius: Radius of processing zone (None = full image)
        visualization_mode: Output mode ("overlay", "mask", "heatmap", "segments")
        
    Returns:
        Visualization of detected anomalies based on selected mode
        
    Technical Details:
        - Each segment is analyzed independently for local variations
        - Z-score calculation identifies statistical outliers
        - Adaptive thresholding adjusts to local image characteristics
        - Can be combined with zone masking for targeted analysis
    """
    # Validate parameters
    grid_size = max(2, min(10, grid_size))
    local_window_size = max(3, local_window_size)
    if local_window_size % 2 == 0:
        local_window_size += 1
    variance_threshold = max(0.1, variance_threshold)
    z_score_threshold = max(0.5, z_score_threshold)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray = image.copy()
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    h, w = gray.shape
    
    # Create zone mask if requested
    if apply_zone_mask:
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        cx = zone_center_x if zone_center_x is not None else w // 2
        cy = zone_center_y if zone_center_y is not None else h // 2
        radius = zone_radius if zone_radius is not None else min(h, w) // 3
        cv2.circle(zone_mask, (cx, cy), radius, 255, -1)
    else:
        zone_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Initialize result and variance maps
    anomaly_mask = np.zeros((h, w), dtype=np.uint8)
    variance_map = np.zeros((h, w), dtype=np.float32)
    z_score_map = np.zeros((h, w), dtype=np.float32)
    
    # Calculate segment dimensions
    segment_h = h // grid_size
    segment_w = w // grid_size
    
    # Store segment statistics for visualization
    segment_stats = []
    
    # Process each segment
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate segment boundaries
            y_start = row * segment_h
            y_end = (row + 1) * segment_h if row < grid_size - 1 else h
            x_start = col * segment_w
            x_end = (col + 1) * segment_w if col < grid_size - 1 else w
            
            # Extract segment
            segment = gray[y_start:y_end, x_start:x_end]
            segment_zone_mask = zone_mask[y_start:y_end, x_start:x_end]
            
            # Skip if no valid pixels in zone
            if np.sum(segment_zone_mask) == 0:
                continue
            
            # Calculate segment statistics for adaptive thresholding
            valid_pixels = segment[segment_zone_mask > 0]
            if len(valid_pixels) > 0:
                segment_mean = np.mean(valid_pixels)
                segment_std = np.std(valid_pixels)
                segment_median = np.median(valid_pixels)
                segment_mad = np.median(np.abs(valid_pixels - segment_median))
            else:
                continue
            
            # Store segment statistics
            segment_stats.append({
                'row': row, 'col': col,
                'mean': segment_mean, 'std': segment_std,
                'median': segment_median, 'mad': segment_mad
            })
            
            # Adaptive threshold for this segment
            if use_adaptive_threshold:
                # Use robust statistics (MAD) for threshold
                robust_std = 1.4826 * segment_mad  # Convert MAD to std equivalent
                adaptive_variance_threshold = variance_threshold * (1 + robust_std / 100)
                adaptive_z_threshold = z_score_threshold * (1 + np.clip(segment_std / segment_mean, 0, 1))
            else:
                adaptive_variance_threshold = variance_threshold
                adaptive_z_threshold = z_score_threshold
            
            # Analyze each pixel in the segment
            seg_h, seg_w = segment.shape
            half_window = local_window_size // 2
            
            for y in range(half_window, seg_h - half_window):
                for x in range(half_window, seg_w - half_window):
                    if segment_zone_mask[y, x] == 0:
                        continue
                    
                    # Get local neighborhood
                    local_region = segment[y-half_window:y+half_window+1,
                                         x-half_window:x+half_window+1]
                    
                    # Calculate local statistics
                    center_value = float(segment[y, x])
                    local_mean = np.mean(local_region)
                    local_std = np.std(local_region)
                    
                    # Calculate variance metrics
                    abs_diff = abs(center_value - local_mean)
                    
                    # Calculate z-score
                    if local_std > 0:
                        z_score = abs_diff / local_std
                    else:
                        z_score = 0
                    
                    # Global coordinates
                    global_y = y_start + y
                    global_x = x_start + x
                    
                    # Store variance metrics
                    variance_map[global_y, global_x] = abs_diff
                    z_score_map[global_y, global_x] = z_score
                    
                    # Check for anomaly
                    is_anomaly = (z_score > adaptive_z_threshold or 
                                abs_diff > adaptive_variance_threshold)
                    
                    if is_anomaly:
                        anomaly_mask[global_y, global_x] = 255
    
    # Post-processing: morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove very small anomalies
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(anomaly_mask, connectivity=8)
    min_anomaly_size = 3
    
    cleaned_mask = np.zeros_like(anomaly_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_anomaly_size:
            cleaned_mask[labels == i] = 255
    
    # Generate visualization
    if visualization_mode == "mask":
        # Binary mask
        result = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        
    elif visualization_mode == "heatmap":
        # Variance heatmap
        # Normalize variance map
        variance_normalized = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX)
        variance_uint8 = variance_normalized.astype(np.uint8)
        
        # Apply zone mask to heatmap
        variance_uint8 = cv2.bitwise_and(variance_uint8, zone_mask)
        
        # Create colored heatmap
        heatmap = cv2.applyColorMap(variance_uint8, cv2.COLORMAP_JET)
        result = heatmap
        
    elif visualization_mode == "segments":
        # Visualize segment grid with statistics
        result = color_image.copy()
        
        # Draw segment boundaries
        for row in range(1, grid_size):
            y = row * segment_h
            cv2.line(result, (0, y), (w, y), (255, 255, 0), 1)
        
        for col in range(1, grid_size):
            x = col * segment_w
            cv2.line(result, (x, 0), (x, h), (255, 255, 0), 1)
        
        # Overlay anomalies
        anomaly_overlay = np.zeros_like(result)
        anomaly_overlay[cleaned_mask > 0] = (0, 0, 255)
        result = cv2.addWeighted(result, 0.7, anomaly_overlay, 0.3, 0)
        
        # Add segment statistics text
        font_scale = 0.3
        for stat in segment_stats:
            y = stat['row'] * segment_h + 15
            x = stat['col'] * segment_w + 5
            text = f"μ:{stat['mean']:.0f}"
            cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 255, 0), 1)
            text2 = f"σ:{stat['std']:.1f}"
            cv2.putText(result, text2, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 255, 0), 1)
        
    elif visualization_mode == "overlay":
        # Overlay on original with statistics
        result = color_image.copy()
        
        # Create colored overlay
        anomaly_overlay = np.zeros_like(result)
        anomaly_overlay[cleaned_mask > 0] = (0, 0, 255)  # Red for anomalies
        
        # Blend
        result = cv2.addWeighted(result, 0.7, anomaly_overlay, 0.3, 0)
        
        # Draw zone boundary if applicable
        if apply_zone_mask:
            contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # Add anomaly count
        anomaly_count = num_labels - 1
        cv2.putText(result, f"Anomalies: {anomaly_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Grid: {grid_size}x{grid_size}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    else:
        result = color_image
    
    return result


# Test code
if __name__ == "__main__":
    # Create test image with various anomalies
    test_size = 400
    test_image = np.ones((test_size, test_size), dtype=np.uint8) * 180
    
    # Add Gaussian noise as background
    noise = np.random.normal(0, 5, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Add localized anomalies in different segments
    anomaly_positions = [
        (100, 100, 10, 50),   # Bright spot
        (300, 100, 15, -50),  # Dark spot
        (100, 300, 5, 30),    # Small bright
        (300, 300, 20, -40),  # Large dark
        (200, 200, 8, 60),    # Center bright
    ]
    
    for x, y, size, intensity in anomaly_positions:
        y1, y2 = max(0, y-size), min(test_size, y+size)
        x1, x2 = max(0, x-size), min(test_size, x+size)
        test_image[y1:y2, x1:x2] = np.clip(
            test_image[y1:y2, x1:x2] + intensity, 0, 255
        )
    
    # Test different visualization modes
    modes = ["overlay", "mask", "heatmap", "segments"]
    
    for i, mode in enumerate(modes):
        result = process_image(
            test_image,
            grid_size=3,
            variance_threshold=15.0,
            visualization_mode=mode,
            apply_zone_mask=True,
            zone_radius=150
        )
        
        cv2.imshow(f"Matrix Variance - {mode}", result)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
