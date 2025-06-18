"""
Defect Validation and False Positive Reduction
==============================================
Validates detected defects using multiple criteria including contrast analysis,
texture consistency, statistical significance, and shape characteristics.
Helps reduce false positives by filtering out detections that don't meet
defect criteria.

This function is particularly useful as a post-processing step after initial
defect detection to improve detection accuracy.
"""
import cv2
import numpy as np
from scipy import stats as scipy_stats
from typing import Optional, Tuple, Dict, List


def process_image(image: np.ndarray,
                  defect_mask: Optional[np.ndarray] = None,
                  min_contrast: float = 10.0,
                  min_area: int = 5,
                  max_area: int = 5000,
                  texture_threshold: float = 0.8,
                  statistical_confidence: float = 0.95,
                  validate_shape: bool = True,
                  min_compactness: float = 0.1,
                  max_compactness: float = 0.9,
                  use_boundary_exclusion: bool = True,
                  boundary_width: int = 3,
                  visualization_mode: str = "overlay") -> np.ndarray:
    """
    Validate and filter detected defects based on multiple criteria.
    
    This function analyzes each detected defect region to determine if it's
    a true defect or a false positive. It uses contrast, texture, statistical,
    and shape-based validation methods.
    
    Args:
        image: Input image (grayscale or color)
        defect_mask: Binary mask of detected defects (None to auto-generate)
        min_contrast: Minimum contrast between defect and surroundings
        min_area: Minimum defect area in pixels
        max_area: Maximum defect area in pixels
        texture_threshold: Texture similarity threshold (0.0-1.0)
        statistical_confidence: Confidence level for statistical tests (0.0-1.0)
        validate_shape: Whether to validate based on shape characteristics
        min_compactness: Minimum compactness (4π*area/perimeter²) for defects
        max_compactness: Maximum compactness for defects
        use_boundary_exclusion: Exclude detections near mask boundaries
        boundary_width: Width of boundary exclusion zone
        visualization_mode: Output mode ("overlay", "analysis", "statistics", "rejected")
        
    Returns:
        Visualization of validated defects based on selected mode
        
    Validation Criteria:
        1. Contrast: Defect must have sufficient contrast with surroundings
        2. Texture: Defect texture should differ from surroundings
        3. Statistical: Defect values must be statistically significant
        4. Shape: Defect shape must be plausible (not too irregular)
        5. Size: Defect must be within reasonable size bounds
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray = image.copy()
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Generate defect mask if not provided
    if defect_mask is None:
        # Simple defect detection for demonstration
        defect_mask = _simple_defect_detection(gray)
    
    # Ensure defect_mask is binary
    _, defect_mask = cv2.threshold(defect_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create boundary exclusion mask if requested
    if use_boundary_exclusion:
        boundary_mask = _create_boundary_mask(defect_mask, boundary_width)
        defect_mask = cv2.bitwise_and(defect_mask, cv2.bitwise_not(boundary_mask))
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        defect_mask, connectivity=8
    )
    
    # Initialize validation results
    validated_mask = np.zeros_like(defect_mask)
    rejected_mask = np.zeros_like(defect_mask)
    validation_info = []
    
    # Validate each component
    for i in range(1, num_labels):
        # Get component properties
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
        
        # Create component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Initialize validation record
        validation = {
            'id': i,
            'area': area,
            'centroid': centroids[i],
            'bbox': (x, y, w, h),
            'passed': True,
            'reasons': []
        }
        
        # Size validation
        if area < min_area:
            validation['passed'] = False
            validation['reasons'].append(f"Too small ({area} < {min_area})")
        elif area > max_area:
            validation['passed'] = False
            validation['reasons'].append(f"Too large ({area} > {max_area})")
        
        # Skip further validation if size check failed
        if validation['passed']:
            # Get defect and surrounding pixels
            defect_pixels = gray[component_mask > 0]
            
            # Create dilated mask for surrounding region
            kernel_size = max(5, int(np.sqrt(area) / 2))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(component_mask, kernel)
            surrounding_mask = cv2.bitwise_and(dilated - component_mask, 255 - defect_mask)
            surrounding_pixels = gray[surrounding_mask > 0]
            
            if len(defect_pixels) > 0 and len(surrounding_pixels) > 10:
                # Contrast validation
                contrast_valid, contrast_value = _validate_contrast(
                    defect_pixels, surrounding_pixels, min_contrast
                )
                validation['contrast'] = contrast_value
                
                if not contrast_valid:
                    validation['passed'] = False
                    validation['reasons'].append(
                        f"Low contrast ({contrast_value:.1f} < {min_contrast})"
                    )
                
                # Texture validation
                texture_valid, texture_similarity = _validate_texture(
                    defect_pixels, surrounding_pixels, texture_threshold
                )
                validation['texture_similarity'] = texture_similarity
                
                if not texture_valid:
                    validation['passed'] = False
                    validation['reasons'].append(
                        f"Similar texture ({texture_similarity:.2f} > {texture_threshold})"
                    )
                
                # Statistical validation
                stat_valid, p_value = _validate_statistical(
                    defect_pixels, surrounding_pixels, statistical_confidence
                )
                validation['p_value'] = p_value
                
                if not stat_valid:
                    validation['passed'] = False
                    validation['reasons'].append(
                        f"Not significant (p={p_value:.3f})"
                    )
                
                # Shape validation
                if validate_shape:
                    shape_valid, compactness = _validate_shape(
                        component_mask, min_compactness, max_compactness
                    )
                    validation['compactness'] = compactness
                    
                    if not shape_valid:
                        validation['passed'] = False
                        validation['reasons'].append(
                            f"Invalid shape (compactness={compactness:.2f})"
                        )
            else:
                validation['passed'] = False
                validation['reasons'].append("Insufficient pixels for analysis")
        
        # Store validation info
        validation_info.append(validation)
        
        # Update masks
        if validation['passed']:
            validated_mask = cv2.bitwise_or(validated_mask, component_mask)
        else:
            rejected_mask = cv2.bitwise_or(rejected_mask, component_mask)
    
    # Generate visualization
    if visualization_mode == "overlay":
        # Overlay validated and rejected defects
        result = color_image.copy()
        
        # Green for validated defects
        valid_overlay = np.zeros_like(result)
        valid_overlay[validated_mask > 0] = (0, 255, 0)
        
        # Red for rejected defects
        reject_overlay = np.zeros_like(result)
        reject_overlay[rejected_mask > 0] = (0, 0, 255)
        
        # Combine overlays
        result = cv2.addWeighted(result, 0.6, valid_overlay, 0.4, 0)
        result = cv2.addWeighted(result, 0.8, reject_overlay, 0.2, 0)
        
        # Add summary
        valid_count = sum(1 for v in validation_info if v['passed'])
        total_count = len(validation_info)
        cv2.putText(result, f"Valid: {valid_count}/{total_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    elif visualization_mode == "analysis":
        # Detailed analysis view
        result = _create_analysis_visualization(
            color_image, validation_info, labels
        )
        
    elif visualization_mode == "statistics":
        # Statistical summary view
        result = _create_statistics_visualization(
            color_image, validation_info, validated_mask, rejected_mask
        )
        
    elif visualization_mode == "rejected":
        # Show only rejected defects with reasons
        result = color_image.copy()
        
        # Highlight rejected regions
        reject_overlay = np.zeros_like(result)
        reject_overlay[rejected_mask > 0] = (0, 0, 255)
        result = cv2.addWeighted(result, 0.7, reject_overlay, 0.3, 0)
        
        # Add rejection reasons
        for info in validation_info:
            if not info['passed'] and info['reasons']:
                cx, cy = info['centroid']
                reason = info['reasons'][0]  # Show first reason
                cv2.putText(result, reason, (int(cx-30), int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    else:
        # Default: return validated mask as color image
        result = cv2.cvtColor(validated_mask, cv2.COLOR_GRAY2BGR)
    
    return result


def _simple_defect_detection(gray: np.ndarray) -> np.ndarray:
    """Simple defect detection for demonstration when no mask is provided."""
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def _create_boundary_mask(mask: np.ndarray, width: int) -> np.ndarray:
    """Create mask for boundary regions."""
    # Find edges of the mask
    edges = cv2.Canny(mask, 50, 150)
    
    # Dilate edges to create boundary region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width*2+1, width*2+1))
    boundary = cv2.dilate(edges, kernel)
    
    return boundary


def _validate_contrast(defect_pixels: np.ndarray, 
                      surrounding_pixels: np.ndarray,
                      min_contrast: float) -> Tuple[bool, float]:
    """Validate defect based on contrast with surroundings."""
    defect_mean = np.mean(defect_pixels)
    surrounding_mean = np.mean(surrounding_pixels)
    
    contrast = abs(defect_mean - surrounding_mean)
    is_valid = contrast >= min_contrast
    
    return is_valid, contrast


def _validate_texture(defect_pixels: np.ndarray,
                     surrounding_pixels: np.ndarray,
                     threshold: float) -> Tuple[bool, float]:
    """Validate defect based on texture difference."""
    # Calculate texture metrics using standard deviation
    defect_std = np.std(defect_pixels)
    surrounding_std = np.std(surrounding_pixels)
    
    # Texture similarity (0 = different, 1 = same)
    if max(defect_std, surrounding_std) > 0:
        similarity = 1.0 - abs(defect_std - surrounding_std) / max(defect_std, surrounding_std)
    else:
        similarity = 1.0
    
    is_valid = similarity < threshold
    
    return is_valid, similarity


def _validate_statistical(defect_pixels: np.ndarray,
                         surrounding_pixels: np.ndarray,
                         confidence: float) -> Tuple[bool, float]:
    """Validate defect using statistical significance test."""
    if len(defect_pixels) < 5 or len(surrounding_pixels) < 5:
        return False, 1.0
    
    # Perform t-test
    try:
        t_stat, p_value = scipy_stats.ttest_ind(defect_pixels, surrounding_pixels)
        is_valid = p_value < (1 - confidence)
    except:
        # Fallback to simple comparison
        p_value = 1.0
        is_valid = False
    
    return is_valid, p_value


def _validate_shape(component_mask: np.ndarray,
                   min_compactness: float,
                   max_compactness: float) -> Tuple[bool, float]:
    """Validate defect based on shape characteristics."""
    # Find contour
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0.0
    
    contour = contours[0]
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate compactness (circularity)
    if perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter * perimeter)
    else:
        compactness = 0.0
    
    is_valid = min_compactness <= compactness <= max_compactness
    
    return is_valid, compactness


def _create_analysis_visualization(image: np.ndarray,
                                 validation_info: List[Dict],
                                 labels: np.ndarray) -> np.ndarray:
    """Create detailed analysis visualization."""
    result = image.copy()
    
    # Color code each defect
    for info in validation_info:
        component_mask = (labels == info['id'])
        
        if info['passed']:
            # Green tint for valid
            result[component_mask] = result[component_mask] * [0.7, 1.0, 0.7]
        else:
            # Red tint for invalid
            result[component_mask] = result[component_mask] * [0.7, 0.7, 1.0]
        
        # Add ID number
        cx, cy = info['centroid']
        cv2.putText(result, str(info['id']), (int(cx-5), int(cy+5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add legend
    y_offset = 30
    for info in validation_info[:10]:  # Show first 10
        color = (0, 255, 0) if info['passed'] else (0, 0, 255)
        text = f"#{info['id']}: "
        
        if info['passed']:
            if 'contrast' in info:
                text += f"C={info['contrast']:.1f}"
        else:
            text += info['reasons'][0] if info['reasons'] else "Failed"
        
        cv2.putText(result, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 20
    
    return result


def _create_statistics_visualization(image: np.ndarray,
                                   validation_info: List[Dict],
                                   validated_mask: np.ndarray,
                                   rejected_mask: np.ndarray) -> np.ndarray:
    """Create statistical summary visualization."""
    h, w = image.shape[:2]
    
    # Create panels
    # Top left: Original with overlay
    panel1 = image.copy()
    overlay = np.zeros_like(panel1)
    overlay[validated_mask > 0] = (0, 255, 0)
    overlay[rejected_mask > 0] = (0, 0, 255)
    panel1 = cv2.addWeighted(panel1, 0.7, overlay, 0.3, 0)
    
    # Top right: Statistics chart
    panel2 = np.ones((h//2, w//2, 3), dtype=np.uint8) * 255
    
    # Calculate statistics
    total = len(validation_info)
    passed = sum(1 for v in validation_info if v['passed'])
    failed = total - passed
    
    if total > 0:
        # Draw pie chart
        center = (w//4, h//4)
        radius = min(w//4, h//4) - 20
        
        # Draw passed segment
        angle1 = 0
        angle2 = int(360 * passed / total)
        cv2.ellipse(panel2, center, (radius, radius), 0, angle1, angle2, 
                   (0, 255, 0), -1)
        
        # Draw failed segment
        cv2.ellipse(panel2, center, (radius, radius), 0, angle2, 360, 
                   (0, 0, 255), -1)
        
        # Add text
        cv2.putText(panel2, f"Valid: {passed} ({passed/total*100:.0f}%)",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(panel2, f"Rejected: {failed} ({failed/total*100:.0f}%)",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Bottom: Failure reasons histogram
    panel3 = np.ones((h//2, w, 3), dtype=np.uint8) * 240
    
    # Count failure reasons
    reason_counts = {}
    for info in validation_info:
        if not info['passed']:
            for reason in info['reasons']:
                # Extract reason type
                if "small" in reason.lower():
                    key = "Too Small"
                elif "large" in reason.lower():
                    key = "Too Large"
                elif "contrast" in reason.lower():
                    key = "Low Contrast"
                elif "texture" in reason.lower():
                    key = "Similar Texture"
                elif "significant" in reason.lower():
                    key = "Not Significant"
                elif "shape" in reason.lower():
                    key = "Invalid Shape"
                else:
                    key = "Other"
                
                reason_counts[key] = reason_counts.get(key, 0) + 1
    
    # Draw histogram
    if reason_counts:
        max_count = max(reason_counts.values())
        bar_width = w // (len(reason_counts) + 1)
        bar_gap = 10
        
        for i, (reason, count) in enumerate(reason_counts.items()):
            x1 = i * bar_width + bar_gap
            x2 = (i + 1) * bar_width - bar_gap
            bar_height = int((h//2 - 60) * count / max_count)
            y1 = h//2 - 30
            y2 = y1 - bar_height
            
            cv2.rectangle(panel3, (x1, y1), (x2, y2), (100, 100, 255), -1)
            cv2.putText(panel3, str(count), (x1 + 5, y2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(panel3, reason, (x1, y1 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    cv2.putText(panel3, "Rejection Reasons", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Combine panels
    top_row = np.hstack([cv2.resize(panel1, (w//2, h//2)), panel2])
    result = np.vstack([top_row, panel3])
    
    return result


# Test code
if __name__ == "__main__":
    # Create test image with various defect-like features
    test_size = 400
    test_image = np.ones((test_size, test_size), dtype=np.uint8) * 200
    
    # Add real defects (high contrast, different texture)
    # Deep pit
    cv2.circle(test_image, (100, 100), 10, 50, -1)
    
    # Contamination spot
    cv2.ellipse(test_image, (300, 100), (20, 15), 45, 0, 360, 100, -1)
    
    # Scratch
    cv2.line(test_image, (50, 200), (150, 250), 80, 3)
    
    # Add false positives (low contrast, similar texture)
    # Slight variation
    cv2.circle(test_image, (200, 200), 8, 190, -1)
    
    # Very small spot
    cv2.circle(test_image, (100, 300), 2, 150, -1)
    
    # Large uniform area
    cv2.rectangle(test_image, (250, 250), (350, 350), 180, -1)
    
    # Add noise
    noise = np.random.normal(0, 5, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Test different visualization modes
    modes = ["overlay", "analysis", "statistics", "rejected"]
    
    for mode in modes:
        result = process_image(
            test_image,
            min_contrast=15.0,
            min_area=10,
            visualization_mode=mode
        )
        
        cv2.imshow(f"Defect Validation - {mode}", result)
    
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
