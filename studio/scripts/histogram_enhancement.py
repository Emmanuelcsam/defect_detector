"""
Histogram Equalization Enhancement - Improve contrast to make low-contrast scratches visible
Redistributes pixel intensities for better dynamic range
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  method: str = "clahe",
                  clahe_clip_limit: float = 2.0,
                  clahe_tile_size: int = 8,
                  global_enhance_color: bool = False,
                  blend_original: float = 0.0,
                  show_histogram: bool = True,
                  apply_to_regions: bool = False,
                  gamma_correction: float = 1.0) -> np.ndarray:
    """
    Apply histogram equalization to enhance low-contrast defects.
    
    This function modifies the image's dynamic range and contrast to make
    low-contrast scratches and defects more visible. It offers both global
    and adaptive (CLAHE) histogram equalization methods.
    
    Args:
        image: Input fiber optic image
        method: "global", "clahe", or "adaptive"
        clahe_clip_limit: Contrast limiting parameter for CLAHE
        clahe_tile_size: Size of grid tiles for CLAHE
        global_enhance_color: Apply to each color channel separately
        blend_original: Blend factor with original (0=full enhancement, 1=original)
        show_histogram: Display histogram comparison
        apply_to_regions: Apply different enhancement to fiber regions
        gamma_correction: Additional gamma correction (1.0 = none)
        
    Returns:
        Contrast-enhanced image with improved defect visibility
    """
    # Prepare working image
    if len(image.shape) == 2:
        working = image.copy()
        is_grayscale = True
    else:
        is_grayscale = False
        if global_enhance_color:
            working = image.copy()
        else:
            # Convert to LAB for luminance-only enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            working = lab[:, :, 0]  # L channel
    
    h, w = working.shape[:2]
    
    # Apply gamma correction if requested
    if gamma_correction != 1.0:
        # Build lookup table
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma_correction)) * 255
                               for i in np.arange(0, 256)]).astype("uint8")
        
        if is_grayscale or not global_enhance_color:
            working = cv2.LUT(working, gamma_table)
        else:
            for i in range(3):
                working[:, :, i] = cv2.LUT(working[:, :, i], gamma_table)
    
    # Store original for histogram comparison
    original_hist = working.copy()
    
    # Apply enhancement based on method
    if method == "global":
        # Global histogram equalization
        if is_grayscale or not global_enhance_color:
            enhanced = cv2.equalizeHist(working)
        else:
            enhanced = working.copy()
            for i in range(3):
                enhanced[:, :, i] = cv2.equalizeHist(working[:, :, i])
                
    elif method == "clahe":
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, 
                               tileGridSize=(clahe_tile_size, clahe_tile_size))
        
        if is_grayscale or not global_enhance_color:
            enhanced = clahe.apply(working)
        else:
            enhanced = working.copy()
            for i in range(3):
                enhanced[:, :, i] = clahe.apply(working[:, :, i])
                
    else:  # adaptive
        # Custom adaptive method - different parameters for different regions
        if hasattr(image, 'core_mask'):
            # Use stored region masks
            core_mask = image.core_mask
            cladding_mask = image.cladding_mask
            coating_mask = image.coating_mask
        else:
            # Create simple radial regions
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            max_dist = min(h, w) // 2
            core_mask = (dist < max_dist * 0.1).astype(np.uint8) * 255
            cladding_mask = ((dist >= max_dist * 0.1) & (dist < max_dist * 0.5)).astype(np.uint8) * 255
            coating_mask = ((dist >= max_dist * 0.5) & (dist < max_dist * 0.9)).astype(np.uint8) * 255
        
        # Apply different CLAHE parameters to each region
        enhanced = np.zeros_like(working)
        
        # Core region - mild enhancement
        clahe_core = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        temp = clahe_core.apply(working)
        enhanced[core_mask > 0] = temp[core_mask > 0]
        
        # Cladding - moderate enhancement
        clahe_cladding = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        temp = clahe_cladding.apply(working)
        enhanced[cladding_mask > 0] = temp[cladding_mask > 0]
        
        # Coating - strong enhancement
        clahe_coating = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        temp = clahe_coating.apply(working)
        enhanced[coating_mask > 0] = temp[coating_mask > 0]
        
        # Background
        background_mask = cv2.bitwise_not(cv2.bitwise_or(
            cv2.bitwise_or(core_mask, cladding_mask), coating_mask))
        enhanced[background_mask > 0] = working[background_mask > 0]
    
    # Blend with original if requested
    if blend_original > 0:
        enhanced = cv2.addWeighted(enhanced, 1 - blend_original, 
                                 working, blend_original, 0)
    
    # Prepare final result
    if not is_grayscale and not global_enhance_color:
        # Put enhanced L channel back
        lab[:, :, 0] = enhanced
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        result = enhanced
    
    # Convert to BGR for display if needed
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # Add histogram visualization if requested
    if show_histogram:
        # Create histogram panel
        hist_h = 200
        hist_w = 256
        hist_panel = np.ones((hist_h, hist_w * 2 + 20, 3), dtype=np.uint8) * 255
        
        # Calculate histograms
        if len(original_hist.shape) == 2:
            hist_orig = cv2.calcHist([original_hist], [0], None, [256], [0, 256])
            hist_enh = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
        else:
            # Use green channel for color images
            hist_orig = cv2.calcHist([original_hist], [1], None, [256], [0, 256])
            hist_enh = cv2.calcHist([enhanced], [1], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_orig, hist_orig, 0, hist_h - 40, cv2.NORM_MINMAX)
        cv2.normalize(hist_enh, hist_enh, 0, hist_h - 40, cv2.NORM_MINMAX)
        
        # Draw original histogram
        for i in range(256):
            cv2.line(hist_panel, 
                    (i, hist_h - 20),
                    (i, hist_h - 20 - int(hist_orig[i])),
                    (100, 100, 100), 1)
        
        # Draw enhanced histogram
        for i in range(256):
            cv2.line(hist_panel,
                    (i + hist_w + 20, hist_h - 20),
                    (i + hist_w + 20, hist_h - 20 - int(hist_enh[i])),
                    (0, 150, 0), 1)
        
        # Add labels
        cv2.putText(hist_panel, "Original", (80, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(hist_panel, "Enhanced", (hist_w + 80, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 1)
        
        # Add to result
        h_result, w_result = result.shape[:2]
        if w_result > h_result:
            # Place histogram at bottom
            hist_resized = cv2.resize(hist_panel, (w_result, hist_h))
            result = np.vstack([result, hist_resized])
        else:
            # Place histogram at right
            hist_resized = cv2.resize(hist_panel, (hist_w * 2 + 20, h_result))
            result = np.hstack([result, hist_resized])
    
    # Add enhancement information
    cv2.putText(result, f"Enhancement: {method.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if method == "clahe":
        cv2.putText(result, f"Clip: {clahe_clip_limit:.1f}, Tile: {clahe_tile_size}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if gamma_correction != 1.0:
        cv2.putText(result, f"Gamma: {gamma_correction:.2f}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Store enhanced data in metadata
    result.enhanced_image = enhanced
    result.enhancement_method = method
    result.enhancement_params = {
        'clahe_clip': clahe_clip_limit,
        'clahe_tile': clahe_tile_size,
        'gamma': gamma_correction,
        'blend': blend_original
    }
    
    return result