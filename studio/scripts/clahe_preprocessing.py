"""
CLAHE Preprocessing with Illumination Correction
================================================
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) with optional
illumination correction for fiber optic images. Particularly effective for 
enhancing local contrast in images with uneven lighting.

This function is optimized for fiber optic end-face images where illumination
may be non-uniform across the field of view.
"""
import cv2
import numpy as np
from typing import Union


def process_image(image: np.ndarray, 
                  clip_limit: float = 2.0,
                  tile_grid_size: int = 8,
                  apply_illumination_correction: bool = True,
                  correction_kernel_size: int = 50,
                  blur_kernel_size: int = 5,
                  blur_after_clahe: bool = True) -> np.ndarray:
    """
    Apply CLAHE preprocessing with optional illumination correction.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast
    while preventing over-amplification of noise. This implementation includes
    advanced illumination correction using a rolling ball algorithm.
    
    Args:
        image: Input image (grayscale or color)
        clip_limit: Threshold for contrast limiting (1.0-10.0, default: 2.0)
                   Higher values = more contrast but potentially more noise
        tile_grid_size: Size of grid for histogram equalization (default: 8)
                       Smaller = more local adaptation, larger = more global
        apply_illumination_correction: Whether to apply illumination correction before CLAHE
        correction_kernel_size: Kernel size for morphological background estimation (10-100)
        blur_kernel_size: Gaussian blur kernel size (must be odd, 3-15)
        blur_after_clahe: Apply Gaussian blur after CLAHE to reduce noise
        
    Returns:
        Preprocessed image with enhanced contrast
        
    Technical Details:
        - Illumination correction uses morphological closing to estimate background
        - Background is subtracted and result is shifted to mid-gray (128)
        - CLAHE is applied per channel for color images
        - Optional Gaussian blur reduces high-frequency noise after enhancement
    """
    # Validate and adjust parameters
    clip_limit = max(1.0, min(10.0, clip_limit))
    tile_grid_size = max(2, min(16, tile_grid_size))
    correction_kernel_size = max(10, min(100, correction_kernel_size))
    blur_kernel_size = max(3, blur_kernel_size)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    # Work with a copy
    result = image.copy()
    
    # Convert to appropriate format for processing
    if len(result.shape) == 3:
        # For color images, convert to LAB color space for better results
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        channels = cv2.split(lab)
        process_channels = [channels[0]]  # Only process L channel
        is_color = True
    else:
        process_channels = [result]
        is_color = False
    
    # Process each channel
    processed_channels = []
    
    for channel in process_channels:
        # Step 1: Illumination Correction (if enabled)
        if apply_illumination_correction:
            # Estimate background using morphological closing
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (correction_kernel_size, correction_kernel_size)
            )
            background = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
            
            # Subtract background and shift to mid-gray
            # Use int16 to handle negative values properly
            corrected = cv2.subtract(
                channel.astype(np.int16), 
                background.astype(np.int16)
            )
            corrected = corrected + 128  # Shift to mid-gray
            
            # Clip and convert back to uint8
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        else:
            corrected = channel
        
        # Step 2: Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=(tile_grid_size, tile_grid_size)
        )
        enhanced = clahe.apply(corrected)
        
        # Step 3: Optional Gaussian blur to reduce noise
        if blur_after_clahe:
            enhanced = cv2.GaussianBlur(
                enhanced, 
                (blur_kernel_size, blur_kernel_size), 
                0
            )
        
        processed_channels.append(enhanced)
    
    # Reconstruct the image
    if is_color:
        # Replace L channel with processed version
        channels[0] = processed_channels[0]
        merged = cv2.merge(channels)
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        result = processed_channels[0]
    
    return result


# Optional: Test code for standalone execution
if __name__ == "__main__":
    # Create a test image with uneven illumination
    test_size = 300
    y, x = np.ogrid[:test_size, :test_size]
    
    # Create base pattern (fiber-like circular gradient)
    center = test_size // 2
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    base_pattern = 255 * np.exp(-(dist**2) / (2 * (test_size/6)**2))
    
    # Add uneven illumination
    illumination = 0.3 + 0.7 * ((x / test_size) * 0.5 + (y / test_size) * 0.5)
    test_image = (base_pattern * illumination).astype(np.uint8)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Test the function
    result = process_image(test_image, clip_limit=3.0, apply_illumination_correction=True)
    
    # Display results
    cv2.imshow("Original", test_image)
    cv2.imshow("CLAHE Preprocessed", result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
