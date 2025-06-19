import cv2
import numpy as np
import os

# Set the backend for Matplotlib to a non-interactive one ('Agg')
# This must be done BEFORE importing pyplot to prevent GUI errors on Linux.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.signal import find_peaks

def apply_binary_filter(image):
    """
    Applies a binary filter to an image to isolate bright regions.
    Based on the reference script.
    Args:
        image (np.array): The input image region (e.g., core or cladding).
    Returns:
        np.array: A binary mask with white pixels representing the bright areas.
    """
    # If the image is already grayscale, use it directly
    if len(image.shape) == 2:
        gray_copy = image.copy()
    else:
        gray_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a hard threshold to create a binary image. Bright pixels become white.
    # Pixels with intensity > 127 become 255 (white), others become 0 (black).
    _, result = cv2.threshold(gray_copy, 127, 255, cv2.THRESH_BINARY)

    # Use a morphological closing operation to fill small holes in the white regions
    # and smooth the boundaries, creating a cleaner mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

def universal_fiber_segmentation(image_path, output_dir='output_universal_refined'):
    """
    Performs universal segmentation with a secondary binary refinement step.
    """
    # --- 1. Preprocessing ---
    if not os.path.exists(image_path):
        print(f"\nError: File not found: '{image_path}'")
        return

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image from '{image_path}'.")
        return

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
    print(f"Successfully loaded and preprocessed image: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    height, width = gray_image.shape

    # --- 2. Multi-Modal Center Finding ---
    print("\n--- Step 1: Multi-Modal Center Finding ---")
    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=width//2,
        param1=50, param2=30, minRadius=10, maxRadius=int(height / 2.5)
    )
    if circles is None:
        print("Initial geometric guess failed. Cannot proceed.")
        return
    hough_center = np.uint16(np.around(circles[0, 0][:2]))
    print(f"  -> Geometric Guess (Hough): ({hough_center[0]}, {hough_center[1]})")

    brightness_threshold = np.percentile(gray_image, 95)
    _, core_thresh_mask = cv2.threshold(blurred_image, brightness_threshold, 255, cv2.THRESH_BINARY)
    M_bright = cv2.moments(core_thresh_mask)
    brightness_center = hough_center
    if M_bright["m00"] != 0:
        cx_bright = int(M_bright["m10"] / M_bright["m00"])
        cy_bright = int(M_bright["m01"] / M_bright["m00"])
        brightness_center = np.array([cx_bright, cy_bright], dtype=np.uint16)
    print(f"  -> Photometric Guess (Core): ({brightness_center[0]}, {brightness_center[1]})")

    lbp_layer = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    texture_threshold = np.percentile(lbp_layer, 25)
    texture_mask = np.where(lbp_layer <= texture_threshold, 255, 0).astype(np.uint8)
    M_texture = cv2.moments(texture_mask)
    texture_center = hough_center
    if M_texture["m00"] != 0:
        cx_texture = int(M_texture["m10"] / M_texture["m00"])
        cy_texture = int(M_texture["m01"] / M_texture["m00"])
        texture_center = np.array([cx_texture, cy_texture], dtype=np.uint16)
    print(f"  -> Textural Guess (Glass): ({texture_center[0]}, {texture_center[1]})")

    final_center = np.mean([hough_center, brightness_center, texture_center], axis=0).astype(np.uint16)
    center_x, center_y = final_center
    print(f"  -> Fused High-Confidence Center: ({center_x}, {center_y})")

    # --- 3. Radial Profile & Boundary Detection ---
    print("\n--- Step 2: Radial Profile Analysis & Boundary Detection ---")
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    change_magnitude_layer = cv2.magnitude(sobel_x, sobel_y)
    max_radius = int(min(center_x, center_y, width - center_x, height - center_y))
    y_coords, x_coords = np.indices((height, width))
    radii_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2).astype(int)
    radial_change = np.array([np.mean(change_magnitude_layer[radii_map == r]) for r in range(max_radius) if np.any(radii_map == r)])
    
    peaks, properties = find_peaks(radial_change, prominence=np.mean(radial_change), distance=10)
    if len(peaks) < 2:
        print("Error: Could not reliably detect two distinct boundaries.")
        return
    top_two_peak_indices = np.argsort(properties['prominences'])[-2:]
    radii = sorted(peaks[top_two_peak_indices])
    core_radius, cladding_radius = radii[0], radii[1]
    print(f"  -> Detected Core Radius: {core_radius}px, Cladding Radius: {cladding_radius}px")

    # --- 4. Mask Generation and ***NEW*** Binary Refinement Step ---
    print("\n--- Step 3: Refining Segments with Binary Filter ---")
    mask_template = np.zeros_like(gray_image)

    # 4a. Isolate the initial Core region
    core_geom_mask = cv2.circle(mask_template.copy(), (center_x, center_y), core_radius, 255, -1)
    initial_core_region = cv2.bitwise_and(original_image, original_image, mask=core_geom_mask)

    # 4b. Apply binary filter and refine the core
    print("  -> Refining core: Keeping only white pixels from binary filter...")
    core_binary_mask = apply_binary_filter(initial_core_region)
    # Keep pixels from the initial region that are also white in the binary mask
    refined_core_region = cv2.bitwise_and(initial_core_region, initial_core_region, mask=core_binary_mask)

    # 4c. Isolate the initial Cladding region
    cladding_outer_mask = cv2.circle(mask_template.copy(), (center_x, center_y), cladding_radius, 255, -1)
    cladding_geom_mask = cv2.subtract(cladding_outer_mask, core_geom_mask)
    initial_cladding_region = cv2.bitwise_and(original_image, original_image, mask=cladding_geom_mask)

    # 4d. Apply binary filter and refine the cladding
    print("  -> Refining cladding: Removing white pixels from binary filter...")
    cladding_binary_mask = apply_binary_filter(initial_cladding_region)
    # Create an inverted mask to remove the white pixel positions
    inverted_cladding_mask = cv2.bitwise_not(cladding_binary_mask)
    # Keep pixels from the initial region that are in the black areas of the binary mask
    refined_cladding_region = cv2.bitwise_and(initial_cladding_region, initial_cladding_region, mask=inverted_cladding_mask)

    # --- 5. Visualization and Final Output ---
    # The diagnostic plot remains the same, showing the geometric boundaries
    # (Code for plotting is omitted for brevity but is the same as before)
    
    # --- 6. Save Final Segmented and Refined Images ---
    # Isolate the ferrule region (this does not get refined)
    ferrule_mask = cv2.bitwise_not(cladding_outer_mask)
    ferrule_region = cv2.bitwise_and(original_image, original_image, mask=ferrule_mask)
    
    # Draw boundaries on a diagnostic image
    diagnostic_image = original_image.copy()
    cv2.circle(diagnostic_image, (center_x, center_y), 3, (0, 255, 255), -1)
    cv2.circle(diagnostic_image, (center_x, center_y), core_radius, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(diagnostic_image, (center_x, center_y), cladding_radius, (0, 0, 255), 2, cv2.LINE_AA)

    # Save the REFINED regions
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_refined_region_core.png'), refined_core_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_refined_region_cladding.png'), refined_cladding_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_region_ferrule.png'), ferrule_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_boundaries_detected.png'), diagnostic_image)
    
    print(f"\n--- Step 4: Segmentation complete. Refined outputs saved in '{output_dir}'. ---")


if __name__ == '__main__':
    print("--- Universal Fiber Segmentation Tool with Binary Refinement ---")
    image_path_input = input("Please enter the full path to the image to analyze: ")
    image_path_cleaned = image_path_input.strip().strip('"').strip("'")
    
    print("-" * 50)
    universal_fiber_segmentation(image_path_cleaned)
