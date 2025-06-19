import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import json

def segment_fiber_with_multimodal_analysis(image_path, output_dir='output_refined'):
    """
    Performs superior segmentation of a fiber optic end-face by first refining
    the fiber's center and then fusing analysis from Intensity, Change Magnitude,
    and Local Texture.
    
    Modified to work with unified system - saves standardized output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result dictionary
    result = {
        'method': 'guess_approach',
        'image_path': image_path,
        'success': False,
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0
    }
    
    # --- 1. Preprocessing: Load, Convert, and Blur ---
    if not os.path.exists(image_path):
        result['error'] = f"File not found: '{image_path}'"
        with open(os.path.join(output_dir, 'guess_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result

    original_image = cv2.imread(image_path)
    if original_image is None:
        result['error'] = f"Could not read image from '{image_path}'."
        with open(os.path.join(output_dir, 'guess_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    height, width = gray_image.shape

    # --- 2. Initial Center Guess (Hypothesis) ---
    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=width//2,
        param1=50, param2=30, minRadius=10, maxRadius=int(height / 2.5)
    )

    if circles is None:
        result['error'] = "Could not detect an initial circle."
        with open(os.path.join(output_dir, 'guess_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
    
    hough_center = np.uint16(np.around(circles[0, 0][:2]))

    # --- 2.5. Center Refinement using Pixel Computations ---
    # Refinement A: Brightness Centroid (find the center of the core)
    brightness_threshold = np.percentile(gray_image, 95)
    _, core_mask = cv2.threshold(blurred_image, brightness_threshold, 255, cv2.THRESH_BINARY)
    M_bright = cv2.moments(core_mask)
    if M_bright["m00"] == 0:
        brightness_center = hough_center
    else:
        cx_bright = int(M_bright["m10"] / M_bright["m00"])
        cy_bright = int(M_bright["m01"] / M_bright["m00"])
        brightness_center = np.array([cx_bright, cy_bright], dtype=np.uint16)

    # Refinement B: Texture Centroid (find the center of the uniform "glassy" area)
    lbp_layer = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    texture_threshold = np.percentile(lbp_layer, 25)
    texture_mask = np.where(lbp_layer <= texture_threshold, 255, 0).astype(np.uint8)
    M_texture = cv2.moments(texture_mask)
    if M_texture["m00"] == 0:
        texture_center = hough_center
    else:
        cx_texture = int(M_texture["m10"] / M_texture["m00"])
        cy_texture = int(M_texture["m01"] / M_texture["m00"])
        texture_center = np.array([cx_texture, cy_texture], dtype=np.uint16)

    # Final Center: Average the three centers for a robust, data-driven result.
    final_center = np.mean([hough_center, brightness_center, texture_center], axis=0).astype(np.uint16)
    center_x, center_y = final_center

    # --- 3. Multi-Modal Profile Calculation (Using the REFINED Center) ---
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    change_magnitude_layer = cv2.magnitude(sobel_x, sobel_y)

    max_radius = int(min(center_x, center_y, width - center_x, height - center_y))
    
    radial_intensity = np.zeros(max_radius)
    radial_change = np.zeros(max_radius)

    # Simplified radial profile calculation
    y_coords, x_coords = np.indices((height, width))
    radii_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2).astype(int)

    for r in range(max_radius):
        mask = radii_map == r
        if np.any(mask):
            radial_intensity[r] = np.mean(blurred_image[mask])
            radial_change[r] = np.mean(change_magnitude_layer[mask])

    # --- 4. Boundary Fusion: Find Radii using All Profiles ---
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(radial_change, prominence=np.mean(radial_change), distance=10)
    
    if len(peaks) < 2:
        result['error'] = "Could not reliably detect two distinct boundaries."
        with open(os.path.join(output_dir, 'guess_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result

    # Sort peaks by their magnitude (sharpness) and take the top two.
    peak_magnitudes = radial_change[peaks]
    top_two_peak_indices = np.argsort(peak_magnitudes)[-2:]
    radii = sorted(peaks[top_two_peak_indices])
    
    core_radius = radii[0]
    cladding_radius = radii[1]

    # --- 5. Generate standardized output ---
    result['success'] = True
    result['center'] = (int(center_x), int(center_y))
    result['core_radius'] = int(core_radius)
    result['cladding_radius'] = int(cladding_radius)
    result['confidence'] = 0.75  # This method has moderate confidence
    
    # Save result data
    with open(os.path.join(output_dir, f'{base_filename}_guess_result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    # --- 6. Visualization ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Multi-Modal Radial Analysis (Guess Approach)', fontsize=16)

    axs[0].plot(radial_intensity, color='blue')
    axs[0].set_title('Average Pixel Intensity vs. Radius')
    axs[0].set_ylabel('Avg. Intensity')
    axs[0].grid(True)
    axs[0].axvline(x=core_radius, color='g', linestyle='--', label=f'Core ({core_radius}px)')
    axs[0].axvline(x=cladding_radius, color='r', linestyle='--', label=f'Cladding ({cladding_radius}px)')
    axs[0].legend()

    axs[1].plot(radial_change, color='orange')
    axs[1].set_title('Average Change Magnitude vs. Radius')
    axs[1].set_xlabel('Radius from Center (pixels)')
    axs[1].set_ylabel('Avg. Change')
    axs[1].grid(True)
    axs[1].plot(peaks, radial_change[peaks], "x", color='purple', markersize=10, label='Peaks')
    axs[1].axvline(x=core_radius, color='g', linestyle='--')
    axs[1].axvline(x=cladding_radius, color='r', linestyle='--')
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    diagnostic_plot_path = os.path.join(output_dir, f'{base_filename}_guess_radial.png')
    plt.savefig(diagnostic_plot_path)
    plt.close()

    # --- 7. Create segmented regions ---
    mask_template = np.zeros_like(gray_image)
    core_mask = cv2.circle(mask_template.copy(), (center_x, center_y), core_radius, 255, -1)
    core_region = cv2.bitwise_and(original_image, original_image, mask=core_mask)

    cladding_outer_mask = cv2.circle(mask_template.copy(), (center_x, center_y), cladding_radius, 255, -1)
    cladding_mask = cv2.subtract(cladding_outer_mask, core_mask)
    cladding_region = cv2.bitwise_and(original_image, original_image, mask=cladding_mask)

    ferrule_mask = cv2.bitwise_not(cladding_outer_mask)
    ferrule_region = cv2.bitwise_and(original_image, original_image, mask=ferrule_mask)

    diagnostic_image = original_image.copy()
    cv2.circle(diagnostic_image, (center_x, center_y), 3, (0, 255, 255), -1)
    cv2.circle(diagnostic_image, (center_x, center_y), core_radius, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(diagnostic_image, (center_x, center_y), cladding_radius, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_guess_core.png'), core_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_guess_cladding.png'), cladding_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_guess_ferrule.png'), ferrule_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_guess_annotated.png'), diagnostic_image)

    return result


if __name__ == '__main__':
    # For standalone testing
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"').strip("'")
    
    result = segment_fiber_with_multimodal_analysis(image_path)
    if result['success']:
        print(f"Success! Center: {result['center']}, Core: {result['core_radius']}, Cladding: {result['cladding_radius']}")
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")
