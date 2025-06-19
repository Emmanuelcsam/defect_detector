import cv2
import numpy as np
import os

# --- MODIFICATION FOR LINUX/HEADLESS ENVIRONMENTS ---
# Set the backend for Matplotlib to a non-interactive one ('Agg')
# This must be done BEFORE importing pyplot.
# This prevents it from trying to initialize a Qt or other GUI window,
# which resolves the "Qt platform plugin" error.
import matplotlib
matplotlib.use('Agg')
# --- END MODIFICATION ---

import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.signal import find_peaks

def universal_fiber_segmentation(image_path, output_dir='output_universal'):
    """
    Performs universal segmentation of a fiber optic end-face by first establishing a
    high-confidence center point through multi-modal fusion, then using that
    center for a precise radial analysis.

    This robust process is:
    1.  Get an initial center guess using Hough Circles (Geometric).
    2.  Refine the center by finding the centroid of the brightest pixels (Photometric).
    3.  Further refine with the centroid of the smoothest texture (Textural).
    4.  Fuse these three sources to get a highly accurate, data-driven center.
    5.  Use this fused center to create 1D radial profiles for Intensity, Gradient, and Texture.
    6.  Pinpoint the core/cladding boundaries from the sharpest peaks in the gradient profile.

    Args:
        image_path (str): The path to the input fiber optic image.
        output_dir (str): The directory to save the output files.
    """
    # --- 1. Preprocessing: Load, Convert, and Blur ---
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

    # --- 2. Multi-Modal Center Finding (The Core Innovation) ---
    print("\n--- Step 1: Multi-Modal Center Finding ---")
    
    # Hypothesis A: Geometric Center Guess (Hough Circles)
    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=width//2,
        param1=50, param2=30, minRadius=10, maxRadius=int(height / 2.5)
    )

    if circles is None:
        print("Initial geometric guess failed. Cannot proceed.")
        return
    hough_center = np.uint16(np.around(circles[0, 0][:2]))
    print(f"  -> Geometric Guess (Hough): ({hough_center[0]}, {hough_center[1]})")

    # Hypothesis B: Photometric Center (Centroid of Brightest Pixels)
    brightness_threshold = np.percentile(gray_image, 95)
    _, core_mask = cv2.threshold(blurred_image, brightness_threshold, 255, cv2.THRESH_BINARY)
    M_bright = cv2.moments(core_mask)
    if M_bright["m00"] == 0:
        print("Warning: Could not find brightness centroid. Relying on other guesses.")
        brightness_center = hough_center
    else:
        cx_bright = int(M_bright["m10"] / M_bright["m00"])
        cy_bright = int(M_bright["m01"] / M_bright["m00"])
        brightness_center = np.array([cx_bright, cy_bright], dtype=np.uint16)
        print(f"  -> Photometric Guess (Core): ({brightness_center[0]}, {brightness_center[1]})")

    # Hypothesis C: Textural Center (Centroid of Smoothest Texture)
    lbp_layer = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    texture_threshold = np.percentile(lbp_layer, 25)
    texture_mask = np.where(lbp_layer <= texture_threshold, 255, 0).astype(np.uint8)
    M_texture = cv2.moments(texture_mask)
    if M_texture["m00"] == 0:
        print("Warning: Could not find texture centroid. Relying on other guesses.")
        texture_center = hough_center
    else:
        cx_texture = int(M_texture["m10"] / M_texture["m00"])
        cy_texture = int(M_texture["m01"] / M_texture["m00"])
        texture_center = np.array([cx_texture, cy_texture], dtype=np.uint16)
        print(f"  -> Textural Guess (Glass): ({texture_center[0]}, {texture_center[1]})")

    # Fusion: Average the centers for a robust, data-driven result.
    final_center = np.mean([hough_center, brightness_center, texture_center], axis=0).astype(np.uint16)
    center_x, center_y = final_center
    print(f"  -> Fused High-Confidence Center: ({center_x}, {center_y})")


    # --- 3. Radial Profile Calculation (Using the Fused Center) ---
    print("\n--- Step 2: Radial Profile Analysis ---")
    
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    change_magnitude_layer = cv2.magnitude(sobel_x, sobel_y)

    max_radius = int(min(center_x, center_y, width - center_x, height - center_y))
    
    y_coords, x_coords = np.indices((height, width))
    radii_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2).astype(int)

    radial_intensity = np.zeros(max_radius)
    radial_change = np.zeros(max_radius)

    for r in range(max_radius):
        mask = radii_map == r
        if np.any(mask):
            radial_intensity[r] = np.mean(blurred_image[mask])
            radial_change[r] = np.mean(change_magnitude_layer[mask])

    print("  -> Generated radial profiles for Intensity and Gradient.")


    # --- 4. Boundary Detection from Gradient Profile ---
    print("\n--- Step 3: Boundary Detection ---")
    
    peaks, properties = find_peaks(radial_change, prominence=np.mean(radial_change), distance=10)
    
    if len(peaks) < 2:
        print("Error: Could not reliably detect two distinct boundaries. Check image quality.")
        return

    top_two_peak_indices = np.argsort(properties['prominences'])[-2:]
    radii = sorted(peaks[top_two_peak_indices])
    
    core_radius = radii[0]
    cladding_radius = radii[1]

    print(f"  -> Detected Core Radius: {core_radius} pixels")
    print(f"  -> Detected Cladding Radius: {cladding_radius} pixels")


    # --- 5. Visualization and Diagnostics ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Universal Radial Analysis (from Fused Center)', fontsize=16)

    axs[0].plot(radial_intensity, color='blue', label='Avg. Intensity')
    axs[0].set_title('Average Pixel Intensity vs. Radius')
    axs[0].set_ylabel('Intensity')
    axs[0].grid(True)
    axs[0].axvline(x=core_radius, color='g', linestyle='--', label=f'Core Boundary ({core_radius}px)')
    axs[0].axvline(x=cladding_radius, color='r', linestyle='--', label=f'Cladding Boundary ({cladding_radius}px)')
    axs[0].legend()

    axs[1].plot(radial_change, color='orange', label='Avg. Gradient')
    axs[1].set_title('Average Change Magnitude (Gradient) vs. Radius')
    axs[1].set_xlabel('Radius from Center (pixels)')
    axs[1].set_ylabel('Gradient Magnitude')
    axs[1].grid(True)
    axs[1].plot(peaks, radial_change[peaks], "x", color='purple', markersize=10, label='Detected Peaks')
    axs[1].axvline(x=core_radius, color='g', linestyle='--')
    axs[1].axvline(x=cladding_radius, color='r', linestyle='--')
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    diagnostic_plot_path = os.path.join(output_dir, f'{base_filename}_radial_analysis.png')
    plt.savefig(diagnostic_plot_path)
    plt.close()
    print(f"\n--- Step 4: Saved diagnostic plot to '{diagnostic_plot_path}' ---")

    # --- 6. Mask Generation and Final Segmentation ---
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
    
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_region_core.png'), core_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_region_cladding.png'), cladding_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_region_ferrule.png'), ferrule_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_boundaries_detected.png'), diagnostic_image)

    print(f"\n--- Step 5: Segmentation complete. Outputs saved in '{output_dir}'. ---")


if __name__ == '__main__':
    print("--- Universal Fiber Optic Segmentation Tool ---")
    print("This script finds the fiber's center using a fusion of geometric,")
    print("photometric, and textural data for maximum accuracy.\n")
    
    image_path_input = input("Please enter the full path to the image to analyze: ")
    image_path_cleaned = image_path_input.strip().strip('"').strip("'")
    
    print("-" * 50)
    universal_fiber_segmentation(image_path_cleaned)
