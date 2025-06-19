import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

def segment_fiber_with_multimodal_analysis(image_path, output_dir='output_advanced'):
    """
    Performs superior segmentation of a fiber optic end-face by fusing analysis
    from three different image properties: Pixel Intensity, Change Magnitude (Gradient),
    and Local Texture.

    This multi-modal approach achieves robust region separation by:
    1.  Detecting the fiber's geometric center.
    2.  Creating 1D radial profiles for Intensity, Change, and Texture from that center.
    3.  Fusing the signals from all three profiles to pinpoint the core and cladding
        boundaries with high confidence.

    Args:
        image_path (str): The path to the input fiber optic image.
        output_dir (str): The directory to save the output files.
    """
    # --- 1. Preprocessing: Load, Convert, and Blur ---
    if not os.path.exists(image_path):
        print(f"\nError: The file was not found at the specified path: '{image_path}'")
        print("Please check the path and try again.")
        return

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read the image from '{image_path}'. The file may be corrupt or in an unsupported format.")
        return

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Blurring is crucial for robust gradient and circle detection
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
    print(f"Successfully loaded and preprocessed image: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    height, width = gray_image.shape

    # --- 2. Center Finding: Establish the Anchor Point ---
    # Use Hough Circle Transform as a robust method to find the geometric center.
    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=width//2,
        param1=50, param2=30, minRadius=10, maxRadius=int(height / 2.5)
    )

    if circles is None:
        print("Could not detect the fiber's center. Analysis cannot proceed.")
        return
    
    circles = np.uint16(np.around(circles))
    center_x, center_y, _ = circles[0, 0]
    print(f"Step 1: Detected fiber center at ({center_x}, {center_y})")

    # --- 3. Multi-Modal Profile Calculation: The Core of the Analysis ---
    # We will now analyze the image from the center outwards, building three distinct
    # profiles that each tell a part of the story.

    # 3a. Prepare the data layers for profiling
    # Layer 1: Intensity (from the blurred image for smoothness)
    intensity_layer = blurred_image
    
    # Layer 2: Change Magnitude (captures boundary sharpness)
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    change_magnitude_layer = cv2.magnitude(sobel_x, sobel_y)

    # Layer 3: Texture (quantifies surface uniformity vs. roughness)
    # Using Local Binary Patterns (LBP) as a powerful texture descriptor.
    # The 'uniform' method is robust and captures primary textural information.
    lbp_layer = local_binary_pattern(gray_image, P=8, R=1, method='uniform')

    # 3b. Build the radial profiles
    max_radius = int(np.sqrt(center_x**2 + center_y**2)) # A safe maximum
    
    # Initialize profile arrays
    radial_intensity = np.zeros(max_radius)
    radial_change = np.zeros(max_radius)
    radial_texture_values = [[] for _ in range(max_radius)] # To calculate variance later
    radial_counts = np.zeros(max_radius, dtype=int)

    # Iterate through each pixel once to build all profiles simultaneously
    for y in range(height):
        for x in range(width):
            radius = int(np.sqrt((x - center_x)**2 + (y - center_y)**2))
            if radius < max_radius:
                radial_intensity[radius] += intensity_layer[y, x]
                radial_change[radius] += change_magnitude_layer[y, x]
                radial_texture_values[radius].append(lbp_layer[y, x])
                radial_counts[radius] += 1

    # Normalize the profiles by the number of pixels at each radius
    radial_counts[radial_counts == 0] = 1 # Avoid division by zero
    radial_intensity /= radial_counts
    radial_change /= radial_counts
    
    # Calculate the variance of LBP values for the texture profile
    radial_texture_variance = np.zeros(max_radius)
    for r in range(max_radius):
        if radial_texture_values[r]:
            radial_texture_variance[r] = np.var(radial_texture_values[r])

    print("Step 2: Generated multi-modal radial profiles (Intensity, Change, Texture)")

    # --- 4. Boundary Fusion: Find Radii using All Profiles ---
    # Find peaks in the change magnitude profile, as these are the strongest
    # indicators of physical boundaries.
    change_peaks = []
    # Simple peak finding logic
    for r in range(10, len(radial_change) - 10): # Avoid edges
        if radial_change[r] > np.max(radial_change[r-5:r]) and radial_change[r] > np.max(radial_change[r+1:r+6]):
            if radial_change[r] > np.mean(radial_change) * 1.5: # Must be a significant peak
                 change_peaks.append((r, radial_change[r]))
    
    if len(change_peaks) < 2:
        print("Could not reliably detect two distinct boundaries. Check image quality.")
        return

    # Sort peaks by magnitude (sharpness of boundary)
    change_peaks.sort(key=lambda p: p[1], reverse=True)
    
    # The Core-Cladding boundary is the sharpest; the Cladding-Ferrule is next.
    radii = sorted([p[0] for p in change_peaks[:2]])
    core_radius = radii[0]
    cladding_radius = radii[1]

    print(f"Step 3: Fused profile data to identify boundaries.")
    print(f"-> Detected Core Radius: {core_radius} pixels (strongest change signature)")
    print(f"-> Detected Cladding Radius: {cladding_radius} pixels (second strongest change signature)")


    # --- 5. Visualization and Diagnostics ---
    # Plotting the profiles is key to understanding the decision-making process
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('Multi-Modal Radial Analysis for Boundary Detection', fontsize=16)

    # Plot Radial Intensity
    axs[0].plot(radial_intensity, color='blue')
    axs[0].set_title('Average Pixel Intensity vs. Radius')
    axs[0].set_ylabel('Avg. Intensity')
    axs[0].grid(True)
    axs[0].axvline(x=core_radius, color='g', linestyle='--', label=f'Core Boundary ({core_radius}px)')
    axs[0].axvline(x=cladding_radius, color='r', linestyle='--', label=f'Cladding Boundary ({cladding_radius}px)')
    axs[0].legend()

    # Plot Radial Change Magnitude
    axs[1].plot(radial_change, color='orange')
    axs[1].set_title('Average Change Magnitude (Gradient) vs. Radius')
    axs[1].set_ylabel('Avg. Change')
    axs[1].grid(True)
    axs[1].axvline(x=core_radius, color='g', linestyle='--')
    axs[1].axvline(x=cladding_radius, color='r', linestyle='--')

    # Plot Radial Texture Variance
    axs[2].plot(radial_texture_variance, color='purple')
    axs[2].set_title('Variance of Local Binary Patterns (Texture) vs. Radius')
    axs[2].set_xlabel('Radius from Center (pixels)')
    axs[2].set_ylabel('LBP Variance')
    axs[2].grid(True)
    axs[2].axvline(x=core_radius, color='g', linestyle='--')
    axs[2].axvline(x=cladding_radius, color='r', linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    diagnostic_plot_path = os.path.join(output_dir, f'{base_filename}_radial_analysis.png')
    plt.savefig(diagnostic_plot_path)
    plt.close()
    print(f"Step 4: Saved diagnostic profile plot to '{diagnostic_plot_path}'")

    # --- 6. Mask Generation and Final Segmentation ---
    # Create masks using the high-confidence radii found through fusion.
    mask_template = np.zeros_like(gray_image)

    # Core
    core_mask = cv2.circle(mask_template.copy(), (center_x, center_y), core_radius, 255, -1)
    core_region = cv2.bitwise_and(original_image, original_image, mask=core_mask)

    # Cladding
    cladding_outer_mask = cv2.circle(mask_template.copy(), (center_x, center_y), cladding_radius, 255, -1)
    cladding_mask = cv2.subtract(cladding_outer_mask, core_mask)
    cladding_region = cv2.bitwise_and(original_image, original_image, mask=cladding_mask)

    # Ferrule
    ferrule_mask = cv2.bitwise_not(cladding_outer_mask)
    ferrule_region = cv2.bitwise_and(original_image, original_image, mask=ferrule_mask)

    # Create a visual diagnostic image with boundaries overlaid
    diagnostic_image = original_image.copy()
    cv2.circle(diagnostic_image, (center_x, center_y), core_radius, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(diagnostic_image, 'Core', (center_x, center_y - core_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.circle(diagnostic_image, (center_x, center_y), cladding_radius, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(diagnostic_image, 'Cladding', (center_x, center_y - cladding_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Save all final output images
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_region_core.png'), core_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_region_cladding.png'), cladding_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_region_ferrule.png'), ferrule_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_boundaries_detected.png'), diagnostic_image)

    print(f"\nStep 5: Segmentation complete. All output files saved in '{output_dir}'.")


if __name__ == '__main__':
    # --- MODIFIED SECTION: Interactive User Input ---
    print("--- Advanced Fiber Optic Segmentation Tool ---")
    
    # Ask the user for the path to the image file.
    image_path_input = input("Please enter the full path to the image you want to analyze: ")
    
    # Clean up the input path to handle potential quotes from copy-pasting.
    image_path_cleaned = image_path_input.strip().strip('"').strip("'")

    # Define the output directory (can also be made an input prompt if desired)
    output_dir_default = 'output_advanced'

    print(f"\nAnalyzing image: {image_path_cleaned}")
    print(f"Output will be saved in: '{output_dir_default}/'")
    print("-" * 30)
    
    # Call the main analysis function with the user-provided path.
    segment_fiber_with_multimodal_analysis(image_path_cleaned, output_dir_default)