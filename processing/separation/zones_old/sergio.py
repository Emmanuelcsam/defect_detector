import cv2
import numpy as np
import os

def segment_fiber_optic_image(image_path, output_dir='output'):
    """
    Analyzes and segments a fiber optic endface image into core, cladding,
    and ferrule regions based on pixel intensity and change magnitude.

    Args:
        image_path (str): The path to the input image.
        output_dir (str): The directory to save the output files.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read the image from '{image_path}'.")
        return

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Apply blur to reduce noise and improve circle detection
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

    print(f"Successfully loaded image: {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # --- 2. Detect the Center of the Fiber ---
    # Use Hough Circle Transform to find the most prominent circle
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=int(gray_image.shape[0] / 3) # Max radius is 1/3 of image height
    )

    if circles is None:
        print("Could not detect the center of the fiber. Try adjusting Hough Circle parameters.")
        return

    # Get the center coordinates of the most prominent circle
    circles = np.uint16(np.around(circles))
    center_x, center_y, _ = circles[0, 0]
    print(f"Detected fiber center at: ({center_x}, {center_y})")

    # --- 3. Analyze Radial Change Magnitude ---
    # Calculate gradient magnitude (similar to the first script)
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    change_magnitude = cv2.magnitude(sobel_x, sobel_y)

    height, width = gray_image.shape
    max_radius = int(np.sqrt(height**2 + width**2) / 2) # Theoretical max radius

    # Calculate the average gradient for each radius from the center
    radial_profile = np.zeros(max_radius)
    radial_counts = np.zeros(max_radius, dtype=int)

    for y in range(height):
        for x in range(width):
            radius = int(np.sqrt((x - center_x)**2 + (y - center_y)**2))
            if radius < max_radius:
                radial_profile[radius] += change_magnitude[y, x]
                radial_counts[radius] += 1

    # Avoid division by zero
    radial_counts[radial_counts == 0] = 1
    radial_profile /= radial_counts

    # --- 4. Identify Core and Cladding Boundaries ---
    # Find peaks in the radial profile. These are the boundaries.
    # We will find the two strongest peaks (local maxima)
    peaks = []
    for r in range(1, len(radial_profile) - 1):
        if radial_profile[r] > radial_profile[r-1] and radial_profile[r] > radial_profile[r+1] and radial_profile[r] > np.mean(radial_profile):
             peaks.append((r, radial_profile[r]))

    if len(peaks) < 2:
        print("Could not reliably detect two distinct boundaries (core/cladding, cladding/ferrule).")
        return

    # Sort peaks by their magnitude (strength of the boundary) and take the top two
    peaks.sort(key=lambda p: p[1], reverse=True)
    radii = sorted([p[0] for p in peaks[:2]])
    core_radius = radii[0]
    cladding_radius = radii[1]

    print(f"Detected Core Radius: {core_radius} pixels")
    print(f"Detected Cladding Radius: {cladding_radius} pixels")

    # --- 5. Create Masks and Output Segmented Regions ---
    mask = np.zeros_like(gray_image)

    # Create Core mask and extract region
    core_mask = cv2.circle(mask.copy(), (center_x, center_y), core_radius, 255, -1)
    core_region = cv2.bitwise_and(original_image, original_image, mask=core_mask)

    # Create Cladding mask and extract region
    cladding_mask_outer = cv2.circle(mask.copy(), (center_x, center_y), cladding_radius, 255, -1)
    cladding_mask = cv2.subtract(cladding_mask_outer, core_mask)
    cladding_region = cv2.bitwise_and(original_image, original_image, mask=cladding_mask)

    # Create Ferrule mask and extract region
    ferrule_mask = cv2.bitwise_not(cladding_mask_outer)
    ferrule_region = cv2.bitwise_and(original_image, original_image, mask=ferrule_mask)

    # Create a diagnostic image with boundaries drawn
    diagnostic_image = original_image.copy()
    cv2.circle(diagnostic_image, (center_x, center_y), core_radius, (0, 255, 0), 2)  # Green for core
    cv2.circle(diagnostic_image, (center_x, center_y), cladding_radius, (0, 0, 255), 2) # Red for cladding

    # Save all the output images
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_core_region.png'), core_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_cladding_region.png'), cladding_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_ferrule_region.png'), ferrule_region)
    cv2.imwrite(os.path.join(output_dir, f'{base_filename}_boundaries_detected.png'), diagnostic_image)

    print(f"\nSegmentation complete. Output files saved in '{output_dir}' directory.")


if __name__ == '__main__':
    # Interactively prompt the user for the input image path
    image_path = input("Please enter the path to the input fiber optic image file: ")

    # Interactively prompt the user for the output directory, providing a default value
    output_dir = input("Enter the directory to save the segmented output files (press Enter for 'output_segmented'): ") or 'output_segmented'

    # Call the main function with the user-provided paths
    segment_fiber_optic_image(image_path, output_dir)
