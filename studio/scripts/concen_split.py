import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def find_fiber_center(image):
    """
    Finds the precise center of the fiber by calculating the centroid of the
    brightest regions (core and ferrule).

    Args:
        image (np.array): Grayscale input image.

    Returns:
        tuple: (cx, cy) integer coordinates of the center.
    """
    # Use Otsu's thresholding to automatically find an optimal value
    # to separate the bright regions (core/ferrule) from the dark (cladding/background).
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the moments of the binary image
    M = cv2.moments(thresh_image)

    # Calculate x,y coordinate of center. This is more robust than finding the brightest pixel.
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # If no contours are found, fall back to the image center
        h, w = image.shape
        cx, cy = w // 2, h // 2

    return cx, cy

def analyze_radial_profile(image, center):
    """
    Calculates the average pixel intensity for each radius moving out from the center.
    It then finds the core and cladding radii by analyzing the derivative of this profile.

    Args:
        image (np.array): Grayscale input image.
        center (tuple): (cx, cy) coordinates of the fiber center.

    Returns:
        tuple: (r_core, r_cladding, radial_profile, derivative)
    """
    cx, cy = center
    h, w = image.shape

    # Create a grid of pixel coordinates
    y, x = np.indices((h, w))

    # Calculate the distance of each pixel from the center (Distance Formula)
    # delta_x^2 + delta_y^2 = r^2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(int) # Convert distances to integer radii

    # Calculate the average intensity for each radius using a fast bin-counting method
    # Sum of intensities for each radius
    tbin = np.bincount(r.ravel(), image.ravel())
    # Number of pixels for each radius
    nr = np.bincount(r.ravel())

    # Avoid division by zero
    radial_profile = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr!=0)

    # We only care about the profile up to the edge of the image
    max_radius = min(cx, cy, w - cx, h - cy)
    radial_profile = radial_profile[:max_radius]

    # --- Find Radii using the Derivative (the "Calculus" part) ---
    # Smooth the profile slightly to reduce noise before taking the derivative
    smoothed_profile = cv2.GaussianBlur(radial_profile, (5, 1), 0).flatten()

    # Calculate the first derivative (gradient) to find points of max change
    derivative = np.gradient(smoothed_profile)

    # The Core -> Cladding boundary is the point of steepest *descent* (min derivative)
    # We search from the center outwards to the first major peak in brightness.
    try:
        # Find the approximate peak of the core's brightness
        core_peak_intensity_idx = np.argmax(smoothed_profile[:len(smoothed_profile)//2])
        # Find the minimum derivative *after* the core's peak
        r_core = core_peak_intensity_idx + np.argmin(derivative[core_peak_intensity_idx:])
    except ValueError:
        print("Warning: Could not robustly determine core radius. Falling back.")
        r_core = np.argmin(derivative)


    # The Cladding -> Ferrule boundary is the point of steepest *ascent* (max derivative)
    try:
        r_cladding = np.argmax(derivative[r_core:]) + r_core
    except ValueError:
        print("Warning: Could not robustly determine cladding radius. Falling back.")
        r_cladding = np.argmax(derivative)


    return int(r_core), int(r_cladding), radial_profile, derivative


def create_and_apply_masks(image, center, r_core, r_cladding):
    """
    Creates pixel-perfect masks using the calculated center and radii and
    applies them to the image.

    Args:
        image (np.array): Grayscale input image.
        center (tuple): (cx, cy) center coordinates.
        r_core (int): Radius of the core.
        r_cladding (int): Radius of the cladding.

    Returns:
        tuple: (isolated_core, isolated_cladding)
    """
    cx, cy = center
    h, w = image.shape
    y, x = np.indices((h, w))

    # Create a distance map using the equation of a circle: (x-cx)^2 + (y-cy)^2
    dist_sq = (x - cx)**2 + (y - cy)**2

    # --- Create Perfect Masks based on Distance Formula ---
    # Core mask: Where distance squared is less than or equal to the core radius squared
    core_mask = (dist_sq <= r_core**2).astype(np.uint8) * 255

    # Cladding mask: Where distance squared is > core_r^2 AND <= cladding_r^2
    cladding_mask = ((dist_sq > r_core**2) & (dist_sq <= r_cladding**2)).astype(np.uint8) * 255

    # Apply masks to the original grayscale image
    isolated_core = cv2.bitwise_and(image, image, mask=core_mask)
    isolated_cladding = cv2.bitwise_and(image, image, mask=cladding_mask)

    return isolated_core, isolated_cladding, core_mask, cladding_mask


def crop_to_content(image, mask):
    """Crops an image to the bounding box of the content in its mask."""
    coords = np.argwhere(mask > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return image[y_min:y_max + 1, x_min:x_max + 1]
    return image # Return original if mask is empty


def process_fiber_image(image_path, output_dir='output_precise'):
    """Main processing pipeline for a single fiber optic image."""
    print(f"\n--- Processing: {image_path} ---")
    if not os.path.exists(image_path):
        print(f"Error: Not found: {image_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Find the precise center of the fiber
    center = find_fiber_center(gray_image)
    print(f"Calculated Center: {center}")

    # 2. Analyze the radial profile to find radii
    r_core, r_cladding, profile, derivative = analyze_radial_profile(gray_image, center)
    print(f"Calculated Radii -> Core: {r_core}px, Cladding: {r_cladding}px")
    
    # 3. Create and apply pixel-perfect masks
    core_img, cladding_img, core_mask, cladding_mask = create_and_apply_masks(gray_image, center, r_core, r_cladding)

    # 4. Crop the results to their content
    cropped_core = crop_to_content(core_img, core_mask)
    cropped_cladding = crop_to_content(cladding_img, cladding_mask)

    # 5. Save everything
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # --- Save analysis plot ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(profile, 'b-', label='Intensity Profile')
    ax2.plot(derivative, 'r-', label='Derivative')
    ax1.axvline(x=r_core, color='g', linestyle='--', label=f'Core Radius ({r_core}px)')
    ax1.axvline(x=r_cladding, color='m', linestyle='--', label=f'Cladding Radius ({r_cladding}px)')
    ax1.set_xlabel('Radius (pixels from center)')
    ax1.set_ylabel('Average Pixel Intensity', color='b')
    ax2.set_ylabel('Intensity Gradient (Derivative)', color='r')
    plt.title(f'Radial Intensity Analysis for {os.path.basename(image_path)}')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(output_dir, f"{base_filename}_analysis_plot.png"))
    plt.close(fig)

    # --- Save image results ---
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_core.png"), cropped_core)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_cladding.png"), cropped_cladding)
    print(f"Successfully saved results to '{output_dir}'")


if __name__ == '__main__':
    # Add the filenames of the images you have uploaded here
    image_filenames = [
        'ima12.jpg',
        'ima18.jpg',
        'ima19.jpg',
        'img35.jpg',
        'img38.jpg',
        'img63.jpg'
    ]
    
    for filename in image_filenames:
        process_fiber_image(filename)