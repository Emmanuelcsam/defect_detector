import cv2
import numpy as np
import os

def segment_image_by_intensity(image_path, intensity_ranges):
    """
    Loads a grayscale image and segments it into multiple images based on
    specified pixel intensity ranges.

    For each intensity range, a new image is created containing only the
    pixels from the original image that fall within that range.

    Args:
        image_path (str): The full path to the input grayscale image.
        intensity_ranges (list of tuples): A list where each tuple defines a
                                            range of intensities, e.g., [(min1, max1), (min2, max2)].
                                            Values should be between 0 and 255.

    Returns:
        None. Images are saved to a new directory.
    """
    # Validate if the image path exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    # Load the image in grayscale mode
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Error: Could not read the image from '{image_path}'. Check the file format.")
        return

    # Create a directory to save the output images
    output_dir = "segmented_regions"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Image loaded successfully. Found {len(intensity_ranges)} intensity ranges to process.")

    # Process each specified intensity range
    for i, (min_val, max_val) in enumerate(intensity_ranges):
        print(f"Processing range {i+1}: Intensity {min_val} to {max_val}...")

        # Create a binary mask for the current intensity range.
        # Pixels within the range [min_val, max_val] will be white (255),
        # and all other pixels will be black (0).
        mask = cv2.inRange(original_image, min_val, max_val)

        # Use the mask to extract the corresponding regions from the original image.
        # The bitwise_and operation keeps original pixel values where the mask is white.
        segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)

        # Construct the output filename
        output_filename = f"region_{i+1}_intensity_{min_val}-{max_val}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Save the resulting segmented image
        cv2.imwrite(output_path, segmented_image)
        print(f"-> Saved segmented image to '{output_path}'")

    print("\nProcessing complete.")


if __name__ == '__main__':
    # --- USER INPUT SECTION ---

    # 1. Specify the path to your input image.
    # Replace this with the path to your image file.
    input_image_file = r"C:\Users\Saem1001\Documents\GitHub\IPPS\processing\output\img (210)_intensity_map.png"

    # 2. Define the pixel intensity ranges you want to extract.
    # Based on your example histogram, we can define three distinct regions.
    # Format: [(min_intensity_1, max_intensity_1), (min_intensity_2, max_intensity_2), ...]
    #
    # For the provided image 'img (210)_intensity_map.jpg', the following ranges
    # correspond to the main features:
    #   - The dark ring and scattered dots.
    #   - The large gray background.
    #   - The bright central area.
    mode_ranges = [
        (80, 130),    # Range for the darkest regions (e.g., the ring)
        (170, 200),   # Range for the main background
        (200, 230)    # Range for the brightest regions (e.g., inside the ring)
    ]

    # --- END OF USER INPUT SECTION ---

    # Call the function to perform the segmentation
    segment_image_by_intensity(input_image_file, mode_ranges)