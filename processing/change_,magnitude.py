import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os

def analyze_image(image_path, output_dir='output'):
    """
    Analyzes an image to detail pixel value changes, generate a change map,
    a JSON file of pixel values, and a histogram of changes.

    Args:
        image_path (str): The path to the input image.
        output_dir (str): The directory to save the output files.
    """
    # --- 1. Load the Image ---
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image from '{image_path}'. Check the file format.")
        return

    print(f"Successfully loaded image: {image_path}")
    print(f"Image dimensions (Height, Width, Channels): {image.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # --- 2. Generate JSON file of all pixel values and coordinates ---
    height, width, _ = image.shape
    pixel_data = []
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x].tolist()  # Convert numpy array to list for JSON
            pixel_data.append({
                'coordinates': {'x': x, 'y': y},
                'bgr_value': pixel_value
            })

    json_path = os.path.join(output_dir, f'{base_filename}_pixel_values.json')
    with open(json_path, 'w') as f:
        json.dump(pixel_data, f, indent=4)
    print(f"Successfully generated JSON file of pixel values at: {json_path}")

    # --- 3. Detail Every Change in Pixel Value ---
    # Convert to grayscale for simpler change detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradients in x and y directions using the Sobel operator
    # Using a 64-bit float to avoid overflow for high-frequency changes
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradient (the "change")
    change_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # --- 4. Generate an Image with a Color Map of Changes ---
    # Normalize the magnitude to the 0-255 range to be used as an image
    change_map_normalized = cv2.normalize(change_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply a colormap for better visualization
    change_map_colored = cv2.applyColorMap(change_map_normalized, cv2.COLORMAP_JET)

    change_map_path = os.path.join(output_dir, f'{base_filename}_change_map.png')
    cv2.imwrite(change_map_path, change_map_colored)
    print(f"Successfully generated color map of changes at: {change_map_path}")


    # --- 5. Generate a Histogram of the Changes ---
    plt.figure(figsize=(10, 6))
    # Use the non-normalized magnitude for an accurate distribution
    plt.hist(change_magnitude.ravel(), bins=256, range=(0.0, np.max(change_magnitude)))
    plt.title('Histogram of Pixel Value Changes')
    plt.xlabel('Change Magnitude (Gradient)')
    plt.ylabel('Frequency')
    plt.grid(True)

    histogram_path = os.path.join(output_dir, f'{base_filename}_change_histogram.png')
    plt.savefig(histogram_path)
    plt.close() # Close the plot to free memory
    print(f"Successfully generated histogram of changes at: {histogram_path}")
    print("\nAnalysis complete.")


if __name__ == '__main__':
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description="""Image Analysis Script using OpenCV and NumPy.
        This script takes an image and generates:
        1. A color map image detailing pixel value changes.
        2. A JSON file with coordinates and values of every pixel.
        3. A histogram of the pixel value changes."""
    )
    parser.add_argument(
        '-i', '--image',
        type=str,
        required=True,
        help='Path to the input image file.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='Directory to save the output files (default: output).'
    )
    args = parser.parse_args()

    analyze_image(args.image, args.output)
