import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os

def analyze_grayscale_image(image_path, output_dir='output'):
    """
    Analyzes an image by converting it to grayscale, generating a black and white
    pixel intensity map, a JSON file of grayscale pixel values, and a histogram
    of these intensities.

    Args:
        image_path (str): The path to the input image.
        output_dir (str): The directory to save the output files.
    """
    # --- 1. Load the Image and Convert to Grayscale ---
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image from '{image_path}'. Check the file format.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Successfully loaded and converted image to grayscale: {image_path}")
    print(f"Grayscale image dimensions (Height, Width): {gray_image.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # --- 2. Generate JSON file of all grayscale pixel values and coordinates ---
    height, width = gray_image.shape
    pixel_data = []
    for y in range(height):
        for x in range(width):
            pixel_value = int(gray_image.item(y, x))  # Get integer grayscale value
            pixel_data.append({
                'coordinates': {'x': x, 'y': y},
                'intensity': pixel_value
            })

    json_path = os.path.join(output_dir, f'{base_filename}_grayscale_pixel_values.json')
    with open(json_path, 'w') as f:
        json.dump(pixel_data, f, indent=4)
    print(f"Successfully generated JSON file of grayscale pixel values at: {json_path}")

    # --- 3. Generate a Black and White Pixel Intensity Map ---
    # The grayscale image itself is the intensity map
    intensity_map_path = os.path.join(output_dir, f'{base_filename}_intensity_map.png')
    cv2.imwrite(intensity_map_path, gray_image)
    print(f"Successfully generated black and white pixel intensity map at: {intensity_map_path}")

    # --- 4. Generate a Histogram of Grayscale Intensities ---
    plt.figure(figsize=(10, 6))
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title('Histogram of Grayscale Pixel Intensities')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.grid(True)

    histogram_path = os.path.join(output_dir, f'{base_filename}_intensity_histogram.png')
    plt.savefig(histogram_path)
    plt.close() # Close the plot to free memory
    print(f"Successfully generated histogram of grayscale intensities at: {histogram_path}")
    print("\nGrayscale image analysis complete.")


if __name__ == '__main__':
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description="""Grayscale Image Analysis Script using OpenCV and NumPy.
        This script takes an image, converts it to grayscale, and generates:
        1. A black and white pixel intensity map.
        2. A JSON file with coordinates and grayscale intensity of every pixel.
        3. A histogram of the grayscale pixel intensities."""
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

    analyze_grayscale_image(args.image, args.output)