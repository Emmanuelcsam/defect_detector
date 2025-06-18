import cv2
import numpy as np
import os
import json

def create_detailed_matrices(input_folder: str, output_folder: str):
    """
    Processes a batch of images from an input folder, converting each into a
    detailed JSON file containing pixel coordinates and BGR intensity values.

    Args:
        input_folder (str): The path to the folder containing images.
        output_folder (str): The path to the folder where JSON files will be saved.
    """
    # Verify the input directory exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input directory not found at '{input_folder}'")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # List of supported image file extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Iterate over all files in the input directory
    image_files = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in supported_extensions)]
    
    if not image_files:
        print(f"No supported image files found in '{input_folder}'.")
        return

    print(f"\nFound {len(image_files)} images to process.")

    for filename in image_files:
        input_image_path = os.path.join(input_folder, filename)
        
        try:
            # Read the image using OpenCV. It's loaded in BGR format by default.
            image = cv2.imread(input_image_path)

            if image is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue

            print(f"Processing: {filename}...")

            # Get image dimensions (height, width, channels)
            height, width, channels = image.shape
            
            # Prepare the main data structure for the JSON file
            image_data = {
                "filename": filename,
                "image_dimensions": {
                    "width": width,
                    "height": height,
                    "channels": channels
                },
                "pixels": []
            }

            # Iterate through each pixel to extract its data
            for y in range(height):
                for x in range(width):
                    # Get the BGR pixel value. Note: OpenCV uses (y, x) indexing.
                    pixel_value = image[y, x]
                    
                    # Create a dictionary for the current pixel's data.
                    # .tolist() converts the numpy array to a standard Python list for JSON serialization.
                    pixel_entry = {
                        "coordinates": {"x": x, "y": y},
                        "bgr_intensity": pixel_value.tolist() 
                    }
                    image_data["pixels"].append(pixel_entry)

            # Define the output JSON filename
            json_filename = os.path.splitext(filename)[0] + '.json'
            output_json_path = os.path.join(output_folder, json_filename)

            # Write the detailed matrix to a JSON file
            with open(output_json_path, 'w') as json_file:
                json.dump(image_data, json_file, indent=4)
            
            print(f"Successfully created JSON for {filename} at {output_json_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    # Ask the user for the input and output directories
    input_dir = input("Please enter the full path to your input image folder: ")
    output_dir = input("Please enter the full path for your output JSON folder: ")
    
    # Clean up paths (e.g., remove quotes if user pastes them)
    input_dir = input_dir.strip().strip('"\'')
    output_dir = output_dir.strip().strip('"\'')

    # Run the main function with the provided paths
    create_detailed_matrices(input_dir, output_dir)