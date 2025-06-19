import cv2
import csv
import os
import sys

def create_csv_for_image(image_path, output_csv_path):
    """
    Reads a single image, converts it to grayscale, and creates a detailed
    CSV file containing information about every pixel.

    Args:
        image_path (str): The full path to the input image file.
        output_csv_path (str): The full path where the output CSV file will be saved.
    """
    try:
        # Read the image using OpenCV
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not read image file: {image_path}. Skipping.")
            return

        # --- Image Dimensions ---
        height, width, channels = original_image.shape
        total_pixels = height * width

        # --- Grayscale Conversion ---
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # --- CSV File Creation ---
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # --- Header Information ---
            csv_writer.writerow(['Image Details'])
            csv_writer.writerow(['Source Image Path', image_path])
            csv_writer.writerow(['Dimensions (Width x Height)', f'{width} x {height}'])
            csv_writer.writerow(['Total Number of Pixels', total_pixels])
            csv_writer.writerow([])  # Add a blank row for spacing
            csv_writer.writerow(['Pixel Data (Grayscale)'])
            csv_writer.writerow([
                'X Coordinate',
                'Y Coordinate',
                'Grayscale Value (0-255)',
                'Binary Representation'
            ])

            # --- Pixel Data Iteration ---
            # Iterate through each pixel and write its data to the CSV
            for y in range(height):
                for x in range(width):
                    grayscale_value = grayscale_image[y, x]
                    binary_representation = bin(grayscale_value)
                    csv_writer.writerow([
                        x,
                        y,
                        grayscale_value,
                        binary_representation
                    ])
        
        print(f"Successfully created CSV for: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"An unexpected error occurred while processing {image_path}: {e}")

def process_directory_to_csvs():
    """
    Asks for a directory of images, then processes each image file into a
    separate, detailed CSV file in a specified output directory.
    """
    # --- Get Input Directory ---
    input_dir = input("Please enter the full path for the directory containing images: ")

    if not os.path.isdir(input_dir):
        print("Error: The specified directory does not exist.")
        return

    # --- Get Output Directory ---
    output_dir = input("Please enter the path for the output CSV directory: ")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("-" * 30)
    print(f"Searching for images in: {input_dir}")
    print(f"Saving CSV files to:    {output_dir}")
    print("-" * 30)

    # --- Supported Image Formats ---
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # --- Iterate and Process Files ---
    found_images = False
    for filename in os.listdir(input_dir):
        # Check if the file has a supported image extension
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            found_images = True
            image_path = os.path.join(input_dir, filename)
            
            # Define the name for the output CSV file
            csv_filename = os.path.splitext(filename)[0] + '_detailed.csv'
            output_csv_path = os.path.join(output_dir, csv_filename)
            
            # Process the individual image
            create_csv_for_image(image_path, output_csv_path)

    if not found_images:
        print("\nNo image files found in the specified directory.")
    else:
        print("\nBatch processing complete.")
        print(f"All CSV files have been saved in: {output_dir}")


if __name__ == '__main__':
    process_directory_to_csvs()
