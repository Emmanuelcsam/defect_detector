import cv2
import csv
import os

def image_to_detailed_csv():
    """
    Asks for the path of an image, converts it to grayscale, and creates a
    detailed CSV file containing information about every subpixel.
    """
    image_path = input("Please enter the full path for the image: ")

    if not os.path.exists(image_path):
        print("Error: The specified file does not exist.")
        return

    try:
        # Read the image in its original format
        original_image = cv2.imread(image_path)
        if original_image is None:
            print("Error: Could not read the image. Please check the file format and integrity.")
            return

        # --- Image Dimensions ---
        height, width, channels = original_image.shape
        total_pixels = height * width

        # --- Grayscale Conversion ---
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # --- CSV File Creation ---
        output_csv_path = os.path.splitext(image_path)[0] + '_detailed.csv'

        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # --- Header Information ---
            csv_writer.writerow(['Image Details'])
            csv_writer.writerow(['Source Image Path', image_path])
            csv_writer.writerow(['Dimensions (Width x Height)', f'{width} x {height}'])
            csv_writer.writerow(['Total Number of Pixels', total_pixels])
            csv_writer.writerow([]) # Add a blank row for spacing
            csv_writer.writerow(['Pixel Data (Grayscale)'])
            csv_writer.writerow([
                'X Coordinate',
                'Y Coordinate',
                'Grayscale Value (0-255)',
                'Binary Representation'
            ])

            # --- Pixel Data Iteration ---
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

        print(f"\nSuccessfully created the detailed CSV file at: {output_csv_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    image_to_detailed_csv()