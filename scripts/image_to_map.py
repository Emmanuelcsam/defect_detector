import cv2
import matplotlib.pyplot as plt
import numpy as np

def create_intensity_map(image_path, output_filename='intensity_map.png'):
    """
    Loads an image, converts it to a 2D intensity map, and saves it.

    Args:
        image_path (str): The path to the input image file.
        output_filename (str): The filename for the output plot image.
    """
    # Load the image in grayscale
    # The '0' flag ensures the image is loaded as a single-channel grayscale image.
    img = cv2.imread(image_path, 0)

    if img is None:
        print(f"Error: Could not read the image file at {image_path}")
        return

    # Get the dimensions of the image
    height, width = img.shape

    # Create a grid of x and y coordinates
    x = np.arange(0, width)
    y = np.arange(0, height)
    X, Y = np.meshgrid(x, y)

    # Flatten the coordinate arrays and the image intensity data
    # We also flip the Y coordinates because image 'y' coordinates typically
    # start from the top, while matplotlib's y-axis starts from the bottom.
    x_coords = X.flatten()
    y_coords = (height - 1) - Y.flatten() # Flip Y for correct orientation in plot
    intensity = img.flatten()

    # --- Plotting ---
    print("Generating plot...")
    plt.figure(figsize=(10, 8))

    # Create a scatter plot
    # 's=1' makes the points small to resemble a continuous image
    # 'cmap' is the color map. 'viridis', 'inferno', or 'gray' are good choices.
    scatter = plt.scatter(x_coords, y_coords, c=intensity, cmap='viridis', s=1)

    # Add a color bar to serve as a legend for the intensity values
    cbar = plt.colorbar(scatter)
    cbar.set_label('Intensity')

    # Set the labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Intensity Map from Image')

    # Ensure the aspect ratio of the plot is equal to the image
    plt.axis('equal')
    plt.xlim([0, width])
    plt.ylim([0, height])


    # Save the plot to a file
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as {output_filename}")

    # To display the plot directly, uncomment the following line:
    # plt.show()


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Replace 'your_image.png' with the actual path to your image file.
    # The script currently expects the image to be in the same directory.
    image_file = r"C:\Users\Saem1001\Documents\GitHub\IPPS\review\output_veridian\img63_cladding.png"
    create_intensity_map(image_file)