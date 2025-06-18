# All necessary imports must be at the top of the file.
import cv2
import numpy as np
import argparse # Needed for standalone execution

# =================================================================================
#  1. GUI-COMPATIBLE FUNCTION
# =================================================================================

def process_image(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.
    If the input image is color, it is converted to LAB color space, CLAHE is
    applied to the L-channel, and then it's converted back to BGR.

    Args:
        image (np.ndarray): The input image from the UI's pipeline.
        clip_limit (float): Threshold for contrast limiting.
        grid_size (int): Size of the grid for histogram equalization (e.g., 8 for an 8x8 grid).

    Returns:
        np.ndarray: The contrast-enhanced image.
    """
    # --- Your image processing logic starts here ---

    processed_image = image.copy()

    # Check if the image is color or grayscale
    if len(processed_image.shape) == 2:
        # It's a grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        processed_image = clahe.apply(processed_image)
    else:
        # It's a color image. Convert to LAB color space to apply CLAHE to the L-channel.
        lab_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l_enhanced = clahe.apply(l)
        
        # Merge the enhanced L-channel back and convert back to BGR
        enhanced_lab_image = cv2.merge((l_enhanced, a, b))
        processed_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    # --- Your image processing logic ends here ---

    return processed_image

# =================================================================================
#  2. STANDALONE EXECUTION
# =================================================================================

if __name__ == '__main__':
    # --- Setup to read arguments from the command line ---
    parser = argparse.ArgumentParser(description='Apply CLAHE to an image.')
    
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to save the processed image file.')
    
    # Add arguments that match the 'process_image' function
    parser.add_argument('--clip_limit', type=float, default=2.0, help='Contrast limiting threshold.')
    parser.add_argument('--grid_size', type=int, default=8, help='Grid size for histogram equalization.')
    
    args = parser.parse_args()

    # --- Script's main logic when run directly ---
    try:
        input_image = cv2.imread(args.input)
        if input_image is None:
            raise FileNotFoundError(f"Error: Could not open or find the image at '{args.input}'")

        print(f"Applying CLAHE with clip_limit={args.clip_limit} and grid_size={args.grid_size}...")
        output_image = process_image(input_image, clip_limit=args.clip_limit, grid_size=args.grid_size)

        cv2.imwrite(args.output, output_image)
        print(f"Successfully processed image and saved it to '{args.output}'")

        # Optional: Display the images
        
        
        print("Press any key to close the image windows...")
        
        

    except Exception as e:
        print(f"An error occurred: {e}")