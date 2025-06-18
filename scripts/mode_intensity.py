"""
Isolate the mode (most frequent) pixel intensity value in an image.
This script calculates the most common pixel intensity and creates a binary
mask, setting all other pixel values to zero.
"""
import cv2
import numpy as np

def process_image(image: np.ndarray,
                  preserve_original_color: bool = False,
                  highlight_color: str = "white") -> np.ndarray:
    """
    Filters an image to show only the pixels with the mode intensity value.

    This function calculates the most frequent pixel value in the grayscale
    representation of the image. It then generates an output image where only
    the pixels corresponding to this mode intensity are kept.

    Args:
        image: Input image (color or grayscale).
        preserve_original_color: If True and input is color, the mode pixels 
                                 will retain their original color.
        highlight_color: The color to use for the mode pixels when not 
                         preserving original color. Supports: "white", 
                         "red", "green", "blue", "yellow", "cyan", "magenta".

    Returns:
        An image where only the mode intensity pixels are non-zero.
    """
    # 1. Handle Color/Grayscale Input
    if len(image.shape) == 3:
        # Input is a color image (BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        # Input is already grayscale
        gray = image
        is_color = False

    # 2. Calculate the Mode Intensity
    # cv2.calcHist expects a list of images
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # Find the peak of the histogram (the mode)
    mode_intensity = np.argmax(hist)

    # 3. Create a mask for the mode intensity
    # cv2.inRange creates a binary mask where pixels within the range are white
    mask = cv2.inRange(gray, int(mode_intensity), int(mode_intensity))

    # 4. Generate the output image
    if is_color:
        # Ensure the final output is a 3-channel BGR image
        if preserve_original_color:
            # Use the mask to extract pixels from the original color image
            result = cv2.bitwise_and(image, image, mask=mask)
        else:
            # Create a 3-channel version of the mask to apply color
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Map color names to BGR tuples
            color_map = {
                "white": (255, 255, 255),
                "red": (0, 0, 255),
                "green": (0, 255, 0),
                "blue": (255, 0, 0),
                "yellow": (0, 255, 255),
                "cyan": (255, 255, 0),
                "magenta": (255, 0, 255)
            }
            # Default to white if the color name is not found
            color_value = color_map.get(highlight_color.lower(), (255, 255, 255))
            
            # Set all pixels in the masked region to the chosen color
            result[mask == 255] = color_value
    else:
        # If the original was grayscale, the output is the binary mask
        result = mask
        
    return result

# Optional: Test block for standalone execution
if __name__ == '__main__':
    print("Testing Mode Intensity Filter script...")

    # Create a synthetic test image (150x150)
    # Fill with value 100, but add a prominent block of value 200 (the mode)
    test_image = np.full((150, 150), 100, dtype=np.uint8)
    test_image[50:100, 50:100] = 200 # This is the mode

    # Add some random noise
    noise = np.random.randint(0, 256, (150, 150), dtype=np.uint8)
    # Create a mask to apply noise only to some areas
    noise_mask = np.random.choice([0, 1], test_image.shape, p=[0.9, 0.1]).astype(np.uint8)
    test_image[noise_mask == 1] = noise[noise_mask == 1]
    
    # Run the processing function
    # The mode should be 200
    filtered_result = process_image(test_image.copy())
    
    print("Test finished. The output window should show only the center square.")
    
    # To visually verify, we can display the images
    # Note: cv2.imshow is for testing only and should not be in the final script
    # for the GUI, as per the guidelines.
    cv2.imshow("Original Test Image", test_image)
    cv2.imshow("Mode Filtered Result (should be 200)", filtered_result)
    
    print("Press any key to close test windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
