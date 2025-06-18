import cv2
import numpy as np

def apply_filter(image):
    result = image.copy()
    
    # Convert to grayscale if needed
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result

def find_inner_white_mask(filtered_image):
    """Find white pixels inside the annulus"""
    # Create mask for inner white pixels
    inner_white_mask = np.zeros_like(filtered_image)
    
    # Find connected components of white pixels
    num_labels, labels = cv2.connectedComponents(filtered_image)
    
    # Find black contours (potential annulus)
    black_contours, _ = cv2.findContours((~filtered_image).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in black_contours:
        # Create a filled version of the contour
        filled_mask = np.zeros_like(filtered_image)
        cv2.drawContours(filled_mask, [contour], -1, 255, -1)
        
        # The inner white pixels are those that are white in the filtered image
        # and inside the filled contour
        inner_white = (filtered_image == 255) & (filled_mask == 255)
        
        # If we found inner white pixels, add to mask
        if np.any(inner_white):
            inner_white_mask = inner_white_mask | inner_white
    
    return inner_white_mask

def extract_white_region(original, inner_white_mask):
    """Extract only the white region (core) from the original image"""
    # Ensure mask is boolean
    inner_white_mask = inner_white_mask.astype(bool)
    
    # Create output image - keep only white region pixels
    white_region_image = original.copy()
    white_region_image[~inner_white_mask] = 0
    
    return white_region_image

def main():
    # Load the image
    image_path = r"C:\Users\Saem1001\Desktop\All Photos\img (87).jpg"
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Apply the filter
    filtered_image = apply_filter(original_image)
    
    # Find inner white pixels (core)
    inner_white_mask = find_inner_white_mask(filtered_image)
    
    # Extract white region
    white_region_image = extract_white_region(original_image, inner_white_mask)
    
    # Count pixels
    white_pixel_count = np.sum(inner_white_mask > 0)
    print(f"Number of white pixels (core): {white_pixel_count}")
    
    # Save the result
    cv2.imwrite("core.png", white_region_image)
    print("\nWhite region (core) saved to: core.png")
    
    return white_region_image

if __name__ == "__main__":
    main()