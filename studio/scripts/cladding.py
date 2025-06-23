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

def find_black_annulus_mask(filtered_image):
    """Find black annulus pixels"""
    # Create mask for black pixels (annulus)
    black_mask = np.zeros_like(filtered_image)
    
    # Black pixels are where filtered_image == 0
    black_pixels = (filtered_image == 0)
    
    # Find black contours
    black_contours, _ = cv2.findContours((~filtered_image).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in black_contours:
        # Create a mask for this black region
        temp_mask = np.zeros_like(filtered_image)
        cv2.drawContours(temp_mask, [contour], -1, 255, -1)
        
        # Create a filled version of the contour
        filled_mask = np.zeros_like(filtered_image)
        cv2.drawContours(filled_mask, [contour], -1, 255, -1)
        
        # Check if there are white pixels inside this black region
        inner_white = (filtered_image == 255) & (filled_mask == 255)
        
        # If we found inner white pixels, this is an annulus
        if np.any(inner_white):
            # Add to black mask (the annulus itself)
            black_mask = black_mask | (temp_mask & black_pixels)
    
    return black_mask

def extract_black_region(original, black_mask):
    """Extract only the black region (cladding) from the original image"""
    # Ensure mask is boolean
    black_mask = black_mask.astype(bool)
    
    # Create output image - keep only black region pixels
    black_region_image = original.copy()
    black_region_image[~black_mask] = 0
    
    return black_region_image

def main():
    # Load the image
    image_path = r"C:\Users\Saem1001\Desktop\All Photos\img (87).jpg"
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Apply the filter
    filtered_image = apply_filter(original_image)
    
    # Find black annulus pixels (cladding)
    black_mask = find_black_annulus_mask(filtered_image)
    
    # Extract black region
    black_region_image = extract_black_region(original_image, black_mask)
    
    # Count pixels
    black_pixel_count = np.sum(black_mask > 0)
    print(f"Number of black pixels (cladding): {black_pixel_count}")
    
    # Save the result
    cv2.imwrite("cladding.png", black_region_image)
    print("\nBlack region (cladding) saved to: cladding.png")
    
    return black_region_image

if __name__ == "__main__":
    main()