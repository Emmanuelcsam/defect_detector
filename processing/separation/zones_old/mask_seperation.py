import cv2
import numpy as np

def apply_filter(image):
    result = image.copy()
    
    # Convert to grayscale if needed
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)  # pixel > 127: set to 255 (white)
    
    # Apply morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # applies mask better to circular shapes
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)  # Fills holes in white regions and smooths boundaries
    
    return result

def find_annulus_and_inner_circle(filtered_image):
    """Find black annulus pixels and white pixels inside the annulus"""
    # Find contours
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the filtered image
    
    # Create masks for black pixels (annulus) and inner white pixels
    black_mask = np.zeros_like(filtered_image)  # Creates empty black mask same size as input
    inner_white_mask = np.zeros_like(filtered_image)
    
    # Black pixels are where filtered_image == 0
    black_pixels = (filtered_image == 0)  # True where pixels are black (value 0), False where pixels are white (value 255)
    
    # Find connected components of white pixels
    num_labels, labels = cv2.connectedComponents(filtered_image)  # Find connected components in the filtered image
    
    # For each black region (potential annulus)
    black_contours, _ = cv2.findContours((~filtered_image).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Inverts image (white→black, black→white), Only retrieves outermost contours, Converts boolean to 8-bit image
    
    # Variables to store circle information
    inner_circle_info = None
    outer_circle_info = None
    annulus_contour = None
    
    for contour in black_contours:
        # Create a mask for this black region
        temp_mask = np.zeros_like(filtered_image)  # Creates temporary mask for current contour
        cv2.drawContours(temp_mask, [contour], -1, 255, -1)  # Draws white filled contour on mask
        
        # Find the bounding box
        x, y, w, h = cv2.boundingRect(contour)  # Finds bounding rectangle of contour
        
        # Check if there are white pixels inside this black region
        # Create a filled version of the contour
        filled_mask = np.zeros_like(filtered_image)
        cv2.drawContours(filled_mask, [contour], -1, 255, -1)
        
        # The inner white pixels are those that are white in the filtered image
        # and inside the filled contour
        inner_white = (filtered_image == 255) & (filled_mask == 255)
        
        # If we found inner white pixels, this is an annulus
        if np.any(inner_white):
            # Add to black mask (the annulus itself)
            black_mask = black_mask | (temp_mask & black_pixels)
            # Add to inner white mask
            inner_white_mask = inner_white_mask | inner_white
            
            # Store the annulus contour for circle calculations
            if annulus_contour is None or cv2.contourArea(contour) > cv2.contourArea(annulus_contour):
                annulus_contour = contour
    
    # Calculate circle parameters if we found an annulus
    if annulus_contour is not None:
        # For outer circle: use the external boundary of the annulus
        # Create a mask that includes both the annulus and its interior
        full_mask = np.zeros_like(filtered_image)
        cv2.drawContours(full_mask, [annulus_contour], -1, 255, -1)
        
        # Find the outer boundary points
        outer_points = np.column_stack(np.where(full_mask > 0))  # Finds coordinates of non-zero pixels in the mask
        if len(outer_points) > 0:
            # Convert to the format OpenCV expects (x, y)
            outer_points_cv = outer_points[:, [1, 0]].astype(np.float32)  # Swaps columns: (y,x) → (x,y) for OpenCV, Converts to float32 for circle fitting
            (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(outer_points_cv)  # Finds minimum enclosing circle, Returns center (x,y) and radius
            outer_circle_info = {
                'center': (outer_x, outer_y),
                'radius': outer_radius,
                'diameter': outer_radius * 2
            }
        
        # For inner circle: use only the inner white pixels
        inner_points = np.column_stack(np.where(inner_white_mask > 0))
        if len(inner_points) > 0:
            # Convert to the format OpenCV expects (x, y)
            inner_points_cv = inner_points[:, [1, 0]].astype(np.float32)
            (inner_x, inner_y), inner_radius = cv2.minEnclosingCircle(inner_points_cv)
            inner_circle_info = {
                'center': (inner_x, inner_y),
                'radius': inner_radius,
                'diameter': inner_radius * 2
            }
    
    return black_mask, inner_white_mask, inner_circle_info, outer_circle_info

def calculate_concentricity(inner_circle_info, outer_circle_info):
    """Calculate the concentricity between inner and outer circles"""
    if inner_circle_info is None or outer_circle_info is None:
        return None
    
    # Calculate the distance between centers
    inner_center = inner_circle_info['center']
    outer_center = outer_circle_info['center']
    
    center_distance = np.sqrt((inner_center[0] - outer_center[0])**2 + 
                             (inner_center[1] - outer_center[1])**2)
    # Euclidean distance formula, Calculates pixel distance between centers, sqrt((x2-x1)² + (y2-y1)²)
    
    # Concentricity can be expressed as:
    # - Absolute distance between centers
    # - Relative to the outer radius (normalized)
    concentricity_info = {
        'center_offset': center_distance,
        'normalized_offset': center_distance / outer_circle_info['radius'] if outer_circle_info['radius'] > 0 else 0,
        'inner_center': inner_center,
        'outer_center': outer_center
    }
    
    return concentricity_info

def segment_original_image(original, black_mask, inner_white_mask):
    """Segment the original image based on the masks"""
    # Ensure masks are boolean
    black_mask = black_mask.astype(bool)
    inner_white_mask = inner_white_mask.astype(bool)
    
    # Create the combined mask (black annulus + inner white)
    combined_mask = black_mask | inner_white_mask
    
    # Remove everything outside the annulus (keep only what's inside the combined mask)
    cleaned_image = original.copy()
    cleaned_image[~combined_mask] = 0
    
    # Separate into images
    # Image 1: Where white pixels were (inside the annulus)
    white_region_image = original.copy()
    white_region_image[~inner_white_mask] = 0
    
    # Image 2: Where black pixels were (the annulus)
    black_region_image = original.copy()
    black_region_image[~black_mask] = 0
    
    # Image 3: Everything outside the annulus
    outside_region_image = original.copy()
    outside_region_image[combined_mask] = 0
    
    return cleaned_image, white_region_image, black_region_image, outside_region_image

def create_visualization_image(original, black_mask, inner_white_mask, 
                             inner_circle_info, outer_circle_info, concentricity_info):
    """Create a visualization image with circles and measurements"""
    # Create overlay visualization with circles
    overlay = original.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay_vis = overlay.copy()
    
    # Color the masks
    overlay_vis[black_mask] = [0, 255, 0]  # Green for annulus
    overlay_vis[inner_white_mask] = [255, 0, 0]  # Red for inner white
    
    # Draw the fitted circles
    if inner_circle_info:
        center = (int(inner_circle_info['center'][0]), int(inner_circle_info['center'][1]))
        radius = int(inner_circle_info['radius'])
        cv2.circle(overlay_vis, center, radius, [255, 255, 0], 2)  # Yellow for inner circle
        cv2.circle(overlay_vis, center, 3, [255, 255, 0], -1)  # Center point
    
    if outer_circle_info:
        center = (int(outer_circle_info['center'][0]), int(outer_circle_info['center'][1]))
        radius = int(outer_circle_info['radius'])
        cv2.circle(overlay_vis, center, radius, [0, 255, 255], 2)  # Cyan for outer circle
        cv2.circle(overlay_vis, center, 3, [0, 255, 255], -1)  # Center point
    
    # If both circles exist, draw a line between centers
    if inner_circle_info and outer_circle_info and concentricity_info:
        inner_center = (int(inner_circle_info['center'][0]), int(inner_circle_info['center'][1]))
        outer_center = (int(outer_circle_info['center'][0]), int(outer_circle_info['center'][1]))
        cv2.line(overlay_vis, inner_center, outer_center, [255, 0, 255], 2)  # Magenta line
        
        # Add text with concentricity measurement
        text = f"Offset: {concentricity_info['center_offset']:.2f}px"
        cv2.putText(overlay_vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, [255, 255, 255], 2, cv2.LINE_AA)
    
    return overlay_vis

def main():
    # Load the image
    image_path = r"C:\Users\Saem1001\Desktop\All Photos\img (219).jpg"
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Apply the filter
    filtered_image = apply_filter(original_image)
    
    # Find black annulus and inner white pixels
    black_mask, inner_white_mask, inner_circle_info, outer_circle_info = find_annulus_and_inner_circle(filtered_image)
    
    # Calculate concentricity
    concentricity_info = calculate_concentricity(inner_circle_info, outer_circle_info)
    
    # Save the positions (as coordinates)
    black_positions = np.column_stack(np.where(black_mask))
    white_positions = np.column_stack(np.where(inner_white_mask))
    outside_positions = np.column_stack(np.where(~(black_mask | inner_white_mask)))
    
    print(f"Number of black pixels (annulus): {len(black_positions)}")
    print(f"Number of white pixels inside annulus: {len(white_positions)}")
    print(f"Number of pixels outside annulus: {len(outside_positions)}")
    
    # Print circle measurements
    print("\n=== CIRCLE MEASUREMENTS ===")
    if inner_circle_info:
        print(f"Inner Circle:")
        print(f"  - Center: ({inner_circle_info['center'][0]:.2f}, {inner_circle_info['center'][1]:.2f})")
        print(f"  - Radius: {inner_circle_info['radius']:.2f} pixels")
        print(f"  - Diameter: {inner_circle_info['diameter']:.2f} pixels")
    else:
        print("Inner Circle: Not detected")
    
    if outer_circle_info:
        print(f"\nOuter Circle:")
        print(f"  - Center: ({outer_circle_info['center'][0]:.2f}, {outer_circle_info['center'][1]:.2f})")
        print(f"  - Radius: {outer_circle_info['radius']:.2f} pixels")
        print(f"  - Diameter: {outer_circle_info['diameter']:.2f} pixels")
    else:
        print("Outer Circle: Not detected")
    
    if concentricity_info:
        print(f"\nConcentricity:")
        print(f"  - Center offset: {concentricity_info['center_offset']:.2f} pixels")
        print(f"  - Normalized offset: {concentricity_info['normalized_offset']:.4f}")
    
    # Segment the original image
    cleaned_image, white_region_image, black_region_image, outside_region_image = segment_original_image(
        original_image, black_mask, inner_white_mask
    )
    
    # Create visualization image
    visualization_image = create_visualization_image(
        original_image, black_mask, inner_white_mask,
        inner_circle_info, outer_circle_info, concentricity_info
    )
    
    # Save the results
    cv2.imwrite("filtered_image.png", filtered_image)
    cv2.imwrite("black_mask.png", black_mask.astype(np.uint8) * 255)
    cv2.imwrite("inner_white_mask.png", inner_white_mask.astype(np.uint8) * 255)
    cv2.imwrite("outside_mask.png", (~(black_mask | inner_white_mask)).astype(np.uint8) * 255)
    cv2.imwrite("cleaned_image.png", cleaned_image)
    cv2.imwrite("white_region_original.png", white_region_image)
    cv2.imwrite("black_region_original.png", black_region_image)
    cv2.imwrite("outside_region_original.png", outside_region_image)
    cv2.imwrite("visualization_overlay.png", visualization_image)
    
    print("\nResults saved to files:")
    print("- filtered_image.png")
    print("- black_mask.png")
    print("- inner_white_mask.png")
    print("- outside_mask.png") 
    print("- cleaned_image.png")
    print("- white_region_original.png")
    print("- black_region_original.png")
    print("- outside_region_original.png")
    print("- visualization_overlay.png")
    
    return {
        'original': original_image,
        'filtered': filtered_image,
        'black_mask': black_mask,
        'inner_white_mask': inner_white_mask,
        'black_positions': black_positions,
        'white_positions': white_positions,
        'outside_positions': outside_positions,
        'cleaned': cleaned_image,
        'white_region': white_region_image,
        'black_region': black_region_image,
        'outside_region': outside_region_image,
        'inner_circle': inner_circle_info,
        'outer_circle': outer_circle_info,
        'concentricity': concentricity_info
    }

if __name__ == "__main__":
    results = main()