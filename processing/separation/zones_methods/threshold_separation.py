import cv2
import numpy as np
import os
import json

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

def find_annulus_and_inner_circle(filtered_image):
    """Find black annulus pixels and white pixels inside the annulus"""
    # Find contours
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create masks for black pixels (annulus) and inner white pixels
    black_mask = np.zeros_like(filtered_image)
    inner_white_mask = np.zeros_like(filtered_image)
    
    # Black pixels are where filtered_image == 0
    black_pixels = (filtered_image == 0)
    
    # Find connected components of white pixels
    num_labels, labels = cv2.connectedComponents(filtered_image)
    
    # For each black region (potential annulus)
    black_contours, _ = cv2.findContours((~filtered_image).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Variables to store circle information
    inner_circle_info = None
    outer_circle_info = None
    annulus_contour = None
    
    for contour in black_contours:
        # Create a mask for this black region
        temp_mask = np.zeros_like(filtered_image)
        cv2.drawContours(temp_mask, [contour], -1, 255, -1)
        
        # Find the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if there are white pixels inside this black region
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
        full_mask = np.zeros_like(filtered_image)
        cv2.drawContours(full_mask, [annulus_contour], -1, 255, -1)
        
        # Find the outer boundary points
        outer_points = np.column_stack(np.where(full_mask > 0))
        if len(outer_points) > 0:
            # Convert to the format OpenCV expects (x, y)
            outer_points_cv = outer_points[:, [1, 0]].astype(np.float32)
            (outer_x, outer_y), outer_radius = cv2.minEnclosingCircle(outer_points_cv)
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
    
    concentricity_info = {
        'center_offset': center_distance,
        'normalized_offset': center_distance / outer_circle_info['radius'] if outer_circle_info['radius'] > 0 else 0,
        'inner_center': inner_center,
        'outer_center': outer_center
    }
    
    return concentricity_info

def segment_with_threshold(image_path, output_dir='output_threshold'):
    """
    Main function modified for unified system
    Returns standardized results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result dictionary
    result = {
        'method': 'threshold_seperation',
        'image_path': image_path,
        'success': False,
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0
    }
    
    # Load image
    if not os.path.exists(image_path):
        result['error'] = f"File not found: '{image_path}'"
        with open(os.path.join(output_dir, 'threshold_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
        
    original_image = cv2.imread(image_path)
    if original_image is None:
        result['error'] = f"Could not read image from '{image_path}'"
        with open(os.path.join(output_dir, 'threshold_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Apply the filter
        filtered_image = apply_filter(original_image)
        
        # Find black annulus and inner white pixels
        black_mask, inner_white_mask, inner_circle_info, outer_circle_info = find_annulus_and_inner_circle(filtered_image)
        
        # Calculate concentricity
        concentricity_info = calculate_concentricity(inner_circle_info, outer_circle_info)
        
        if inner_circle_info and outer_circle_info:
            # Use outer circle center as main center (usually more stable)
            result['success'] = True
            result['center'] = (int(outer_circle_info['center'][0]), int(outer_circle_info['center'][1]))
            result['core_radius'] = int(inner_circle_info['radius'])
            result['cladding_radius'] = int(outer_circle_info['radius'])
            result['confidence'] = 0.6  # Moderate confidence for threshold method
            
            # Adjust confidence based on concentricity
            if concentricity_info and concentricity_info['normalized_offset'] < 0.1:
                result['confidence'] = 0.7
        else:
            # Fallback: try to find any circular structure
            contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest)
                result['center'] = (int(x), int(y))
                result['cladding_radius'] = int(radius)
                result['core_radius'] = int(radius * 0.3)  # Estimate
                result['confidence'] = 0.4
                result['success'] = True
            else:
                result['error'] = "Could not detect fiber structure"
                with open(os.path.join(output_dir, f'{base_filename}_threshold_result.json'), 'w') as f:
                    json.dump(result, f, indent=4)
                return result
        
        # Save result data
        with open(os.path.join(output_dir, f'{base_filename}_threshold_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
        # Segment the original image
        cleaned_image, white_region_image, black_region_image, outside_region_image = segment_original_image(
            original_image, black_mask, inner_white_mask
        )
        
        # Create visualization image
        visualization_image = create_visualization_image(
            original_image, black_mask, inner_white_mask,
            inner_circle_info, outer_circle_info, concentricity_info
        )
        
        # Save the results with standardized names
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_threshold_filtered.png"), filtered_image)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_threshold_core.png"), white_region_image)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_threshold_cladding.png"), black_region_image)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_threshold_ferrule.png"), outside_region_image)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_threshold_annotated.png"), visualization_image)
        
    except Exception as e:
        result['error'] = str(e)
        with open(os.path.join(output_dir, f'{base_filename}_threshold_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
    return result

def segment_original_image(original, black_mask, inner_white_mask):
    """Segment the original image based on the masks"""
    # Ensure masks are boolean
    black_mask = black_mask.astype(bool)
    inner_white_mask = inner_white_mask.astype(bool)
    
    # Create the combined mask (black annulus + inner white)
    combined_mask = black_mask | inner_white_mask
    
    # Remove everything outside the annulus
    cleaned_image = original.copy()
    cleaned_image[~combined_mask] = 0
    
    # Separate into images
    white_region_image = original.copy()
    white_region_image[~inner_white_mask] = 0
    
    black_region_image = original.copy()
    black_region_image[~black_mask] = 0
    
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
    """Main function for standalone testing"""
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"').strip("'")
    
    result = segment_with_threshold(image_path)
    if result['success']:
        print(f"Success! Center: {result['center']}, Core: {result['core_radius']}, Cladding: {result['cladding_radius']}")
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()