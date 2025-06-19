import numpy as np
import cv2 as cv
import os
import json

def preprocess_image(img):
    """
    This function takes in the raw greyscale image,
    runs a contrast enhancement (CLAHE),
    and runs a canny filter to make the circles easier to see.
    """
    assert img is not None, "image could not be read"
   
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
    clahe_image = clahe.apply(img)
   
    canny_image = cv.Canny(clahe_image, 100, 200, apertureSize= 3)
 
    # apply a gaussian blur to improve yield of circle recognition
    img_blurred = cv.GaussianBlur(canny_image,(5, 5), 0)
 
    processed_dict = {
        "blurred_canny": img_blurred,
        "original_img": img          
    }
    return processed_dict

def circle_extract(image: np.ndarray, x0: int, y0: int, radius: int) -> np.ndarray:
    """Takes in a grayscale image"""
    arr = image.copy()
    rows, cols = arr.shape
 
    for i in range(rows):
        for j in range(cols):
            # test against the equation for a circle
            if np.sqrt(np.square(i - y0) + np.square(j - x0)) > (radius):
                arr[i][j] = 0
            else:
                pass
    output_image = arr
    return output_image
 
def core_mask(image: np.ndarray, x0: int, y0: int, radius: int) -> np.ndarray:
    """Takes in a grayscale image"""
    arr = image.copy()
    rows, cols = arr.shape
 
    for i in range(rows):
        for j in range(cols):
            # test against the equation for a circle
            if np.sqrt(np.square(i - y0) + np.square(j - x0)) <= (radius):
                arr[i][j] = 0
            else:
                pass
    output_image = arr
    return output_image

def segment_with_hough(image_path, output_dir='output_hough'):
    """
    Main function modified for unified system
    Returns standardized results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result dictionary
    result = {
        'method': 'hough_seperation',
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
        with open(os.path.join(output_dir, 'hough_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
        
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        result['error'] = f"Could not read image from '{image_path}'"
        with open(os.path.join(output_dir, 'hough_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Preprocess
        preproc = preprocess_image(img)
        proc_img = preproc["blurred_canny"]
        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
       
        # Detect Cladding
        cladding_circle = cv.HoughCircles(proc_img, cv.HOUGH_GRADIENT, dp=1, minDist=10,
                                 param1=50, param2=40, minRadius=100, maxRadius=500)
       
        if cladding_circle is None:
            result['error'] = "Could not detect cladding circle"
            with open(os.path.join(output_dir, f'{base_filename}_hough_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result
            
        cladding_x0 = int(cladding_circle[0][0][0])
        cladding_y0 = int(cladding_circle[0][0][1])
        cladding_radius = int(cladding_circle[0][0][2])
        
        # Extract cladding region
        cladding = circle_extract(img, cladding_x0, cladding_y0, cladding_radius)
        cladding_copy = cladding.copy()
       
        # Enhance cladding for core detection
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
        clahe_cladding = clahe.apply(cladding)
        canny_clad = cv.Canny(clahe_cladding, 100, 200)
        canny_clad_blurred = cv.GaussianBlur(canny_clad,(7, 7), 0)
     
        # Detect Core
        core_circle = cv.HoughCircles(canny_clad_blurred, cv.HOUGH_GRADIENT, dp=1, minDist=10,
                              param1=50, param2=40, minRadius=20, maxRadius=80)
       
        if core_circle is None:
            # Try with enhanced parameters
            clahe_adjusted_cladding = clahe.apply(cladding)
            new_cladding_canny = cv.Canny(clahe_adjusted_cladding, 100, 200)
            new_canny_clad_blurred = cv.GaussianBlur(new_cladding_canny,(7, 7), 0)
            
            core_circle = cv.HoughCircles(new_canny_clad_blurred, cv.HOUGH_GRADIENT, dp=1, minDist=10,
                        param1=50, param2=40, minRadius=20, maxRadius=60)
     
        if core_circle is None:
            # Use estimate based on cladding
            core_x0 = cladding_x0
            core_y0 = cladding_y0
            core_radius = int(cladding_radius * 0.3)  # Typical ratio
            confidence = 0.5
        else:
            core_x0 = int(core_circle[0][0][0])
            core_y0 = int(core_circle[0][0][1])
            core_radius = int(core_circle[0][0][2])
            confidence = 0.7
        
        # Set results
        result['success'] = True
        result['center'] = (cladding_x0, cladding_y0)  # Use cladding center as main center
        result['core_radius'] = core_radius
        result['cladding_radius'] = cladding_radius
        result['confidence'] = confidence
        
        # Save result data
        with open(os.path.join(output_dir, f'{base_filename}_hough_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
        # Create output images
        core = circle_extract(cladding, core_x0, core_y0, core_radius)
        cladding_only = core_mask(cladding_copy, core_x0, core_y0, core_radius)
        
        # Draw circles on original image
        cv.circle(cimg, (core_x0, core_y0), core_radius, (0, 255, 0), 2)
        cv.circle(cimg, (core_x0, core_y0), 2, (0, 0, 255), 3)
        cv.circle(cimg, (cladding_x0, cladding_y0), cladding_radius, (255, 255, 0), 2)
        cv.circle(cimg, (cladding_x0, cladding_y0), 2, (0, 0, 255), 3)
        
        # Save images with standardized names
        cv.imwrite(os.path.join(output_dir, f'{base_filename}_hough_annotated.png'), cimg)
        cv.imwrite(os.path.join(output_dir, f'{base_filename}_hough_core.png'), core)
        cv.imwrite(os.path.join(output_dir, f'{base_filename}_hough_cladding.png'), cladding_only)
        
    except Exception as e:
        result['error'] = str(e)
        with open(os.path.join(output_dir, f'{base_filename}_hough_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
    return result

def run_analysis(image_path, output_dir='output_hough'):
    """Wrapper function for compatibility"""
    return segment_with_hough(image_path, output_dir)

if __name__ == "__main__":
    # For standalone testing
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"').strip("'")
    
    result = segment_with_hough(image_path)
    if result['success']:
        print(f"Success! Center: {result['center']}, Core: {result['core_radius']}, Cladding: {result['cladding_radius']}")
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")