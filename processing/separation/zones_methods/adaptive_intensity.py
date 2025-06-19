import cv2
import numpy as np
import os
import json
from scipy.signal import find_peaks

def adaptive_segment_image(image_path, peak_prominence=1000, output_dir="adaptive_segmented_regions"):
    """
    Automatically segments an image by finding the most prominent intensity peaks
    in its histogram and creating separate images for each region.
    
    Modified to work with unified system - returns standardized results
    """
    # Initialize result dictionary
    result = {
        'method': 'adaptive_intensity_approach',
        'image_path': image_path,
        'success': False,
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0
    }
    
    if not os.path.exists(image_path):
        result['error'] = f"File not found: '{image_path}'"
        with open(os.path.join(output_dir, 'adaptive_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        result['error'] = f"Could not read image from '{image_path}'"
        with open(os.path.join(output_dir, 'adaptive_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    try:
        histogram = cv2.calcHist([original_image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()

        peaks, properties = find_peaks(histogram, prominence=peak_prominence, width=(None, None), rel_height=1.0)

        if len(peaks) == 0:
            result['error'] = f"No significant intensity peaks found with prominence={peak_prominence}"
            with open(os.path.join(output_dir, f'{base_filename}_adaptive_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result

        # Convert peak boundaries to standard Python integers
        left_bases = [int(x) for x in properties['left_ips']]
        right_bases = [int(x) for x in properties['right_ips']]
        
        intensity_ranges = list(zip(left_bases, right_bases))
        
        # Try to identify core and cladding regions based on intensity
        # Typically, core is brightest, cladding is medium, ferrule is darkest
        regions_info = []
        
        for i, (min_val, max_val) in enumerate(intensity_ranges):
            peak_intensity = peaks[i]
            
            # Create mask for this intensity range
            mask = cv2.inRange(original_image, min_val, max_val)
            
            # Find contours in this region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                
                # Calculate average intensity for region identification
                masked_pixels = original_image[mask > 0]
                avg_intensity = np.mean(masked_pixels) if len(masked_pixels) > 0 else peak_intensity
                
                regions_info.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'avg_intensity': avg_intensity,
                    'peak_intensity': peak_intensity,
                    'contour': largest_contour,
                    'mask': mask
                })
                
                # Save segmented region
                segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)
                output_filename = f"{base_filename}_adaptive_region_{i+1}_intensity_{peak_intensity}.png"
                cv2.imwrite(os.path.join(output_dir, output_filename), segmented_image)
        
        # Sort regions by average intensity (brightest first)
        regions_info.sort(key=lambda x: x['avg_intensity'], reverse=True)
        
        # Try to identify core and cladding
        if len(regions_info) >= 2:
            # Assume brightest region contains the core
            # Second brightest likely contains cladding
            core_region = regions_info[0]
            cladding_region = regions_info[1]
            
            # Use the center from the larger region (usually cladding)
            if cladding_region['radius'] > core_region['radius']:
                result['center'] = cladding_region['center']
                result['cladding_radius'] = cladding_region['radius']
                result['core_radius'] = core_region['radius']
            else:
                # Something might be wrong, but use what we have
                result['center'] = core_region['center']
                result['core_radius'] = int(core_region['radius'] * 0.5)  # Estimate
                result['cladding_radius'] = core_region['radius']
            
            result['success'] = True
            result['confidence'] = 0.5  # Lower confidence for this method
            
        elif len(regions_info) == 1:
            # Only one region found, estimate the structure
            region = regions_info[0]
            result['center'] = region['center']
            result['cladding_radius'] = region['radius']
            result['core_radius'] = int(region['radius'] * 0.3)  # Typical ratio
            result['success'] = True
            result['confidence'] = 0.3  # Very low confidence
        else:
            result['error'] = "Could not identify fiber structure from intensity regions"
            with open(os.path.join(output_dir, f'{base_filename}_adaptive_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result
        
        # Save result data
        with open(os.path.join(output_dir, f'{base_filename}_adaptive_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
        # Create annotated visualization
        annotated = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        if result['success']:
            cv2.circle(annotated, result['center'], 3, (0, 255, 255), -1)
            cv2.circle(annotated, result['center'], result['core_radius'], (0, 255, 0), 2)
            cv2.circle(annotated, result['center'], result['cladding_radius'], (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_adaptive_annotated.png"), annotated)
        
    except Exception as e:
        result['error'] = str(e)
        with open(os.path.join(output_dir, f'{base_filename}_adaptive_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
    
    return result


def main():
    """Main function for standalone testing"""
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"').strip("'")
    
    result = adaptive_segment_image(image_path)
    if result['success']:
        print(f"Success! Center: {result['center']}, Core: {result['core_radius']}, Cladding: {result['cladding_radius']}")
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()