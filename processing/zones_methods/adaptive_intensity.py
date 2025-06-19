import cv2
import numpy as np
import os
import json
from scipy.signal import find_peaks

def adaptive_segment_image(image_path, peak_prominence=500, output_dir="adaptive_segmented_regions"):
    """
    Automatically segments an image by finding the most prominent intensity peaks
    in its histogram and creating separate images for each region.
    
    Modified with CLAHE enhancement for low-contrast images
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

    # --- ENHANCEMENT START ---
    # Apply CLAHE to enhance local contrast before analysis
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(original_image)
    # --- ENHANCEMENT END ---

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Calculate histogram on the ENHANCED image
        histogram = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()

        # Find peaks with lowered prominence for low-contrast images
        peaks, properties = find_peaks(histogram, prominence=peak_prominence, width=(None, None), rel_height=1.0)

        if len(peaks) == 0:
            # Try with even lower prominence
            peaks, properties = find_peaks(histogram, prominence=peak_prominence/2, width=(None, None), rel_height=1.0)
        
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
            
            # Create mask for this intensity range on ENHANCED image
            mask = cv2.inRange(enhanced_image, min_val, max_val)
            
            # Find contours in this region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                
                # Calculate average intensity for region identification
                masked_pixels = enhanced_image[mask > 0]
                avg_intensity = np.mean(masked_pixels) if len(masked_pixels) > 0 else peak_intensity
                
                regions_info.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'avg_intensity': avg_intensity,
                    'peak_intensity': peak_intensity,
                    'contour': largest_contour,
                    'mask': mask,
                    'area': cv2.contourArea(largest_contour)
                })
                
                # Save segmented region from ORIGINAL image
                segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)
                output_filename = f"{base_filename}_adaptive_region_{i+1}_intensity_{peak_intensity}.png"
                cv2.imwrite(os.path.join(output_dir, output_filename), segmented_image)
        
        # Sort regions by average intensity (brightest first)
        regions_info.sort(key=lambda x: x['avg_intensity'], reverse=True)
        
        # Filter out very small regions
        regions_info = [r for r in regions_info if r['area'] > 100]
        
        # Try to identify core and cladding
        if len(regions_info) >= 2:
            # Look for concentric circles
            # The brightest region should be inside a darker region for typical fiber optics
            core_region = None
            cladding_region = None
            
            # Find the brightest small region (likely core)
            for i, region in enumerate(regions_info):
                if region['radius'] < enhanced_image.shape[0] * 0.3:  # Core should be relatively small
                    core_region = region
                    break
            
            # Find a larger region that could be cladding
            if core_region:
                for region in regions_info:
                    if region['radius'] > core_region['radius'] * 1.5 and region['radius'] < enhanced_image.shape[0] * 0.5:
                        cladding_region = region
                        break
            
            if core_region and cladding_region:
                # Use the center from the larger region (usually more stable)
                result['center'] = cladding_region['center']
                result['core_radius'] = core_region['radius']
                result['cladding_radius'] = cladding_region['radius']
                result['success'] = True
                result['confidence'] = 0.6  # Moderate confidence
            else:
                # Fallback: use the two largest regions
                regions_info.sort(key=lambda x: x['radius'])
                if len(regions_info) >= 2:
                    result['center'] = regions_info[-1]['center']
                    result['core_radius'] = regions_info[0]['radius']
                    result['cladding_radius'] = regions_info[-1]['radius']
                    result['success'] = True
                    result['confidence'] = 0.5
                    
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
        
        # Create annotated visualization on ORIGINAL image
        annotated = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        if result['success']:
            cv2.circle(annotated, result['center'], 3, (0, 255, 255), -1)
            cv2.circle(annotated, result['center'], result['core_radius'], (0, 255, 0), 2)
            cv2.circle(annotated, result['center'], result['cladding_radius'], (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_adaptive_annotated.png"), annotated)
        
        # Save the enhanced image for debugging
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_adaptive_enhanced.png"), enhanced_image)
        
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
    
    # Use lower prominence for low-contrast images
    result = adaptive_segment_image(image_path, peak_prominence=500)
    if result['success']:
        print(f"Success! Center: {result['center']}, Core: {result['core_radius']}, Cladding: {result['cladding_radius']}")
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()
