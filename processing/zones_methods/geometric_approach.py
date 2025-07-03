import cv2
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional

def segment_with_geometric(image_path: str, output_dir: str = 'output') -> Dict[str, Any]:
    """
    Analyzes and segments a fiber optic endface image using geometric approach
    with Hough Circle Transform and radial gradient analysis.
    
    Returns a standardized result dictionary for integration with separation.py
    """
    try:
        # Validate input
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f"The file '{image_path}' does not exist."
            }
        
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {
                'success': False,
                'error': f"Could not read the image from '{image_path}'."
            }
        
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # Apply blur to reduce noise and improve circle detection
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # --- Detect the Center of the Fiber ---
        # Use Hough Circle Transform to find the most prominent circle
        circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=int(gray_image.shape[0] / 3)
        )
        
        if circles is None:
            return {
                'success': False,
                'error': "Could not detect the center of the fiber using Hough Circle Transform."
            }
        
        # Get the center coordinates of the most prominent circle
        circles = np.uint16(np.around(circles))
        center_x, center_y, detected_radius = circles[0, 0]
        
        # --- Analyze Radial Change Magnitude ---
        # Calculate gradient magnitude
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        change_magnitude = cv2.magnitude(sobel_x, sobel_y)
        
        height, width = gray_image.shape
        max_radius = int(np.sqrt(height**2 + width**2) / 2)
        
        # Calculate the average gradient for each radius from the center
        radial_profile = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius, dtype=int)
        
        for y in range(height):
            for x in range(width):
                radius = int(np.sqrt((x - center_x)**2 + (y - center_y)**2))
                if radius < max_radius:
                    radial_profile[radius] += change_magnitude[y, x]
                    radial_counts[radius] += 1
        
        # Avoid division by zero
        radial_counts[radial_counts == 0] = 1
        radial_profile /= radial_counts
        
        # --- Identify Core and Cladding Boundaries ---
        # Find peaks in the radial profile
        peaks = []
        mean_profile = np.mean(radial_profile)
        
        for r in range(1, len(radial_profile) - 1):
            if (radial_profile[r] > radial_profile[r-1] and 
                radial_profile[r] > radial_profile[r+1] and 
                radial_profile[r] > mean_profile):
                peaks.append((r, radial_profile[r]))
        
        if len(peaks) < 2:
            # If we can't find two distinct peaks, use fallback strategy
            # Use the detected circle radius as core radius
            core_radius = float(detected_radius)
            # Estimate cladding radius based on typical fiber geometry
            cladding_radius = core_radius * 3.0  # Typical ratio
            confidence = 0.4  # Lower confidence
        else:
            # Sort peaks by their magnitude and take the top two
            peaks.sort(key=lambda p: p[1], reverse=True)
            radii = sorted([p[0] for p in peaks[:2]])
            core_radius = float(radii[0])
            cladding_radius = float(radii[1])
            
            # Calculate confidence based on peak strengths
            peak_strengths = [p[1] for p in peaks[:2]]
            confidence = min(1.0, (peak_strengths[0] + peak_strengths[1]) / (2 * np.max(radial_profile)))
        
        # Validate detected radii
        if core_radius <= 0 or core_radius >= min(height, width) / 2:
            return {
                'success': False,
                'error': "Detected core radius is invalid"
            }
        
        if cladding_radius <= core_radius:
            # Adjust cladding radius if it's not larger than core
            cladding_radius = core_radius * 2.5
            confidence *= 0.8  # Reduce confidence
        
        # Create diagnostic image (optional, for visualization)
        diagnostic_image = original_image.copy()
        cv2.circle(diagnostic_image, (int(center_x), int(center_y)), int(core_radius), (0, 255, 0), 2)
        cv2.circle(diagnostic_image, (int(center_x), int(center_y)), int(cladding_radius), (0, 0, 255), 2)
        cv2.circle(diagnostic_image, (int(center_x), int(center_y)), 3, (255, 255, 0), -1)  # Center point
        
        # Save diagnostic image
        diagnostic_path = os.path.join(output_dir, f'{base_filename}_geometric_diagnostic.png')
        cv2.imwrite(diagnostic_path, diagnostic_image)
        
        # Save radial profile plot if matplotlib is available
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(radial_profile[:int(max_radius*0.6)], 'b-', linewidth=1)
            plt.axvline(x=core_radius, color='g', linestyle='--', label=f'Core: {core_radius:.1f}px')
            plt.axvline(x=cladding_radius, color='r', linestyle='--', label=f'Cladding: {cladding_radius:.1f}px')
            plt.xlabel('Radius (pixels)')
            plt.ylabel('Average Gradient Magnitude')
            plt.title('Radial Gradient Profile')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            profile_path = os.path.join(output_dir, f'{base_filename}_geometric_radial_profile.png')
            plt.savefig(profile_path)
            plt.close()
        except ImportError:
            pass  # Skip if matplotlib not available
        
        # Return standardized result
        return {
            'success': True,
            'center': (float(center_x), float(center_y)),
            'core_radius': core_radius,
            'cladding_radius': cladding_radius,
            'confidence': confidence,
            'method_details': {
                'detected_circles': len(circles[0]) if circles is not None else 0,
                'num_peaks_found': len(peaks),
                'hough_detected_radius': float(detected_radius)
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error in geometric approach: {str(e)}"
        }


# Keep the original function name for backward compatibility
def segment_fiber_optic_image(image_path: str, output_dir: str = 'output') -> Dict[str, Any]:
    """Alias for segment_with_geometric for backward compatibility"""
    return segment_with_geometric(image_path, output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="""Geometric Fiber Optic Segmentation.
        Uses Hough Circle Transform and radial gradient analysis."""
    )
    parser.add_argument('-i', '--image', type=str, required=True, 
                       help='Path to the input fiber optic image file.')
    parser.add_argument('-o', '--output', type=str, default='output_geometric', 
                       help='Directory to save the output files.')
    args = parser.parse_args()
    
    result = segment_with_geometric(args.image, args.output)
    
    if result['success']:
        print(f"✓ Segmentation successful!")
        print(f"  Center: {result['center']}")
        print(f"  Core radius: {result['core_radius']:.2f} pixels")
        print(f"  Cladding radius: {result['cladding_radius']:.2f} pixels")
        print(f"  Confidence: {result['confidence']:.3f}")
    else:
        print(f"✗ Segmentation failed: {result['error']}")