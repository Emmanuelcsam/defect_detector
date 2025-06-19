import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skimage.feature import canny

def get_edge_points(image, sigma=1.5, low_threshold=0.1, high_threshold=0.3):
    """
    Uses the Canny edge detector to extract a sparse set of high-confidence
    edge points from the image, decoupling geometry from illumination.

    Returns:
        np.array: An (N, 2) array of [x, y] coordinates for N edge points.
    """
    # Convert image to float for scikit-image's Canny implementation
    image_float = image.astype(float) / 255.0
    edges = canny(image_float, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    points = np.argwhere(edges).astype(float) # argwhere returns (row, col) which is (y, x)
    return points[:, ::-1] # Return as (x, y)

def generate_hypotheses_ransac(points, num_iterations=1000, inlier_threshold=1.0):
    """
    Generates a highly robust initial guess for the center and radii using a custom
    RANSAC and Radial Histogram Voting scheme.
    """
    best_score = -1
    best_params = None

    for i in range(num_iterations):
        # 1. Hypothesize: Randomly sample 3 points and find the circumcenter
        if len(points) < 3:
            continue
            
        sample_indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Using a standard formula to find the circumcenter of a triangle
        D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(D) < 1e-6: continue # Avoid degenerate cases (collinear points)

        ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
        uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D
        
        # Validate center is within reasonable bounds
        if ux < 0 or uy < 0 or ux > 10000 or uy > 10000:
            continue
            
        center_hypothesis = np.array([ux, uy])

        # 2. Score via Radial Histogram Voting
        distances = np.linalg.norm(points - center_hypothesis, axis=1)
        
        # Filter out unreasonable distances
        reasonable_distances = distances[distances < 1000]
        if len(reasonable_distances) < 10:
            continue
            
        # Create a histogram of distances (radii)
        hist, bin_edges = np.histogram(reasonable_distances, bins=50, range=(0, 500))
        
        # Find the two largest peaks in the histogram
        peak_indices = np.argsort(hist)[-2:] # Get indices of the two highest bins
        
        # Score is the sum of the heights of the two peaks
        score = np.sum(hist[peak_indices])
        
        if score > best_score:
            best_score = score
            r1_guess = bin_edges[peak_indices[0]] + (bin_edges[1] - bin_edges[0])/2
            r2_guess = bin_edges[peak_indices[1]] + (bin_edges[1] - bin_edges[0])/2
            
            # Ensure radii are reasonable
            if r1_guess < 500 and r2_guess < 500:
                best_params = [ux, uy, min(r1_guess, r2_guess), max(r1_guess, r2_guess)]

    return best_params

def refine_fit_least_squares(points, initial_params):
    """
    Performs the ultimate refinement using Non-Linear Least Squares to minimize
    the true geometric distance of edge points to the two-circle model.
    """
    def residuals(params, points):
        """The objective function to minimize."""
        cx, cy, r1, r2 = params
        center = np.array([cx, cy])
        # Calculate distance of each point to the center
        distances = np.linalg.norm(points - center, axis=1)
        
        # For each point, the error is the distance to the *nearer* of the two circles
        res1 = np.abs(distances - r1)
        res2 = np.abs(distances - r2)
        
        return np.minimum(res1, res2)

    # Set reasonable bounds for parameters
    max_coord = np.max(points) * 1.5
    bounds = (
        [0, 0, 5, 10],  # Lower bounds: cx, cy, r1, r2
        [max_coord, max_coord, max_coord/2, max_coord/2]  # Upper bounds
    )
    
    # Use scipy's Levenberg-Marquardt implementation with bounds
    try:
        result = least_squares(residuals, initial_params, args=(points,), 
                             method='trf', bounds=bounds)
        
        # Validate the result
        cx, cy, r1, r2 = result.x
        if cx > 0 and cy > 0 and r1 > 0 and r2 > 0 and r1 < 1000 and r2 < 1000:
            return result.x
        else:
            # Return initial params if optimization went wrong
            return initial_params
    except:
        return initial_params

def create_final_masks(image_shape, params):
    """Creates final masks using the ultra-precise parameters."""
    h, w = image_shape
    cx, cy, r_core, r_cladding = params
    
    # Ensure radii are ordered correctly
    if r_core > r_cladding:
        r_core, r_cladding = r_cladding, r_core

    # Create the distance matrix using matrix operations (Linear Algebra)
    y, x = np.mgrid[:h, :w]
    dist_sq = (x - cx)**2 + (y - cy)**2
    
    # Create masks based on the equation of a circle and a washer (annulus)
    core_mask = (dist_sq <= r_core**2).astype(np.uint8) * 255
    cladding_mask = ((dist_sq > r_core**2) & (dist_sq <= r_cladding**2)).astype(np.uint8) * 255
    
    return core_mask, cladding_mask

def process_fiber_image_veridian(image_path, output_dir='output_veridian'):
    """
    Main processing function modified for unified system
    Returns standardized results
    """
    # Initialize result dictionary
    result = {
        'method': 'computational_separation',
        'image_path': image_path,
        'success': False,
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0
    }
    
    if not os.path.exists(image_path): 
        result['error'] = f"File not found: {image_path}"
        with open(os.path.join(output_dir, 'computational_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
        
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    original_image = cv2.imread(image_path)
    if original_image is None:
        result['error'] = f"Could not read image from '{image_path}'"
        with open(os.path.join(output_dir, 'computational_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
        
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # STAGE 1: Extract unbiased geometric features
        edge_points = get_edge_points(gray_image, sigma=1.5)
        
        if len(edge_points) < 10:
            result['error'] = "Insufficient edge points detected"
            with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result
        
        # STAGE 2: Generate robust hypothesis with RANSAC
        initial_guess = generate_hypotheses_ransac(edge_points)
        if initial_guess is None: 
            result['error'] = "RANSAC failed to find a suitable hypothesis"
            with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result

        # STAGE 3: Perform ultimate refinement with Non-Linear Least Squares
        final_params = refine_fit_least_squares(edge_points, initial_guess)
        cx, cy, r1, r2 = final_params
        r_core, r_cladding = min(r1, r2), max(r1, r2)
        
        # Validate the results are within image bounds
        h, w = gray_image.shape
        if not (0 < cx < w and 0 < cy < h):
            result['error'] = f"Center ({cx}, {cy}) is outside image bounds ({w}, {h})"
            with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result
            
        if r_cladding > min(w, h) / 2:
            result['error'] = f"Cladding radius {r_cladding} is too large for image size"
            with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result
        
        # Set results
        result['success'] = True
        result['center'] = (int(cx), int(cy))
        result['core_radius'] = int(r_core)
        result['cladding_radius'] = int(r_cladding)
        result['confidence'] = 0.8  # High confidence for geometric method
        
        # Save result data
        with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
            json.dump(result, f, indent=4)

        # STAGE 4: Generate final masks and output
        core_mask, cladding_mask = create_final_masks(gray_image.shape, final_params)
        isolated_core = cv2.bitwise_and(gray_image, gray_image, mask=core_mask)
        isolated_cladding = cv2.bitwise_and(gray_image, gray_image, mask=cladding_mask)
        
        # Cropping
        coords_core = np.argwhere(core_mask > 0)
        if coords_core.size > 0:
            y_min, x_min = coords_core.min(axis=0); y_max, x_max = coords_core.max(axis=0)
            isolated_core = isolated_core[y_min:y_max+1, x_min:x_max+1]

        coords_cladding = np.argwhere(cladding_mask > 0)
        if coords_cladding.size > 0:
            y_min, x_min = coords_cladding.min(axis=0); y_max, x_max = coords_cladding.max(axis=0)
            isolated_cladding = isolated_cladding[y_min:y_max+1, x_min:x_max+1]
            
        # Save Diagnostic Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(original_image)
        # Plot the refined circles
        circle1 = plt.Circle((cx, cy), r_core, color='lime', fill=False, linewidth=2, label='Core')
        circle2 = plt.Circle((cx, cy), r_cladding, color='cyan', fill=False, linewidth=2, label='Cladding')
        plt.gca().add_artist(circle1)
        plt.gca().add_artist(circle2)
        plt.scatter(edge_points[:, 0], edge_points[:, 1], s=1, c='red', alpha=0.3, label='Edge Points')
        plt.title(f'Computational Geometric Fit')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{base_filename}_computational_fit.png"))
        plt.close()

        # Save Image Results with standardized names
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_computational_core.png"), isolated_core)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_computational_cladding.png"), isolated_cladding)
        
        # Create annotated image
        annotated = original_image.copy()
        cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 255, 255), -1)
        cv2.circle(annotated, (int(cx), int(cy)), int(r_core), (0, 255, 0), 2)
        cv2.circle(annotated, (int(cx), int(cy)), int(r_cladding), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_computational_annotated.png"), annotated)
        
    except Exception as e:
        result['error'] = str(e)
        with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
    return result

def main():
    """Main function for standalone testing"""
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"').strip("'")
    
    result = process_fiber_image_veridian(image_path)
    if result['success']:
        print(f"Success! Center: {result['center']}, Core: {result['core_radius']}, Cladding: {result['cladding_radius']}")
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()