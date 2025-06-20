import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - MUST BE BEFORE pyplot import
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import json
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

def generate_hypotheses_ransac(points, num_iterations=2000, inlier_threshold=1.5, image_shape=None):
    """
    Generates a highly robust initial guess for the center and radii using a custom
    RANSAC and Radial Histogram Voting scheme with improved parameters for low-contrast images.
    """
    best_score = -1
    best_params = None
    
    # Use image shape to constrain search if provided
    max_radius = 500  # Default max radius
    if image_shape is not None:
        h, w = image_shape
        max_radius = min(h, w) * 0.45  # Maximum 45% of image size

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
        if image_shape is not None:
            if ux < 0 or uy < 0 or ux > w or uy > h:
                continue
        else:
            if ux < 0 or uy < 0 or ux > 10000 or uy > 10000:
                continue
            
        center_hypothesis = np.array([ux, uy])

        # 2. Score via Radial Histogram Voting
        distances = np.linalg.norm(points - center_hypothesis, axis=1)
        
        # Filter out unreasonable distances
        reasonable_distances = distances[distances < max_radius]
        if len(reasonable_distances) < 10:
            continue
            
        # Create a histogram of distances (radii)
        hist, bin_edges = np.histogram(reasonable_distances, bins=50, range=(0, max_radius))
        
        # Find the two largest peaks in the histogram
        peak_indices = np.argsort(hist)[-2:] # Get indices of the two highest bins
        
        # Score is the sum of the heights of the two peaks
        score = np.sum(hist[peak_indices])
        
        if score > best_score:
            best_score = score
            r1_guess = bin_edges[peak_indices[0]] + (bin_edges[1] - bin_edges[0])/2
            r2_guess = bin_edges[peak_indices[1]] + (bin_edges[1] - bin_edges[0])/2
            
            # Ensure radii are reasonable
            if r1_guess < max_radius and r2_guess < max_radius:
                best_params = [ux, uy, min(r1_guess, r2_guess), max(r1_guess, r2_guess)]

    return best_params

def refine_fit_least_squares(points, initial_params, image_shape):
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

    # Set reasonable bounds for parameters based on image size
    h, w = image_shape
    max_coord = max(h, w)
    max_radius = min(h, w) * 0.45  # Maximum 45% of smallest dimension
    
    bounds = (
        [0, 0, 5, 10],  # Lower bounds: cx, cy, r1, r2
        [w, h, max_radius, max_radius]  # Upper bounds
    )
    
    # Use scipy's Levenberg-Marquardt implementation with bounds
    try:
        result = least_squares(residuals, initial_params, args=(points,), 
                             method='trf', bounds=bounds, max_nfev=1000)
        
        # Validate the result
        cx, cy, r1, r2 = result.x
        if cx > 0 and cy > 0 and r1 > 0 and r2 > 0 and cx < w and cy < h:
            # Additional validation: ensure radii are reasonable
            if r1 <= max_radius and r2 <= max_radius:
                return result.x
        
        # If validation fails, return constrained version of initial params
        cx, cy, r1, r2 = initial_params
        cx = np.clip(cx, 0, w)
        cy = np.clip(cy, 0, h)
        r1 = np.clip(r1, 5, max_radius)
        r2 = np.clip(r2, 10, max_radius)
        return [cx, cy, r1, r2]
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return constrained initial params
        cx, cy, r1, r2 = initial_params
        cx = np.clip(cx, 0, w)
        cy = np.clip(cy, 0, h)
        r1 = np.clip(r1, 5, max_radius)
        r2 = np.clip(r2, 10, max_radius)
        return [cx, cy, r1, r2]

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
    Main processing function modified for unified system with enhanced contrast
    Returns standardized results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
        
    original_image = cv2.imread(image_path)
    if original_image is None:
        result['error'] = f"Could not read image from '{image_path}'"
        with open(os.path.join(output_dir, 'computational_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        return result
        
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # ---- ENHANCEMENT START ----
    # Enhance contrast using histogram equalization for low-contrast images
    gray_image = cv2.equalizeHist(gray_image)
    # ---- ENHANCEMENT END ----
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        h, w = gray_image.shape
        image_shape = (h, w)
        
        # STAGE 1: Extract unbiased geometric features
        edge_points = get_edge_points(gray_image, sigma=1.5)
        
        if len(edge_points) < 10:
            result['error'] = "Insufficient edge points detected"
            with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
                json.dump(result, f, indent=4)
            return result
        
        # STAGE 2: Generate robust hypothesis with enhanced RANSAC
        initial_guess = generate_hypotheses_ransac(edge_points, image_shape=image_shape)
        if initial_guess is None: 
            # Try with more relaxed parameters
            edge_points_relaxed = get_edge_points(gray_image, sigma=2.0, low_threshold=0.05, high_threshold=0.2)
            initial_guess = generate_hypotheses_ransac(edge_points_relaxed, num_iterations=3000, image_shape=image_shape)
            
            if initial_guess is None:
                result['error'] = "RANSAC failed to find a suitable hypothesis"
                with open(os.path.join(output_dir, f'{base_filename}_computational_result.json'), 'w') as f:
                    json.dump(result, f, indent=4)
                return result

        # STAGE 3: Perform ultimate refinement with Non-Linear Least Squares
        final_params = refine_fit_least_squares(edge_points, initial_guess, image_shape)
        cx, cy, r1, r2 = final_params
        r_core, r_cladding = min(r1, r2), max(r1, r2)
        
        # Final validation with more reasonable constraints
        max_allowed_radius = min(w, h) * 0.45  # Maximum 45% of smallest dimension
        
        if r_cladding > max_allowed_radius:
            # Scale down proportionally
            scale_factor = max_allowed_radius / r_cladding
            r_cladding = max_allowed_radius
            r_core = r_core * scale_factor
            print(f"Warning: Scaled down radii by factor {scale_factor:.2f}")
        
        # Ensure minimum radius constraints
        if r_core < 5:
            r_core = 5
        if r_cladding < r_core + 10:
            r_cladding = r_core + 10
            
        # Validate center is well within bounds
        margin = 5
        if not (margin < cx < w - margin and margin < cy < h - margin):
            result['error'] = f"Center ({cx:.1f}, {cy:.1f}) is too close to image bounds"
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
        # Subsample edge points for visualization
        if len(edge_points) > 1000:
            indices = np.random.choice(len(edge_points), 1000, replace=False)
            edge_points_vis = edge_points[indices]
        else:
            edge_points_vis = edge_points
        plt.scatter(edge_points_vis[:, 0], edge_points_vis[:, 1], s=1, c='red', alpha=0.3, label='Edge Points')
        plt.title(f'Computational Geometric Fit (Enhanced)')
        plt.legend()
        plt.axis('equal')
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