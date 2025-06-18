import cv2
import numpy as np
import os
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
        sample_indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Using a standard formula to find the circumcenter of a triangle
        D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(D) < 1e-6: continue # Avoid degenerate cases (collinear points)

        ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
        uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D
        
        center_hypothesis = np.array([ux, uy])

        # 2. Score via Radial Histogram Voting
        distances = np.linalg.norm(points - center_hypothesis, axis=1)
        # Create a histogram of distances (radii)
        hist, bin_edges = np.histogram(distances, bins=200, range=(0, np.max(distances)))
        
        # Find the two largest peaks in the histogram
        peak_indices = np.argsort(hist)[-2:] # Get indices of the two highest bins
        
        # Score is the sum of the heights of the two peaks
        score = np.sum(hist[peak_indices])
        
        if score > best_score:
            best_score = score
            r1_guess = bin_edges[peak_indices[0]] + (bin_edges[1] - bin_edges[0])/2
            r2_guess = bin_edges[peak_indices[1]] + (bin_edges[1] - bin_edges[0])/2
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

    # Use scipy's Levenberg-Marquardt implementation
    result = least_squares(residuals, initial_params, args=(points,), method='lm')
    return result.x

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
    print(f"\n--- Veridian Pipeline commencing for: {image_path} ---")
    if not os.path.exists(image_path): print(f"Error: Not found: {image_path}"); return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # STAGE 1: Extract unbiased geometric features
    edge_points = get_edge_points(gray_image, sigma=1.5)
    print(f"Stage 1: Extracted {len(edge_points)} edge points using Canny.")
    
    # STAGE 2: Generate robust hypothesis with RANSAC
    initial_guess = generate_hypotheses_ransac(edge_points)
    if initial_guess is None: print("RANSAC failed to find a suitable hypothesis."); return
    print(f"Stage 2: RANSAC initial guess -> Center:({initial_guess[0]:.2f}, {initial_guess[1]:.2f}), Radii:({initial_guess[2]:.2f}, {initial_guess[3]:.2f})")

    # STAGE 3: Perform ultimate refinement with Non-Linear Least Squares
    final_params = refine_fit_least_squares(edge_points, initial_guess)
    cx, cy, r1, r2 = final_params
    r_core, r_cladding = min(r1, r2), max(r1, r2)
    print(f"Stage 3: Final refined parameters -> Center:({cx:.4f}, {cy:.4f}), Radii:({r_core:.4f}, {r_cladding:.4f})")

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
        
    # --- Save Diagnostic Plot ---
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    # Plot the refined circles
    circle1 = plt.Circle((cx, cy), r_core, color='lime', fill=False, linewidth=2, label='Fitted Core')
    circle2 = plt.Circle((cx, cy), r_cladding, color='cyan', fill=False, linewidth=2, label='Fitted Cladding')
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)
    plt.scatter(edge_points[:, 0], edge_points[:, 1], s=1, c='red', alpha=0.3, label='Canny Edge Points')
    plt.title(f'Final Geometric Fit for {os.path.basename(image_path)}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_veridian_fit.png"))
    plt.close()

    # --- Save Image Results ---
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_core.png"), isolated_core)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_cladding.png"), isolated_cladding)
    print(f"Stage 4: Successfully saved Veridian results to '{output_dir}'")

if __name__ == '__main__':
    image_filenames = [
        r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img63.jpg'
    ]
    for filename in image_filenames:
        # Check if file exists before processing
        if os.path.exists(filename):
            process_fiber_image_veridian(filename)
        else:
            print(f"Skipping {filename}, file not found.")