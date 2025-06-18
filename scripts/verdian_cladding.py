"""
Veridian Cladding Detector - v1.0

Detects and isolates the fiber optic cladding (annulus) with sub-pixel accuracy 
using a multi-stage geometric fitting pipeline (Canny -> RANSAC -> Levenberg-Marquardt).
This script is designed to be robust against illumination gradients.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from skimage.feature import canny

# --- HELPER FUNCTIONS (Identical for both Core and Cladding scripts) ---

def _get_edge_points(image: np.ndarray, sigma: float, low_thresh: float, high_thresh: float) -> np.ndarray:
    """Extracts unbiased edge points using the Canny algorithm."""
    image_float = image.astype(float) / 255.0
    edges = canny(image_float, sigma=sigma, low_threshold=low_thresh, high_threshold=high_thresh)
    points = np.argwhere(edges).astype(float)
    return points[:, ::-1] # Return as (x, y) coordinates

def _generate_ransac_hypothesis(points: np.ndarray) -> (list | None):
    """Generates a robust initial guess via RANSAC and Radial Histogram Voting."""
    best_score = -1
    best_params = None
    num_iterations = 750 # Balanced for speed and robustness

    for _ in range(num_iterations):
        sample_indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample_indices]
        D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(D) < 1e-6: continue

        ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
        uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D
        center_hypothesis = np.array([ux, uy])
        
        distances = np.linalg.norm(points - center_hypothesis, axis=1)
        hist, bin_edges = np.histogram(distances, bins=200, range=(0, np.max(distances)))
        
        peak_indices = np.argsort(hist)[-2:]
        score = np.sum(hist[peak_indices])
        
        if score > best_score:
            best_score = score
            r1_guess = bin_edges[peak_indices[0]] + (bin_edges[1] - bin_edges[0]) / 2
            r2_guess = bin_edges[peak_indices[1]] + (bin_edges[1] - bin_edges[0]) / 2
            best_params = [ux, uy, min(r1_guess, r2_guess), max(r1_guess, r2_guess)]
    
    return best_params

def _refine_fit(points: np.ndarray, initial_params: list) -> np.ndarray:
    """Refines the geometric fit using Non-Linear Least Squares."""
    def residuals(params, points):
        cx, cy, r1, r2 = params
        distances = np.linalg.norm(points - np.array([cx, cy]), axis=1)
        return np.minimum(np.abs(distances - r1), np.abs(distances - r2))

    result = least_squares(residuals, initial_params, args=(points,), method='lm', ftol=1e-6)
    return result.x

# --- MAIN PROCESSING FUNCTION ---

def process_image(image: np.ndarray, 
                  canny_sigma: float = 1.5,
                  highlight_boundaries: bool = True,
                  crop_output: bool = True) -> np.ndarray:
    """
    Isolates the fiber optic cladding using the Veridian geometric fitting pipeline.
    
    Args:
        image: Input fiber optic image (color or grayscale).
        canny_sigma: Sigma for Gaussian blur in Canny edge detector. Controls sensitivity.
        highlight_boundaries: If True, draws circles on the cladding boundaries.
        crop_output: If True, crops the output image to the cladding's bounding box.
        
    Returns:
        An image showing the isolated cladding, cropped and/or highlighted.
    """
    try:
        # 1. Ensure Grayscale for Analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Extract Edge Points
        edge_points = _get_edge_points(gray, sigma=canny_sigma, low_thresh=0.1, high_thresh=0.3)
        if len(edge_points) < 20:
            print("Warning: Not enough edge points detected. Try adjusting Canny sigma.")
            return image

        # 3. Get Initial Guess via RANSAC
        initial_guess = _generate_ransac_hypothesis(edge_points)
        if initial_guess is None:
            print("Warning: RANSAC could not form a stable hypothesis.")
            return image

        # 4. Refine Fit to Sub-pixel Accuracy
        final_params = _refine_fit(edge_points, initial_guess)
        cx, cy, r1, r2 = final_params
        r_core, r_cladding = min(r1, r2), max(r1, r2)
        
        # 5. Create Mask using the Equation of a Washer (Annulus)
        h, w = gray.shape
        y_grid, x_grid = np.mgrid[:h, :w]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        
        # Condition: distance squared must be > core radius squared AND <= cladding radius squared
        cladding_mask = ((dist_sq > r_core**2) & (dist_sq <= r_cladding**2)).astype(np.uint8) * 255
        
        # 6. Isolate the Cladding
        if len(image.shape) == 3:
            cladding_mask_color = cv2.cvtColor(cladding_mask, cv2.COLOR_GRAY2BGR)
            isolated_cladding = cv2.bitwise_and(image, cladding_mask_color)
        else:
            isolated_cladding = cv2.bitwise_and(gray, gray, mask=cladding_mask)
            
        # 7. Final Touches (Highlighting & Cropping)
        result = isolated_cladding
        if highlight_boundaries:
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            # Draw inner (green) and outer (cyan) boundaries
            cv2.circle(result, (int(round(cx)), int(round(cy))), int(round(r_core)), (0, 255, 0), 1)
            cv2.circle(result, (int(round(cx)), int(round(cy))), int(round(r_cladding)), (255, 255, 0), 1)

        if crop_output:
            coords = np.argwhere(cladding_mask > 0)
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                result = result[y_min:y_max+1, x_min:x_max+1]
        
        return result

    except Exception as e:
        print(f"Error in veridian_cladding_detector: {e}")
        return image # Return original image on error
