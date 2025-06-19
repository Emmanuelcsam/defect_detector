import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from skimage.feature import canny

# Ensure a non-interactive backend for matplotlib for Linux/headless compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- HELPER FUNCTIONS ---

def apply_binary_refinement_mask(region, keep_white=True):
    """
    Applies a binary filter to a segmented region to remove artifacts.
    - For the core (keep_white=True), it keeps the bright areas.
    - For the cladding (keep_white=False), it removes the bright areas.
    """
    if np.all(region == 0): return region

    if len(region.shape) > 2:
        gray_copy = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray_copy = region.copy()

    _, binary_mask = cv2.threshold(gray_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    if keep_white:
        return cv2.bitwise_and(region, region, mask=binary_mask)
    else:
        inverted_mask = cv2.bitwise_not(binary_mask)
        return cv2.bitwise_and(region, region, mask=inverted_mask)

def run_ransac_lsq_fit(points):
    """
    (From zara.py) Runs one instance of RANSAC + LSQ fitting.
    Returns [cx, cy, r1, r2] or None if it fails.
    """
    if len(points) < 20: return None

    def residuals(params, points_arg):
        cx, cy, r1, r2 = params
        distances = np.linalg.norm(points_arg - np.array([cx, cy]), axis=1)
        return np.minimum(np.abs(distances - r1), np.abs(distances - r2))

    best_score, best_params_ransac = -1, None
    for _ in range(250):
        try:
            p1, p2, p3 = points[np.random.choice(len(points), 3, replace=False)]
        except ValueError:
            return None
        D = 2 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        if abs(D) < 1e-6: continue
        ux = ((p1[0]**2+p1[1]**2)*(p2[1]-p3[1])+(p2[0]**2+p2[1]**2)*(p3[1]-p1[1])+(p3[0]**2+p3[1]**2)*(p1[1]-p2[1]))/D
        uy = ((p1[0]**2+p1[1]**2)*(p3[0]-p2[0])+(p2[0]**2+p2[1]**2)*(p1[0]-p3[0])+(p3[0]**2+p3[1]**2)*(p2[0]-p1[0]))/D
        distances = np.linalg.norm(points - np.array([ux, uy]), axis=1)
        hist, bin_edges = np.histogram(distances, bins=100, range=(0, np.max(distances)))
        score = np.sum(np.sort(hist)[-2:])
        if score > best_score:
            best_score, peak_indices = score, np.argsort(hist)[-2:]
            best_params_ransac = [ux, uy, bin_edges[peak_indices[0]], bin_edges[peak_indices[1]]]
    
    if best_params_ransac is None: return None
    
    result = least_squares(residuals, best_params_ransac, args=(points,), method='lm', max_nfev=100)
    return result.x

# --- MAIN ANALYSIS PIPELINE ---

def ultimate_consensus_segmentation(image_path, output_dir='output_ultimate_consensus'):
    """
    Implements a multi-hypothesis, validated consensus approach including Hough Transforms.
    """
    print(f"\n--- Ultimate Consensus Pipeline Initiated for: {image_path} ---")
    if not os.path.exists(image_path): print(f"Error: File not found: {image_path}"); return
    os.makedirs(output_dir, exist_ok=True)

    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
    height, width, _ = original_image.shape
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # STAGE 0: PRE-CALCULATE DATA MAPS
    print("\n--- Stage 0: Pre-calculating Data Maps ---")
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    change_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # STAGE 1: GENERATE & TEST HYPOTHESES
    print("\n--- Stage 1: Generating and Testing All Hypotheses ---")
    hypotheses = []
    
    # --- Methods based on Binary Views ---
    views = {
        'Canny': (canny(gray_image, sigma=1.5) * 255).astype(np.uint8),
        'Otsu': cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        'Adaptive': cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    }

    for method_name, view in views.items():
        print(f"\n  -> Testing '{method_name}' View Hypothesis...")
        points = None
        if method_name == 'Canny':
            points = np.argwhere(view > 0)[:, ::-1].astype(float)
        else:
            contours, _ = cv2.findContours(view, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                points = np.vstack([c for c in contours]).squeeze().astype(float)
        
        if points is None or len(points) < 50:
            print("     - Not enough feature points. Skipping.")
            continue
            
        geom_params = run_ransac_lsq_fit(points)
        if geom_params is None:
            print("     - Geometric fit failed. Skipping.")
            continue
        
        cx_geom, cy_geom, r1_geom, r2_geom = geom_params
        
        # Validate against gradient data
        y_coords, x_coords = np.indices((height, width))
        radii_map = np.sqrt((x_coords - cx_geom)**2 + (y_coords - cy_geom)**2)
        max_r = int(np.max(radii_map))
        radial_change = np.array([np.mean(change_magnitude[radii_map.astype(int) == r]) for r in range(max_r) if np.any(radii_map.astype(int) == r)])
        
        data_peaks, _ = find_peaks(radial_change, prominence=np.mean(radial_change), distance=20)
        if len(data_peaks) < 2:
            print("     - Validation failed: could not find two gradient peaks. Skipping.")
            continue
            
        data_radii = sorted(data_peaks[np.argsort(radial_change[data_peaks])[-2:]])
        alignment_score = abs(sorted([r1_geom, r2_geom])[0] - data_radii[0]) + abs(sorted([r1_geom, r2_geom])[1] - data_radii[1])
        
        print(f"     - Score: {alignment_score:.2f}")
        hypotheses.append({'method': method_name, 'params': geom_params, 'score': alignment_score})

    # --- NEW: Hough Circle Hypothesis ---
    print("\n  -> Testing 'Hough Circle' Hypothesis...")
    hough_circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10, param1=50, param2=30, minRadius=10)
    if hough_circles is not None and len(hough_circles[0]) >= 2:
        # Take the two most prominent Hough circles as the boundaries
        circles = sorted(hough_circles[0], key=lambda c: c[2]) # Sort by radius
        cx_hough = np.mean([c[0] for c in circles[:2]]) # Average center
        cy_hough = np.mean([c[1] for c in circles[:2]])
        r1_hough, r2_hough = circles[0][2], circles[1][2]

        y_coords, x_coords = np.indices((height, width))
        radii_map = np.sqrt((x_coords - cx_hough)**2 + (y_coords - cy_hough)**2)
        max_r = int(np.max(radii_map))
        radial_change = np.array([np.mean(change_magnitude[radii_map.astype(int) == r]) for r in range(max_r) if np.any(radii_map.astype(int) == r)])

        data_peaks, _ = find_peaks(radial_change, prominence=np.mean(radial_change), distance=20)
        if len(data_peaks) >= 2:
            data_radii = sorted(data_peaks[np.argsort(radial_change[data_peaks])[-2:]])
            alignment_score = abs(r1_hough - data_radii[0]) + abs(r2_hough - data_radii[1])
            print(f"     - Score: {alignment_score:.2f}")
            hypotheses.append({'method': 'Hough', 'params': [cx_hough, cy_hough, r1_hough, r2_hough], 'score': alignment_score})
        else:
            print("     - Validation failed: could not find two gradient peaks. Skipping.")
    else:
        print("     - Hough transform failed to find at least two circles. Skipping.")
        
    # STAGE 2: SELECT BEST HYPOTHESIS AND SEGMENT
    if not hypotheses: print("\nError: All hypothesis methods failed."); return

    best_hypothesis = min(hypotheses, key=lambda x: x['score'])
    print(f"\n--- Stage 2: Consensus! Best alignment from '{best_hypothesis['method']}' (Score: {best_hypothesis['score']:.2f}) ---")
    
    cx, cy, r1, r2 = best_hypothesis['params']
    core_radius, cladding_radius = int(round(min(r1, r2))), int(round(max(r1, r2)))
    final_center = (int(round(cx)), int(round(cy)))

    # STAGE 3: PER-SEGMENT REFINEMENT & OUTPUT
    print("\n--- Stage 3: Applying Final Segmentation and Artifact Removal ---")
    mask_template = np.zeros_like(gray_image)
    core_mask = cv2.circle(mask_template.copy(), final_center, core_radius, 255, -1)
    cladding_outer = cv2.circle(mask_template.copy(), final_center, cladding_radius, 255, -1)
    cladding_mask = cv2.subtract(cladding_outer, core_mask)
    
    # Refine and save
    refined_core = apply_binary_refinement_mask(cv2.bitwise_and(original_image, original_image, mask=core_mask), keep_white=True)
    refined_cladding = apply_binary_refinement_mask(cv2.bitwise_and(original_image, original_image, mask=cladding_mask), keep_white=False)
    cv2.imwrite(os.path.join(output_dir, f'1_refined_core.png'), refined_core)
    cv2.imwrite(os.path.join(output_dir, f'2_refined_cladding.png'), refined_cladding)
    print("  -> Saved refined core and cladding regions.")

    # STAGE 4: FINAL REPORTING
    print("\n--- Stage 4: Generating Final Reports ---")
    diagnostic_image = original_image.copy()
    cv2.circle(diagnostic_image, final_center, core_radius, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(diagnostic_image, final_center, cladding_radius, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, f'3_final_boundaries.png'), diagnostic_image)
    
    # Final chart
    plt.figure(figsize=(12, 6))
    y_coords, x_coords = np.indices((height, width))
    radii_map = np.sqrt((x_coords - final_center[0])**2 + (y_coords - final_center[1])**2)
    max_r = int(np.max(radii_map))
    final_radial_change = np.array([np.mean(change_magnitude[radii_map.astype(int) == r]) for r in range(max_r) if np.any(radii_map.astype(int) == r)])
    
    plt.plot(final_radial_change, label='Change Gradient from Final Center', color='red', linewidth=2)
    plt.title(f"Final Analysis - Winning Method: {best_hypothesis['method']} (Score: {best_hypothesis['score']:.2f})", fontsize=16)
    plt.axvline(x=core_radius, color='g', linestyle='--', label=f'Core Boundary ({core_radius}px)')
    plt.axvline(x=cladding_radius, color='b', linestyle='--', label=f'Cladding Boundary ({cladding_radius}px)')
    plt.xlabel('Radius from Center (pixels)'); plt.ylabel('Gradient Magnitude'); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, f'4_final_analysis_chart.png'))
    plt.close()

    print("  -> Saved final boundary image and analysis chart.")
    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        ultimate_consensus_segmentation(sys.argv[1])
    else:
        print("Usage: python your_script_name.py /path/to/your/image.png")