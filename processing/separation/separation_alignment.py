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
    if np.all(region == 0): return region # Skip if region is empty

    if len(region.shape) > 2:
        gray_copy = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray_copy = region.copy()

    # Otsu's thresholding is excellent for finding the natural break between light/dark in a region
    _, binary_mask = cv2.threshold(gray_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    if keep_white:
        # For the core, we keep the original pixels where the mask is white
        return cv2.bitwise_and(region, region, mask=binary_mask)
    else:
        # For the cladding, we keep original pixels where the mask is black
        inverted_mask = cv2.bitwise_not(binary_mask)
        return cv2.bitwise_and(region, region, mask=inverted_mask)

def run_single_geometric_fit(points):
    """
    (From zara.py) Runs one instance of RANSAC + LSQ fitting.
    Returns [cx, cy, r1, r2] or None if it fails.
    """
    if len(points) < 20: return None # Not enough points to fit

    def residuals(params, points_arg):
        cx, cy, r1, r2 = params
        distances = np.linalg.norm(points_arg - np.array([cx, cy]), axis=1)
        # The error is the distance to the *nearer* of the two circles
        return np.minimum(np.abs(distances - r1), np.abs(distances - r2))

    best_score, best_params_ransac = -1, None
    for _ in range(250): # RANSAC iterations
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
            best_score = score
            peak_indices = np.argsort(hist)[-2:]
            best_params_ransac = [ux, uy, bin_edges[peak_indices[0]], bin_edges[peak_indices[1]]]
    
    if best_params_ransac is None: return None
    
    result = least_squares(residuals, best_params_ransac, args=(points,), method='lm', max_nfev=100)
    return result.x

# --- MAIN ANALYSIS PIPELINE ---

def multi_hypothesis_segmentation(image_path, output_dir='output_multi_hypothesis'):
    """
    Implements a multi-hypothesis, validated consensus approach for ultimate segmentation.
    """
    print(f"\n--- Multi-Hypothesis Pipeline Initiated for: {image_path} ---")
    if not os.path.exists(image_path): print(f"Error: File not found: {image_path}"); return
    os.makedirs(output_dir, exist_ok=True)

    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
    height, width = gray_image.shape
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # --- STAGE 0: PRE-CALCULATE DATA MAPS ---
    print("\n--- Stage 0: Pre-calculating Data Maps ---")
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    change_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # --- STAGE 1: GENERATE MULTIPLE HYPOTHESIS VIEWS ---
    print("\n--- Stage 1: Generating Multiple Binary 'Views' of the Image ---")
    # View 1: Canny Edges
    view_canny = (canny(gray_image, sigma=1.5) * 255).astype(np.uint8) #
    # View 2: Otsu's Global Threshold
    _, view_otsu = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # View 3: Adaptive Threshold
    view_adaptive = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

    views = {'Canny': view_canny, 'Otsu': view_otsu, 'Adaptive': view_adaptive}
    for name, view_img in views.items():
        cv2.imwrite(os.path.join(output_dir, f'0_view_{name}.png'), view_img)
    print("  -> Saved all binary views for inspection.")

    # --- STAGE 2: TEST EACH HYPOTHESIS AND FIND CONSENSUS ---
    print("\n--- Stage 2: Testing Each Hypothesis for Data-Geometric Alignment ---")
    hypotheses = []

    for method_name, view in views.items():
        print(f"\n  -> Testing '{method_name}' Hypothesis...")
        
        # Step 2a: Extract points from the current view
        if method_name == 'Canny':
            points = np.argwhere(view > 0)[:, ::-1].astype(float)
        else: # For Otsu and Adaptive, get points from contours
            contours, _ = cv2.findContours(view, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("     - No contours found in this view. Skipping.")
                continue
            points = np.vstack([c for c in contours]).squeeze().astype(float)

        if len(points) < 50:
            print("     - Not enough feature points in this view. Skipping.")
            continue
        print(f"     - Extracted {len(points)} feature points.")
        
        # Step 2b: Run geometric fit on these points
        geom_params = run_single_geometric_fit(points)
        if geom_params is None:
            print("     - Geometric fit failed for this view. Skipping.")
            continue
        cx_geom, cy_geom, r1_geom, r2_geom = geom_params
        geom_radii = sorted([r1_geom, r2_geom])
        print(f"     - Geometric fit produced radii: {geom_radii[0]:.2f}, {geom_radii[1]:.2f}")
        
        # Step 2c: Validate against gradient data
        y_coords, x_coords = np.indices((height, width))
        radii_map = np.sqrt((x_coords - cx_geom)**2 + (y_coords - cy_geom)**2)
        max_radius = int(np.max(radii_map))
        radial_change = np.array([np.mean(change_magnitude[radii_map.astype(int) == r]) for r in range(max_radius) if np.any(radii_map.astype(int) == r)])
        
        data_peaks, _ = find_peaks(radial_change, prominence=np.mean(radial_change), distance=20)
        if len(data_peaks) < 2:
            print("     - Could not find two gradient peaks for validation. Skipping.")
            continue
        data_radii = sorted(data_peaks[np.argsort(radial_change[data_peaks])[-2:]])
        
        alignment_score = abs(geom_radii[0] - data_radii[0]) + abs(geom_radii[1] - data_radii[1])
        print(f"     - Data-driven peaks at {data_radii[0]}, {data_radii[1]}. Alignment Score: {alignment_score:.2f}")

        hypotheses.append({
            'method': method_name,
            'params': geom_params,
            'score': alignment_score,
            'radial_profile': radial_change
        })

    # --- STAGE 3: SELECT BEST HYPOTHESIS AND SEGMENT ---
    if not hypotheses:
        print("\nError: All hypothesis testing methods failed. Unable to find a solution.")
        return

    best_hypothesis = min(hypotheses, key=lambda x: x['score'])
    print(f"\n--- Stage 3: Consensus Achieved! Best alignment from '{best_hypothesis['method']}' method with score {best_hypothesis['score']:.2f} ---")

    cx, cy, r1, r2 = best_hypothesis['params']
    final_radii = sorted([r1, r2])
    core_radius, cladding_radius = int(round(final_radii[0])), int(round(final_radii[1]))
    final_center = (int(round(cx)), int(round(cy)))

    print(f"  -> Final Parameters: Center({final_center}), Core Radius({core_radius}), Cladding Radius({cladding_radius})")

    # --- STAGE 4: PER-SEGMENT REFINEMENT & OUTPUT ---
    print("\n--- Stage 4: Applying Final Segmentation and Artifact Removal ---")
    mask_template = np.zeros_like(gray_image)
    # The equation of a circle/washer is used here for masking
    core_mask = cv2.circle(mask_template.copy(), final_center, core_radius, 255, -1)
    cladding_outer = cv2.circle(mask_template.copy(), final_center, cladding_radius, 255, -1)
    cladding_mask = cv2.subtract(cladding_outer, core_mask)
    
    # Refine Core: keep bright pixels
    initial_core = cv2.bitwise_and(original_image, original_image, mask=core_mask)
    refined_core = apply_binary_refinement_mask(initial_core, keep_white=True)
    
    # Refine Cladding: remove bright pixels
    initial_cladding = cv2.bitwise_and(original_image, original_image, mask=cladding_mask)
    refined_cladding = apply_binary_refinement_mask(initial_cladding, keep_white=False)

    print("  -> Saved refined core and cladding regions.")
    cv2.imwrite(os.path.join(output_dir, f'1_refined_core.png'), refined_core)
    cv2.imwrite(os.path.join(output_dir, f'2_refined_cladding.png'), refined_cladding)

    # --- STAGE 5: FINAL REPORTING & VISUALIZATION ---
    print("\n--- Stage 5: Generating Final Reports ---")
    diagnostic_image = original_image.copy()
    cv2.circle(diagnostic_image, final_center, core_radius, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(diagnostic_image, final_center, cladding_radius, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, f'3_final_boundaries.png'), diagnostic_image)

    # Final Analysis Chart
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    radial_profile = best_hypothesis['radial_profile']
    axs.plot(radial_profile, label='Change Gradient of Winning Hypothesis', color='red', linewidth=2)
    axs.set_title(f"Final Analysis - Winning Method: {best_hypothesis['method']} (Score: {best_hypothesis['score']:.2f})", fontsize=16)
    axs.set_xlabel('Radius from Center (pixels)')
    axs.set_ylabel('Gradient Magnitude')
    axs.axvline(x=core_radius, color='g', linestyle='--', label=f'Core Boundary ({core_radius}px)')
    axs.axvline(x=cladding_radius, color='b', linestyle='--', label=f'Cladding Boundary ({cladding_radius}px)')
    axs.grid(True)
    axs.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'4_final_analysis_chart.png'))
    plt.close()

    print("  -> Saved final boundary image and analysis chart.")
    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        multi_hypothesis_segmentation(sys.argv[1])
    else:
        print("Usage: python your_script_name.py /path/to/your/image.png")
