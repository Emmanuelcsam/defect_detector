# -*- coding: utf-8 -*-
"""
================================================================================
 Hybrid Geometric-Pixel Fiber Analyzer
================================================================================
Version: 2.0
Author: Gemini
Date: 19 June 2024

Description:
This script provides a highly robust, second-generation pipeline for segmenting
fiber optic end-face images. It directly addresses the instability of simpler
consensus models by creating a deep, symbiotic relationship between geometric
and pixel-based analysis at each stage of the process.

--------------------------------------------------------------------------------
Methodology Overview (Iterative Refinement & Fusion):
--------------------------------------------------------------------------------
This script uses a more intelligent, iterative pipeline to achieve superior
accuracy and stability.

Phase 1: Symbiotic Hypothesis Generation
    - A custom RANSAC algorithm is used to generate geometric hypotheses
      (center, radius) from Canny edge points.
    - Crucially, each hypothesis is immediately scored not just on its geometric
      fit (histogram of edge points) but also on a pixel-value metric: the
      prominence of peaks in its corresponding radial gradient profile. This
      fusion of geometry and pixel data produces a single, high-quality
      initial guess and rejects nonsensical candidates early.

Phase 2: Pixel-Guided Center Refinement
    - Using the strong initial hypothesis, the algorithm refines the center
      position. It creates a "cost map" in a small window around the initial
      center guess.
    - The cost for each pixel is based on how well a circle centered there
      encloses a bright, uniform core (a pixel-value metric).
    - The pixel with the optimal score becomes the new high-confidence center,
      effectively using pixel data to perfect a geometric starting point.

Phase 3: Model-Based Boundary Identification
    - With a reliable center, a 1D radial profile of the image's intensity
      and gradient is created.
    - Instead of naive peak-finding, a model of the fiber's physical
      properties is used to interpret the profile. The algorithm searches for
      a sequence of gradient peaks that corresponds to the expected intensity
      pattern: (bright core -> dark cladding -> ferrule).
    - This model-based approach makes the boundary detection highly resilient
      to noise, scratches, and debris that would otherwise create false peaks.

Phase 4: Stable Masking and Reporting
    - The final, high-confidence parameters (center from Phase 2, radii from
      Phase 3) are used to generate the segmentation masks.
    - The previous unstable `least_squares` refinement step has been REMOVED
      to prevent the possibility of mathematical divergence, which was the
      primary failure point in the prior version. Accuracy is now derived from
      the robust, multi-stage analysis itself.
    - A comprehensive suite of reports, including segmented images, data, and
      diagnostic plots, is generated.

--------------------------------------------------------------------------------
Dependencies:
--------------------------------------------------------------------------------
- opencv-python
- numpy
- matplotlib
- scipy
- scikit-image

Install with: pip install opencv-python numpy matplotlib scipy scikit-image
"""

# --- Core Imports ---
import os
import json
import shlex
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# --- Scientific Computing Imports ---
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt

# --- Analysis & Optimization Imports ---
from scipy.signal import find_peaks, savgol_filter
from skimage.feature import canny

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_radial_profiles(image: np.ndarray, center: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the 1D radial intensity and gradient profiles from a center."""
    h, w = image.shape
    cx, cy = center
    
    # Create a map of distances from the center
    y_grid, x_grid = np.ogrid[:h, :w]
    r_map = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    max_r = int(min(cx, cy, w - cx, h - cy))
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Bin pixels by integer radius
    r_map_int = r_map.astype(int)
    counts = np.bincount(r_map_int.ravel())
    
    # Calculate mean intensity and gradient for each radius
    intensity_sum = np.bincount(r_map_int.ravel(), weights=image.ravel())
    gradient_sum = np.bincount(r_map_int.ravel(), weights=grad_mag.ravel())
    
    # Handle potential division by zero
    valid_counts = counts > 0
    intensity_profile = np.zeros_like(counts, dtype=float)
    gradient_profile = np.zeros_like(counts, dtype=float)
    
    intensity_profile[valid_counts] = intensity_sum[valid_counts] / counts[valid_counts]
    gradient_profile[valid_counts] = gradient_sum[valid_counts] / counts[valid_counts]
    
    return intensity_profile[:max_r], gradient_profile[:max_r]


# =============================================================================
# MAIN ANALYSIS CLASS: HybridFiberAnalyzer
# =============================================================================

class HybridFiberAnalyzer:
    """
    Encapsulates the integrated geometric-pixel fiber segmentation pipeline.
    """
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.original_image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.edge_points: Optional[np.ndarray] = None
        self.height, self.width = 0, 0
        self.final_params: Dict[str, float] = {}
        self.report_data: Dict[str, Any] = {}

    def run_full_pipeline(self) -> bool:
        """Executes the entire segmentation pipeline."""
        try:
            print("\n" + "="*70)
            print(f"🔬 Analyzing: {self.image_path.name}")
            print("="*70)
            
            # --- Execute each phase sequentially ---
            self._load_and_preprocess()

            # Phase 1: Get a single, strong initial guess
            print("\n--- PHASE 1: Symbiotic Hypothesis Generation ---")
            initial_hypothesis = self._generate_initial_hypothesis()
            if initial_hypothesis is None:
                raise RuntimeError("Failed to generate a valid initial hypothesis.")
            print(f"  - Generated initial hypothesis: Center=({initial_hypothesis['cx']:.1f}, {initial_hypothesis['cy']:.1f}), Radius={initial_hypothesis['r_clad']:.1f}")

            # Phase 2: Refine the center using pixel brightness
            print("\n--- PHASE 2: Pixel-Guided Center Refinement ---")
            refined_center = self._refine_center(initial_hypothesis)
            print(f"  - Refined center to: ({refined_center[0]:.2f}, {refined_center[1]:.2f})")

            # Phase 3: Use the refined center to find boundaries with a model
            print("\n--- PHASE 3: Model-Based Boundary Identification ---")
            final_radii = self._find_boundaries_with_model(refined_center)
            if final_radii is None:
                raise RuntimeError("Failed to identify core and cladding boundaries from radial profile.")
            print(f"  - Identified final boundaries: Core Radius={final_radii['r_core']:.2f}, Cladding Radius={final_radii['r_cladding']:.2f}")

            # Phase 4: Assemble final parameters and report
            print("\n--- PHASE 4: Finalizing and Reporting ---")
            self.final_params = {
                'cx': refined_center[0], 'cy': refined_center[1],
                'r_core': final_radii['r_core'], 'r_cladding': final_radii['r_cladding']
            }
            self._finalize_and_report()
            
            print("\n" + "="*70)
            print(f"✅ Analysis Complete for: {self.image_path.name}")
            print("="*70)
            return True

        except Exception as e:
            print(f"\n❌ FATAL ERROR during processing of {self.image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_and_preprocess(self):
        """Loads, converts, and prepares the image for analysis."""
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")

        self.height, self.width = self.original_image.shape[:2]
        
        # Use a blurred image for most operations to reduce noise
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 1.5)
        
        # Extract Canny edge points from the blurred image
        edges = canny(blurred / 255.0, sigma=1, low_threshold=0.1, high_threshold=0.2)
        self.edge_points = np.argwhere(edges).astype(float)[:, ::-1] # (x, y) format
        print(f"  - Image loaded ({self.width}x{self.height}), extracted {len(self.edge_points)} edge points.")

    def _generate_initial_hypothesis(self) -> Optional[Dict[str, float]]:
        """
        Uses a symbiotic RANSAC approach to find the best single initial guess.
        Scores hypotheses based on both geometric and pixel-value evidence.
        """
        points = self.edge_points
        if len(points) < 50: return None

        best_score = -1
        best_hypothesis = None
        
        # Iterate to find the best possible geometric and pixel-wise circle
        for _ in range(1500): # More iterations for higher robustness
            # 1. Geometric hypothesis from 3 random edge points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[sample_indices]
            
            # Circumcenter calculation
            D = 2 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            if abs(D) < 1e-6: continue
            ux = ((p1[0]**2+p1[1]**2)*(p2[1]-p3[1]) + (p2[0]**2+p2[1]**2)*(p3[1]-p1[1]) + (p3[0]**2+p3[1]**2)*(p1[1]-p2[1])) / D
            cy = ((p1[0]**2+p1[1]**2)*(p3[0]-p2[0]) + (p2[0]**2+p2[1]**2)*(p1[0]-p3[0]) + (p3[0]**2+p3[1]**2)*(p2[0]-p1[0])) / D
            
            if not (0 <= ux < self.width and 0 <= cy < self.height): continue

            # 2. Score the hypothesis
            # Geometric score: How many other edge points support this circle?
            distances = np.linalg.norm(points - [ux, cy], axis=1)
            # Find the most prominent radius from the edge points
            hist, bin_edges = np.histogram(distances, bins=100, range=(0, self.width / 2))
            if np.sum(hist) == 0: continue
            
            strongest_radius_idx = np.argmax(hist)
            r_hyp = bin_edges[strongest_radius_idx] + (bin_edges[1] - bin_edges[0]) / 2
            geometric_score = np.max(hist)

            # Pixel-value score: How "peaky" is the gradient profile for this hypothesis?
            _, grad_profile = get_radial_profiles(self.gray_image, (ux, cy))
            if len(grad_profile) < 20: continue
            
            # Find all prominent gradient peaks
            peaks, props = find_peaks(grad_profile, prominence=(np.std(grad_profile)), distance=15)
            pixel_score = np.sum(props['prominences']) if len(peaks) > 0 else 0

            # 3. Symbiotic final score
            # We want a circle that is geometrically sound AND has sharp gradient transitions
            total_score = geometric_score * pixel_score

            if total_score > best_score:
                best_score = total_score
                best_hypothesis = {'cx': ux, 'cy': cy, 'r_clad': r_hyp}
        
        return best_hypothesis

    def _refine_center(self, hypothesis: Dict[str, float]) -> Tuple[float, float]:
        """
        Refines the center by finding the location within a search window that
        maximizes the brightness of the enclosed core.
        """
        cx, cy = int(hypothesis['cx']), int(hypothesis['cy'])
        r_clad = hypothesis['r_clad']
        
        # A plausible core radius is a fraction of the cladding radius
        r_core_guess = r_clad * 0.5 
        
        # Search in a small window around the initial guess
        window_size = 15 # Search +/- 15 pixels
        best_score = -1
        best_center = (cx, cy)
        
        y_grid, x_grid = np.ogrid[:self.height, :self.width]

        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                temp_cx, temp_cy = cx + dx, cy + dy
                
                # Create a circular mask for the potential core
                dist_sq = (x_grid - temp_cx)**2 + (y_grid - temp_cy)**2
                core_mask = (dist_sq <= r_core_guess**2)
                
                # Score is the mean brightness within the mask
                # A good center will perfectly frame the bright core
                score = np.mean(self.gray_image[core_mask]) if np.any(core_mask) else 0
                
                if score > best_score:
                    best_score = score
                    best_center = (temp_cx, temp_cy)

        return best_center
        
    def _find_boundaries_with_model(self, center: Tuple[float, float]) -> Optional[Dict[str, float]]:
        """
        Analyzes the radial profiles using a physical model of the fiber to
        robustly identify core and cladding boundaries.
        """
        intensity_profile, gradient_profile = get_radial_profiles(self.gray_image, center)
        
        # Smooth the gradient profile for robust peak finding
        if len(gradient_profile) < 21: return None
        grad_smooth = savgol_filter(gradient_profile, 21, 3)

        # Find ALL significant gradient peaks (our boundary candidates)
        peaks, props = find_peaks(grad_smooth, prominence=(np.std(grad_smooth)*0.5), distance=10)
        if len(peaks) < 2:
            print("  - Warning: Could not find at least two significant gradient peaks.")
            return None

        # --- Model-Based Interpretation ---
        best_pair = None
        highest_score = -1

        # Iterate through all pairs of peaks to find the one that best fits our model
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                r1 = peaks[i]
                r2 = peaks[j]
                
                # Model constraint 1: Core must be bright
                # The average intensity inside the inner radius must be high
                avg_intensity_core = np.mean(intensity_profile[0:r1])
                
                # Model constraint 2: Cladding must be darker than core
                avg_intensity_cladding = np.mean(intensity_profile[r1:r2])
                
                # A simple scoring function based on our model
                # Score is high if core is bright and there's a large drop to the cladding
                score = (avg_intensity_core - avg_intensity_cladding) * (props['prominences'][i] + props['prominences'][j])
                
                if score > highest_score:
                    highest_score = score
                    best_pair = {'r_core': float(r1), 'r_cladding': float(r2)}
        
        return best_pair

    def _finalize_and_report(self):
        """Assembles the final report data."""
        self.report_data = {
            'source_file': str(self.image_path),
            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'image_info': {'width': self.width, 'height': self.height},
            'final_parameters': self.final_params
        }

    def save_outputs(self, output_dir: Path):
        """Saves all visual and data outputs."""
        print("\n--- Generating and Saving All Outputs ---")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.image_path.stem}_{timestamp}"
        
        # 1. Generate Masks and Regions
        params = self.final_params
        y, x = np.mgrid[:self.height, :self.width]
        dist_sq = (x - params['cx'])**2 + (y - params['cy'])**2
        
        core_mask = (dist_sq <= params['r_core']**2).astype(np.uint8) * 255
        cladding_mask = ((dist_sq > params['r_core']**2) & (dist_sq <= params['r_cladding']**2)).astype(np.uint8) * 255
        ferrule_mask = cv2.bitwise_not(cv2.circle(np.zeros_like(core_mask), (int(params['cx']), int(params['cy'])), int(params['r_cladding']), 255, -1))
        
        regions = {
            'core': cv2.bitwise_and(self.original_image, self.original_image, mask=core_mask),
            'cladding': cv2.bitwise_and(self.original_image, self.original_image, mask=cladding_mask),
            'ferrule': cv2.bitwise_and(self.original_image, self.original_image, mask=ferrule_mask)
        }
        
        # 2. Save Segmented Images
        for name, image in regions.items():
            path = output_dir / f"{base_name}_region_{name}.png"
            cv2.imwrite(str(path), image)
            print(f"  - Saved region: {path.name}")
            
        # 3. Save Data Report
        json_path = output_dir / f"{base_name}_report.json"
        with open(json_path, 'w') as f:
            json.dump(self.report_data, f, cls=NumpyEncoder, indent=4)
        print(f"  - Saved data report: {json_path.name}")
        
        # 4. Save Diagnostic and Summary Plots
        self._save_diagnostic_plots(output_dir / f"{base_name}_diagnostics.png")
        print(f"  - Saved diagnostic plots: {base_name}_diagnostics.png")

    def _save_diagnostic_plots(self, save_path: Path):
        """Saves summary and radial plots in a single figure."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 14))
        fig.suptitle(f"Hybrid Analysis for {self.image_path.name}", fontsize=16)

        # --- a) Final Summary Image ---
        ax = axes[0, 0]
        vis_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        params = self.final_params
        center_pt = (int(params['cx']), int(params['cy']))
        ax.imshow(vis_img)
        ax.add_artist(plt.Circle(center_pt, params['r_core'], color='lime', fill=False, linewidth=2, label='Core'))
        ax.add_artist(plt.Circle(center_pt, params['r_cladding'], color='cyan', fill=False, linewidth=2, label='Cladding'))
        ax.scatter(center_pt[0], center_pt[1], c='red', s=40, marker='+')
        ax.set_title("Final Segmentation")
        ax.axis('off')

        # --- b) Edge Points Used ---
        ax = axes[0, 1]
        ax.imshow(self.gray_image, cmap='gray')
        ax.scatter(self.edge_points[:, 0], self.edge_points[:, 1], s=1, c='red', alpha=0.1)
        ax.set_title("Canny Edge Points Used for Analysis")
        ax.axis('off')

        # --- c) & d) Radial Profiles ---
        intensity_p, gradient_p = get_radial_profiles(self.gray_image, (params['cx'], params['cy']))
        axes[1, 0].plot(intensity_p, color='dodgerblue')
        axes[1, 0].set_title('Radial Intensity Profile')
        axes[1, 0].set_ylabel("Average Intensity")
        axes[1, 0].grid(True, linestyle=':')
        
        axes[1, 1].plot(gradient_p, color='orangered')
        axes[1, 1].set_title('Radial Gradient Profile')
        axes[1, 1].set_ylabel("Average Gradient Magnitude")
        axes[1, 1].grid(True, linestyle=':')

        for ax in [axes[1, 0], axes[1, 1]]:
            ax.axvline(params['r_core'], color='lime', linestyle='--', label=f"Core: {params['r_core']:.1f}px")
            ax.axvline(params['r_cladding'], color='cyan', linestyle='--', label=f"Clad: {params['r_cladding']:.1f}px")
            ax.set_xlabel("Radius from Center (pixels)")
            ax.legend()
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# =============================================================================
# COMMAND-LINE INTERFACE (CLI) & MAIN EXECUTION
# =============================================================================

def main():
    """Main function with interactive command-line workflow."""
    print("\n" + "="*80)
    print(" Welcome to the Hybrid Geometric-Pixel Fiber Analyzer (v2.0) ".center(80, "="))
    print("="*80)
    
    while True:
        try:
            paths_str = input("Enter path(s) to image(s) (use quotes for paths with spaces):\n> ").strip()
            if not paths_str:
                print("No paths entered. Exiting.")
                break
                
            image_paths = [Path(p) for p in shlex.split(paths_str)]
            
            # Filter for existing files
            valid_paths = [p for p in image_paths if p.is_file()]
            if len(valid_paths) != len(image_paths):
                print("Warning: One or more paths were not valid files and will be skipped.")
            
            if not valid_paths:
                print("No valid image files to process.")
                continue

            output_dir_str = input("Enter output directory (default: 'hybrid_analysis_results'): ").strip()
            output_dir = Path(output_dir_str) if output_dir_str else Path("hybrid_analysis_results")

            for img_path in valid_paths:
                analyzer = HybridFiberAnalyzer(img_path)
                if analyzer.run_full_pipeline():
                    analyzer.save_outputs(output_dir)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting program.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*80)
        if input("Analyze more images? (y/n): ").lower().strip() != 'y':
            break

    print("\n" + "="*80)
    print(" All processing complete. Goodbye! ".center(80, "="))
    print("="*80)


if __name__ == '__main__':
    main()
