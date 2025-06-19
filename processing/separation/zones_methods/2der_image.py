import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from tkinter import filedialog, Tk
import json
import os
import glob
import warnings

# Suppress runtime warnings from calculations
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# START OF FIX: Add a custom JSON encoder to handle NumPy data types
# ==============================================================================
class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder for numpy data types.
    This teaches the json.dump function how to handle numeric types from NumPy.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# ==============================================================================
# END OF FIX
# ==============================================================================


class AdvancedUnifiedFiberSeparator:
    """
    Advanced Unified Fiber Optic Region Separator.

    This class integrates multiple analysis methods to achieve robust segmentation
    of fiber optic end-face images. It uses a consensus-based approach to
    determine the final core and cladding boundaries, making it resilient to
    defects, artifacts, and variations in image quality.
    """

    def __init__(self, image_path, priors=None):
        """
        Initialize the separator with the image and optional priors.
        """
        self.image_path = image_path
        self.priors = priors if priors is not None else {}
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # Apply physical constraints, using priors if available
        self.min_core_radius = self.priors.get('avg_min_core_radius', 15)
        self.min_cladding_thickness = self.priors.get('avg_min_cladding_thickness', 15)
        avg_clad_ratio = self.priors.get('avg_cladding_radius_ratio', 0.8)
        self.max_cladding_radius = min(self.width, self.height) * 0.5 * avg_clad_ratio

        print(f"Loaded image: {self.width}x{self.height} pixels")
        if self.priors:
            print("Applied learned priors to analysis parameters.")

    def analyze(self):
        """
        Run the complete unified analysis pipeline.
        """
        print("\n" + "="*70)
        print("ADVANCED UNIFIED FIBER OPTIC SEPARATOR")
        print("="*70)

        self.find_robust_center()
        self.create_preprocessing_variants()
        self.run_all_analyses()
        self.establish_consensus_boundaries()
        self.create_final_masks()
        self.apply_artifact_removal()
        self.extract_regions_and_output()

        print("\nAnalysis Complete!")
        return self.results

    def find_robust_center(self):
        """Find the fiber's center using a weighted average of multiple methods."""
        print("\nStage 1: Finding Robust Center")
        print("-" * 50)
        centers, weights = [], []

        smoothed = cv2.GaussianBlur(self.gray, (11, 11), 5)
        _, bright_mask = cv2.threshold(smoothed, np.percentile(smoothed, 98), 255, cv2.THRESH_BINARY)
        moments = cv2.moments(bright_mask)
        if moments['m00'] > 0:
            cx, cy = moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
            centers.append((cx, cy)); weights.append(2.0)
            print(f"  Brightness centroid: ({cx:.1f}, {cy:.1f})")

        circles = cv2.HoughCircles(
            cv2.GaussianBlur(self.gray, (9, 9), 2), cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=self.width // 2, param1=50, param2=30,
            minRadius=int(self.min_core_radius), maxRadius=int(self.max_cladding_radius)
        )
        if circles is not None:
            cx, cy, _ = circles[0, 0]
            centers.append((cx, cy)); weights.append(1.5)
            print(f"  Hough circle center: ({cx:.1f}, {cy:.1f})")

        if not centers:
            self.center_x, self.center_y = self.width / 2, self.height / 2
        else:
            centers_arr, weights_arr = np.array(centers), np.array(weights)
            self.center_x, self.center_y = np.average(centers_arr[:, 0], weights=weights_arr), np.average(centers_arr[:, 1], weights=weights_arr)

        print(f"\n  Final weighted center: ({self.center_x:.1f}, {self.center_y:.1f})")

    def create_preprocessing_variants(self):
        """Create a list of preprocessed images for robust analysis."""
        print("\nStage 2: Creating Preprocessing Variants")
        print("-" * 50)
        self.preprocessing_variants = [{'name': 'Original', 'image': self.gray.copy()}]
        noise_level = np.std(self.gray - cv2.GaussianBlur(self.gray, (5, 5), 1))
        print(f"  Estimated noise level: {noise_level:.2f}")

        sigmas = [2, 3, 4] if noise_level > 100 else [1, 2, 3]
        for sigma in sigmas:
            self.preprocessing_variants.append({
                'name': f'Gaussian_σ{sigma}',
                'image': cv2.GaussianBlur(self.gray, (0, 0), sigma)
            })
        self.preprocessing_variants.append({
            'name': 'Bilateral',
            'image': cv2.bilateralFilter(self.gray, 9, 75, 75)
        })
        print(f"  Created {len(self.preprocessing_variants)} preprocessing variants.")

    def run_all_analyses(self):
        """Execute all boundary detection methods and collect candidates."""
        print("\nStage 3: Running All Analysis Methods")
        print("-" * 50)
        self.all_boundaries = []

        for variant in self.preprocessing_variants:
            self._analyze_with_derivatives(variant)

        self._analyze_with_contours()
        self._analyze_with_ransac()

        print(f"\n  Collected a total of {len(self.all_boundaries)} boundary candidates.")

    def _analyze_with_derivatives(self, variant):
        """Find boundaries using radial profile derivatives."""
        profiles = self._compute_radial_profiles(variant['image'])
        second_deriv_min = -profiles['second_derivative']
        peaks, props = find_peaks(second_deriv_min, distance=self.min_cladding_thickness, prominence=np.std(second_deriv_min) * 0.1)
        valid_peaks = [p for p in peaks if self.min_core_radius < p < self.max_cladding_radius]

        if len(valid_peaks) >= 2:
            prominences = props['prominences']
            sorted_by_prominence = sorted(zip(valid_peaks, prominences), key=lambda x: x[1], reverse=True)
            top_two = sorted([p[0] for p in sorted_by_prominence[:2]])
            self.all_boundaries.append({'method': '2nd_Derivative', 'boundaries': top_two, 'confidence': 1.0, 'profiles': profiles})
            print(f"    - Derivative Method ({variant['name']}): Found boundaries {top_two}")

    def _analyze_with_contours(self):
        """Find boundaries using adaptive thresholding and contour analysis."""
        _, thresh = cv2.threshold(cv2.GaussianBlur(self.gray, (11, 11), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: return
        
        circles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: continue
            
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter > 0 else 0
            
            if circularity > 0.7 and np.sqrt((x-self.center_x)**2 + (y-self.center_y)**2) < 50:
                 if self.min_core_radius < radius < self.max_cladding_radius:
                    circles.append(radius)
        
        if len(circles) >= 2:
            sorted_radii = sorted(circles)
            self.all_boundaries.append({'method': 'Contours', 'boundaries': [sorted_radii[0], sorted_radii[-1]], 'confidence': 0.8})
            print(f"    - Contour Method: Found boundaries {[sorted_radii[0], sorted_radii[-1]]}")

    def _analyze_with_ransac(self):
        """Find boundaries using RANSAC on Canny edge points."""
        edges = cv2.Canny(cv2.GaussianBlur(self.gray, (5, 5), 1.5), 50, 150)
        edge_points = np.column_stack(np.where(edges.T > 0))

        if len(edge_points) < 50: return

        dists = np.linalg.norm(edge_points - [self.center_x, self.center_y], axis=1)
        hist, bins = np.histogram(dists, bins=100, range=(0, self.max_cladding_radius * 1.2))
        bin_width = bins[1] - bins[0]
        peaks, _ = find_peaks(hist, distance=int(self.min_cladding_thickness / bin_width if bin_width > 0 else 5), prominence=len(edge_points)*0.01)

        if len(peaks) >= 2:
            peak_heights = hist[peaks]
            top_two_indices = np.argsort(peak_heights)[-2:]
            radii = sorted([bins[peaks[i]] + bin_width/2 for i in top_two_indices])
            self.all_boundaries.append({'method': 'RANSAC', 'boundaries': radii, 'confidence': 0.9})
            print(f"    - RANSAC Method: Found boundaries {radii}")

    def establish_consensus_boundaries(self):
        """Use clustering (DBSCAN) to find consensus boundaries from all candidates."""
        print("\nStage 4: Establishing Consensus Boundaries via Clustering")
        print("-" * 50)

        if not self.all_boundaries:
            raise ValueError("No boundaries detected! Image may not be a fiber optic end face or is of very poor quality.")

        inner_radii = np.array([b['boundaries'][0] for b in self.all_boundaries]).reshape(-1, 1)
        outer_radii = np.array([b['boundaries'][1] for b in self.all_boundaries]).reshape(-1, 1)

        def find_cluster_median(data, eps=10, min_samples=2):
            if len(data) < min_samples: return int(np.median(data)) if len(data) > 0 else None
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
            labels = clustering.labels_
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(counts) == 0: return int(np.median(data))
            
            largest_cluster_label = unique_labels[np.argmax(counts)]
            cluster_points = data[labels == largest_cluster_label]
            return int(np.median(cluster_points))

        inner_boundary = find_cluster_median(inner_radii)
        outer_boundary = find_cluster_median(outer_radii)

        if inner_boundary is None or outer_boundary is None or inner_boundary >= outer_boundary:
             raise ValueError("Could not establish a valid consensus for boundaries.")

        if outer_boundary - inner_boundary < self.min_cladding_thickness:
            center = (inner_boundary + outer_boundary) / 2
            inner_boundary = int(center - self.min_cladding_thickness / 2)
            outer_boundary = int(center + self.min_cladding_thickness / 2)
            print("  Warning: Initial consensus boundaries were too close. Enforcing minimum separation.")

        self.consensus_boundaries = [inner_boundary, outer_boundary]
        self.final_radii = {'radius1': float(inner_boundary), 'radius2': float(outer_boundary)}
        self.final_center = {'x': float(self.center_x), 'y': float(self.center_y)}

        print(f"  Consensus Core Boundary: {self.consensus_boundaries[0]} pixels")
        print(f"  Consensus Cladding Boundary: {self.consensus_boundaries[1]} pixels")
        print(f"  Resulting Cladding thickness: {self.consensus_boundaries[1] - self.consensus_boundaries[0]} pixels")

    def create_final_masks(self):
        """Create final segmentation masks using the consensus boundaries."""
        print("\nStage 5: Creating Final Masks")
        print("-" * 50)
        Y, X = np.ogrid[:self.height, :self.width]
        dist_map = np.sqrt((X - self.final_center['x'])**2 + (Y - self.final_center['y'])**2)
        
        self.masks = {
            'core': (dist_map <= self.final_radii['radius1']).astype(np.uint8) * 255,
            'cladding': ((dist_map > self.final_radii['radius1']) & (dist_map <= self.final_radii['radius2'])).astype(np.uint8) * 255,
            'ferrule': (dist_map > self.final_radii['radius2']).astype(np.uint8) * 255
        }
        for name, mask in self.masks.items():
            print(f"  {name.capitalize()} mask created.")

    def apply_artifact_removal(self):
        """Apply morphological operations to clean up the final binary masks."""
        print("\nStage 6: Applying Binary Artifact Removal")
        print("-" * 50)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for name in ['core', 'cladding', 'ferrule']:
            mask = self.masks[name]
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            self.masks[name] = mask
        print("  All masks refined.")
        
    def extract_regions_and_output(self):
        """Extract the final regions and generate all output files and reports."""
        print("\nStage 7: Generating Final Output")
        print("-" * 50)

        output_dir = 'unified_segmentation_results'
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path_prefix = os.path.join(output_dir, base_filename)

        self.regions = {}
        region_stats = {}
        for name, mask in self.masks.items():
            self.regions[name] = cv2.bitwise_and(self.original, self.original, mask=mask)
            pixel_count = int(np.sum(mask > 0))
            region_stats[name] = {
                'pixel_count': pixel_count,
                'percentage': float(100 * pixel_count / (self.width * self.height))
            }
            cv2.imwrite(f"{output_path_prefix}_{name}.png", self.regions[name])
            print(f"  Saved segmented region: {name}.png")

        self.results = {
            'image_info': {'path': self.image_path, 'width': self.width, 'height': self.height},
            'final_center': self.final_center,
            'consensus_boundaries': self.consensus_boundaries,
            'cladding_thickness': self.final_radii['radius2'] - self.final_radii['radius1'],
            'regions': region_stats,
            'boundary_candidates': self.all_boundaries,
            'applied_priors': self.priors
        }
        
        for b in self.results['boundary_candidates']:
            if 'profiles' in b and b['profiles'] is not None:
                for key, value in b['profiles'].items():
                    if isinstance(value, np.ndarray):
                        b['profiles'][key] = value.tolist()

        report_path = f"{output_path_prefix}_analysis_report.json"
        
        # ==============================================================================
        # START OF FIX: Use the custom NumpyEncoder when dumping to JSON
        # ==============================================================================
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=4, cls=NumpyEncoder)
        # ==============================================================================
        # END OF FIX
        # ==============================================================================
            
        print(f"  Saved analysis report: {os.path.basename(report_path)}")

        self._create_visualization(output_path_prefix)
        print(f"  Saved diagnostic visualization: {os.path.basename(output_path_prefix)}_visualization.png")

    def _create_visualization(self, output_path_prefix):
        """Create a comprehensive visualization of the analysis results."""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(self.final_center['x'] + self.final_radii['radius1'] * np.cos(theta), self.final_center['y'] + self.final_radii['radius1'] * np.sin(theta), 'lime', linewidth=2)
        ax1.plot(self.final_center['x'] + self.final_radii['radius2'] * np.cos(theta), self.final_center['y'] + self.final_radii['radius2'] * np.sin(theta), 'cyan', linewidth=2)
        ax1.set_title('Final Consensus Boundaries')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        mask_overlay = np.zeros_like(self.original)
        mask_overlay[self.masks['core'] > 0] = [255, 0, 0]
        mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0]
        ax2.imshow(mask_overlay)
        ax2.set_title('Final Region Masks')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        composite = cv2.bitwise_and(self.original, self.original, mask=(self.masks['core'] | self.masks['cladding']))
        ax3.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        ax3.set_title('Segmented Core & Cladding')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, :])
        rep_profile = next((b['profiles'] for b in self.all_boundaries if 'profiles' in b and b['profiles']), None)
        if rep_profile:
            ax4.plot(rep_profile['radii'], rep_profile['intensity_smooth'], label='Smoothed Intensity', color='blue')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(rep_profile['radii'], rep_profile['gradient'], label='Gradient', color='orange', alpha=0.7)
            ax4.set_xlabel('Radius from Center (pixels)')
            ax4.set_ylabel('Intensity', color='blue')
            ax4_twin.set_ylabel('Gradient Magnitude', color='orange')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
        
        ax4.set_title('Radial Profile Analysis')
        ax4.axvline(x=self.consensus_boundaries[0], color='lime', linestyle='--', linewidth=2, label=f'Core Boundary ({self.consensus_boundaries[0]}px)')
        ax4.axvline(x=self.consensus_boundaries[1], color='cyan', linestyle='--', linewidth=2, label=f'Cladding Boundary ({self.consensus_boundaries[1]}px)')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_visualization.png", dpi=150)
        plt.close()

    def _compute_radial_profiles(self, img):
        """Compute average radial profiles for intensity and its derivatives."""
        max_radius = int(min(self.center_x, self.center_y, self.width - self.center_x, self.height - self.center_y)) - 1
        if max_radius <= 0: return None
        
        y, x = np.ogrid[:self.height, :self.width]
        r = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        r_int = r.round().astype(int)
        
        pixel_counts = np.bincount(r_int.ravel())
        pixel_sums = np.bincount(r_int.ravel(), weights=img.ravel())
        
        radial_mean = pixel_sums / pixel_counts.clip(1)
        radial_mean = radial_mean[:max_radius]

        smooth_intensity = gaussian_filter1d(radial_mean, sigma=2)
        gradient = np.gradient(smooth_intensity)
        second_derivative = np.gradient(gradient)
        
        return {
            'radii': np.arange(len(radial_mean)),
            'intensity_smooth': smooth_intensity,
            'gradient': np.abs(gradient),
            'second_derivative': second_derivative
        }


def create_dataset_entry(image_path, dataset_path):
    """Analyzes a single image and saves its report to the dataset directory."""
    print(f"\nAnalyzing '{os.path.basename(image_path)}' to add to dataset...")
    try:
        separator = AdvancedUnifiedFiberSeparator(image_path, priors=None)
        separator.analyze()
        
        report_filename = os.path.splitext(os.path.basename(image_path))[0] + "_report.json"
        report_path = os.path.join(dataset_path, report_filename)
        
        with open(report_path, 'w') as f:
            # Use the custom encoder here as well
            json.dump(separator.results, f, indent=4, cls=NumpyEncoder)
        print(f"Successfully added analysis report to dataset: {report_path}")

    except Exception as e:
        print(f"An error occurred while creating dataset entry for {image_path}: {e}")
        import traceback
        traceback.print_exc()

def load_dataset_priors(dataset_path):
    """Loads all analysis reports from a dataset and computes statistical priors."""
    print(f"\nLoading dataset from: {dataset_path}")
    json_files = glob.glob(os.path.join(dataset_path, "*_report.json"))
    
    if not json_files:
        print("Warning: No analysis reports found in the dataset directory. Using default parameters.")
        return {}

    all_priors = {'cladding_radius_ratios': [], 'cladding_thicknesses': []}

    for f_path in json_files:
        with open(f_path, 'r') as f:
            try:
                data = json.load(f)
                boundaries = data.get('consensus_boundaries')
                img_info = data.get('image_info')
                if boundaries and img_info and len(boundaries) == 2:
                    r_core, r_clad = boundaries
                    width = img_info['width']
                    all_priors['cladding_radius_ratios'].append(r_clad / width)
                    all_priors['cladding_thicknesses'].append(r_clad - r_core)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {f_path}. Skipping.")
    
    if not all_priors['cladding_radius_ratios']:
        print("Warning: Dataset reports were invalid or empty. Using default parameters.")
        return {}
        
    priors = {
        'avg_cladding_radius_ratio': np.median(all_priors['cladding_radius_ratios']),
        'avg_min_cladding_thickness': np.percentile(all_priors['cladding_thicknesses'], 25),
        'avg_min_core_radius': np.percentile(all_priors['cladding_thicknesses'], 25)
    }
    
    print("Successfully loaded priors from dataset:")
    print(f"  - Average Cladding Radius Ratio: {priors['avg_cladding_radius_ratio']:.3f}")
    print(f"  - Robust Min Cladding Thickness: {priors['avg_min_cladding_thickness']:.1f} px")
    return priors


def main():
    """Main function to drive the user interface and analysis."""
    root = Tk()
    root.withdraw()

    print("--- Advanced Fiber Optic Segmentation with Dataset Learning ---")
    choice = input(
        "Choose an option:\n"
        "  [1] Create a new dataset\n"
        "  [2] Add images to an existing dataset\n"
        "  [3] Analyze an image using a dataset\n"
        "Enter your choice (1, 2, or 3): "
    )

    if choice in ['1', '2']:
        title = "Select Directory for New Dataset" if choice == '1' else "Select Existing Dataset Directory"
        dataset_path = filedialog.askdirectory(title=title)
        if not dataset_path:
            print("Operation cancelled.")
            return

        if choice == '1' and os.listdir(dataset_path):
             if input("Warning: Directory is not empty. Continue? (y/n): ").lower() != 'y':
                return

        image_paths = filedialog.askopenfilenames(title="Select Images for Dataset", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not image_paths:
            print("Operation cancelled.")
            return
            
        for img_path in image_paths:
            # The create_dataset_entry function now handles exceptions internally
            try:
                separator = AdvancedUnifiedFiberSeparator(img_path, priors=None)
                # We need to call analyze and then save the report from its results
                results = separator.analyze()
                report_filename = os.path.splitext(os.path.basename(img_path))[0] + "_report.json"
                report_path = os.path.join(dataset_path, report_filename)
                with open(report_path, 'w') as f:
                    json.dump(results, f, indent=4, cls=NumpyEncoder)
                print(f"Successfully created and saved analysis report to dataset: {report_filename}")
            except Exception as e:
                 print(f"FATAL ERROR processing {os.path.basename(img_path)}: {e}")
        print("\nDataset processing complete.")

    elif choice == '3':
        image_path = filedialog.askopenfilename(title="Select Image to Analyze", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not image_path:
            print("Operation cancelled.")
            return

        dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        priors = load_dataset_priors(dataset_path) if dataset_path else {}

        try:
            separator = AdvancedUnifiedFiberSeparator(image_path, priors)
            separator.analyze()
        except Exception as e:
            print(f"\n--- An error occurred during analysis ---")
            print(f"Error: {e}")
            print("This can happen with very unusual or low-quality images.")
            import traceback
            traceback.print_exc()

    else:
        print("Invalid choice. Please run the script again.")

if __name__ == '__main__':
    main()