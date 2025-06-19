import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')
import json
import os

class CurvatureFiberAnalyzer:
    """
    Fiber optic region detection based on intensity curve inflection points
    Finds where the intensity starts to "cave in" using second derivative analysis
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        print(f"Loaded image: {self.width}x{self.height} pixels")
    
    def analyze(self):
        """Run analysis"""
        print("\nStarting Curvature-Based Fiber Analysis...")
        print("=" * 60)
        
        # Step 1: Find center
        self.find_center()
        
        # Step 2: Compute radial intensity profile
        self.compute_radial_intensity_profile()
        
        # Step 3: Find gradient peaks and inflection points
        self.find_boundaries()
        
        # Step 4: Define regions based on inflection points
        self.define_regions()
        
        # Step 5: Create masks
        self.create_masks()
        
        # Step 6: Extract regions
        self.extract_regions()
        
        # Step 7: Generate visualizations and report
        self.generate_output()
        
        print("\nAnalysis Complete!")
        return self.results
    
    def find_center(self):
        """Find image center using brightest region"""
        print("\nStep 1: Finding center...")
        
        # Apply some smoothing
        smoothed = cv2.GaussianBlur(self.gray, (5, 5), 1)
        
        # Find brightest region
        threshold = np.percentile(smoothed, 95)
        bright_mask = smoothed > threshold
        
        # Get centroid
        moments = cv2.moments(bright_mask.astype(np.uint8))
        if moments['m00'] > 0:
            self.center_x = moments['m10'] / moments['m00']
            self.center_y = moments['m01'] / moments['m00']
        else:
            self.center_x = self.width // 2
            self.center_y = self.height // 2
        
        print(f"  Center: ({self.center_x:.1f}, {self.center_y:.1f})")
    
    def compute_radial_intensity_profile(self):
        """Compute radial intensity profile"""
        print("\nStep 2: Computing radial intensity profile...")
        
        # Maximum radius
        max_radius = int(min(self.center_x, self.center_y, 
                            self.width - self.center_x, 
                            self.height - self.center_y))
        
        self.radii = np.arange(max_radius)
        
        # Sample along multiple angles
        num_angles = 360
        angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        intensity_profiles = []
        
        for angle in angles:
            # Points along this radial line
            x_coords = self.center_x + self.radii * np.cos(angle)
            y_coords = self.center_y + self.radii * np.sin(angle)
            
            # Sample intensities
            intensities = []
            for x, y in zip(x_coords, y_coords):
                if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                    intensities.append(self.gray[int(y), int(x)])
            
            if len(intensities) == len(self.radii):
                intensity_profiles.append(intensities)
        
        # Compute median profile (robust to outliers)
        self.intensity_profile = np.median(intensity_profiles, axis=0)
        
        # Apply smoothing for derivative calculations
        self.intensity_profile_smooth = gaussian_filter1d(self.intensity_profile, sigma=2)
        
        print(f"  Profile computed for {len(self.radii)} radial positions")
    
    def find_boundaries(self):
        """Find boundaries using gradient peaks and curvature analysis"""
        print("\nStep 3: Finding boundaries using curvature analysis...")
        
        # First derivative (gradient)
        self.first_derivative = np.gradient(self.intensity_profile_smooth)
        
        # Second derivative (curvature)
        self.second_derivative = np.gradient(self.first_derivative)
        
        # Find gradient peaks (for reference)
        gradient_magnitude = np.abs(self.first_derivative)
        peaks, _ = find_peaks(gradient_magnitude, distance=10)
        
        if len(peaks) == 0:
            raise ValueError("No gradient peaks found!")
        
        # Sort peaks by magnitude
        peak_magnitudes = gradient_magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        
        # Get the two highest peaks
        if len(peaks) >= 2:
            self.gradient_peaks = sorted([peaks[sorted_indices[0]], peaks[sorted_indices[1]]])
        else:
            self.gradient_peaks = [peaks[sorted_indices[0]]]
        
        print(f"  Gradient peaks at: {self.gradient_peaks} pixels")
        
        # Find inflection points around the peaks
        self.boundaries = self._find_inflection_boundaries()
        
        print(f"  Cladding boundaries at: {self.boundaries} pixels")
    
    def _find_inflection_boundaries(self):
        """Find where the curve starts to cave in before/after gradient peaks"""
        boundaries = []
        
        # For the first peak, find where curvature starts changing significantly before it
        if len(self.gradient_peaks) >= 1:
            first_peak = self.gradient_peaks[0]
            
            # Look backwards from the peak
            start_search = max(0, first_peak - 30)  # Search window
            search_region = self.second_derivative[start_search:first_peak]
            
            if len(search_region) > 0:
                # Find where second derivative starts changing significantly
                # This is where the curve starts to "cave in"
                threshold = np.std(self.second_derivative) * 0.1
                
                # Find the last point before the peak where curvature is still small
                for i in range(len(search_region) - 1, -1, -1):
                    if abs(search_region[i]) < threshold:
                        boundaries.append(start_search + i)
                        break
                else:
                    # If not found, use a point before the peak
                    boundaries.append(max(0, first_peak - 5))
            else:
                boundaries.append(max(0, first_peak - 5))
        
        # For the second peak, find where curvature settles after it
        if len(self.gradient_peaks) >= 2:
            second_peak = self.gradient_peaks[1]
            
            # Look forward from the peak
            end_search = min(len(self.second_derivative), second_peak + 30)
            search_region = self.second_derivative[second_peak:end_search]
            
            if len(search_region) > 0:
                # Find where second derivative settles down
                threshold = np.std(self.second_derivative) * 0.1
                
                # Find the first point after the peak where curvature becomes small
                for i in range(len(search_region)):
                    if abs(search_region[i]) < threshold:
                        boundaries.append(second_peak + i)
                        break
                else:
                    # If not found, use a point after the peak
                    boundaries.append(min(len(self.radii) - 1, second_peak + 5))
            else:
                boundaries.append(min(len(self.radii) - 1, second_peak + 5))
        
        return boundaries
    
    def define_regions(self):
        """Define regions based on inflection boundaries"""
        print("\nStep 4: Defining regions based on inflection points...")
        
        self.regions = {}
        
        if len(self.boundaries) >= 2:
            # Standard case: two boundaries found
            boundary1, boundary2 = self.boundaries
            
            # Core: inside first boundary
            self.regions['core'] = {
                'start': 0,
                'end': boundary1,
                'name': 'core'
            }
            
            # Cladding: between the boundaries
            self.regions['cladding'] = {
                'start': boundary1,
                'end': boundary2,
                'name': 'cladding'
            }
            
            # Note: Ferrule will be defined in create_masks as everything else
            
        else:
            # Fallback if boundaries not properly detected
            print("  Warning: Could not detect proper boundaries, using gradient peaks")
            if len(self.gradient_peaks) >= 2:
                self.regions['core'] = {
                    'start': 0,
                    'end': self.gradient_peaks[0],
                    'name': 'core'
                }
                
                self.regions['cladding'] = {
                    'start': self.gradient_peaks[0],
                    'end': self.gradient_peaks[1],
                    'name': 'cladding'
                }
        
        # Analyze intensity in each region
        for region_name, region in self.regions.items():
            start = region['start']
            end = region['end']
            
            if end > start:
                region_intensity = self.intensity_profile[start:end+1]
                region['mean_intensity'] = np.mean(region_intensity)
                region['std_intensity'] = np.std(region_intensity)
                region['median_intensity'] = np.median(region_intensity)
                
                print(f"  {region_name}: radius {start}-{end}px, "
                      f"intensity={region['mean_intensity']:.1f}±{region['std_intensity']:.1f}")
    
    def create_masks(self):
        """Create masks for each region"""
        print("\nStep 5: Creating region masks...")
        
        # Create distance map
        Y, X = np.ogrid[:self.height, :self.width]
        distance_map = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        self.masks = {}
        
        # Core mask (circular region)
        if 'core' in self.regions:
            core_radius = self.radii[self.regions['core']['end']]
            self.masks['core'] = (distance_map <= core_radius).astype(np.uint8) * 255
        
        # Cladding mask (annular region)
        if 'cladding' in self.regions:
            inner_radius = self.radii[self.regions['cladding']['start']]
            outer_radius = self.radii[self.regions['cladding']['end']]
            self.masks['cladding'] = ((distance_map > inner_radius) & 
                                     (distance_map <= outer_radius)).astype(np.uint8) * 255
        
        # Ferrule mask (everything else - not limited to any radius)
        # This is the entire rest of the image
        ferrule_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        # Remove core and cladding regions from ferrule
        if 'core' in self.masks:
            ferrule_mask[self.masks['core'] > 0] = 0
        if 'cladding' in self.masks:
            ferrule_mask[self.masks['cladding'] > 0] = 0
        
        self.masks['ferrule'] = ferrule_mask
        
        # Print pixel counts
        for region_name, mask in self.masks.items():
            pixel_count = np.sum(mask > 0)
            print(f"  {region_name}: {pixel_count} pixels")
    
    def extract_regions(self):
        """Extract regions using masks"""
        print("\nStep 6: Extracting regions...")
        
        self.extracted_regions = {}
        
        for region_name, mask in self.masks.items():
            # Extract region
            region = cv2.bitwise_and(self.original, self.original, mask=mask)
            self.extracted_regions[region_name] = region
    
    def generate_output(self, output_dir='curvature_analysis_results'):
        """Generate visualizations and save results"""
        print("\nStep 7: Generating output...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save extracted regions
        for region_name, region in self.extracted_regions.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}.png'), region)
        
        # Save masks
        for region_name, mask in self.masks.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}_mask.png'), mask)
        
        # Create visualizations
        self._create_visualizations(output_dir)
        
        # Generate report
        self.results = {
            'center': {'x': float(self.center_x), 'y': float(self.center_y)},
            'gradient_peaks': [int(p) for p in self.gradient_peaks],
            'cladding_boundaries': [int(b) for b in self.boundaries],
            'regions': {}
        }
        
        # Add region info
        for region_name, mask in self.masks.items():
            if region_name in self.regions:
                region = self.regions[region_name]
                self.results['regions'][region_name] = {
                    'radial_range': [region['start'], region['end']],
                    'mean_intensity': float(region['mean_intensity']),
                    'std_intensity': float(region['std_intensity']),
                    'median_intensity': float(region['median_intensity']),
                    'pixel_count': int(np.sum(mask > 0))
                }
            else:
                # For ferrule
                self.results['regions'][region_name] = {
                    'pixel_count': int(np.sum(mask > 0))
                }
        
        # Save JSON report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")
    
    def _create_visualizations(self, output_dir):
        """Create visualization plots"""
        # 1. Comprehensive analysis plot
        fig, axes = plt.subplots(4, 1, figsize=(10, 14))
        fig.suptitle('Curvature-Based Fiber Analysis', fontsize=16)
        
        # Intensity profile
        ax1 = axes[0]
        ax1.plot(self.radii, self.intensity_profile, 'b-', linewidth=2, label='Intensity')
        ax1.set_ylabel('Pixel Intensity')
        ax1.set_title('Radial Intensity Profile')
        ax1.grid(True, alpha=0.3)
        
        # Mark boundaries
        if len(self.boundaries) >= 2:
            ax1.axvline(x=self.boundaries[0], color='g', linestyle='--', linewidth=2,
                       label=f'Start of cladding ({self.boundaries[0]}px)')
            ax1.axvline(x=self.boundaries[1], color='r', linestyle='--', linewidth=2,
                       label=f'End of cladding ({self.boundaries[1]}px)')
        ax1.legend()
        
        # First derivative (gradient)
        ax2 = axes[1]
        ax2.plot(self.radii, self.first_derivative, 'orange', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('dI/dr')
        ax2.set_title('First Derivative (Gradient)')
        ax2.grid(True, alpha=0.3)
        
        # Mark gradient peaks
        for peak in self.gradient_peaks:
            ax2.axvline(x=peak, color='red', linestyle=':', linewidth=1.5)
            ax2.plot(peak, self.first_derivative[peak], 'ro', markersize=8)
        
        # Second derivative (curvature)
        ax3 = axes[2]
        ax3.plot(self.radii, self.second_derivative, 'green', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('d²I/dr²')
        ax3.set_title('Second Derivative (Curvature)')
        ax3.grid(True, alpha=0.3)
        
        # Mark inflection boundaries
        if len(self.boundaries) >= 2:
            ax3.axvline(x=self.boundaries[0], color='g', linestyle='--', linewidth=2)
            ax3.axvline(x=self.boundaries[1], color='r', linestyle='--', linewidth=2)
        
        # Combined view with shaded regions
        ax4 = axes[3]
        ax4.plot(self.radii, self.intensity_profile, 'b-', linewidth=2)
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('Pixel Intensity')
        ax4.set_title('Identified Regions')
        ax4.grid(True, alpha=0.3)
        
        # Shade regions
        alpha = 0.3
        if 'core' in self.regions:
            r = self.regions['core']
            ax4.axvspan(r['start'], r['end'], alpha=alpha, color='red', label='Core')
        
        if 'cladding' in self.regions:
            r = self.regions['cladding']
            ax4.axvspan(r['start'], r['end'], alpha=alpha, color='green', label='Cladding')
            
            # Annotate cladding region
            mid_point = (r['start'] + r['end']) / 2
            ax4.annotate('CLADDING', xy=(mid_point, ax4.get_ylim()[0]), 
                        xytext=(mid_point, ax4.get_ylim()[0] + 5),
                        ha='center', fontsize=10, fontweight='bold')
        
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'curvature_analysis.png'), dpi=150)
        plt.close()
        
        # 2. Results visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Region Detection Results', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Mark center
        axes[0, 0].plot(self.center_x, self.center_y, 'r+', markersize=10, markeredgewidth=2)
        
        # Combined masks
        mask_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if 'core' in self.masks:
            mask_overlay[self.masks['core'] > 0] = [255, 0, 0]  # Red
        if 'cladding' in self.masks:
            mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0]  # Green
        if 'ferrule' in self.masks:
            mask_overlay[self.masks['ferrule'] > 0] = [0, 0, 255]  # Blue
        
        axes[0, 1].imshow(mask_overlay)
        axes[0, 1].set_title('Region Masks (R=Core, G=Cladding, B=Ferrule)')
        axes[0, 1].axis('off')
        
        # Cladding region highlight
        if 'cladding' in self.extracted_regions:
            axes[1, 0].imshow(cv2.cvtColor(self.extracted_regions['cladding'], cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Extracted Cladding Region')
            axes[1, 0].axis('off')
        
        # Ferrule region (showing it includes corners)
        if 'ferrule' in self.extracted_regions:
            axes[1, 1].imshow(cv2.cvtColor(self.extracted_regions['ferrule'], cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('Ferrule Region (entire background)')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results.png'), dpi=150)
        plt.close()
        
        # 3. Inflection point visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot intensity with shaded cladding
        ax.plot(self.radii, self.intensity_profile, 'b-', linewidth=2, label='Intensity')
        
        if len(self.boundaries) >= 2:
            # Shade cladding region
            ax.axvspan(self.boundaries[0], self.boundaries[1], 
                      alpha=0.2, color='green', label='Cladding Region')
            
            # Mark inflection points
            ax.axvline(x=self.boundaries[0], color='g', linestyle='--', linewidth=2,
                      label='Start (inflection)')
            ax.axvline(x=self.boundaries[1], color='r', linestyle='--', linewidth=2,
                      label='End (inflection)')
            
            # Mark gradient peaks for reference
            for i, peak in enumerate(self.gradient_peaks):
                ax.axvline(x=peak, color='orange', linestyle=':', linewidth=1.5,
                          label='Gradient peak' if i == 0 else '')
        
        ax.set_xlabel('Radius (pixels)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title('Cladding Detection: Region Between Inflection Points', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inflection_detection.png'), dpi=150)
        plt.close()


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Create test image if not provided
        image_path = 'test_fiber.jpg'
        
        if not os.path.exists(image_path):
            print("Creating test fiber image...")
            
            # Create test image with clear inflection points
            size = 400
            img = np.zeros((size, size, 3), dtype=np.uint8)
            center = size // 2
            
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - center)**2 + (Y - center)**2)
            
            # Create smooth intensity profile with clear inflection points
            intensity = np.ones((size, size)) * 140  # Base
            
            # Use smooth transitions
            for y in range(size):
                for x in range(size):
                    r = dist[y, x]
                    
                    if r < 40:
                        # Core - bright
                        intensity[y, x] = 160
                    elif r < 50:
                        # Transition to cladding - smooth curve
                        t = (r - 40) / 10
                        intensity[y, x] = 160 - 30 * (3*t**2 - 2*t**3)  # Smooth cubic
                    elif r < 90:
                        # Cladding - darker
                        intensity[y, x] = 130
                    elif r < 100:
                        # Transition from cladding - smooth curve
                        t = (r - 90) / 10
                        intensity[y, x] = 130 + 10 * (3*t**2 - 2*t**3)  # Smooth cubic
                    else:
                        # Ferrule
                        intensity[y, x] = 140
            
            # Apply to all channels
            for i in range(3):
                img[:, :, i] = intensity.astype(np.uint8)
            
            # Add slight noise
            noise = np.random.normal(0, 1, img.shape)
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(image_path, img)
            print(f"Test image saved as {image_path}")
    
    # Run analysis
    try:
        analyzer = CurvatureFiberAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(f"Gradient peaks at: {results['gradient_peaks']} pixels")
        print(f"Cladding boundaries (inflection points) at: {results['cladding_boundaries']} pixels")
        
        if 'cladding' in results['regions']:
            cladding = results['regions']['cladding']
            print(f"\nCladding region: radius {cladding['radial_range'][0]}-{cladding['radial_range'][1]}px")
            print(f"  Mean intensity: {cladding['mean_intensity']:.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
