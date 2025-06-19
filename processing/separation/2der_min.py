import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
import warnings
warnings.filterwarnings('ignore')
import json
import os

class SecondDerivativeMinAnalyzer:
    """
    Fiber optic region detection based on second derivative minima
    The cladding boundaries are at the two minima of the second derivative
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
        print("\nStarting Second Derivative Minima Analysis...")
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
        """Find boundaries using minima of second derivative"""
        print("\nStep 3: Finding boundaries using second derivative minima...")
        
        # First derivative (gradient)
        self.first_derivative = np.gradient(self.intensity_profile_smooth)
        
        # Second derivative (curvature)
        self.second_derivative = np.gradient(self.first_derivative)
        
        # Apply slight smoothing to second derivative to reduce noise
        self.second_derivative_smooth = gaussian_filter1d(self.second_derivative, sigma=1)
        
        # Find minima in second derivative (negative peaks)
        # Invert the second derivative to find minima as peaks
        inverted_second_derivative = -self.second_derivative_smooth
        
        # Find peaks in inverted second derivative (which are minima in original)
        minima_indices, properties = find_peaks(
            inverted_second_derivative,
            distance=10,  # Minimum distance between minima
            prominence=None  # Will be calculated
        )
        
        if len(minima_indices) == 0:
            raise ValueError("No second derivative minima found!")
        
        # Get the values at these minima (in smoothed second derivative)
        minima_values = self.second_derivative_smooth[minima_indices]
        
        # If we have more than 2 minima, we need to be smart about selection
        if len(minima_indices) > 2:
            print(f"  Found {len(minima_indices)} minima, selecting the two most prominent...")
            
            # Calculate prominence for each minimum
            prominences = properties.get('prominences', 
                                       peak_prominences(inverted_second_derivative, minima_indices)[0])
            
            # Sort by prominence (highest first)
            prom_indices = np.argsort(prominences)[::-1]
            
            # Take the two most prominent minima
            selected_indices = [minima_indices[prom_indices[0]], 
                              minima_indices[prom_indices[1]]]
            
            # Sort by position
            self.boundaries = sorted(selected_indices)
        elif len(minima_indices) == 2:
            # Perfect - we have exactly two minima
            self.boundaries = sorted(minima_indices.tolist())
        elif len(minima_indices) == 1:
            # Only one minimum found
            self.boundaries = [minima_indices[0]]
            print("  Warning: Only one second derivative minimum found!")
        else:
            raise ValueError("No second derivative minima found!")
        
        print(f"  Second derivative minima at: {self.boundaries} pixels")
        print("  (These are points of maximum negative curvature where intensity curve caves in most)")
        
        # Print minimum values
        for i, boundary in enumerate(self.boundaries):
            value = self.second_derivative_smooth[boundary]
            print(f"  Minimum {i+1}: radius={boundary}px, value={value:.4f}")
    
    def define_regions(self):
        """Define regions based on second derivative minima"""
        print("\nStep 4: Defining regions based on second derivative minima...")
        
        self.regions = {}
        
        if len(self.boundaries) >= 2:
            # Standard case: two minima found
            boundary1, boundary2 = self.boundaries
            
            # Core: inside first minimum
            self.regions['core'] = {
                'start': 0,
                'end': boundary1,
                'name': 'core'
            }
            
            # Cladding: between the two minima (where curve has maximum negative curvature)
            self.regions['cladding'] = {
                'start': boundary1,
                'end': boundary2,
                'name': 'cladding'
            }
            
            # Ferrule: everything outside the cladding (defined in create_masks)
            
        elif len(self.boundaries) == 1:
            # Only one minimum found - estimate second boundary
            print("  Warning: Only one minimum found, estimating second boundary")
            boundary1 = self.boundaries[0]
            
            # Estimate second boundary (could be refined)
            estimated_boundary2 = min(boundary1 * 2, len(self.radii) - 1)
            
            self.regions['core'] = {
                'start': 0,
                'end': boundary1,
                'name': 'core'
            }
            
            self.regions['cladding'] = {
                'start': boundary1,
                'end': estimated_boundary2,
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
    
    def generate_output(self, output_dir='second_derivative_results'):
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
            'second_derivative_minima': [int(b) for b in self.boundaries],
            'second_derivative_values': [float(self.second_derivative_smooth[b]) for b in self.boundaries],
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
        fig.suptitle('Second Derivative Minima Fiber Analysis', fontsize=16)
        
        # Intensity profile
        ax1 = axes[0]
        ax1.plot(self.radii, self.intensity_profile, 'b-', linewidth=2, label='Intensity')
        ax1.set_ylabel('Pixel Intensity')
        ax1.set_title('Radial Intensity Profile')
        ax1.grid(True, alpha=0.3)
        
        # Mark boundaries
        if len(self.boundaries) >= 2:
            ax1.axvline(x=self.boundaries[0], color='purple', linestyle='--', linewidth=2,
                       label=f'2nd deriv min 1 ({self.boundaries[0]}px)')
            ax1.axvline(x=self.boundaries[1], color='purple', linestyle='--', linewidth=2,
                       label=f'2nd deriv min 2 ({self.boundaries[1]}px)')
        ax1.legend()
        
        # First derivative (gradient)
        ax2 = axes[1]
        ax2.plot(self.radii, self.first_derivative, 'orange', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('dI/dr')
        ax2.set_title('First Derivative (Gradient)')
        ax2.grid(True, alpha=0.3)
        
        # Second derivative (curvature)
        ax3 = axes[2]
        ax3.plot(self.radii, self.second_derivative_smooth, 'green', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('d²I/dr²')
        ax3.set_title('Second Derivative (Curvature) - Minima Define Cladding Boundaries')
        ax3.grid(True, alpha=0.3)
        
        # Mark the minima clearly
        if len(self.boundaries) >= 2:
            for i, boundary in enumerate(self.boundaries):
                ax3.axvline(x=boundary, color='purple', linestyle='--', linewidth=2)
                ax3.plot(boundary, self.second_derivative_smooth[boundary], 'mo', 
                        markersize=10, markeredgewidth=2, markerfacecolor='purple')
                ax3.annotate(f'Min {i+1}', 
                           xy=(boundary, self.second_derivative_smooth[boundary]),
                           xytext=(boundary + 5, self.second_derivative_smooth[boundary] - 0.05),
                           fontsize=10, fontweight='bold', color='purple')
        
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
            y_pos = ax4.get_ylim()[0] + (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.1
            ax4.annotate('CLADDING', xy=(mid_point, y_pos), 
                        xytext=(mid_point, y_pos),
                        ha='center', fontsize=10, fontweight='bold')
        
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'second_derivative_analysis.png'), dpi=150)
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
        
        # 3. Second derivative minima visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot second derivative
        ax.plot(self.radii, self.second_derivative_smooth, 'g-', linewidth=2, label='Second Derivative')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        if len(self.boundaries) >= 2:
            # Shade cladding region
            ax.axvspan(self.boundaries[0], self.boundaries[1], 
                      alpha=0.2, color='yellow', label='Cladding Region')
            
            # Mark minima prominently
            for i, boundary in enumerate(self.boundaries):
                ax.plot(boundary, self.second_derivative_smooth[boundary], 'mo', 
                       markersize=15, markeredgewidth=3, markerfacecolor='purple',
                       label=f'Minimum {i+1} ({boundary}px)')
                
                # Draw vertical lines from minima
                min_y = min(ax.get_ylim()[0], self.second_derivative_smooth[boundary])
                ax.vlines(boundary, ymin=min_y, 
                         ymax=0,  # Draw to zero line
                         colors='purple', linestyles='--', linewidth=2)
        
        ax.set_xlabel('Radius (pixels)', fontsize=12)
        ax.set_ylabel('d²I/dr² (Second Derivative)', fontsize=12)
        ax.set_title('Cladding Detection: Region Between Second Derivative Minima', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'second_derivative_minima.png'), dpi=150)
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
            print("Note: Looking for minima in second derivative (maximum negative curvature)")
            
            # Create test image with clear second derivative minima
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
        analyzer = SecondDerivativeMinAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(f"Second derivative minima at: {results['second_derivative_minima']} pixels")
        print(f"Minima values: {[f'{v:.4f}' for v in results['second_derivative_values']]}")
        
        if 'cladding' in results['regions']:
            cladding = results['regions']['cladding']
            print(f"\nCladding region: radius {cladding['radial_range'][0]}-{cladding['radial_range'][1]}px")
            print(f"  Mean intensity: {cladding['mean_intensity']:.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()