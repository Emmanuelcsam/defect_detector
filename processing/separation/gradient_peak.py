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

class GradientPeakFiberAnalyzer:
    """
    Fiber optic region detection based on intensity gradient peaks
    Cladding is defined as the region between the two highest gradient peaks
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
        print("\nStarting Gradient Peak-Based Analysis...")
        print("=" * 60)
        
        # Step 1: Find center
        self.find_center()
        
        # Step 2: Compute radial intensity profile
        self.compute_radial_intensity_profile()
        
        # Step 3: Find gradient peaks
        self.find_gradient_peaks()
        
        # Step 4: Define regions based on peaks
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
        
        # Light smoothing to reduce noise
        self.intensity_profile_smooth = gaussian_filter1d(self.intensity_profile, sigma=1)
        
        print(f"  Profile computed for {len(self.radii)} radial positions")
    
    def find_gradient_peaks(self):
        """Find the two highest peaks in the intensity gradient"""
        print("\nStep 3: Finding gradient peaks...")
        
        # Compute gradient (derivative) of intensity profile
        # This shows where intensity is changing fastest
        self.intensity_gradient = np.abs(np.gradient(self.intensity_profile_smooth))
        
        # Find all peaks in the gradient
        peaks, properties = find_peaks(
            self.intensity_gradient,
            height=None,  # Will filter later
            distance=10   # Minimum distance between peaks
        )
        
        if len(peaks) == 0:
            raise ValueError("No gradient peaks found!")
        
        # Get peak heights
        peak_heights = self.intensity_gradient[peaks]
        
        # Sort peaks by height (descending)
        peak_indices = np.argsort(peak_heights)[::-1]
        
        # Take the two highest peaks
        if len(peaks) >= 2:
            top_two_indices = peak_indices[:2]
            self.boundary_peaks = sorted([peaks[i] for i in top_two_indices])
        else:
            # Only one peak found
            self.boundary_peaks = [peaks[peak_indices[0]]]
            print("  Warning: Only one gradient peak found!")
        
        print(f"  Found gradient peaks at radii: {self.boundary_peaks} pixels")
        
        # Print peak characteristics
        for i, peak_radius in enumerate(self.boundary_peaks):
            height = self.intensity_gradient[peak_radius]
            print(f"  Peak {i+1}: radius={peak_radius}px, gradient={height:.2f}")
    
    def define_regions(self):
        """Define regions based on gradient peaks"""
        print("\nStep 4: Defining regions based on gradient peaks...")
        
        self.regions = {}
        
        if len(self.boundary_peaks) >= 2:
            # Standard case: two peaks found
            peak1, peak2 = self.boundary_peaks
            
            # Core: inside first peak
            self.regions['core'] = {
                'start': 0,
                'end': peak1,
                'name': 'core'
            }
            
            # Cladding: between the two peaks
            self.regions['cladding'] = {
                'start': peak1,
                'end': peak2,
                'name': 'cladding'
            }
            
            # Ferrule: outside second peak
            self.regions['ferrule'] = {
                'start': peak2,
                'end': len(self.radii) - 1,
                'name': 'ferrule'
            }
            
        elif len(self.boundary_peaks) == 1:
            # Only one peak: assume it's the core-cladding boundary
            peak = self.boundary_peaks[0]
            
            self.regions['core'] = {
                'start': 0,
                'end': peak,
                'name': 'core'
            }
            
            # Estimate cladding extent (could be refined based on intensity patterns)
            cladding_end = min(peak * 2, len(self.radii) - 1)
            
            self.regions['cladding'] = {
                'start': peak,
                'end': cladding_end,
                'name': 'cladding'
            }
            
            self.regions['ferrule'] = {
                'start': cladding_end,
                'end': len(self.radii) - 1,
                'name': 'ferrule'
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
        
        for region_name, region in self.regions.items():
            # Create annular mask
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            
            inner_radius = self.radii[region['start']] if region['start'] < len(self.radii) else 0
            outer_radius = self.radii[region['end']] if region['end'] < len(self.radii) else np.max(distance_map)
            
            # Fill mask
            mask_region = (distance_map >= inner_radius) & (distance_map <= outer_radius)
            mask[mask_region] = 255
            
            self.masks[region_name] = mask
            
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
    
    def generate_output(self, output_dir='gradient_peak_results'):
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
            'gradient_peaks': [int(p) for p in self.boundary_peaks],
            'regions': {}
        }
        
        for region_name, region in self.regions.items():
            self.results['regions'][region_name] = {
                'radial_range': [region['start'], region['end']],
                'mean_intensity': float(region['mean_intensity']),
                'std_intensity': float(region['std_intensity']),
                'median_intensity': float(region['median_intensity']),
                'pixel_count': int(np.sum(self.masks[region_name] > 0))
            }
        
        # Save JSON report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")
    
    def _create_visualizations(self, output_dir):
        """Create visualization plots"""
        # 1. Main analysis plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Gradient Peak-Based Fiber Analysis', fontsize=16)
        
        # Intensity profile
        ax1 = axes[0]
        ax1.plot(self.radii, self.intensity_profile, 'b-', linewidth=2, label='Intensity')
        ax1.set_ylabel('Pixel Intensity')
        ax1.set_title('Radial Intensity Profile')
        ax1.grid(True, alpha=0.3)
        
        # Mark boundaries
        colors = ['g', 'r']
        labels = ['Core-Cladding', 'Cladding-Ferrule']
        for i, peak in enumerate(self.boundary_peaks):
            color = colors[i % len(colors)]
            label = labels[i % len(labels)]
            ax1.axvline(x=peak, color=color, linestyle='--', linewidth=2,
                       label=f'{label} ({peak}px)')
        ax1.legend()
        
        # Intensity gradient (derivative)
        ax2 = axes[1]
        ax2.plot(self.radii, self.intensity_gradient, 'orange', linewidth=2)
        ax2.set_ylabel('|dI/dr|')
        ax2.set_title('Intensity Gradient (Derivative)')
        ax2.grid(True, alpha=0.3)
        
        # Mark the peaks
        for peak in self.boundary_peaks:
            ax2.axvline(x=peak, color='r', linestyle='--', linewidth=2)
            ax2.plot(peak, self.intensity_gradient[peak], 'ro', markersize=10)
        
        # Annotate peaks
        for i, peak in enumerate(self.boundary_peaks):
            ax2.annotate(f'Peak {i+1}', 
                        xy=(peak, self.intensity_gradient[peak]),
                        xytext=(peak + 10, self.intensity_gradient[peak] + 0.1),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        # Region visualization
        ax3 = axes[2]
        ax3.plot(self.radii, self.intensity_profile, 'b-', linewidth=2)
        ax3.set_xlabel('Radius (pixels)')
        ax3.set_ylabel('Pixel Intensity')
        ax3.set_title('Identified Regions')
        ax3.grid(True, alpha=0.3)
        
        # Shade regions
        alpha = 0.3
        if 'core' in self.regions:
            r = self.regions['core']
            ax3.axvspan(r['start'], r['end'], alpha=alpha, color='red', label='Core')
        
        if 'cladding' in self.regions:
            r = self.regions['cladding']
            ax3.axvspan(r['start'], r['end'], alpha=alpha, color='green', label='Cladding')
        
        if 'ferrule' in self.regions:
            r = self.regions['ferrule']
            end = min(r['end'], len(self.radii)-1)
            ax3.axvspan(r['start'], end, alpha=alpha, color='blue', label='Ferrule')
        
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_analysis.png'), dpi=150)
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
        
        # Intensity statistics
        ax = axes[1, 1]
        regions = []
        intensities = []
        errors = []
        
        for region_name in ['core', 'cladding', 'ferrule']:
            if region_name in self.regions:
                regions.append(region_name.capitalize())
                intensities.append(self.regions[region_name]['mean_intensity'])
                errors.append(self.regions[region_name]['std_intensity'])
        
        if regions:
            x_pos = np.arange(len(regions))
            bars = ax.bar(x_pos, intensities, yerr=errors, capsize=10, 
                          color=['red', 'green', 'blue'][:len(regions)], alpha=0.7)
            ax.set_xlabel('Region')
            ax.set_ylabel('Mean Intensity')
            ax.set_title('Region Intensity Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(regions)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, intensity in zip(bars, intensities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{intensity:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results.png'), dpi=150)
        plt.close()
        
        # 3. Simple gradient peak visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot gradient with filled area
        ax.fill_between(self.radii, 0, self.intensity_gradient, alpha=0.3, color='orange')
        ax.plot(self.radii, self.intensity_gradient, 'orange', linewidth=2)
        
        # Mark peaks prominently
        for i, peak in enumerate(self.boundary_peaks):
            ax.axvline(x=peak, color='red', linestyle='--', linewidth=3, alpha=0.7)
            ax.plot(peak, self.intensity_gradient[peak], 'ro', markersize=15)
            
            # Label
            ax.text(peak, self.intensity_gradient[peak] + 0.05, f'Peak {i+1}\n({peak}px)', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Highlight cladding region
        if len(self.boundary_peaks) >= 2:
            ax.axvspan(self.boundary_peaks[0], self.boundary_peaks[1], 
                      alpha=0.2, color='green', label='Cladding Region')
        
        ax.set_xlabel('Radius (pixels)', fontsize=12)
        ax.set_ylabel('Intensity Gradient |dI/dr|', fontsize=12)
        ax.set_title('Cladding Detection: Region Between Two Highest Gradient Peaks', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_peaks.png'), dpi=150)
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
            
            # Create test image with clear gradient peaks
            size = 400
            img = np.zeros((size, size, 3), dtype=np.uint8)
            center = size // 2
            
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - center)**2 + (Y - center)**2)
            
            # Create intensity pattern with sharp transitions
            # Core: bright (radius 0-50)
            # Cladding: darker (radius 50-100)  
            # Ferrule: medium (radius 100+)
            
            # Base intensity
            intensity = np.ones((size, size)) * 140  # Ferrule level
            
            # Cladding region (darker)
            cladding_mask = (dist >= 50) & (dist <= 100)
            intensity[cladding_mask] = 120
            
            # Core region (brightest)
            core_mask = dist < 50
            intensity[core_mask] = 160
            
            # Create sharp transitions
            # Smooth transitions to create clear gradient peaks
            from scipy.ndimage import gaussian_filter
            intensity = gaussian_filter(intensity, sigma=2)
            
            # Apply to all channels
            for i in range(3):
                img[:, :, i] = intensity.astype(np.uint8)
            
            # Add some noise
            noise = np.random.normal(0, 2, img.shape)
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(image_path, img)
            print(f"Test image saved as {image_path}")
    
    # Run analysis
    try:
        analyzer = GradientPeakFiberAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(f"Gradient peaks found at: {results['gradient_peaks']} pixels")
        print("\nRegion definitions:")
        for region_name, region_data in results['regions'].items():
            print(f"  {region_name}: radius {region_data['radial_range'][0]}-{region_data['radial_range'][1]}px")
            print(f"    Mean intensity: {region_data['mean_intensity']:.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()