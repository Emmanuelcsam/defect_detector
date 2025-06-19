import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')
import json
import os

class DataDrivenFiberAnalyzer:
    """
    Fully data-driven fiber optic analysis
    No assumptions - regions detected purely from intensity and gradient patterns
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Results storage
        self.results = {}
        self.visualizations = []
        
    def analyze(self):
        """Run complete data-driven analysis"""
        print("Starting Data-Driven Fiber Optic Analysis...")
        print("=" * 60)
        
        # Step 1: Preprocessing
        self.preprocess()
        
        # Step 2: Find center using weighted centroid
        self.find_center()
        
        # Step 3: Compute radial profiles
        self.compute_radial_profiles()
        
        # Step 4: Detect boundaries from gradient peaks
        self.detect_boundaries()
        
        # Step 5: Analyze regions between boundaries
        self.analyze_regions()
        
        # Step 6: Create masks based on detected regions
        self.create_masks()
        
        # Step 7: Extract and clean regions
        self.extract_regions()
        
        # Step 8: Generate report
        self.generate_report()
        
        # Save results
        self.save_results()
        
        print("\nAnalysis Complete!")
        return self.results
    
    def preprocess(self):
        """Preprocessing with minimal assumptions"""
        print("\nStep 1: Preprocessing...")
        
        # Basic denoising
        self.denoised = cv2.fastNlMeansDenoising(self.gray, None, 10, 7, 21)
        
        # Compute gradients
        self.grad_x = cv2.Sobel(self.denoised, cv2.CV_64F, 1, 0, ksize=3)
        self.grad_y = cv2.Sobel(self.denoised, cv2.CV_64F, 0, 1, ksize=3)
        self.gradient_magnitude = np.sqrt(self.grad_x**2 + self.grad_y**2)
        
        # Compute local variance
        self.local_variance = self._compute_local_variance(self.denoised, window_size=5)
        
    def _compute_local_variance(self, img, window_size=5):
        """Compute local variance"""
        # Local mean
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        local_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
        
        # Local variance
        local_sq_mean = cv2.filter2D(img.astype(np.float32)**2, -1, kernel)
        variance = local_sq_mean - local_mean**2
        variance[variance < 0] = 0
        
        return variance
    
    def find_center(self):
        """Find center using intensity-weighted centroid"""
        print("\nStep 2: Finding center...")
        
        # Use brightest region to find center
        threshold = np.percentile(self.denoised, 90)
        bright_mask = self.denoised > threshold
        
        # Calculate centroid
        moments = cv2.moments(bright_mask.astype(np.uint8))
        if moments['m00'] > 0:
            self.center_x = moments['m10'] / moments['m00']
            self.center_y = moments['m01'] / moments['m00']
        else:
            self.center_x = self.width // 2
            self.center_y = self.height // 2
        
        print(f"  Center found at: ({self.center_x:.1f}, {self.center_y:.1f})")
    
    def compute_radial_profiles(self):
        """Compute comprehensive radial profiles"""
        print("\nStep 3: Computing radial profiles...")
        
        # Create radial coordinates
        Y, X = np.ogrid[:self.height, :self.width]
        self.distance_map = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        # Maximum radius to analyze
        max_radius = int(min(self.center_x, self.center_y, 
                            self.width - self.center_x, 
                            self.height - self.center_y))
        
        # Initialize profile arrays
        num_radii = max_radius
        self.radii = np.arange(num_radii)
        
        # Arrays to store profiles
        intensity_profiles = []
        gradient_profiles = []
        variance_profiles = []
        
        # Sample along many angles
        num_angles = 360
        angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        print("  Sampling radial profiles...")
        for angle in angles:
            # Sample points along this radial line
            x_coords = self.center_x + self.radii * np.cos(angle)
            y_coords = self.center_y + self.radii * np.sin(angle)
            
            # Extract values along line
            intensity_line = []
            gradient_line = []
            variance_line = []
            
            for x, y in zip(x_coords, y_coords):
                if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                    intensity_line.append(self.denoised[int(y), int(x)])
                    gradient_line.append(self.gradient_magnitude[int(y), int(x)])
                    variance_line.append(self.local_variance[int(y), int(x)])
            
            if len(intensity_line) == len(self.radii):
                intensity_profiles.append(intensity_line)
                gradient_profiles.append(gradient_line)
                variance_profiles.append(variance_line)
        
        # Compute median profiles (robust to outliers)
        self.intensity_profile = np.median(intensity_profiles, axis=0)
        self.gradient_profile = np.median(gradient_profiles, axis=0)
        self.variance_profile = np.median(variance_profiles, axis=0)
        
        # Compute derivatives
        self.intensity_derivative = np.abs(np.gradient(self.intensity_profile))
        
        # Smooth profiles slightly to reduce noise
        self.intensity_profile_smooth = gaussian_filter1d(self.intensity_profile, sigma=1)
        self.gradient_profile_smooth = gaussian_filter1d(self.gradient_profile, sigma=1)
        
        print(f"  Computed profiles for {len(self.radii)} radial positions")
    
    def detect_boundaries(self):
        """Detect boundaries from gradient peaks"""
        print("\nStep 4: Detecting boundaries from gradient peaks...")
        
        # Find peaks in gradient profile
        # Use smoothed gradient to reduce noise
        peaks, properties = find_peaks(
            self.gradient_profile_smooth,
            height=None,  # No minimum height requirement
            distance=10,  # Minimum 10 pixels between peaks
            prominence=None  # Calculate prominence
        )
        
        # Calculate prominences
        prominences = peak_prominences(self.gradient_profile_smooth, peaks)[0]
        
        # Sort peaks by prominence
        peak_data = [(peaks[i], prominences[i], self.gradient_profile_smooth[peaks[i]]) 
                     for i in range(len(peaks))]
        peak_data.sort(key=lambda x: x[1], reverse=True)  # Sort by prominence
        
        print(f"  Found {len(peaks)} gradient peaks")
        
        # Take the most prominent peaks as boundaries
        self.boundaries = []
        
        # We expect 2 main boundaries for a fiber
        num_boundaries = min(2, len(peak_data))
        
        for i in range(num_boundaries):
            peak_idx, prominence, height = peak_data[i]
            self.boundaries.append(peak_idx)
            print(f"  Boundary {i+1}: radius={peak_idx}px, prominence={prominence:.1f}, height={height:.1f}")
        
        # Sort boundaries by radius
        self.boundaries.sort()
        
        # Validate boundaries by checking intensity changes
        validated_boundaries = []
        for boundary in self.boundaries:
            if self._validate_boundary(boundary):
                validated_boundaries.append(boundary)
        
        self.boundaries = validated_boundaries
        print(f"  Validated {len(self.boundaries)} boundaries")
    
    def _validate_boundary(self, boundary_idx):
        """Validate boundary by checking intensity change"""
        if boundary_idx < 5 or boundary_idx >= len(self.intensity_profile) - 5:
            return False
        
        # Check intensity change across boundary
        before = np.mean(self.intensity_profile[max(0, boundary_idx-5):boundary_idx])
        after = np.mean(self.intensity_profile[boundary_idx:min(len(self.intensity_profile), boundary_idx+5)])
        
        change = abs(before - after)
        
        # Significant change threshold
        threshold = np.std(self.intensity_profile) * 0.2
        
        return change > threshold
    
    def analyze_regions(self):
        """Analyze regions between boundaries"""
        print("\nStep 5: Analyzing regions between boundaries...")
        
        self.regions = []
        
        # Create region segments
        if len(self.boundaries) == 0:
            # No boundaries found - single region
            self.regions.append({
                'start': 0,
                'end': len(self.intensity_profile) - 1,
                'type': 'unknown'
            })
        elif len(self.boundaries) == 1:
            # One boundary - two regions
            self.regions.append({
                'start': 0,
                'end': self.boundaries[0],
                'type': 'inner'
            })
            self.regions.append({
                'start': self.boundaries[0],
                'end': len(self.intensity_profile) - 1,
                'type': 'outer'
            })
        else:
            # Multiple boundaries - analyze each segment
            # Region before first boundary
            self.regions.append({
                'start': 0,
                'end': self.boundaries[0],
                'type': 'core'
            })
            
            # Regions between boundaries
            for i in range(len(self.boundaries) - 1):
                self.regions.append({
                    'start': self.boundaries[i],
                    'end': self.boundaries[i + 1],
                    'type': 'intermediate'
                })
            
            # Region after last boundary
            self.regions.append({
                'start': self.boundaries[-1],
                'end': len(self.intensity_profile) - 1,
                'type': 'outer'
            })
        
        # Analyze intensity characteristics of each region
        for region in self.regions:
            start = region['start']
            end = region['end']
            
            if end > start:
                region_intensity = self.intensity_profile[start:end]
                region_gradient = self.gradient_profile[start:end]
                region_variance = self.variance_profile[start:end]
                
                region['mean_intensity'] = np.mean(region_intensity)
                region['median_intensity'] = np.median(region_intensity)
                region['std_intensity'] = np.std(region_intensity)
                region['min_intensity'] = np.min(region_intensity)
                region['max_intensity'] = np.max(region_intensity)
                region['mean_gradient'] = np.mean(region_gradient)
                region['mean_variance'] = np.mean(region_variance)
        
        # Identify region types based on characteristics
        self._identify_region_types()
        
        # Print region analysis
        print("\n  Region Analysis:")
        for i, region in enumerate(self.regions):
            print(f"  Region {i+1} ({region['type']}): radius {region['start']}-{region['end']}px")
            print(f"    Intensity: mean={region['mean_intensity']:.1f}, "
                  f"median={region['median_intensity']:.1f}, "
                  f"range=[{region['min_intensity']:.1f}, {region['max_intensity']:.1f}]")
    
    def _identify_region_types(self):
        """Identify region types based on their characteristics"""
        if len(self.regions) < 3:
            return
        
        # For a typical fiber with 3 regions:
        # 1. Core (innermost) - usually brightest
        # 2. Cladding (middle) - specific intensity pattern
        # 3. Ferrule (outer) - background
        
        # Find the region with minimum mean intensity
        intensities = [(i, r['mean_intensity']) for i, r in enumerate(self.regions)]
        intensities.sort(key=lambda x: x[1])
        
        # The darkest region is likely the cladding
        darkest_idx = intensities[0][0]
        
        # Check if this makes sense spatially
        if len(self.regions) >= 3:
            # Typical pattern: core (0), cladding (1), ferrule (2)
            # But let's verify with intensity patterns
            
            # If the middle region is darkest, it's likely cladding
            if darkest_idx == 1 and len(self.regions) == 3:
                self.regions[0]['type'] = 'core'
                self.regions[1]['type'] = 'cladding'
                self.regions[2]['type'] = 'ferrule'
            else:
                # Use intensity patterns to determine
                # Core should be central and relatively bright
                # Cladding should show as a distinct intensity dip
                # Ferrule is the outer region
                
                # Find region with intensity dip pattern
                for i, region in enumerate(self.regions):
                    if i > 0 and i < len(self.regions) - 1:  # Not first or last
                        # Check if this region has lower intensity than neighbors
                        prev_intensity = self.regions[i-1]['mean_intensity']
                        next_intensity = self.regions[i+1]['mean_intensity']
                        curr_intensity = region['mean_intensity']
                        
                        if curr_intensity < prev_intensity and curr_intensity < next_intensity:
                            # This is likely the cladding
                            self.regions[i]['type'] = 'cladding'
                            if i > 0:
                                self.regions[0]['type'] = 'core'
                            if i < len(self.regions) - 1:
                                self.regions[-1]['type'] = 'ferrule'
                            break
    
    def create_masks(self):
        """Create masks for each detected region"""
        print("\nStep 6: Creating region masks...")
        
        self.masks = {}
        
        # Initialize masks
        for region_type in ['core', 'cladding', 'ferrule']:
            self.masks[region_type] = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create masks based on regions and distance map
        for region in self.regions:
            if region['type'] in self.masks:
                # Create annular mask for this region
                if region['end'] < len(self.radii):
                    inner_radius = self.radii[region['start']]
                    outer_radius = self.radii[region['end']]
                    
                    mask = ((self.distance_map >= inner_radius) & 
                           (self.distance_map < outer_radius))
                    
                    self.masks[region['type']][mask] = 255
        
        # Ensure masks are mutually exclusive
        # Priority: core > cladding > ferrule
        if np.any(self.masks['core']):
            self.masks['cladding'][self.masks['core'] > 0] = 0
            self.masks['ferrule'][self.masks['core'] > 0] = 0
        
        if np.any(self.masks['cladding']):
            self.masks['ferrule'][self.masks['cladding'] > 0] = 0
        
        # Morphological cleanup
        for region_type in self.masks:
            if np.any(self.masks[region_type]):
                # Remove small holes
                self.masks[region_type] = morphology.remove_small_holes(
                    self.masks[region_type] > 0, area_threshold=50
                ).astype(np.uint8) * 255
        
        print("  Masks created for detected regions")
    
    def extract_regions(self):
        """Extract regions using masks"""
        print("\nStep 7: Extracting regions...")
        
        self.extracted_regions = {}
        
        for region_type, mask in self.masks.items():
            if np.any(mask):
                # Extract region
                region = cv2.bitwise_and(self.original, self.original, mask=mask)
                self.extracted_regions[region_type] = region
                
                # Calculate statistics
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                valid_pixels = gray_region[mask > 0]
                
                print(f"  {region_type}: {len(valid_pixels)} pixels, "
                      f"intensity={np.mean(valid_pixels):.1f}±{np.std(valid_pixels):.1f}")
    
    def generate_report(self):
        """Generate analysis report"""
        print("\nStep 8: Generating report...")
        
        self.results = {
            'image_info': {
                'width': self.width,
                'height': self.height,
                'center': {'x': float(self.center_x), 'y': float(self.center_y)}
            },
            'boundaries': [int(b) for b in self.boundaries],
            'regions': []
        }
        
        # Add region information
        for region in self.regions:
            region_info = {
                'type': region['type'],
                'radial_range': [region['start'], region['end']],
                'intensity_stats': {
                    'mean': float(region['mean_intensity']),
                    'median': float(region['median_intensity']),
                    'std': float(region['std_intensity']),
                    'min': float(region['min_intensity']),
                    'max': float(region['max_intensity'])
                },
                'gradient_mean': float(region['mean_gradient']),
                'variance_mean': float(region['mean_variance'])
            }
            self.results['regions'].append(region_info)
        
        # Add pixel counts
        self.results['pixel_counts'] = {}
        for region_type, mask in self.masks.items():
            self.results['pixel_counts'][region_type] = int(np.sum(mask > 0))
    
    def save_results(self, output_dir='fiber_analysis_output'):
        """Save all results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save extracted regions
        for region_type, region in self.extracted_regions.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_type}.png'), region)
        
        # Save masks
        for region_type, mask in self.masks.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_type}_mask.png'), mask)
        
        # Create visualizations
        self._create_visualizations(output_dir)
        
        # Save JSON report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")
    
    def _create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        # 1. Radial profiles with detected boundaries
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Radial Profile Analysis', fontsize=16)
        
        # Intensity profile
        ax1 = axes[0, 0]
        ax1.plot(self.radii, self.intensity_profile, 'b-', linewidth=2)
        ax1.set_xlabel('Radius (pixels)')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Radial Intensity Profile')
        ax1.grid(True, alpha=0.3)
        
        # Mark boundaries
        for i, boundary in enumerate(self.boundaries):
            color = 'g' if i == 0 else 'r'
            label = 'Core Boundary' if i == 0 else 'Cladding Boundary'
            ax1.axvline(x=boundary, color=color, linestyle='--', linewidth=2,
                       label=f'{label} ({boundary}px)')
        ax1.legend()
        
        # Gradient profile
        ax2 = axes[0, 1]
        ax2.plot(self.radii, self.gradient_profile, 'orange', linewidth=2)
        ax2.set_xlabel('Radius (pixels)')
        ax2.set_ylabel('Gradient Magnitude')
        ax2.set_title('Radial Gradient Profile')
        ax2.grid(True, alpha=0.3)
        
        # Mark peaks
        for boundary in self.boundaries:
            ax2.axvline(x=boundary, color='r', linestyle='--', linewidth=2)
            if boundary < len(self.gradient_profile):
                ax2.plot(boundary, self.gradient_profile[boundary], 
                        'rx', markersize=10, markeredgewidth=3)
        
        # Variance profile
        ax3 = axes[1, 0]
        ax3.plot(self.radii, self.variance_profile, 'g-', linewidth=2)
        ax3.set_xlabel('Radius (pixels)')
        ax3.set_ylabel('Local Variance')
        ax3.set_title('Radial Variance Profile')
        ax3.grid(True, alpha=0.3)
        
        for boundary in self.boundaries:
            ax3.axvline(x=boundary, color='r', linestyle='--', linewidth=2)
        
        # Intensity derivative
        ax4 = axes[1, 1]
        ax4.plot(self.radii, self.intensity_derivative, 'm-', linewidth=2)
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('|dI/dr|')
        ax4.set_title('Intensity Derivative Profile')
        ax4.grid(True, alpha=0.3)
        
        for boundary in self.boundaries:
            ax4.axvline(x=boundary, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radial_profiles.png'), dpi=150)
        plt.close()
        
        # 2. Multi-modal analysis (similar to reference)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Multi-Modal Radial Analysis for Boundary Detection', fontsize=16)
        
        # Intensity
        ax1 = axes[0]
        ax1.plot(self.radii, self.intensity_profile, 'b-', linewidth=2, label='Avg. Intensity')
        ax1.set_ylabel('Avg. Intensity')
        ax1.set_title('Average Pixel Intensity vs. Radius')
        ax1.grid(True, alpha=0.3)
        
        for i, boundary in enumerate(self.boundaries):
            color = 'g' if i == 0 else 'r'
            label = 'Core Boundary' if i == 0 else 'Cladding Boundary'
            ax1.axvline(x=boundary, color=color, linestyle='--', linewidth=2,
                       label=f'{label} ({boundary}px)')
        ax1.legend()
        
        # Gradient
        ax2 = axes[1]
        ax2.plot(self.radii, self.gradient_profile, 'orange', linewidth=2, label='Avg. Gradient')
        ax2.set_ylabel('Avg. Change')
        ax2.set_title('Average Change Magnitude (Gradient) vs. Radius')
        ax2.grid(True, alpha=0.3)
        
        for boundary in self.boundaries:
            ax2.axvline(x=boundary, color='r', linestyle='--', linewidth=2)
            if boundary < len(self.gradient_profile):
                ax2.plot(boundary, self.gradient_profile[boundary], 
                        'rx', markersize=12, markeredgewidth=3)
        ax2.annotate('Detected Peaks', xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top')
        
        # Variance/Texture
        ax3 = axes[2]
        ax3.plot(self.radii, self.variance_profile, 'purple', linewidth=2)
        ax3.set_ylabel('LBP Variance')
        ax3.set_xlabel('Radius from Center (pixels)')
        ax3.set_title('Variance of Local Binary Patterns (Texture) vs. Radius')
        ax3.grid(True, alpha=0.3)
        
        for boundary in self.boundaries:
            ax3.axvline(x=boundary, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multimodal_analysis.png'), dpi=150)
        plt.close()
        
        # 3. Final results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data-Driven Region Detection Results', fontsize=16)
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Gradient magnitude
        im1 = axes[0, 1].imshow(self.gradient_magnitude, cmap='hot')
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Combined masks
        mask_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if 'core' in self.masks and np.any(self.masks['core']):
            mask_overlay[self.masks['core'] > 0] = [255, 0, 0]  # Red
        if 'cladding' in self.masks and np.any(self.masks['cladding']):
            mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0]  # Green
        if 'ferrule' in self.masks and np.any(self.masks['ferrule']):
            mask_overlay[self.masks['ferrule'] > 0] = [0, 0, 255]  # Blue
        
        axes[0, 2].imshow(mask_overlay)
        axes[0, 2].set_title('Detected Regions (R=Core, G=Cladding, B=Ferrule)')
        axes[0, 2].axis('off')
        
        # Individual regions
        for i, (region_type, region) in enumerate(self.extracted_regions.items()):
            if i < 3:
                axes[1, i].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f'{region_type.capitalize()} Region')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_results.png'), dpi=150)
        plt.close()
        
        # 4. Intensity distribution plot
        if len(self.regions) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            x_pos = []
            means = []
            stds = []
            labels = []
            
            for i, region in enumerate(self.regions):
                x_pos.append(i)
                means.append(region['mean_intensity'])
                stds.append(region['std_intensity'])
                labels.append(f"{region['type']}\n({region['start']}-{region['end']}px)")
            
            ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                   color=['red', 'green', 'blue'][:len(x_pos)])
            ax.set_xlabel('Region')
            ax.set_ylabel('Intensity')
            ax.set_title('Mean Intensity by Region')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (x, y, s) in enumerate(zip(x_pos, means, stds)):
                ax.text(x, y + s + 1, f'{y:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'region_intensities.png'), dpi=150)
            plt.close()


# Test script
if __name__ == "__main__":
    import sys
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'fiber_endface.jpg'
    
    # Create test image if needed
    if not os.path.exists(image_path):
        print("Creating test image...")
        
        # Create realistic test image
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        center = 200
        
        Y, X = np.ogrid[:400, :400]
        dist = np.sqrt((X - center)**2 + (Y - center)**2)
        
        # Create intensity pattern similar to the graphs
        # Core region (0-40px): ~140 intensity
        # Cladding region (40-80px): ~135 intensity (darker)
        # Ferrule region (80+px): ~137 intensity
        
        # Start with base
        img[:, :] = [137, 137, 137]
        
        # Core (brightest)
        core_mask = dist <= 40
        img[core_mask] = [140, 140, 140]
        
        # Add intensity variation in core
        core_variation = np.random.normal(0, 2, np.sum(core_mask))
        for i in range(3):
            img[:, :, i][core_mask] = np.clip(img[:, :, i][core_mask] + core_variation, 0, 255)
        
        # Cladding (darkest) - between 40 and 80 pixels
        cladding_mask = (dist > 40) & (dist <= 80)
        img[cladding_mask] = [135, 135, 135]
        
        # Add transitions
        for r in range(38, 43):
            transition_mask = (dist > r) & (dist <= r + 1)
            t = (r - 38) / 5
            intensity = int(140 * (1 - t) + 135 * t)
            img[transition_mask] = [intensity, intensity, intensity]
        
        for r in range(78, 83):
            transition_mask = (dist > r) & (dist <= r + 1)
            t = (r - 78) / 5
            intensity = int(135 * (1 - t) + 137 * t)
            img[transition_mask] = [intensity, intensity, intensity]
        
        # Add noise
        noise = np.random.normal(0, 1, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(image_path, img)
        print(f"Test image saved as {image_path}")
    
    # Run analysis
    try:
        analyzer = DataDrivenFiberAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY:")
        print("=" * 60)
        
        print(f"\nDetected {len(results['boundaries'])} boundaries at: {results['boundaries']} pixels")
        
        print("\nRegion Analysis:")
        for region in results['regions']:
            print(f"\n{region['type'].upper()} Region:")
            print(f"  Radial range: {region['radial_range'][0]}-{region['radial_range'][1]} pixels")
            print(f"  Mean intensity: {region['intensity_stats']['mean']:.1f}")
            print(f"  Intensity range: [{region['intensity_stats']['min']:.1f}, {region['intensity_stats']['max']:.1f}]")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()