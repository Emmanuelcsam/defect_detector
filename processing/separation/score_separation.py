import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_prominences
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import json
import os

class RobustFiberAnalyzer:
    """
    Robust fiber optic region detection using statistical consensus
    Analyzes multiple preprocessing versions to avoid artifacts
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Storage for multiple analyses
        self.analyses = []
        self.preprocessing_methods = []
        
        print(f"Loaded image: {self.width}x{self.height} pixels")
    
    def analyze(self):
        """Run robust multi-threshold analysis"""
        print("\nStarting Robust Multi-Threshold Fiber Analysis...")
        print("=" * 60)
        
        # Step 1: Find center (common for all analyses)
        self.find_center()
        
        # Step 2: Create multiple preprocessed versions
        self.create_preprocessed_versions()
        
        # Step 3: Analyze each version
        self.analyze_all_versions()
        
        # Step 4: Find consensus boundaries
        self.find_consensus_boundaries()
        
        # Step 5: Create final masks using consensus
        self.create_final_masks()
        
        # Step 6: Extract regions
        self.extract_regions()
        
        # Step 7: Generate comprehensive output
        self.generate_output()
        
        print("\nRobust Analysis Complete!")
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
    
    def create_preprocessed_versions(self):
        """Create multiple preprocessed versions with different filters and thresholds"""
        print("\nStep 2: Creating multiple preprocessed versions...")
        
        self.preprocessed_images = []
        
        # 1. Original grayscale
        self.preprocessed_images.append({
            'name': 'Original',
            'image': self.gray,
            'description': 'Original grayscale'
        })
        
        # 2. Gaussian blur (multiple sigmas)
        for sigma in [1, 2, 3]:
            blurred = cv2.GaussianBlur(self.gray, (0, 0), sigma)
            self.preprocessed_images.append({
                'name': f'Gaussian_σ{sigma}',
                'image': blurred,
                'description': f'Gaussian blur σ={sigma}'
            })
        
        # 3. Median filter (for salt-and-pepper noise)
        for ksize in [3, 5]:
            median = median_filter(self.gray, size=ksize)
            self.preprocessed_images.append({
                'name': f'Median_k{ksize}',
                'image': median,
                'description': f'Median filter {ksize}x{ksize}'
            })
        
        # 4. Bilateral filter (edge-preserving)
        bilateral = cv2.bilateralFilter(self.gray, 9, 75, 75)
        self.preprocessed_images.append({
            'name': 'Bilateral',
            'image': bilateral,
            'description': 'Bilateral filter'
        })
        
        # 5. Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(self.gray)
        self.preprocessed_images.append({
            'name': 'CLAHE',
            'image': enhanced,
            'description': 'Adaptive histogram equalization'
        })
        
        # 6. Morphological operations (to remove small artifacts)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(self.gray, cv2.MORPH_OPEN, kernel)
        self.preprocessed_images.append({
            'name': 'Morphological',
            'image': opened,
            'description': 'Morphological opening'
        })
        
        # 7. Statistical filtering (remove outliers)
        # Replace pixels that deviate too much from local mean
        statistical = self._statistical_filter(self.gray)
        self.preprocessed_images.append({
            'name': 'Statistical',
            'image': statistical,
            'description': 'Statistical outlier removal'
        })
        
        print(f"  Created {len(self.preprocessed_images)} preprocessed versions")
    
    def _statistical_filter(self, img, window_size=5, z_threshold=2):
        """Remove statistical outliers"""
        filtered = img.copy()
        padded = np.pad(img, window_size//2, mode='reflect')
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                mean = np.mean(window)
                std = np.std(window)
                
                # If pixel deviates too much from local statistics, replace with median
                if abs(img[i, j] - mean) > z_threshold * std:
                    filtered[i, j] = np.median(window)
        
        return filtered
    
    def analyze_all_versions(self):
        """Analyze each preprocessed version"""
        print("\nStep 3: Analyzing all preprocessed versions...")
        
        self.analyses = []
        
        for i, prep in enumerate(self.preprocessed_images):
            print(f"\n  Analyzing version {i+1}/{len(self.preprocessed_images)}: {prep['name']}")
            
            try:
                # Compute radial profile for this version
                profile_data = self._compute_radial_profile(prep['image'])
                
                # Find boundaries using second derivative minima
                boundaries = self._find_boundaries(profile_data)
                
                # Store analysis results
                self.analyses.append({
                    'name': prep['name'],
                    'boundaries': boundaries,
                    'profile_data': profile_data,
                    'valid': len(boundaries) >= 2
                })
                
                if len(boundaries) >= 2:
                    print(f"    Found boundaries at: {boundaries} pixels")
                else:
                    print(f"    Warning: Only {len(boundaries)} boundaries found")
                    
            except Exception as e:
                print(f"    Error analyzing {prep['name']}: {e}")
                self.analyses.append({
                    'name': prep['name'],
                    'boundaries': [],
                    'valid': False
                })
    
    def _compute_radial_profile(self, img):
        """Compute comprehensive radial profile statistics"""
        # Maximum radius
        max_radius = int(min(self.center_x, self.center_y, 
                            self.width - self.center_x, 
                            self.height - self.center_y))
        
        radii = np.arange(max_radius)
        
        # Initialize storage for statistics
        intensity_values = [[] for _ in range(max_radius)]
        
        # Sample along multiple angles
        num_angles = 360
        angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        for angle in angles:
            # Points along this radial line
            x_coords = self.center_x + radii * np.cos(angle)
            y_coords = self.center_y + radii * np.sin(angle)
            
            # Sample intensities
            for r, (x, y) in enumerate(zip(x_coords, y_coords)):
                if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                    intensity_values[r].append(img[int(y), int(x)])
        
        # Compute statistics for each radius
        profile_stats = {
            'radii': radii,
            'mean': np.zeros(max_radius),
            'median': np.zeros(max_radius),
            'mode': np.zeros(max_radius),
            'variance': np.zeros(max_radius),
            'std': np.zeros(max_radius)
        }
        
        for r in range(max_radius):
            if intensity_values[r]:
                values = np.array(intensity_values[r])
                profile_stats['mean'][r] = np.mean(values)
                profile_stats['median'][r] = np.median(values)
                profile_stats['variance'][r] = np.var(values)
                profile_stats['std'][r] = np.std(values)
                
                # Mode calculation
                hist, bins = np.histogram(values, bins=20)
                mode_idx = np.argmax(hist)
                profile_stats['mode'][r] = (bins[mode_idx] + bins[mode_idx + 1]) / 2
        
        # Use median as primary profile (most robust to outliers)
        profile_stats['primary_profile'] = profile_stats['median']
        
        # Apply smoothing
        profile_stats['smooth_profile'] = gaussian_filter1d(
            profile_stats['primary_profile'], sigma=2
        )
        
        return profile_stats
    
    def _find_boundaries(self, profile_data):
        """Find boundaries using second derivative minima"""
        # First derivative
        first_derivative = np.gradient(profile_data['smooth_profile'])
        
        # Second derivative
        second_derivative = np.gradient(first_derivative)
        
        # Smooth second derivative
        second_derivative_smooth = gaussian_filter1d(second_derivative, sigma=1)
        
        # Find minima
        inverted = -second_derivative_smooth
        minima_indices, properties = find_peaks(
            inverted,
            distance=10,
            prominence=None
        )
        
        if len(minima_indices) == 0:
            return []
        
        # If more than 2 minima, select most prominent
        if len(minima_indices) > 2:
            prominences = peak_prominences(inverted, minima_indices)[0]
            prom_indices = np.argsort(prominences)[::-1]
            selected = [minima_indices[prom_indices[0]], 
                       minima_indices[prom_indices[1]]]
            return sorted(selected)
        else:
            return sorted(minima_indices.tolist())
    
    def find_consensus_boundaries(self):
        """Find consensus boundaries from multiple analyses"""
        print("\nStep 4: Finding consensus boundaries...")
        
        # Collect all valid boundaries
        all_boundaries = []
        valid_analyses = [a for a in self.analyses if a['valid']]
        
        print(f"  Valid analyses: {len(valid_analyses)}/{len(self.analyses)}")
        
        for analysis in valid_analyses:
            all_boundaries.extend(analysis['boundaries'])
        
        if not all_boundaries:
            raise ValueError("No valid boundaries found in any analysis!")
        
        # Cluster boundaries to find consensus
        # Use DBSCAN-like approach
        boundary_clusters = self._cluster_boundaries(all_boundaries)
        
        # Select the two most common boundary positions
        if len(boundary_clusters) >= 2:
            # Sort by number of votes (descending)
            sorted_clusters = sorted(boundary_clusters, 
                                   key=lambda x: len(x), 
                                   reverse=True)
            
            # Take median of top two clusters
            boundary1 = int(np.median(sorted_clusters[0]))
            boundary2 = int(np.median(sorted_clusters[1]))
            
            self.consensus_boundaries = sorted([boundary1, boundary2])
        else:
            print("  Warning: Could not find two distinct boundary clusters")
            # Fallback: use median of all boundaries
            if len(all_boundaries) >= 2:
                sorted_boundaries = sorted(all_boundaries)
                mid = len(sorted_boundaries) // 2
                self.consensus_boundaries = [sorted_boundaries[0], sorted_boundaries[-1]]
            else:
                raise ValueError("Insufficient boundaries for consensus")
        
        print(f"  Consensus boundaries: {self.consensus_boundaries} pixels")
        
        # Calculate confidence based on agreement
        self._calculate_confidence(all_boundaries)
    
    def _cluster_boundaries(self, boundaries, tolerance=5):
        """Cluster nearby boundaries"""
        boundaries = sorted(boundaries)
        clusters = []
        
        for b in boundaries:
            # Find cluster within tolerance
            added = False
            for cluster in clusters:
                if any(abs(b - c) <= tolerance for c in cluster):
                    cluster.append(b)
                    added = True
                    break
            
            if not added:
                clusters.append([b])
        
        return clusters
    
    def _calculate_confidence(self, all_boundaries):
        """Calculate confidence scores for consensus boundaries"""
        self.confidence_scores = []
        
        for consensus_boundary in self.consensus_boundaries:
            # Count how many analyses found a boundary near this position
            count = sum(1 for b in all_boundaries 
                       if abs(b - consensus_boundary) <= 5)
            
            confidence = count / len(self.analyses)
            self.confidence_scores.append(confidence)
        
        print(f"  Confidence scores: {[f'{c:.2%}' for c in self.confidence_scores]}")
    
    def create_final_masks(self):
        """Create masks using consensus boundaries"""
        print("\nStep 5: Creating final masks using consensus boundaries...")
        
        # Create distance map
        Y, X = np.ogrid[:self.height, :self.width]
        distance_map = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        self.masks = {}
        
        # Create radii array for indexing
        max_radius = int(np.max(distance_map))
        radii = np.arange(max_radius)
        
        if len(self.consensus_boundaries) >= 2:
            boundary1, boundary2 = self.consensus_boundaries
            
            # Core: inside first boundary
            if boundary1 < len(radii):
                core_radius = radii[boundary1]
                self.masks['core'] = (distance_map <= core_radius).astype(np.uint8) * 255
            
            # Cladding: between boundaries
            if boundary1 < len(radii) and boundary2 < len(radii):
                inner_radius = radii[boundary1]
                outer_radius = radii[boundary2]
                self.masks['cladding'] = ((distance_map > inner_radius) & 
                                         (distance_map <= outer_radius)).astype(np.uint8) * 255
            
            # Ferrule: everything else
            ferrule_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
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
            region = cv2.bitwise_and(self.original, self.original, mask=mask)
            self.extracted_regions[region_name] = region
    
    def generate_output(self, output_dir='robust_analysis_results'):
        """Generate comprehensive output"""
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
            'consensus_boundaries': [int(b) for b in self.consensus_boundaries],
            'confidence_scores': [float(c) for c in self.confidence_scores],
            'num_analyses': len(self.analyses),
            'num_valid_analyses': len([a for a in self.analyses if a['valid']]),
            'individual_results': []
        }
        
        # Add individual analysis results
        for analysis in self.analyses:
            self.results['individual_results'].append({
                'method': analysis['name'],
                'boundaries': analysis.get('boundaries', []),
                'valid': analysis['valid']
            })
        
        # Save JSON report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")
    
    def _create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        # 1. Multi-analysis comparison
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Multi-Threshold Analysis Results', fontsize=16)
        axes = axes.flatten()
        
        # Plot first 9 analyses
        for i, analysis in enumerate(self.analyses[:9]):
            if i < len(axes):
                ax = axes[i]
                
                if analysis['valid'] and 'profile_data' in analysis:
                    profile = analysis['profile_data']['smooth_profile']
                    radii = analysis['profile_data']['radii']
                    
                    ax.plot(radii, profile, 'b-', linewidth=1)
                    
                    # Mark boundaries
                    for b in analysis['boundaries']:
                        ax.axvline(x=b, color='r', linestyle='--', alpha=0.7)
                    
                    ax.set_title(f"{analysis['name']}\nBoundaries: {analysis['boundaries']}")
                else:
                    ax.text(0.5, 0.5, f"{analysis['name']}\nNo valid boundaries", 
                           ha='center', va='center', transform=ax.transAxes)
                
                ax.set_xlabel('Radius (px)')
                ax.set_ylabel('Intensity')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multi_analysis_comparison.png'), dpi=150)
        plt.close()
        
        # 2. Consensus visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Consensus Analysis Results', fontsize=16)
        
        # Use the first valid analysis for visualization
        valid_analysis = next((a for a in self.analyses if a['valid']), None)
        
        if valid_analysis:
            profile_data = valid_analysis['profile_data']
            
            # Intensity profile with consensus boundaries
            ax1 = axes[0, 0]
            ax1.plot(profile_data['radii'], profile_data['mean'], 'b-', 
                    linewidth=2, label='Mean', alpha=0.7)
            ax1.plot(profile_data['radii'], profile_data['median'], 'g-', 
                    linewidth=2, label='Median', alpha=0.7)
            ax1.plot(profile_data['radii'], profile_data['mode'], 'r-', 
                    linewidth=2, label='Mode', alpha=0.7)
            
            # Mark consensus boundaries
            for i, b in enumerate(self.consensus_boundaries):
                ax1.axvline(x=b, color='purple', linestyle='--', linewidth=3,
                           label=f'Consensus {i+1} ({b}px)')
            
            ax1.set_xlabel('Radius (pixels)')
            ax1.set_ylabel('Intensity')
            ax1.set_title('Statistical Profiles with Consensus Boundaries')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Variance profile
            ax2 = axes[0, 1]
            ax2.plot(profile_data['radii'], profile_data['variance'], 'orange', linewidth=2)
            ax2.set_xlabel('Radius (pixels)')
            ax2.set_ylabel('Variance')
            ax2.set_title('Radial Variance Profile')
            ax2.grid(True, alpha=0.3)
            
            for b in self.consensus_boundaries:
                ax2.axvline(x=b, color='purple', linestyle='--', linewidth=2)
        
        # Boundary histogram
        ax3 = axes[1, 0]
        all_boundaries = []
        for a in self.analyses:
            if a['valid']:
                all_boundaries.extend(a['boundaries'])
        
        if all_boundaries:
            ax3.hist(all_boundaries, bins=50, alpha=0.7, color='blue', edgecolor='black')
            
            # Mark consensus boundaries
            for b in self.consensus_boundaries:
                ax3.axvline(x=b, color='red', linestyle='--', linewidth=3,
                           label=f'Consensus: {b}px')
            
            ax3.set_xlabel('Boundary Position (pixels)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Boundary Detection Histogram')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Confidence visualization
        ax4 = axes[1, 1]
        if self.confidence_scores:
            bars = ax4.bar(['Boundary 1', 'Boundary 2'], 
                           self.confidence_scores,
                           color=['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' 
                                  for c in self.confidence_scores])
            
            # Add percentage labels
            for bar, score in zip(bars, self.confidence_scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.1%}', ha='center', va='bottom')
            
            ax4.set_ylabel('Confidence Score')
            ax4.set_title('Boundary Detection Confidence')
            ax4.set_ylim(0, 1.1)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'consensus_analysis.png'), dpi=150)
        plt.close()
        
        # 3. Final results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Robust Detection Final Results', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[0, 0].plot(self.center_x, self.center_y, 'r+', markersize=10, markeredgewidth=2)
        
        # Mask overlay
        mask_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if 'core' in self.masks:
            mask_overlay[self.masks['core'] > 0] = [255, 0, 0]
        if 'cladding' in self.masks:
            mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0]
        if 'ferrule' in self.masks:
            mask_overlay[self.masks['ferrule'] > 0] = [0, 0, 255]
        
        axes[0, 1].imshow(mask_overlay)
        axes[0, 1].set_title('Detected Regions (R=Core, G=Cladding, B=Ferrule)')
        axes[0, 1].axis('off')
        
        # Cladding region
        if 'cladding' in self.extracted_regions:
            axes[1, 0].imshow(cv2.cvtColor(self.extracted_regions['cladding'], cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Extracted Cladding')
            axes[1, 0].axis('off')
        
        # Method comparison text
        ax_text = axes[1, 1]
        ax_text.axis('off')
        
        summary_text = "Analysis Summary:\n\n"
        summary_text += f"Total preprocessing methods: {len(self.analyses)}\n"
        summary_text += f"Valid detections: {len([a for a in self.analyses if a['valid']])}\n"
        summary_text += f"\nConsensus boundaries: {self.consensus_boundaries} px\n"
        summary_text += f"Confidence: {[f'{c:.1%}' for c in self.confidence_scores]}\n\n"
        
        summary_text += "Individual Results:\n"
        for a in self.analyses[:5]:  # Show first 5
            summary_text += f"{a['name']}: {a['boundaries'] if a['valid'] else 'Failed'}\n"
        
        if len(self.analyses) > 5:
            summary_text += f"... and {len(self.analyses) - 5} more"
        
        ax_text.text(0.1, 0.9, summary_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_results.png'), dpi=150)
        plt.close()


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Create test image
        image_path = 'test_fiber.jpg'
        
        if not os.path.exists(image_path):
            print("Creating test fiber image with artifacts...")
            
            size = 400
            img = np.zeros((size, size, 3), dtype=np.uint8)
            center = size // 2
            
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - center)**2 + (Y - center)**2)
            
            # Base intensity pattern
            intensity = np.ones((size, size)) * 140
            
            # Core
            core_mask = dist < 50
            intensity[core_mask] = 160
            
            # Smooth transition to cladding
            for r in range(45, 55):
                mask = (dist >= r) & (dist < r + 1)
                t = (r - 45) / 10
                intensity[mask] = 160 - 30 * (3*t**2 - 2*t**3)
            
            # Cladding
            cladding_mask = (dist >= 55) & (dist < 95)
            intensity[cladding_mask] = 130
            
            # Smooth transition to ferrule
            for r in range(90, 100):
                mask = (dist >= r) & (dist < r + 1)
                t = (r - 90) / 10
                intensity[mask] = 130 + 10 * (3*t**2 - 2*t**3)
            
            # Add some artifacts (defects)
            # Sharp gradient artifact
            artifact_mask = (dist >= 120) & (dist < 125)
            intensity[artifact_mask] = 80  # Sharp dip
            
            # Add contamination spots
            for _ in range(10):
                x = np.random.randint(50, size-50)
                y = np.random.randint(50, size-50)
                radius = np.random.randint(3, 8)
                cv2.circle(intensity, (x, y), radius, 
                          np.random.randint(50, 200), -1)
            
            # Apply to all channels
            for i in range(3):
                img[:, :, i] = intensity.astype(np.uint8)
            
            # Add noise
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(image_path, img)
            print(f"Test image with artifacts saved as {image_path}")
    
    # Run analysis
    try:
        analyzer = RobustFiberAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("ROBUST ANALYSIS SUMMARY:")
        print("=" * 60)
        print(f"Consensus boundaries: {results['consensus_boundaries']} pixels")
        print(f"Confidence scores: {[f'{c:.1%}' for c in results['confidence_scores']]}")
        print(f"Valid analyses: {results['num_valid_analyses']}/{results['num_analyses']}")
        
        print("\nBoundary agreement across methods:")
        for i, boundary in enumerate(results['consensus_boundaries']):
            agreements = sum(1 for r in results['individual_results'] 
                           if r['valid'] and any(abs(b - boundary) <= 5 for b in r['boundaries']))
            print(f"  Boundary {i+1} ({boundary}px): {agreements}/{results['num_valid_analyses']} methods agree")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
