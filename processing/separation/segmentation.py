import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from scipy.optimize import least_squares, minimize
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings('ignore')
import json
import os

class ImprovedUnifiedFiberSeparator:
    """
    Improved Unified Fiber Optic Region Separator
    
    Key improvements:
    - Better center finding with validation
    - Minimum boundary separation constraints
    - Intensity pattern validation
    - Robust peak detection with physical constraints
    - Adaptive smoothing based on noise level
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Reinforcement learning parameters
        self.max_iterations = 50
        self.convergence_threshold = 0.5  # pixels
        self.alignment_reward_factor = 10.0
        self.misalignment_penalty_factor = 5.0
        
        # Physical constraints for fiber optics
        self.min_core_radius = 20  # Minimum expected core radius
        self.min_cladding_thickness = 20  # Minimum cladding thickness
        self.max_cladding_radius = min(self.width, self.height) * 0.4  # Maximum reasonable cladding
        
        print(f"Loaded image: {self.width}x{self.height} pixels")
    
    def analyze(self):
        """Run complete unified analysis with improvements"""
        print("\n" + "="*70)
        print("IMPROVED UNIFIED FIBER OPTIC SEPARATOR")
        print("="*70)
        
        # Stage 1: Find and validate center
        self.find_and_validate_center()
        
        # Stage 2: Create preprocessing variants with noise assessment
        self.create_preprocessing_variants()
        
        # Stage 3: Find data-driven boundaries with constraints
        self.find_constrained_boundaries()
        self.establish_ground_truth_boundaries()
        
        # Stage 4: Validate intensity pattern
        if not self.validate_intensity_pattern():
            print("\n⚠ Warning: Intensity pattern doesn't match typical fiber optic profile")
            print("  Attempting alternative analysis...")
            self.alternative_boundary_detection()
        
        # Stage 5: Initialize geometric parameters
        self.initialize_geometric_fitting()
        
        # Stage 6: Reinforcement alignment loop
        self.reinforcement_alignment_loop()
        
        # Stage 7: Create final masks with converged boundaries
        self.create_final_masks()
        
        # Stage 8: Apply binary artifact removal
        self.apply_artifact_removal()
        
        # Stage 9: Extract regions and generate output
        self.extract_regions_and_output()
        
        print("\nAnalysis Complete!")
        return self.results
    
    def find_and_validate_center(self):
        """Find center with multiple methods and validate"""
        print("\nStage 1: Finding and Validating Center")
        print("-" * 50)
        
        centers = []
        weights = []
        
        # Method 1: Brightness centroid (most reliable for good images)
        smoothed = cv2.GaussianBlur(self.gray, (9, 9), 2)
        threshold = np.percentile(smoothed, 95)
        bright_mask = smoothed > threshold
        
        moments = cv2.moments(bright_mask.astype(np.uint8))
        if moments['m00'] > 0:
            cx1 = moments['m10'] / moments['m00']
            cy1 = moments['m01'] / moments['m00']
            centers.append((cx1, cy1))
            weights.append(2.0)  # High weight
            print(f"  Brightness centroid: ({cx1:.1f}, {cy1:.1f})")
        
        # Method 2: Maximum intensity location
        # Apply heavy smoothing to find general bright area
        heavily_smoothed = cv2.GaussianBlur(self.gray, (31, 31), 10)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heavily_smoothed)
        centers.append(max_loc)
        weights.append(1.0)
        print(f"  Maximum intensity: {max_loc}")
        
        # Method 3: Weighted centroid of bright pixels
        # Use top 10% brightest pixels
        threshold_10 = np.percentile(self.gray, 90)
        bright_pixels_mask = self.gray > threshold_10
        y_coords, x_coords = np.where(bright_pixels_mask)
        if len(x_coords) > 0:
            # Weight by intensity
            intensities = self.gray[bright_pixels_mask]
            cx3 = np.average(x_coords, weights=intensities)
            cy3 = np.average(y_coords, weights=intensities)
            centers.append((cx3, cy3))
            weights.append(1.5)
            print(f"  Weighted centroid: ({cx3:.1f}, {cy3:.1f})")
        
        # Calculate weighted average center
        if centers:
            centers_array = np.array(centers)
            weights_array = np.array(weights)
            self.center_x = np.average(centers_array[:, 0], weights=weights_array)
            self.center_y = np.average(centers_array[:, 1], weights=weights_array)
        else:
            # Fallback to image center
            self.center_x = self.width / 2
            self.center_y = self.height / 2
        
        print(f"\n  Final center: ({self.center_x:.1f}, {self.center_y:.1f})")
        
        # Validate center is reasonable
        margin = 0.2  # Center should be within middle 60% of image
        if (self.center_x < self.width * margin or self.center_x > self.width * (1-margin) or
            self.center_y < self.height * margin or self.center_y > self.height * (1-margin)):
            print("  ⚠ Warning: Center is near image edge, adjusting...")
            self.center_x = np.clip(self.center_x, self.width * margin, self.width * (1-margin))
            self.center_y = np.clip(self.center_y, self.height * margin, self.height * (1-margin))
            print(f"  Adjusted center: ({self.center_x:.1f}, {self.center_y:.1f})")
    
    def create_preprocessing_variants(self):
        """Create preprocessing variants with noise assessment"""
        print("\nStage 2: Creating Preprocessing Variants")
        print("-" * 50)
        
        # Assess noise level in image
        noise_level = self._estimate_noise_level()
        print(f"  Estimated noise level: {noise_level:.2f}")
        
        self.preprocessing_variants = []
        
        # Adjust preprocessing based on noise
        if noise_level < 5:
            # Low noise - use minimal preprocessing
            sigma_values = [0.5, 1, 1.5]
            median_sizes = [3]
        elif noise_level < 10:
            # Medium noise
            sigma_values = [1, 2, 3]
            median_sizes = [3, 5]
        else:
            # High noise - stronger preprocessing
            sigma_values = [2, 3, 4]
            median_sizes = [5, 7]
        
        # Always include original
        self.preprocessing_variants.append({
            'name': 'Original',
            'image': self.gray.copy()
        })
        
        # Gaussian variants
        for sigma in sigma_values:
            blurred = cv2.GaussianBlur(self.gray, (0, 0), sigma)
            self.preprocessing_variants.append({
                'name': f'Gaussian_σ{sigma}',
                'image': blurred
            })
        
        # Median filter variants
        for size in median_sizes:
            filtered = median_filter(self.gray, size=size)
            self.preprocessing_variants.append({
                'name': f'Median_{size}',
                'image': filtered
            })
        
        # Bilateral filter
        bilateral = cv2.bilateralFilter(self.gray, 9, 75, 75)
        self.preprocessing_variants.append({
            'name': 'Bilateral',
            'image': bilateral
        })
        
        # Savitzky-Golay filter (preserves features while smoothing)
        # Apply to each row and column
        savgol = self.gray.copy().astype(float)
        for i in range(self.height):
            if self.width > 21:  # Ensure window size is appropriate
                savgol[i, :] = savgol_filter(savgol[i, :], 21, 3)
        for j in range(self.width):
            if self.height > 21:
                savgol[:, j] = savgol_filter(savgol[:, j], 21, 3)
        self.preprocessing_variants.append({
            'name': 'Savitzky-Golay',
            'image': savgol.astype(np.uint8)
        })
        
        print(f"  Created {len(self.preprocessing_variants)} preprocessing variants")
    
    def _estimate_noise_level(self):
        """Estimate noise level in image using local variance"""
        # Use Median Absolute Deviation (MAD) for robust noise estimation
        # Apply to high-frequency component
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 1)
        diff = self.gray.astype(float) - blurred.astype(float)
        
        # MAD estimation
        mad = np.median(np.abs(diff - np.median(diff)))
        noise_estimate = 1.4826 * mad  # Scale factor for Gaussian noise
        
        return noise_estimate
    
    def find_constrained_boundaries(self):
        """Find boundaries with physical constraints"""
        print("\nStage 3: Finding Boundaries with Physical Constraints")
        print("-" * 50)
        
        self.all_boundaries = []
        
        for variant in self.preprocessing_variants:
            print(f"\n  Processing variant: {variant['name']}")
            
            # Compute radial profiles with larger smoothing window
            profiles = self._compute_robust_radial_profiles(variant['image'])
            
            # Method 1: Second derivative minima with constraints
            second_deriv_boundaries = self._find_constrained_second_derivative_minima(profiles)
            if second_deriv_boundaries and len(second_deriv_boundaries) >= 2:
                self.all_boundaries.append({
                    'method': '2nd_derivative',
                    'variant': variant['name'],
                    'boundaries': sorted(second_deriv_boundaries[:2]),
                    'confidence': 1.0,
                    'profiles': profiles
                })
                print(f"    2nd derivative minima: {sorted(second_deriv_boundaries[:2])}")
            
            # Method 2: Gradient peaks with separation constraint
            gradient_boundaries = self._find_constrained_gradient_peaks(profiles)
            if gradient_boundaries and len(gradient_boundaries) >= 2:
                self.all_boundaries.append({
                    'method': 'gradient',
                    'variant': variant['name'],
                    'boundaries': sorted(gradient_boundaries[:2]),
                    'confidence': 0.8,
                    'profiles': profiles
                })
                print(f"    Gradient peaks: {sorted(gradient_boundaries[:2])}")
            
            # Method 3: Intensity drop analysis (new method)
            intensity_boundaries = self._find_intensity_drop_boundaries(profiles)
            if intensity_boundaries and len(intensity_boundaries) >= 2:
                self.all_boundaries.append({
                    'method': 'intensity_drop',
                    'variant': variant['name'],
                    'boundaries': sorted(intensity_boundaries[:2]),
                    'confidence': 0.7,
                    'profiles': profiles
                })
                print(f"    Intensity drops: {sorted(intensity_boundaries[:2])}")
    
    def _compute_robust_radial_profiles(self, img):
        """Compute radial profiles with improved robustness"""
        max_radius = int(min(self.center_x, self.center_y,
                            self.width - self.center_x,
                            self.height - self.center_y))
        
        radii = np.arange(max_radius)
        num_angles = 360
        angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        intensity_profiles = []
        
        for angle in angles:
            x_coords = self.center_x + radii * np.cos(angle)
            y_coords = self.center_y + radii * np.sin(angle)
            
            intensities = []
            for x, y in zip(x_coords, y_coords):
                if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                    # Use bilinear interpolation for smoother profiles
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)
                    
                    dx, dy = x - x0, y - y0
                    
                    intensity = (1 - dx) * (1 - dy) * img[y0, x0] + \
                               dx * (1 - dy) * img[y0, x1] + \
                               (1 - dx) * dy * img[y1, x0] + \
                               dx * dy * img[y1, x1]
                    
                    intensities.append(intensity)
            
            if len(intensities) == len(radii):
                intensity_profiles.append(intensities)
        
        # Use median for robustness
        median_intensity = np.median(intensity_profiles, axis=0)
        
        # Adaptive smoothing based on profile characteristics
        # Stronger smoothing for noisier profiles
        profile_noise = np.std(np.diff(median_intensity))
        if profile_noise > 5:
            sigma = 3
        elif profile_noise > 2:
            sigma = 2
        else:
            sigma = 1.5
        
        smooth_intensity = gaussian_filter1d(median_intensity, sigma=sigma)
        
        # Compute derivatives with appropriate smoothing
        first_derivative = np.gradient(smooth_intensity)
        first_derivative_smooth = gaussian_filter1d(first_derivative, sigma=1)
        
        second_derivative = np.gradient(first_derivative_smooth)
        second_derivative_smooth = gaussian_filter1d(second_derivative, sigma=1)
        
        return {
            'radii': radii,
            'intensity': median_intensity,
            'intensity_smooth': smooth_intensity,
            'gradient': np.abs(first_derivative_smooth),
            'gradient_signed': first_derivative_smooth,
            'second_derivative': second_derivative_smooth
        }
    
    def _find_constrained_second_derivative_minima(self, profiles):
        """Find second derivative minima with separation constraints"""
        second_deriv = profiles['second_derivative']
        inverted = -second_deriv
        
        # Find all peaks (minima in original)
        peaks, properties = find_peaks(
            inverted,
            distance=self.min_cladding_thickness,  # Enforce minimum separation
            prominence=np.std(inverted) * 0.1      # Require some prominence
        )
        
        if len(peaks) == 0:
            return None
        
        # Filter peaks by physical constraints
        valid_peaks = []
        for peak in peaks:
            if (peak > self.min_core_radius and 
                peak < self.max_cladding_radius):
                valid_peaks.append(peak)
        
        if len(valid_peaks) < 2:
            return None
        
        # Sort by prominence
        prominences = []
        for peak in valid_peaks:
            prom = peak_prominences(inverted, [peak])[0][0]
            prominences.append(prom)
        
        sorted_indices = np.argsort(prominences)[::-1]
        sorted_peaks = [valid_peaks[i] for i in sorted_indices]
        
        # Ensure proper separation
        selected_peaks = [sorted_peaks[0]]
        for peak in sorted_peaks[1:]:
            if abs(peak - selected_peaks[0]) >= self.min_cladding_thickness:
                selected_peaks.append(peak)
                break
        
        return selected_peaks if len(selected_peaks) >= 2 else None
    
    def _find_constrained_gradient_peaks(self, profiles):
        """Find gradient peaks with separation constraints"""
        gradient = profiles['gradient']
        
        # Find peaks with constraints
        peaks, properties = find_peaks(
            gradient,
            distance=self.min_cladding_thickness,
            height=np.percentile(gradient, 75)  # Only consider significant peaks
        )
        
        if len(peaks) < 2:
            return None
        
        # Filter by location constraints
        valid_peaks = []
        for peak in peaks:
            if (peak > self.min_core_radius and 
                peak < self.max_cladding_radius):
                valid_peaks.append(peak)
        
        if len(valid_peaks) < 2:
            return None
        
        # Sort by peak height
        peak_heights = gradient[valid_peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        
        return [valid_peaks[sorted_indices[0]], valid_peaks[sorted_indices[1]]]
    
    def _find_intensity_drop_boundaries(self, profiles):
        """Find boundaries based on intensity drops"""
        intensity = profiles['intensity_smooth']
        
        # Look for regions where intensity drops significantly
        # Calculate relative intensity change
        rel_change = np.zeros_like(intensity)
        for i in range(1, len(intensity)):
            if intensity[i-1] > 0:
                rel_change[i] = (intensity[i-1] - intensity[i]) / intensity[i-1]
        
        # Find peaks in relative change
        peaks, _ = find_peaks(
            rel_change,
            distance=self.min_cladding_thickness,
            height=0.05  # At least 5% relative drop
        )
        
        if len(peaks) < 2:
            return None
        
        # Sort by drop magnitude
        drop_magnitudes = rel_change[peaks]
        sorted_indices = np.argsort(drop_magnitudes)[::-1]
        
        return [peaks[sorted_indices[0]], peaks[sorted_indices[1]]]
    
    def validate_intensity_pattern(self):
        """Validate that intensity pattern matches fiber optic expectations"""
        print("\nValidating Intensity Pattern")
        print("-" * 50)
        
        if not hasattr(self, 'ground_truth_boundaries') or len(self.ground_truth_boundaries) < 2:
            return False
        
        # Sample intensities at three regions
        core_radius = self.ground_truth_boundaries[0] - 10
        cladding_radius = (self.ground_truth_boundaries[0] + self.ground_truth_boundaries[1]) // 2
        ferrule_radius = self.ground_truth_boundaries[1] + 10
        
        intensities = []
        for test_radius in [core_radius, cladding_radius, ferrule_radius]:
            if test_radius >= 0 and test_radius < min(self.width/2, self.height/2):
                # Sample at multiple angles
                angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
                samples = []
                for angle in angles:
                    x = int(self.center_x + test_radius * np.cos(angle))
                    y = int(self.center_y + test_radius * np.sin(angle))
                    if 0 <= x < self.width and 0 <= y < self.height:
                        samples.append(self.gray[y, x])
                
                if samples:
                    intensities.append(np.median(samples))
                else:
                    intensities.append(None)
            else:
                intensities.append(None)
        
        if all(i is not None for i in intensities):
            core_int, clad_int, ferr_int = intensities
            print(f"  Sampled intensities - Core: {core_int:.1f}, Cladding: {clad_int:.1f}, Ferrule: {ferr_int:.1f}")
            
            # Expected pattern: Cladding should be darkest
            # Core should be brightest or at least brighter than cladding
            # Ferrule should be between core and cladding or similar to core
            
            pattern_valid = (clad_int < core_int) and (clad_int < ferr_int)
            
            if pattern_valid:
                print("  ✓ Intensity pattern matches expected fiber optic profile")
                return True
            else:
                print("  ✗ Intensity pattern does not match expected profile")
                return False
        
        return False
    
    def alternative_boundary_detection(self):
        """Alternative boundary detection when standard pattern fails"""
        print("\nApplying Alternative Boundary Detection")
        print("-" * 50)
        
        # Method 1: Look for concentric rings of similar intensity
        # This handles cases where the fiber might have unusual coatings or damage
        
        # Compute radial intensity profile
        max_radius = int(min(self.center_x, self.center_y,
                            self.width - self.center_x,
                            self.height - self.center_y))
        
        radii = np.arange(max_radius)
        radial_profile = []
        radial_variance = []
        
        for r in radii:
            # Sample points at this radius
            angles = np.linspace(0, 2*np.pi, max(8, int(2 * np.pi * r / 10)), endpoint=False)
            samples = []
            
            for angle in angles:
                x = int(self.center_x + r * np.cos(angle))
                y = int(self.center_y + r * np.sin(angle))
                if 0 <= x < self.width and 0 <= y < self.height:
                    samples.append(self.gray[y, x])
            
            if samples:
                radial_profile.append(np.median(samples))
                radial_variance.append(np.std(samples))
            else:
                radial_profile.append(0)
                radial_variance.append(0)
        
        radial_profile = np.array(radial_profile)
        radial_variance = np.array(radial_variance)
        
        # Smooth profiles
        radial_profile_smooth = gaussian_filter1d(radial_profile, sigma=2)
        radial_variance_smooth = gaussian_filter1d(radial_variance, sigma=2)
        
        # Look for regions of low variance (uniform regions)
        # These often indicate distinct fiber regions
        variance_threshold = np.percentile(radial_variance_smooth, 30)
        low_variance = radial_variance_smooth < variance_threshold
        
        # Find transitions between low and high variance regions
        transitions = np.diff(low_variance.astype(int))
        rising_edges = np.where(transitions == 1)[0]
        falling_edges = np.where(transitions == -1)[0]
        
        # Combine with gradient information
        gradient = np.abs(np.gradient(radial_profile_smooth))
        gradient_peaks, _ = find_peaks(gradient, distance=20)
        
        # Find best boundary candidates
        boundary_candidates = []
        
        # Add variance-based boundaries
        for edge in np.concatenate([rising_edges, falling_edges]):
            if self.min_core_radius < edge < self.max_cladding_radius:
                boundary_candidates.append(edge)
        
        # Add gradient-based boundaries
        for peak in gradient_peaks:
            if self.min_core_radius < peak < self.max_cladding_radius:
                boundary_candidates.append(peak)
        
        # Cluster nearby candidates
        if boundary_candidates:
            boundary_candidates = sorted(set(boundary_candidates))
            
            # Merge nearby candidates
            merged_boundaries = []
            last_boundary = boundary_candidates[0]
            cluster = [last_boundary]
            
            for boundary in boundary_candidates[1:]:
                if boundary - last_boundary < 10:  # Within 10 pixels
                    cluster.append(boundary)
                else:
                    merged_boundaries.append(int(np.median(cluster)))
                    cluster = [boundary]
                last_boundary = boundary
            
            if cluster:
                merged_boundaries.append(int(np.median(cluster)))
            
            # Select two boundaries with proper separation
            if len(merged_boundaries) >= 2:
                # Sort by gradient magnitude at boundary
                boundary_scores = [(b, gradient[b]) for b in merged_boundaries]
                boundary_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Select boundaries with minimum separation
                selected = [boundary_scores[0][0]]
                for boundary, score in boundary_scores[1:]:
                    if abs(boundary - selected[0]) >= self.min_cladding_thickness:
                        selected.append(boundary)
                        break
                
                if len(selected) >= 2:
                    alternative_boundaries = sorted(selected)
                    print(f"  Alternative boundaries found: {alternative_boundaries}")
                    
                    # Add to boundary detections
                    self.all_boundaries.append({
                        'method': 'alternative',
                        'variant': 'variance_analysis',
                        'boundaries': alternative_boundaries,
                        'confidence': 0.6,
                        'profiles': {
                            'radii': radii,
                            'intensity': radial_profile,
                            'intensity_smooth': radial_profile_smooth,
                            'gradient': gradient,
                            'variance': radial_variance_smooth
                        }
                    })
    
    def establish_ground_truth_boundaries(self):
        """Establish consensus ground truth from all detections"""
        print("\nEstablishing Ground Truth Boundaries")
        print("-" * 50)
        
        if not self.all_boundaries:
            raise ValueError("No boundaries detected! Image may not be a fiber optic end face.")
        
        # Group boundaries by method for analysis
        method_groups = {}
        for detection in self.all_boundaries:
            method = detection['method']
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].extend(detection['boundaries'])
        
        print("\nBoundary detections by method:")
        for method, boundaries in method_groups.items():
            print(f"  {method}: {len(boundaries)//2} detections")
        
        # Weight boundaries by confidence and method
        weighted_inner_boundaries = []
        weighted_outer_boundaries = []
        
        for detection in self.all_boundaries:
            weight = detection['confidence']
            # 2nd derivative gets extra weight
            if detection['method'] == '2nd_derivative':
                weight *= 1.5
            
            # Ensure boundaries are properly ordered
            b1, b2 = sorted(detection['boundaries'][:2])
            
            # Add weighted votes
            weighted_inner_boundaries.extend([b1] * int(weight * 10))
            weighted_outer_boundaries.extend([b2] * int(weight * 10))
        
        # Find consensus for each boundary separately
        if weighted_inner_boundaries and weighted_outer_boundaries:
            # Use robust statistics (median) for consensus
            inner_boundary = int(np.median(weighted_inner_boundaries))
            outer_boundary = int(np.median(weighted_outer_boundaries))
            
            # Ensure minimum separation
            if outer_boundary - inner_boundary < self.min_cladding_thickness:
                print(f"  Warning: Boundaries too close ({inner_boundary}, {outer_boundary})")
                print(f"  Enforcing minimum separation of {self.min_cladding_thickness}px")
                
                # Adjust boundaries to maintain minimum separation
                center = (inner_boundary + outer_boundary) / 2
                inner_boundary = int(center - self.min_cladding_thickness / 2)
                outer_boundary = int(center + self.min_cladding_thickness / 2)
            
            self.ground_truth_boundaries = [inner_boundary, outer_boundary]
        else:
            raise ValueError("Could not establish consensus boundaries")
        
        print(f"  Ground truth boundaries: {self.ground_truth_boundaries} pixels")
        print(f"  Cladding thickness: {self.ground_truth_boundaries[1] - self.ground_truth_boundaries[0]} pixels")
    
    def initialize_geometric_fitting(self):
        """Initialize geometric parameters for fitting"""
        print("\nStage 4: Initializing Geometric Fitting")
        print("-" * 50)
        
        self.geometric_params = {
            'center_x': self.center_x,
            'center_y': self.center_y,
            'radius1': float(self.ground_truth_boundaries[0]),
            'radius2': float(self.ground_truth_boundaries[1])
        }
        
        self.alignment_score = self._calculate_alignment_score()
        
        print(f"  Initial geometric parameters:")
        print(f"    Center: ({self.geometric_params['center_x']:.1f}, "
              f"{self.geometric_params['center_y']:.1f})")
        print(f"    Radii: {self.geometric_params['radius1']:.1f}, "
              f"{self.geometric_params['radius2']:.1f}")
        print(f"  Initial alignment score: {self.alignment_score:.3f}")
    
    def reinforcement_alignment_loop(self):
        """Main reinforcement loop with improved convergence"""
        print("\nStage 5: Reinforcement Alignment Loop")
        print("-" * 50)
        
        best_params = self.geometric_params.copy()
        best_score = self.alignment_score
        
        # Track convergence history
        score_history = [self.alignment_score]
        
        for iteration in range(self.max_iterations):
            print(f"\n  Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get edge points for current geometric fitting
            edge_points = self._get_edge_points_near_boundaries()
            
            if len(edge_points) > 50:
                # Fit circles to edge points
                new_params = self._fit_circles_with_constraints(edge_points)
                
                # Calculate new alignment score
                old_score = self.alignment_score
                self.geometric_params = new_params
                self.alignment_score = self._calculate_alignment_score()
                
                score_history.append(self.alignment_score)
                
                # Reinforcement logic
                improvement = self.alignment_score - old_score
                
                if improvement > 0:
                    print(f"    ✓ Reward! Score improved by {improvement:.3f}")
                    best_params = new_params.copy()
                    best_score = self.alignment_score
                    self._apply_reinforcement(improvement)
                else:
                    print(f"    ✗ Penalty! Score decreased by {-improvement:.3f}")
                    self._apply_penalty(improvement)
                
                print(f"    Current score: {self.alignment_score:.3f}")
                print(f"    Distance from ground truth: "
                      f"R1={abs(self.geometric_params['radius1'] - self.ground_truth_boundaries[0]):.2f}px, "
                      f"R2={abs(self.geometric_params['radius2'] - self.ground_truth_boundaries[1]):.2f}px")
                
                # Check convergence
                if (abs(self.geometric_params['radius1'] - self.ground_truth_boundaries[0]) < self.convergence_threshold and
                    abs(self.geometric_params['radius2'] - self.ground_truth_boundaries[1]) < self.convergence_threshold):
                    print("\n  ✓ Converged to ground truth boundaries!")
                    break
                
                # Check if stuck in local minimum
                if len(score_history) > 5:
                    recent_scores = score_history[-5:]
                    if np.std(recent_scores) < 0.001:
                        print("    Stuck in local minimum, applying perturbation...")
                        self._apply_perturbation()
            else:
                print("    Insufficient edge points, adjusting directly toward ground truth")
                self._adjust_toward_ground_truth()
        
        # Use best parameters found
        self.geometric_params = best_params
        self.alignment_score = best_score
        
        print(f"\n  Final alignment score: {self.alignment_score:.3f}")
        print(f"  Final geometric parameters:")
        print(f"    Center: ({self.geometric_params['center_x']:.1f}, "
              f"{self.geometric_params['center_y']:.1f})")
        print(f"    Radii: {self.geometric_params['radius1']:.1f}, "
              f"{self.geometric_params['radius2']:.1f}")
    
    def _calculate_alignment_score(self):
        """Calculate alignment score between geometric and data-driven boundaries"""
        dist1 = abs(self.geometric_params['radius1'] - self.ground_truth_boundaries[0])
        dist2 = abs(self.geometric_params['radius2'] - self.ground_truth_boundaries[1])
        
        # Exponential decay score
        score = np.exp(-(dist1 + dist2) / 10.0)
        
        # Bonus for maintaining minimum separation
        separation = self.geometric_params['radius2'] - self.geometric_params['radius1']
        if separation >= self.min_cladding_thickness:
            score *= 1.1
        
        return min(score, 1.0)
    
    def _get_edge_points_near_boundaries(self):
        """Get edge points with improved detection"""
        # Multi-scale edge detection
        edges_combined = np.zeros((self.height, self.width), dtype=float)
        
        # Canny at multiple scales
        for sigma in [1.0, 1.5, 2.0]:
            blurred = cv2.GaussianBlur(self.gray, (0, 0), sigma)
            edges = cv2.Canny(blurred, 50, 150)
            edges_combined += edges.astype(float) / 255.0
        
        # Normalize
        edges_combined /= 3.0
        
        # Threshold
        edge_mask = edges_combined > 0.3
        
        # Get points near boundaries
        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - self.geometric_params['center_x'])**2 + 
                                  (Y - self.geometric_params['center_y'])**2)
        
        # Points near boundaries with adaptive tolerance
        tolerance = max(10, self.min_cladding_thickness * 0.3)
        
        near_boundary1 = (np.abs(dist_from_center - self.geometric_params['radius1']) < tolerance) & edge_mask
        near_boundary2 = (np.abs(dist_from_center - self.geometric_params['radius2']) < tolerance) & edge_mask
        
        points_mask = near_boundary1 | near_boundary2
        points = np.column_stack(np.where(points_mask))
        
        return points[:, [1, 0]]  # Convert to (x, y)
    
    def _fit_circles_with_constraints(self, edge_points):
        """Fit circles with minimum separation constraint"""
        cx, cy = self.geometric_params['center_x'], self.geometric_params['center_y']
        distances = np.sqrt((edge_points[:, 0] - cx)**2 + (edge_points[:, 1] - cy)**2)
        
        # Cluster points
        mid_radius = (self.geometric_params['radius1'] + self.geometric_params['radius2']) / 2
        inner_points = edge_points[distances < mid_radius]
        outer_points = edge_points[distances >= mid_radius]
        
        new_params = self.geometric_params.copy()
        
        # Fit inner circle
        if len(inner_points) > 10:
            inner_circle = self._fit_single_circle_robust(inner_points)
            if inner_circle is not None:
                new_params['center_x'] = inner_circle[0]
                new_params['center_y'] = inner_circle[1]
                new_params['radius1'] = inner_circle[2]
        
        # Fit outer circle
        if len(outer_points) > 10:
            outer_circle = self._fit_single_circle_robust(outer_points)
            if outer_circle is not None:
                # Use weighted average center
                new_params['center_x'] = 0.5 * new_params['center_x'] + 0.5 * outer_circle[0]
                new_params['center_y'] = 0.5 * new_params['center_y'] + 0.5 * outer_circle[1]
                new_params['radius2'] = outer_circle[2]
        
        # Enforce minimum separation
        if new_params['radius2'] - new_params['radius1'] < self.min_cladding_thickness:
            # Adjust to maintain separation
            center_radius = (new_params['radius1'] + new_params['radius2']) / 2
            new_params['radius1'] = center_radius - self.min_cladding_thickness / 2
            new_params['radius2'] = center_radius + self.min_cladding_thickness / 2
        
        return new_params
    
    def _fit_single_circle_robust(self, points):
        """Robust circle fitting using RANSAC + least squares"""
        best_circle = None
        best_inliers = 0
        
        # RANSAC iterations
        for _ in range(100):
            # Random sample
            if len(points) < 3:
                break
            
            sample_idx = np.random.choice(len(points), min(3, len(points)), replace=False)
            sample_points = points[sample_idx]
            
            # Fit circle to sample
            if len(sample_points) >= 3:
                # Calculate circle from 3 points
                p1, p2, p3 = sample_points[:3]
                
                # Calculate circle center and radius
                ax, ay = p1
                bx, by = p2
                cx, cy = p3
                
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-10:
                    continue
                
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                
                radius = np.sqrt((ux - ax)**2 + (uy - ay)**2)
                
                # Count inliers
                distances = np.sqrt((points[:, 0] - ux)**2 + (points[:, 1] - uy)**2)
                inliers = np.abs(distances - radius) < 5
                num_inliers = np.sum(inliers)
                
                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    best_circle = [ux, uy, radius]
        
        # Refine with least squares on all inliers
        if best_circle is not None and best_inliers > len(points) * 0.3:
            def residuals(params, points):
                cx, cy, r = params
                distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
                return distances - r
            
            try:
                result = least_squares(residuals, best_circle, args=(points,))
                if result.success:
                    return result.x
            except:
                pass
        
        return best_circle
    
    def _apply_reinforcement(self, improvement):
        """Apply reinforcement with adaptive learning rate"""
        # Adaptive learning rate based on improvement magnitude
        alpha = min(0.3, improvement * self.alignment_reward_factor)
        
        # Move toward ground truth
        self.geometric_params['radius1'] = (1 - alpha) * self.geometric_params['radius1'] + \
                                          alpha * self.ground_truth_boundaries[0]
        self.geometric_params['radius2'] = (1 - alpha) * self.geometric_params['radius2'] + \
                                          alpha * self.ground_truth_boundaries[1]
    
    def _apply_penalty(self, improvement):
        """Apply penalty with stronger correction"""
        alpha = min(0.5, abs(improvement) * self.misalignment_penalty_factor)
        
        self.geometric_params['radius1'] = (1 - alpha) * self.geometric_params['radius1'] + \
                                          alpha * self.ground_truth_boundaries[0]
        self.geometric_params['radius2'] = (1 - alpha) * self.geometric_params['radius2'] + \
                                          alpha * self.ground_truth_boundaries[1]
    
    def _apply_perturbation(self):
        """Apply small perturbation to escape local minima"""
        noise_scale = 2.0
        self.geometric_params['radius1'] += np.random.normal(0, noise_scale)
        self.geometric_params['radius2'] += np.random.normal(0, noise_scale)
        
        # Ensure constraints
        self.geometric_params['radius1'] = max(self.min_core_radius, 
                                              self.geometric_params['radius1'])
        self.geometric_params['radius2'] = max(self.geometric_params['radius1'] + self.min_cladding_thickness,
                                              self.geometric_params['radius2'])
    
    def _adjust_toward_ground_truth(self):
        """Direct adjustment when edge fitting fails"""
        alpha = 0.3
        self.geometric_params['radius1'] = (1 - alpha) * self.geometric_params['radius1'] + \
                                          alpha * self.ground_truth_boundaries[0]
        self.geometric_params['radius2'] = (1 - alpha) * self.geometric_params['radius2'] + \
                                          alpha * self.ground_truth_boundaries[1]
    
    def create_final_masks(self):
        """Create masks using converged boundaries"""
        print("\nStage 6: Creating Final Masks")
        print("-" * 50)
        
        Y, X = np.ogrid[:self.height, :self.width]
        distance_map = np.sqrt((X - self.geometric_params['center_x'])**2 + 
                              (Y - self.geometric_params['center_y'])**2)
        
        self.masks = {}
        
        self.masks['core'] = (distance_map <= self.geometric_params['radius1']).astype(np.uint8) * 255
        self.masks['cladding'] = ((distance_map > self.geometric_params['radius1']) & 
                                 (distance_map <= self.geometric_params['radius2'])).astype(np.uint8) * 255
        self.masks['ferrule'] = (distance_map > self.geometric_params['radius2']).astype(np.uint8) * 255
        
        for region_name, mask in self.masks.items():
            pixel_count = np.sum(mask > 0)
            print(f"  {region_name}: {pixel_count} pixels")
    
    def apply_artifact_removal(self):
        """Apply binary filter artifact removal"""
        print("\nStage 7: Applying Binary Artifact Removal")
        print("-" * 50)
        
        initial_regions = {}
        for name, mask in self.masks.items():
            initial_regions[name] = cv2.bitwise_and(self.original, self.original, mask=mask)
        
        # Core refinement
        print("  Refining core region...")
        if 'core' in initial_regions:
            core_gray = cv2.cvtColor(initial_regions['core'], cv2.COLOR_BGR2GRAY)
            
            if np.any(self.masks['core']):
                valid_pixels = core_gray[self.masks['core'] > 0]
                if len(valid_pixels) > 0:
                    # Adaptive threshold based on distribution
                    threshold = np.percentile(valid_pixels, 25)
                    _, bright_mask = cv2.threshold(core_gray, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Clean up
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
                    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
                    
                    # Keep largest component
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
                    if num_labels > 1:
                        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        bright_mask = (labels == largest_label).astype(np.uint8) * 255
                    
                    self.masks['core'] = cv2.bitwise_and(self.masks['core'], bright_mask)
        
        # Cladding refinement
        print("  Refining cladding region...")
        if 'cladding' in initial_regions:
            clad_gray = cv2.cvtColor(initial_regions['cladding'], cv2.COLOR_BGR2GRAY)
            
            if np.any(self.masks['cladding']):
                valid_pixels = clad_gray[self.masks['cladding'] > 0]
                if len(valid_pixels) > 0:
                    # Remove bright outliers
                    threshold = np.percentile(valid_pixels, 75)
                    _, dark_mask = cv2.threshold(clad_gray, threshold, 255, cv2.THRESH_BINARY_INV)
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
                    
                    self.masks['cladding'] = cv2.bitwise_and(self.masks['cladding'], dark_mask)
        
        print("  Artifact removal complete")
    
    def extract_regions_and_output(self):
        """Extract final regions and generate output"""
        print("\nStage 8: Extracting Regions and Generating Output")
        print("-" * 50)
        
        self.regions = {}
        region_stats = {}
        
        for region_name, mask in self.masks.items():
            self.regions[region_name] = cv2.bitwise_and(self.original, self.original, mask=mask)
            
            pixel_count = int(np.sum(mask > 0))
            region_stats[region_name] = {
                'pixel_count': pixel_count,
                'percentage': float(100 * pixel_count / (self.width * self.height))
            }
        
        self.results = {
            'image_info': {
                'width': int(self.width),
                'height': int(self.height),
                'center': {
                    'x': float(self.geometric_params['center_x']),
                    'y': float(self.geometric_params['center_y'])
                }
            },
            'ground_truth_boundaries': [int(b) for b in self.ground_truth_boundaries],
            'converged_boundaries': [
                float(self.geometric_params['radius1']),
                float(self.geometric_params['radius2'])
            ],
            'cladding_thickness': float(self.geometric_params['radius2'] - self.geometric_params['radius1']),
            'alignment_score': float(self.alignment_score),
            'alignment_error': {
                'radius1': float(abs(self.geometric_params['radius1'] - self.ground_truth_boundaries[0])),
                'radius2': float(abs(self.geometric_params['radius2'] - self.ground_truth_boundaries[1]))
            },
            'regions': region_stats,
            'method_counts': {
                '2nd_derivative': len([b for b in self.all_boundaries if b['method'] == '2nd_derivative']),
                'gradient': len([b for b in self.all_boundaries if b['method'] == 'gradient']),
                'intensity_drop': len([b for b in self.all_boundaries if b['method'] == 'intensity_drop']),
                'alternative': len([b for b in self.all_boundaries if b['method'] == 'alternative'])
            }
        }
        
        output_dir = 'improved_unified_results'
        os.makedirs(output_dir, exist_ok=True)
        
        for region_name, region in self.regions.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}.png'), region)
            print(f"  Saved {region_name}: {region_stats[region_name]['pixel_count']} pixels "
                  f"({region_stats[region_name]['percentage']:.1f}%)")
        
        for region_name, mask in self.masks.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}_mask.png'), mask)
        
        self._create_improved_visualization(output_dir)
        
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")
    
    def _create_improved_visualization(self, output_dir):
        """Create improved visualization with diagnostics"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
        
        # Original with boundaries
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        ax1.plot(self.geometric_params['center_x'], self.geometric_params['center_y'], 
                'r+', markersize=15, markeredgewidth=2)
        
        theta = np.linspace(0, 2*np.pi, 100)
        for r, color, label in [(self.geometric_params['radius1'], 'g', 'Core'),
                               (self.geometric_params['radius2'], 'r', 'Cladding')]:
            x = self.geometric_params['center_x'] + r * np.cos(theta)
            y = self.geometric_params['center_y'] + r * np.sin(theta)
            ax1.plot(x, y, color=color, linewidth=2, label=f'{label} boundary')
        
        ax1.set_title('Final Boundaries')
        ax1.legend()
        ax1.axis('off')
        
        # Masks
        ax2 = fig.add_subplot(gs[0, 1])
        mask_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_overlay[self.masks['core'] > 0] = [255, 0, 0]
        mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0]
        mask_overlay[self.masks['ferrule'] > 0] = [0, 0, 255]
        ax2.imshow(mask_overlay)
        ax2.set_title('Region Masks')
        ax2.axis('off')
        
        # Composite
        ax3 = fig.add_subplot(gs[0, 2])
        composite = np.zeros_like(self.original)
        for name, region in self.regions.items():
            mask = self.masks[name] > 0
            composite[mask] = region[mask]
        ax3.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        ax3.set_title('Separated Regions')
        ax3.axis('off')
        
        # Find best profile
        best_profile = max(self.all_boundaries, key=lambda x: x['confidence'])['profiles']
        
        # Intensity profile
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(best_profile['radii'], best_profile['intensity'], 'b-', linewidth=2, alpha=0.5)
        ax4.plot(best_profile['radii'], best_profile['intensity_smooth'], 'b-', linewidth=2)
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('Intensity')
        ax4.set_title('Radial Intensity Profile')
        ax4.grid(True, alpha=0.3)
        
        # Mark boundaries
        for i, (b, label) in enumerate(zip(self.ground_truth_boundaries, ['Inner', 'Outer'])):
            ax4.axvline(x=b, color=['g', 'r'][i], linestyle='--', linewidth=2, 
                       label=f'{label} boundary ({b}px)')
        
        # Shade regions
        if len(self.ground_truth_boundaries) >= 2:
            ax4.axvspan(0, self.ground_truth_boundaries[0], alpha=0.1, color='red', label='Core')
            ax4.axvspan(self.ground_truth_boundaries[0], self.ground_truth_boundaries[1], 
                       alpha=0.1, color='green', label='Cladding')
            ax4.axvspan(self.ground_truth_boundaries[1], len(best_profile['radii'])-1, 
                       alpha=0.1, color='blue', label='Ferrule')
        
        ax4.legend()
        
        # Gradient
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(best_profile['radii'], best_profile['gradient'], 'orange', linewidth=2)
        ax5.set_xlabel('Radius (pixels)')
        ax5.set_ylabel('|dI/dr|')
        ax5.set_title('Gradient Magnitude')
        ax5.grid(True, alpha=0.3)
        for b in self.ground_truth_boundaries:
            ax5.axvline(x=b, color='r', linestyle='--', alpha=0.7)
        
        # Second derivative
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(best_profile['radii'], best_profile['second_derivative'], 'green', linewidth=2)
        ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax6.set_xlabel('Radius (pixels)')
        ax6.set_ylabel('d²I/dr²')
        ax6.set_title('Second Derivative')
        ax6.grid(True, alpha=0.3)
        for b in self.ground_truth_boundaries:
            ax6.axvline(x=b, color='r', linestyle='--', alpha=0.7)
        
        # Statistics
        ax7 = fig.add_subplot(gs[2, 2])
        stats_text = f"""Analysis Summary:

Detection Methods:
  2nd derivative: {self.results['method_counts']['2nd_derivative']}
  Gradient: {self.results['method_counts']['gradient']}
  Intensity drop: {self.results['method_counts']['intensity_drop']}
  Alternative: {self.results['method_counts']['alternative']}

Boundaries:
  Ground truth: {self.ground_truth_boundaries}
  Converged: [{self.geometric_params['radius1']:.1f}, {self.geometric_params['radius2']:.1f}]
  
Cladding thickness: {self.results['cladding_thickness']:.1f}px

Alignment:
  Score: {self.alignment_score:.3f}
  Error R1: {self.results['alignment_error']['radius1']:.2f}px
  Error R2: {self.results['alignment_error']['radius2']:.2f}px
"""
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, 
                fontsize=10, fontfamily='monospace', verticalalignment='top')
        ax7.axis('off')
        
        # Individual regions
        for i, (name, region) in enumerate(self.regions.items()):
            ax = fig.add_subplot(gs[3, i])
            ax.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            ax.set_title(f'{name.capitalize()}\n({self.results["regions"][name]["percentage"]:.1f}%)')
            ax.axis('off')
        
        plt.suptitle('Improved Fiber Separation Analysis', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analysis_visualization.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python improved_unified_separator.py <image_path>")
        sys.exit(1)
    
    try:
        separator = ImprovedUnifiedFiberSeparator(image_path)
        results = separator.analyze()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE - SUMMARY:")
        print("="*70)
        print(f"Center: ({results['image_info']['center']['x']:.1f}, "
              f"{results['image_info']['center']['y']:.1f})")
        print(f"Boundaries: {results['ground_truth_boundaries']} pixels")
        print(f"Cladding thickness: {results['cladding_thickness']:.1f} pixels")
        print(f"Alignment score: {results['alignment_score']:.3f}")
        print(f"Alignment errors: R1={results['alignment_error']['radius1']:.2f}px, "
              f"R2={results['alignment_error']['radius2']:.2f}px")
        print("\nRegion sizes:")
        for region, data in results['regions'].items():
            print(f"  {region}: {data['pixel_count']} pixels ({data['percentage']:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()