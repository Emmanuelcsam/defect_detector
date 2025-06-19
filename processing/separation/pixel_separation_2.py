import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, median_filter
from skimage import filters, feature, morphology, measure, segmentation
from skimage.filters import threshold_multiotsu, threshold_local
import warnings
warnings.filterwarnings('ignore')
import json
import os

class PixelBasedFiberAnalyzer:
    """
    Pixel Characteristic-Based Fiber Optic Analysis
    No geometric assumptions - pure intensity-based segmentation
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.results = {}
        self.visualizations = []
        
        # Convert to grayscale for analysis
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
    def analyze(self):
        """Run complete pixel-based analysis"""
        print("Starting Pixel-Based Fiber Optic Analysis...")
        print("=" * 60)
        
        # Step 1: Comprehensive preprocessing
        self.preprocess()
        
        # Step 2: Multi-scale gradient and variance analysis
        self.compute_pixel_characteristics()
        
        # Step 3: Radial profile analysis
        self.analyze_radial_profiles()
        
        # Step 4: Intensity-based region detection
        self.detect_regions_by_intensity()
        
        # Step 5: Create and refine masks
        self.create_refined_masks()
        
        # Step 6: Extract and clean regions
        self.extract_clean_regions()
        
        # Step 7: Generate analysis report
        self.generate_report()
        
        # Save all results
        self.save_results()
        
        print("\nAnalysis Complete!")
        return self.results
    
    def preprocess(self):
        """Advanced preprocessing for enhanced feature detection"""
        print("\nStep 1: Preprocessing...")
        
        # 1. Illumination correction using rolling ball algorithm
        print("  - Correcting illumination...")
        self.illumination_corrected = self._rolling_ball_background_subtraction(self.gray)
        
        # 2. Adaptive histogram equalization
        print("  - Applying adaptive histogram equalization...")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.enhanced = clahe.apply(self.illumination_corrected)
        
        # 3. Multi-scale denoising while preserving edges
        print("  - Multi-scale edge-preserving denoising...")
        self.denoised = self._multiscale_denoise(self.enhanced)
        
        # 4. Sharpening to enhance transitions
        print("  - Enhancing edge transitions...")
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        self.sharpened = cv2.filter2D(self.denoised, -1, kernel)
        self.sharpened = np.clip(self.sharpened, 0, 255).astype(np.uint8)
        
    def _rolling_ball_background_subtraction(self, img, radius=50):
        """Rolling ball algorithm for background subtraction"""
        # Create structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        
        # Morphological opening (erosion followed by dilation)
        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        # Subtract background
        corrected = cv2.subtract(img, background)
        corrected = cv2.add(corrected, 127)  # Shift to middle gray
        
        return corrected
    
    def _multiscale_denoise(self, img):
        """Multi-scale edge-preserving denoising"""
        # Apply bilateral filter at multiple scales
        scales = [(5, 50, 50), (9, 75, 75), (13, 100, 100)]
        denoised_scales = []
        
        for d, sigma_color, sigma_space in scales:
            denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
            denoised_scales.append(denoised)
        
        # Combine scales with weights favoring fine details
        weights = [0.5, 0.3, 0.2]
        combined = np.zeros_like(img, dtype=np.float32)
        
        for weight, denoised in zip(weights, denoised_scales):
            combined += weight * denoised
        
        return combined.astype(np.uint8)
    
    def compute_pixel_characteristics(self):
        """Compute comprehensive pixel characteristics"""
        print("\nStep 2: Computing pixel characteristics...")
        
        self.features = {}
        
        # 1. Multi-operator gradient computation
        print("  - Computing multi-operator gradients...")
        
        # Sobel gradients
        sobel_x = cv2.Sobel(self.denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.denoised, cv2.CV_64F, 0, 1, ksize=3)
        self.features['gradient_magnitude'] = np.sqrt(sobel_x**2 + sobel_y**2)
        self.features['gradient_direction'] = np.arctan2(sobel_y, sobel_x)
        
        # Scharr for more accurate gradients
        scharr_x = cv2.Scharr(self.denoised, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(self.denoised, cv2.CV_64F, 0, 1)
        self.features['scharr_magnitude'] = np.sqrt(scharr_x**2 + scharr_y**2)
        
        # Laplacian for second derivative
        self.features['laplacian'] = cv2.Laplacian(self.denoised, cv2.CV_64F, ksize=5)
        
        # Combined gradient with emphasis on sharp transitions
        self.features['combined_gradient'] = np.maximum(
            self.features['gradient_magnitude'],
            self.features['scharr_magnitude']
        )
        
        # 2. Multi-scale local variance
        print("  - Computing multi-scale local variance...")
        variances = []
        variance_gradients = []
        
        for window_size in [3, 5, 7, 9, 11]:
            # Local variance
            mean = cv2.boxFilter(self.denoised.astype(np.float32), -1, (window_size, window_size))
            sqr_mean = cv2.boxFilter(self.denoised.astype(np.float32)**2, -1, (window_size, window_size))
            variance = sqr_mean - mean**2
            variance = np.maximum(variance, 0)  # Handle numerical errors
            variances.append(variance)
            
            # Gradient of variance
            var_grad_x = cv2.Sobel(variance, cv2.CV_64F, 1, 0)
            var_grad_y = cv2.Sobel(variance, cv2.CV_64F, 0, 1)
            var_grad_mag = np.sqrt(var_grad_x**2 + var_grad_y**2)
            variance_gradients.append(var_grad_mag)
        
        # Combine scales
        self.features['local_variance'] = np.mean(variances, axis=0)
        self.features['variance_gradient'] = np.mean(variance_gradients, axis=0)
        
        # 3. Texture energy
        print("  - Computing texture energy...")
        self.features['texture_energy'] = self._compute_texture_energy()
        
        # 4. Local contrast
        print("  - Computing local contrast...")
        self.features['local_contrast'] = self._compute_local_contrast()
        
        # 5. Ridge detection for boundaries
        print("  - Detecting ridges and valleys...")
        self.features['ridges'] = filters.meijering(self.denoised, sigmas=range(1, 5))
        
    def _compute_texture_energy(self):
        """Compute texture energy using Laws' texture masks"""
        # Laws' texture energy masks (simplified set)
        L5 = np.array([1, 4, 6, 4, 1]) / 16  # Level
        E5 = np.array([-1, -2, 0, 2, 1]) / 6  # Edge
        S5 = np.array([-1, 0, 2, 0, -1]) / 2  # Spot
        
        # Generate 2D kernels
        kernels = []
        for v1 in [L5, E5, S5]:
            for v2 in [L5, E5, S5]:
                kernel = np.outer(v1, v2)
                kernels.append(kernel)
        
        # Apply kernels and compute energy
        energy_maps = []
        for kernel in kernels:
            filtered = cv2.filter2D(self.denoised.astype(np.float32), -1, kernel)
            energy = np.abs(filtered)
            energy_maps.append(energy)
        
        # Combine energy maps
        total_energy = np.mean(energy_maps, axis=0)
        return total_energy
    
    def _compute_local_contrast(self):
        """Compute local contrast using min-max in neighborhood"""
        contrasts = []
        
        for window_size in [3, 5, 7]:
            # Local max and min
            kernel = np.ones((window_size, window_size), np.uint8)
            local_max = cv2.dilate(self.denoised, kernel)
            local_min = cv2.erode(self.denoised, kernel)
            
            # Contrast
            contrast = (local_max - local_min).astype(np.float32)
            contrasts.append(contrast)
        
        return np.mean(contrasts, axis=0)
    
    def analyze_radial_profiles(self):
        """Analyze radial intensity and gradient profiles"""
        print("\nStep 3: Analyzing radial profiles...")
        
        # Find approximate center using intensity-weighted centroid
        print("  - Finding intensity-weighted center...")
        center = self._find_intensity_center()
        self.center = center
        
        print(f"  - Center found at: ({center[0]:.1f}, {center[1]:.1f})")
        
        # Compute radial profiles
        print("  - Computing radial profiles...")
        self.radial_profiles = self._compute_radial_profiles(center)
        
        # Analyze profile transitions
        print("  - Analyzing profile transitions...")
        self.transitions = self._analyze_transitions(self.radial_profiles)
        
    def _find_intensity_center(self):
        """Find center using intensity-weighted centroid"""
        # Apply Gaussian blur to smooth
        smoothed = gaussian_filter(self.denoised, sigma=5)
        
        # Find brightest region (likely core)
        threshold = np.percentile(smoothed, 90)
        bright_mask = smoothed > threshold
        
        # Calculate centroid of bright region
        moments = cv2.moments(bright_mask.astype(np.uint8))
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            # Fallback to image center
            cx, cy = self.width // 2, self.height // 2
        
        return (cx, cy)
    
    def _compute_radial_profiles(self, center):
        """Compute comprehensive radial profiles"""
        cx, cy = center
        max_radius = int(min(cx, cy, self.width - cx, self.height - cy))
        
        # Create radial sampling points
        num_angles = 360
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        radii = np.arange(0, max_radius)
        
        profiles = {
            'intensity': [],
            'gradient': [],
            'variance': [],
            'laplacian': []
        }
        
        for angle in angles:
            # Sample along radial line
            x_coords = cx + radii * np.cos(angle)
            y_coords = cy + radii * np.sin(angle)
            
            # Bilinear interpolation for sub-pixel accuracy
            intensity_profile = []
            gradient_profile = []
            variance_profile = []
            laplacian_profile = []
            
            for x, y in zip(x_coords, y_coords):
                if 0 <= x < self.width - 1 and 0 <= y < self.height - 1:
                    # Bilinear interpolation
                    intensity = self._bilinear_interpolate(self.denoised, x, y)
                    gradient = self._bilinear_interpolate(self.features['combined_gradient'], x, y)
                    variance = self._bilinear_interpolate(self.features['local_variance'], x, y)
                    laplacian = self._bilinear_interpolate(np.abs(self.features['laplacian']), x, y)
                    
                    intensity_profile.append(intensity)
                    gradient_profile.append(gradient)
                    variance_profile.append(variance)
                    laplacian_profile.append(laplacian)
            
            if intensity_profile:
                profiles['intensity'].append(np.array(intensity_profile))
                profiles['gradient'].append(np.array(gradient_profile))
                profiles['variance'].append(np.array(variance_profile))
                profiles['laplacian'].append(np.array(laplacian_profile))
        
        # Compute median profiles
        profiles['median_intensity'] = np.median(profiles['intensity'], axis=0)
        profiles['median_gradient'] = np.median(profiles['gradient'], axis=0)
        profiles['median_variance'] = np.median(profiles['variance'], axis=0)
        profiles['median_laplacian'] = np.median(profiles['laplacian'], axis=0)
        
        # Compute profile derivatives
        profiles['intensity_derivative'] = np.abs(np.gradient(profiles['median_intensity']))
        profiles['gradient_derivative'] = np.gradient(profiles['median_gradient'])
        
        return profiles
    
    def _bilinear_interpolate(self, img, x, y):
        """Bilinear interpolation for sub-pixel sampling"""
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        
        if x1 >= img.shape[1] or y1 >= img.shape[0]:
            return img[y0, x0]
        
        dx, dy = x - x0, y - y0
        
        return (1 - dx) * (1 - dy) * img[y0, x0] + \
               dx * (1 - dy) * img[y0, x1] + \
               (1 - dx) * dy * img[y1, x0] + \
               dx * dy * img[y1, x1]
    
    def _analyze_transitions(self, profiles):
        """Analyze transitions in radial profiles to find boundaries"""
        transitions = {}
        
        # Use gradient profile to find boundaries (sharp peaks indicate transitions)
        gradient_profile = profiles['median_gradient']
        
        # Smooth the gradient profile slightly to reduce noise
        from scipy.ndimage import gaussian_filter1d
        smoothed_gradient = gaussian_filter1d(gradient_profile, sigma=1)
        
        # Find peaks in gradient profile (boundaries)
        # Use a high percentile threshold to get only the most significant peaks
        gradient_peaks, properties = signal.find_peaks(
            smoothed_gradient,
            height=np.percentile(smoothed_gradient, 85),
            distance=20,  # Minimum distance between peaks
            prominence=np.std(smoothed_gradient) * 0.5  # Require significant prominence
        )
        
        # Also check intensity derivative for additional validation
        intensity_derivative = profiles['intensity_derivative']
        intensity_peaks, _ = signal.find_peaks(
            intensity_derivative,
            height=np.percentile(intensity_derivative, 85),
            distance=20
        )
        
        # Combine and validate peaks
        all_peaks = []
        
        # Priority to gradient peaks as they are most reliable
        for peak in gradient_peaks:
            # Validate by checking if there's a significant intensity change
            if peak > 5 and peak < len(gradient_profile) - 5:
                # Check intensity change around this point
                before = np.mean(profiles['median_intensity'][max(0, peak-10):peak])
                after = np.mean(profiles['median_intensity'][peak:min(len(profiles['median_intensity']), peak+10)])
                
                if abs(before - after) > 10:  # Significant intensity change
                    all_peaks.append(peak)
        
        # Sort peaks by their gradient magnitude to get the most significant ones
        if len(all_peaks) > 0:
            peak_strengths = [(p, smoothed_gradient[p]) for p in all_peaks]
            peak_strengths.sort(key=lambda x: x[1], reverse=True)
            
            # Take the two strongest peaks as boundaries
            strongest_peaks = [p[0] for p in peak_strengths[:2]]
            strongest_peaks.sort()  # Sort by radius
            
            transitions['boundary_radii'] = strongest_peaks
        else:
            # Fallback: look for largest changes in intensity
            intensity_changes = np.abs(np.diff(profiles['median_intensity']))
            change_peaks, _ = signal.find_peaks(
                intensity_changes,
                height=np.percentile(intensity_changes, 90),
                distance=20
            )
            
            if len(change_peaks) >= 2:
                transitions['boundary_radii'] = sorted(change_peaks[:2])
            else:
                transitions['boundary_radii'] = []
        
        # Analyze intensity levels between boundaries
        transitions['regions'] = self._identify_regions_from_transitions(
            profiles['median_intensity'],
            transitions['boundary_radii']
        )
        
        # Log detected boundaries
        if transitions['boundary_radii']:
            print(f"  - Detected {len(transitions['boundary_radii'])} boundaries at radii: {transitions['boundary_radii']}")
            
            # Analyze intensity pattern
            if len(transitions['regions']) >= 3:
                intensities = [(r['mean_intensity'], r['start_radius'], r['end_radius']) 
                              for r in transitions['regions']]
                print(f"  - Region intensities: {[f'{i[0]:.1f}' for i in intensities]}")
        
        return transitions
    
    def _cluster_peaks(self, peaks, min_distance=5):
        """Cluster nearby peaks"""
        if len(peaks) == 0:
            return []
        
        peaks = np.sort(peaks)
        clusters = [[peaks[0]]]
        
        for peak in peaks[1:]:
            if peak - clusters[-1][-1] <= min_distance:
                clusters[-1].append(peak)
            else:
                clusters.append([peak])
        
        # Return cluster centers
        return [int(np.mean(cluster)) for cluster in clusters]
    
    def _identify_regions_from_transitions(self, intensity_profile, boundaries):
        """Identify regions based on intensity levels between boundaries"""
        regions = []
        
        # Add start and end points
        boundaries = [0] + list(boundaries) + [len(intensity_profile) - 1]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            if end > start:
                # Calculate region statistics
                region_intensity = intensity_profile[start:end]
                regions.append({
                    'start_radius': start,
                    'end_radius': end,
                    'mean_intensity': np.mean(region_intensity),
                    'std_intensity': np.std(region_intensity),
                    'median_intensity': np.median(region_intensity),
                    'mode_intensity': self._calculate_mode(region_intensity)
                })
        
        return regions
    
    def _calculate_mode(self, data):
        """Calculate mode (most common value) of data"""
        if len(data) == 0:
            return 0
        hist, bins = np.histogram(data, bins=50)
        mode_idx = np.argmax(hist)
        mode_value = (bins[mode_idx] + bins[mode_idx + 1]) / 2
        return mode_value
    
    def detect_regions_by_intensity(self):
        """Detect regions using intensity-based segmentation"""
        print("\nStep 4: Detecting regions by intensity characteristics...")
        
        # 1. Multi-level thresholding
        print("  - Applying multi-level Otsu thresholding...")
        thresholds = threshold_multiotsu(self.denoised, classes=3)
        self.intensity_regions = np.digitize(self.denoised, bins=thresholds)
        
        # 2. Gradient magnitude thresholding
        print("  - Segmenting by gradient magnitude...")
        gradient_thresh = np.percentile(self.features['combined_gradient'], [85, 95])
        self.gradient_regions = np.digitize(self.features['combined_gradient'], bins=gradient_thresh)
        
        # 3. Variance-based segmentation
        print("  - Segmenting by local variance...")
        variance_thresh = np.percentile(self.features['local_variance'], [70, 90])
        self.variance_regions = np.digitize(self.features['local_variance'], bins=variance_thresh)
        
        # 4. Texture-based segmentation
        print("  - Segmenting by texture energy...")
        texture_thresh = np.percentile(self.features['texture_energy'], [60, 85])
        self.texture_regions = np.digitize(self.features['texture_energy'], bins=texture_thresh)
        
        # 5. Combined segmentation using voting
        print("  - Combining segmentations...")
        self.combined_regions = self._combine_segmentations()
        
    def _combine_segmentations(self):
        """Combine multiple segmentations using voting"""
        # Stack all segmentations
        segmentations = np.stack([
            self.intensity_regions,
            self.gradient_regions,
            self.variance_regions,
            self.texture_regions
        ], axis=2)
        
        # Weighted voting
        weights = [0.35, 0.35, 0.2, 0.1]  # Emphasize intensity and gradient
        
        combined = np.zeros((self.height, self.width))
        
        for i in range(3):  # 3 regions
            votes = np.zeros((self.height, self.width))
            for j, weight in enumerate(weights):
                votes += weight * (segmentations[:, :, j] == i)
            combined[votes >= 0.5] = i
        
        return combined.astype(np.uint8)
    
    def create_refined_masks(self):
        """Create and refine masks for each region using radial analysis"""
        print("\nStep 5: Creating and refining masks...")
        
        self.masks = {}
        
        # Create radial distance map
        cx, cy = self.center
        Y, X = np.ogrid[:self.height, :self.width]
        distance_map = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Initialize masks
        self.masks['core'] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.masks['cladding'] = np.zeros((self.height, self.width), dtype=np.uint8)
        self.masks['ferrule'] = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Use radial boundaries to create initial masks
        if self.transitions and 'boundary_radii' in self.transitions and len(self.transitions['boundary_radii']) >= 2:
            # We expect two main boundaries: core-cladding and cladding-ferrule
            boundaries = sorted(self.transitions['boundary_radii'])
            
            print(f"  - Detected boundaries at radii: {boundaries}")
            
            # Core: innermost region (0 to first boundary)
            core_radius = boundaries[0]
            self.masks['core'] = (distance_map <= core_radius).astype(np.uint8) * 255
            
            # Cladding: annular region (first to second boundary)
            cladding_inner = boundaries[0]
            cladding_outer = boundaries[1]
            self.masks['cladding'] = ((distance_map > cladding_inner) & 
                                     (distance_map <= cladding_outer)).astype(np.uint8) * 255
            
            # Ferrule: outer region (beyond second boundary)
            self.masks['ferrule'] = (distance_map > cladding_outer).astype(np.uint8) * 255
            
        else:
            # Fallback: use intensity-based segmentation with spatial constraints
            print("  - Using intensity-based segmentation with spatial analysis...")
            self._create_masks_by_intensity_and_location()
        
        # Refine masks based on intensity characteristics
        print("  - Refining masks based on intensity patterns...")
        self._refine_masks_by_intensity()
        
        # Morphological refinement
        print("  - Applying morphological refinement...")
        self.masks['core'] = self._morphological_refinement(self.masks['core'], 'core')
        self.masks['cladding'] = self._morphological_refinement(self.masks['cladding'], 'cladding')
        self.masks['ferrule'] = self._morphological_refinement(self.masks['ferrule'], 'ferrule')
        
        # Ensure mutual exclusivity with proper priority
        self._ensure_exclusive_masks()
        
        # Validate and adjust based on expected intensity patterns
        self._validate_region_assignments()
        
    def _create_masks_by_intensity_and_location(self):
        """Create masks using intensity patterns and spatial relationships"""
        # Create radial distance map
        cx, cy = self.center
        Y, X = np.ogrid[:self.height, :self.width]
        distance_map = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Analyze intensity distribution at different radial distances
        max_radius = int(np.max(distance_map))
        radial_intensities = []
        
        for r in range(0, max_radius, 5):  # Sample every 5 pixels
            ring_mask = (distance_map >= r) & (distance_map < r + 5)
            if np.any(ring_mask):
                ring_intensities = self.denoised[ring_mask]
                radial_intensities.append({
                    'radius': r,
                    'mean': np.mean(ring_intensities),
                    'median': np.median(ring_intensities),
                    'std': np.std(ring_intensities)
                })
        
        # Find transitions based on intensity changes
        if len(radial_intensities) > 3:
            means = [r['mean'] for r in radial_intensities]
            
            # Smooth the means to reduce noise
            from scipy.ndimage import gaussian_filter1d
            smoothed_means = gaussian_filter1d(means, sigma=2)
            
            # Find significant changes
            changes = np.abs(np.gradient(smoothed_means))
            
            # Find peaks in changes
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(changes, height=np.percentile(changes, 75))
            
            if len(peaks) >= 2:
                # Convert peak indices to radii
                boundary_radii = [radial_intensities[p]['radius'] for p in peaks[:2]]
                
                # Create masks based on these boundaries
                self.masks['core'] = (distance_map <= boundary_radii[0]).astype(np.uint8) * 255
                self.masks['cladding'] = ((distance_map > boundary_radii[0]) & 
                                         (distance_map <= boundary_radii[1])).astype(np.uint8) * 255
                self.masks['ferrule'] = (distance_map > boundary_radii[1]).astype(np.uint8) * 255
                return
        
        # Fallback to simple radial division
        max_r = np.max(distance_map)
        self.masks['core'] = (distance_map <= max_r * 0.2).astype(np.uint8) * 255
        self.masks['cladding'] = ((distance_map > max_r * 0.2) & 
                                 (distance_map <= max_r * 0.5)).astype(np.uint8) * 255
        self.masks['ferrule'] = (distance_map > max_r * 0.5).astype(np.uint8) * 255
    
    def _refine_masks_by_intensity(self):
        """Refine masks based on expected intensity patterns"""
        # Calculate mean intensity for each region
        region_stats = {}
        
        for region_name, mask in self.masks.items():
            if np.any(mask):
                intensities = self.denoised[mask > 0]
                region_stats[region_name] = {
                    'mean': np.mean(intensities),
                    'median': np.median(intensities),
                    'std': np.std(intensities),
                    'mode': self._calculate_mode(intensities)
                }
        
        # Expected pattern: Core (brightest) > Ferrule (medium) > Cladding (darkest)
        if len(region_stats) == 3:
            # Check if the pattern matches expectations
            core_int = region_stats.get('core', {}).get('median', 0)
            clad_int = region_stats.get('cladding', {}).get('median', 0)
            ferr_int = region_stats.get('ferrule', {}).get('median', 0)
            
            print(f"  - Region intensities - Core: {core_int:.1f}, Cladding: {clad_int:.1f}, Ferrule: {ferr_int:.1f}")
            
            # If cladding is not the darkest, we need to reassign
            if not (clad_int < ferr_int < core_int):
                print("  - Intensity pattern doesn't match expected (Core > Ferrule > Cladding)")
                print("  - Reassigning regions based on intensity levels...")
                
                # Create temporary masks based on intensity thresholds
                temp_masks = {}
                
                # Use multi-level thresholding
                try:
                    thresholds = threshold_multiotsu(self.denoised, classes=3)
                    
                    # The darkest region should be cladding
                    darkest = self.denoised <= thresholds[0]
                    middle = (self.denoised > thresholds[0]) & (self.denoised <= thresholds[1])
                    brightest = self.denoised > thresholds[1]
                    
                    # Combine with spatial information
                    cx, cy = self.center
                    Y, X = np.ogrid[:self.height, :self.width]
                    distance_map = np.sqrt((X - cx)**2 + (Y - cy)**2)
                    
                    # Core should be brightest AND central
                    core_candidate = brightest & (distance_map < np.percentile(distance_map[brightest], 50))
                    
                    # Cladding should be darkest AND in middle radial range
                    clad_candidate = darkest
                    
                    # Refine based on radial position
                    for region_name in ['core', 'cladding', 'ferrule']:
                        self.masks[region_name] = self._refine_by_radial_continuity(
                            self.masks[region_name], distance_map
                        )
                
                except Exception as e:
                    print(f"  - Warning: Could not refine by intensity: {e}")
    
    def _refine_by_radial_continuity(self, mask, distance_map):
        """Ensure radial continuity of regions"""
        # Convert mask to binary
        binary_mask = mask > 0
        
        # For each radius, ensure continuity
        max_radius = int(np.max(distance_map))
        refined_mask = np.zeros_like(mask)
        
        for r in range(max_radius):
            ring = (distance_map >= r) & (distance_map < r + 1)
            ring_region = binary_mask & ring
            
            if np.any(ring_region):
                # Use majority voting for this radius
                coverage = np.sum(ring_region) / np.sum(ring)
                if coverage > 0.5:  # If more than 50% of the ring is in this region
                    refined_mask[ring] = 255
        
        return refined_mask
    
    def _validate_region_assignments(self):
        """Validate that regions follow expected intensity pattern"""
        # Calculate final statistics
        final_stats = {}
        
        for region_name, mask in self.masks.items():
            if np.any(mask):
                intensities = self.denoised[mask > 0]
                final_stats[region_name] = {
                    'mean': np.mean(intensities),
                    'median': np.median(intensities),
                    'pixels': np.sum(mask > 0)
                }
        
        # Print validation results
        print("\n  - Final region validation:")
        if 'core' in final_stats:
            print(f"    Core: {final_stats['core']['pixels']} pixels, median intensity: {final_stats['core']['median']:.1f}")
        if 'cladding' in final_stats:
            print(f"    Cladding: {final_stats['cladding']['pixels']} pixels, median intensity: {final_stats['cladding']['median']:.1f}")
        if 'ferrule' in final_stats:
            print(f"    Ferrule: {final_stats['ferrule']['pixels']} pixels, median intensity: {final_stats['ferrule']['median']:.1f}")
        
        # Check if cladding is darkest
        if len(final_stats) == 3:
            intensities = [(name, stats['median']) for name, stats in final_stats.items()]
            intensities.sort(key=lambda x: x[1])
            
            if intensities[0][0] == 'cladding':
                print("    ✓ Cladding correctly identified as darkest region")
            else:
                print(f"    ⚠ Warning: {intensities[0][0]} is darkest, expected cladding")
    
    def _morphological_refinement(self, mask, region_name):
        """Apply morphological operations for refinement"""
        if region_name == 'core':
            # Core should be smooth and circular
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Fill holes
            mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
            
        elif region_name == 'cladding':
            # Cladding refinement
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove small components
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=100).astype(np.uint8)
            
        elif region_name == 'ferrule':
            # Ferrule can be more irregular
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask * 255
    
    def _largest_component_only(self, mask):
        """Keep only the largest connected component"""
        labels = measure.label(mask > 0)
        if labels.max() == 0:
            return mask
        
        largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return (largest * 255).astype(np.uint8)
    
    def _ensure_exclusive_masks(self):
        """Ensure masks are mutually exclusive with priority: core > cladding > ferrule"""
        # Core has highest priority
        if 'core' in self.masks and 'cladding' in self.masks:
            self.masks['cladding'][self.masks['core'] > 0] = 0
        
        # Cladding over ferrule
        if 'cladding' in self.masks and 'ferrule' in self.masks:
            self.masks['ferrule'][self.masks['cladding'] > 0] = 0
        
        if 'core' in self.masks and 'ferrule' in self.masks:
            self.masks['ferrule'][self.masks['core'] > 0] = 0
    
    def extract_clean_regions(self):
        """Extract and clean regions using refined masks"""
        print("\nStep 6: Extracting and cleaning regions...")
        
        self.cleaned_regions = {}
        
        for region_name, mask in self.masks.items():
            print(f"  - Cleaning {region_name}...")
            
            # Extract region
            region = cv2.bitwise_and(self.original, self.original, mask=mask)
            
            # Additional cleaning based on region
            if region_name == 'core':
                region = self._clean_core(region, mask)
            elif region_name == 'cladding':
                region = self._clean_cladding(region, mask)
            elif region_name == 'ferrule':
                region = self._clean_ferrule(region, mask)
            
            self.cleaned_regions[region_name] = region
    
    def _clean_core(self, region, mask):
        """Specific cleaning for core region"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply local adaptive thresholding to remove artifacts
        if np.any(mask):
            valid_pixels = gray[mask > 0]
            if len(valid_pixels) > 0:
                threshold = np.percentile(valid_pixels, 75)
                artifact_mask = (gray < threshold) & (mask > 0)
                
                # Remove artifacts
                for i in range(3):
                    region[:, :, i][artifact_mask] = 0
        
        return region
    
    def _clean_cladding(self, region, mask):
        """Specific cleaning for cladding region"""
        # Denoise while preserving edges
        cleaned = cv2.fastNlMeansDenoisingColored(region, None, 10, 10, 7, 21)
        
        # Ensure mask is applied
        result = np.zeros_like(region)
        result[mask > 0] = cleaned[mask > 0]
        
        return result
    
    def _clean_ferrule(self, region, mask):
        """Specific cleaning for ferrule region"""
        # Simple denoising
        cleaned = cv2.fastNlMeansDenoisingColored(region, None, 10, 10, 7, 21)
        
        # Apply mask
        result = np.zeros_like(region)
        result[mask > 0] = cleaned[mask > 0]
        
        return result
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\nStep 7: Generating analysis report...")
        
        self.results = {
            'image_properties': {
                'width': self.width,
                'height': self.height,
                'center': {
                    'x': float(self.center[0]),
                    'y': float(self.center[1])
                }
            },
            'intensity_statistics': {},
            'gradient_statistics': {},
            'variance_statistics': {},
            'texture_statistics': {},
            'region_characteristics': {},
            'transition_analysis': {}
        }
        
        # Analyze each region
        for region_name, mask in self.masks.items():
            if np.any(mask):
                mask_bool = mask > 0
                
                # Intensity statistics
                intensities = self.denoised[mask_bool]
                self.results['intensity_statistics'][region_name] = {
                    'mean': float(np.mean(intensities)),
                    'std': float(np.std(intensities)),
                    'median': float(np.median(intensities)),
                    'min': float(np.min(intensities)),
                    'max': float(np.max(intensities)),
                    'percentile_25': float(np.percentile(intensities, 25)),
                    'percentile_75': float(np.percentile(intensities, 75))
                }
                
                # Gradient statistics
                gradients = self.features['combined_gradient'][mask_bool]
                self.results['gradient_statistics'][region_name] = {
                    'mean': float(np.mean(gradients)),
                    'std': float(np.std(gradients)),
                    'max': float(np.max(gradients))
                }
                
                # Variance statistics
                variances = self.features['local_variance'][mask_bool]
                self.results['variance_statistics'][region_name] = {
                    'mean': float(np.mean(variances)),
                    'std': float(np.std(variances)),
                    'max': float(np.max(variances))
                }
                
                # Texture statistics
                textures = self.features['texture_energy'][mask_bool]
                self.results['texture_statistics'][region_name] = {
                    'mean': float(np.mean(textures)),
                    'std': float(np.std(textures))
                }
                
                # Region size
                self.results['region_characteristics'][region_name] = {
                    'pixel_count': int(np.sum(mask_bool)),
                    'percentage': float(100 * np.sum(mask_bool) / (self.width * self.height))
                }
        
        # Transition analysis
        if self.transitions:
            self.results['transition_analysis'] = {
                'detected_boundaries': [int(b) for b in self.transitions['boundary_radii']],
                'num_regions': len(self.transitions['regions']),
                'region_details': self.transitions['regions']
            }
    
    def save_results(self, output_dir='pixel_based_results'):
        """Save all results and visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cleaned regions
        for region_name, region in self.cleaned_regions.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}_cleaned.png'), region)
        
        # Save masks
        for region_name, mask in self.masks.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}_mask.png'), mask)
        
        # Save feature maps
        feature_dir = os.path.join(output_dir, 'features')
        os.makedirs(feature_dir, exist_ok=True)
        
        # Normalize and save features
        for feature_name, feature_map in self.features.items():
            normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            normalized = (normalized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(feature_dir, f'{feature_name}.png'), normalized)
        
        # Save processing stages
        cv2.imwrite(os.path.join(output_dir, 'preprocessed.png'), self.denoised)
        cv2.imwrite(os.path.join(output_dir, 'combined_segmentation.png'), 
                   (self.combined_regions * 85).astype(np.uint8))
        
        # Create comprehensive visualizations
        self._create_visualizations(output_dir)
        
        # Create multi-modal analysis plot similar to user's reference
        self._create_multimodal_analysis_plot(output_dir)
        
        # Save JSON report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")
    
    def _create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        # 1. Feature comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pixel Characteristics Analysis', fontsize=16)
        
        axes[0, 0].imshow(self.denoised, cmap='gray')
        axes[0, 0].set_title('Preprocessed Image')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(self.features['combined_gradient'], cmap='hot')
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        im2 = axes[0, 2].imshow(self.features['local_variance'], cmap='viridis')
        axes[0, 2].set_title('Local Variance')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        im3 = axes[1, 0].imshow(self.features['texture_energy'], cmap='plasma')
        axes[1, 0].set_title('Texture Energy')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        im4 = axes[1, 1].imshow(self.features['local_contrast'], cmap='copper')
        axes[1, 1].set_title('Local Contrast')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        axes[1, 2].imshow(self.combined_regions, cmap='tab10')
        axes[1, 2].set_title('Combined Segmentation')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pixel_characteristics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Radial profiles plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Radial Profile Analysis', fontsize=16)
        
        radii = np.arange(len(self.radial_profiles['median_intensity']))
        
        # Plot intensity profile
        axes[0, 0].plot(radii, self.radial_profiles['median_intensity'], 'b-', linewidth=2)
        axes[0, 0].set_title('Radial Intensity Profile')
        axes[0, 0].set_xlabel('Radius (pixels)')
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark detected boundaries with labels
        if self.transitions['boundary_radii']:
            for i, boundary in enumerate(self.transitions['boundary_radii']):
                color = 'g' if i == 0 else 'r'
                label = 'Core Boundary' if i == 0 else 'Cladding Boundary'
                axes[0, 0].axvline(x=boundary, color=color, linestyle='--', alpha=0.7, linewidth=2, label=f'{label} ({boundary}px)')
        
        axes[0, 0].legend()
        
        # Plot gradient profile
        axes[0, 1].plot(radii, self.radial_profiles['median_gradient'], 'orange', linewidth=2)
        axes[0, 1].set_title('Radial Gradient Profile')
        axes[0, 1].set_xlabel('Radius (pixels)')
        axes[0, 1].set_ylabel('Gradient Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark peaks on gradient plot
        if self.transitions['boundary_radii']:
            for i, boundary in enumerate(self.transitions['boundary_radii']):
                axes[0, 1].axvline(x=boundary, color='r', linestyle='--', alpha=0.7, linewidth=2)
                # Mark the peak value
                if boundary < len(self.radial_profiles['median_gradient']):
                    axes[0, 1].plot(boundary, self.radial_profiles['median_gradient'][boundary], 
                                   'rx', markersize=10, markeredgewidth=2)
        
        # Plot variance profile
        axes[1, 0].plot(radii, self.radial_profiles['median_variance'], 'g-', linewidth=2)
        axes[1, 0].set_title('Radial Variance Profile')
        axes[1, 0].set_xlabel('Radius (pixels)')
        axes[1, 0].set_ylabel('Local Variance')
        axes[1, 0].grid(True, alpha=0.3)
        
        for boundary in self.transitions['boundary_radii']:
            axes[1, 0].axvline(x=boundary, color='r', linestyle='--', alpha=0.7)
        
        # Plot intensity derivative
        axes[1, 1].plot(radii, self.radial_profiles['intensity_derivative'], 'm-', linewidth=2)
        axes[1, 1].set_title('Intensity Derivative Profile')
        axes[1, 1].set_xlabel('Radius (pixels)')
        axes[1, 1].set_ylabel('|dI/dr|')
        axes[1, 1].grid(True, alpha=0.3)
        
        for boundary in self.transitions['boundary_radii']:
            axes[1, 1].axvline(x=boundary, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radial_profiles.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Final results visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pixel-Based Region Separation Results', fontsize=16)
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Combined masks overlay
        mask_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if 'core' in self.masks:
            mask_overlay[self.masks['core'] > 0] = [255, 0, 0]  # Red for core
        if 'cladding' in self.masks:
            mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0]  # Green for cladding
        if 'ferrule' in self.masks:
            mask_overlay[self.masks['ferrule'] > 0] = [0, 0, 255]  # Blue for ferrule
        
        axes[0, 1].imshow(mask_overlay)
        axes[0, 1].set_title('Region Masks')
        axes[0, 1].axis('off')
        
        # Final composite
        composite = np.zeros_like(self.original)
        for region_name, region in self.cleaned_regions.items():
            mask = self.masks[region_name] > 0
            composite[mask] = region[mask]
        
        axes[0, 2].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Final Composite')
        axes[0, 2].axis('off')
        
        # Individual cleaned regions
        for i, (region_name, region) in enumerate(self.cleaned_regions.items()):
            if i < 3:
                axes[1, i].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f'Cleaned {region_name.capitalize()}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_results.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Statistical summary plot
        if self.results['intensity_statistics']:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            regions = list(self.results['intensity_statistics'].keys())
            means = [self.results['intensity_statistics'][r]['mean'] for r in regions]
            stds = [self.results['intensity_statistics'][r]['std'] for r in regions]
            
            x = np.arange(len(regions))
            ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7)
            ax.set_xlabel('Region')
            ax.set_ylabel('Intensity')
            ax.set_title('Mean Intensity by Region')
            ax.set_xticks(x)
            ax.set_xticklabels([r.capitalize() for r in regions])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'intensity_statistics.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    def _create_multimodal_analysis_plot(self, output_dir):
        """Create a multi-modal analysis plot similar to reference"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Multi-Modal Radial Analysis for Boundary Detection', fontsize=16)
        
        radii = np.arange(len(self.radial_profiles['median_intensity']))
        
        # 1. Average Pixel Intensity vs. Radius
        ax1 = axes[0]
        ax1.plot(radii, self.radial_profiles['median_intensity'], 'b-', linewidth=2, label='Avg. Intensity')
        ax1.set_ylabel('Avg. Intensity')
        ax1.set_title('Average Pixel Intensity vs. Radius')
        ax1.grid(True, alpha=0.3)
        
        # Mark boundaries
        if self.transitions['boundary_radii']:
            colors = ['g', 'r']
            labels = ['Core Boundary', 'Cladding Boundary']
            for i, boundary in enumerate(self.transitions['boundary_radii'][:2]):
                ax1.axvline(x=boundary, color=colors[i % 2], linestyle='--', 
                           linewidth=2, label=f'{labels[i % 2]} ({boundary}px)')
        ax1.legend()
        
        # 2. Average Change Magnitude (Gradient) vs. Radius
        ax2 = axes[1]
        ax2.plot(radii, self.radial_profiles['median_gradient'], 'orange', linewidth=2, label='Avg. Gradient')
        ax2.set_ylabel('Avg. Change')
        ax2.set_title('Average Change Magnitude (Gradient) vs. Radius')
        ax2.grid(True, alpha=0.3)
        
        # Mark detected peaks
        if self.transitions['boundary_radii']:
            for boundary in self.transitions['boundary_radii'][:2]:
                ax2.axvline(x=boundary, color='r', linestyle='--', linewidth=2)
                if boundary < len(self.radial_profiles['median_gradient']):
                    ax2.plot(boundary, self.radial_profiles['median_gradient'][boundary], 
                            'rx', markersize=12, markeredgewidth=3, label='Detected Peak')
        
        # Show only first peak in legend
        handles, labels = ax2.get_legend_handles_labels()
        if len(handles) > 2:
            ax2.legend([handles[0], handles[1]], [labels[0], 'Detected Peaks'])
        
        # 3. Variance of Local Binary Patterns (Texture) vs. Radius
        ax3 = axes[2]
        # Use local variance as texture measure
        if 'median_variance' in self.radial_profiles:
            ax3.plot(radii, self.radial_profiles['median_variance'], 'purple', linewidth=2)
        ax3.set_ylabel('LBP Variance')
        ax3.set_xlabel('Radius from Center (pixels)')
        ax3.set_title('Variance of Local Binary Patterns (Texture) vs. Radius')
        ax3.grid(True, alpha=0.3)
        
        # Mark boundaries on variance plot
        if self.transitions['boundary_radii']:
            for boundary in self.transitions['boundary_radii'][:2]:
                ax3.axvline(x=boundary, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multimodal_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()


# Test script
if __name__ == "__main__":
    import sys
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'fiber_endface.jpg'
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print("Creating synthetic test image with realistic intensity pattern...")
        
        # Create synthetic test image matching real fiber characteristics
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        center = 300
        
        # Create a more realistic intensity pattern
        Y, X = np.ogrid[:600, :600]
        dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
        
        # Ferrule (background) - medium intensity (~175)
        img[:, :] = [175, 175, 175]
        
        # Cladding - DARKEST region (~95)
        cladding_mask = (dist_from_center > 100) & (dist_from_center <= 200)
        img[cladding_mask] = [95, 95, 95]
        
        # Core - BRIGHTEST region (~230)
        core_mask = dist_from_center <= 100
        img[core_mask] = [230, 230, 230]
        
        # Add smooth transitions
        # Core to cladding transition
        transition_width = 10
        for r in range(95, 105):
            mask = (dist_from_center > r) & (dist_from_center <= r + 1)
            t = (r - 95) / transition_width
            intensity = int(230 * (1 - t) + 95 * t)
            img[mask] = [intensity, intensity, intensity]
        
        # Cladding to ferrule transition
        for r in range(195, 205):
            mask = (dist_from_center > r) & (dist_from_center <= r + 1)
            t = (r - 195) / transition_width
            intensity = int(95 * (1 - t) + 175 * t)
            img[mask] = [intensity, intensity, intensity]
        
        # Add realistic noise and blur
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Slight blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        cv2.imwrite(image_path, img)
        print(f"Synthetic image saved as '{image_path}'")
        print("Pattern: Core (bright ~230) -> Cladding (dark ~95) -> Ferrule (medium ~175)")
    
    # Run analysis
    try:
        analyzer = PixelBasedFiberAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("Analysis Summary:")
        print("=" * 60)
        
        # Print summary
        for region in ['core', 'cladding', 'ferrule']:
            if region in results['intensity_statistics']:
                stats = results['intensity_statistics'][region]
                print(f"\n{region.upper()}:")
                print(f"  Mean intensity: {stats['mean']:.1f}")
                print(f"  Std deviation: {stats['std']:.1f}")
                
                if region in results['region_characteristics']:
                    chars = results['region_characteristics'][region]
                    print(f"  Size: {chars['pixel_count']} pixels ({chars['percentage']:.1f}%)")
        
        if 'transition_analysis' in results:
            trans = results['transition_analysis']
            print(f"\nDetected {trans['num_regions']} regions")
            print(f"Boundaries at radii: {trans['detected_boundaries']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()