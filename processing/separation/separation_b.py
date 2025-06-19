import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import least_squares
from skimage import filters, feature, morphology, exposure, measure
from skimage.filters import threshold_otsu, threshold_local, threshold_multiotsu
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import warnings
warnings.filterwarnings('ignore')

class FiberOpticAnalyzer:
    """
    Advanced Fiber Optic End-Face Analysis System
    Implements multi-stage computational analysis for precise region separation
    """
    
    def __init__(self, image_path):
        """Initialize analyzer with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.results = {}
        self.visualizations = []
        
    def stage1_image_preparation(self):
        """
        Stage 1: Initial Image Preparation and Feature Enhancement
        Implements greyscale conversion, heatmap generation, and feature calculation
        """
        print("=== STAGE 1: Image Preparation and Feature Enhancement ===")
        
        # 1.1 Greyscale Conversion with optimal channel weighting
        print("1.1 Converting to greyscale with perceptual weighting...")
        self.greyscale = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for enhanced local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.greyscale_enhanced = clahe.apply(self.greyscale)
        
        # 1.2 Illumination Correction using homomorphic filtering
        print("1.2 Applying illumination correction...")
        self.illumination_corrected = self._homomorphic_filter(self.greyscale_enhanced)
        
        # 1.3 Noise Reduction with edge-preserving filter
        print("1.3 Applying edge-preserving noise reduction...")
        self.denoised = cv2.bilateralFilter(self.illumination_corrected, 9, 75, 75)
        
        # 1.4 Heatmap Generation and Recolorization
        print("1.4 Generating multi-scale heatmap...")
        self.heatmap, self.colored_heatmap = self._generate_advanced_heatmap(self.denoised)
        
        # 1.5 Comprehensive Feature Calculation
        print("1.5 Calculating comprehensive feature maps...")
        self.features = self._calculate_features()
        
        # Visualize Stage 1 results
        self._visualize_stage1()
        
    def _homomorphic_filter(self, img):
        """Apply homomorphic filtering for illumination correction"""
        # Convert to float and add small value to avoid log(0)
        img_float = img.astype(np.float32) + 1.0
        
        # Apply log transform
        img_log = np.log(img_float)
        
        # Apply FFT
        img_fft = np.fft.fft2(img_log)
        
        # Create high-pass filter
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        
        # Gaussian high-pass filter
        D0 = 30  # cutoff frequency
        n = 2    # filter order
        H = np.zeros((rows, cols), np.float32)
        
        for i in range(rows):
            for j in range(cols):
                D = np.sqrt((i-crow)**2 + (j-ccol)**2)
                H[i,j] = 1 - np.exp(-(D**2)/(2*(D0**2)))
        
        # Apply filter
        img_fft_filtered = img_fft * H
        
        # Inverse FFT
        img_filtered = np.real(np.fft.ifft2(img_fft_filtered))
        
        # Exponential transform
        img_exp = np.exp(img_filtered) - 1.0
        
        # Normalize to 8-bit
        img_normalized = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
        
        return img_normalized.astype(np.uint8)
    
    def _generate_advanced_heatmap(self, img):
        """Generate multi-scale heatmap with custom colorization"""
        # Multi-scale analysis
        scales = [1, 2, 4]
        heatmaps = []
        
        for scale in scales:
            # Gaussian pyramid for multi-scale
            scaled = cv2.GaussianBlur(img, (scale*2+1, scale*2+1), scale)
            
            # Normalize to 0-1
            normalized = scaled.astype(np.float32) / 255.0
            
            # Apply gamma correction for enhanced contrast
            gamma = 0.7
            corrected = np.power(normalized, gamma)
            
            heatmaps.append(corrected)
        
        # Combine multi-scale heatmaps
        combined_heatmap = np.mean(heatmaps, axis=0)
        
        # Apply custom colormap for sharp transitions
        # Create custom colormap with sharp transitions
        colormap = self._create_custom_colormap()
        colored = cv2.applyColorMap((combined_heatmap * 255).astype(np.uint8), colormap)
        
        return combined_heatmap, colored
    
    def _create_custom_colormap(self):
        """Create custom colormap with sharp transitions"""
        # Define color points for sharp transitions
        colors = np.array([
            [0, 0, 128],      # Dark blue for background
            [0, 128, 255],    # Light blue for ferrule
            [255, 255, 0],    # Yellow for cladding boundary
            [255, 128, 0],    # Orange for cladding
            [255, 0, 0],      # Red for core boundary
            [255, 255, 255]   # White for core
        ], dtype=np.uint8)
        
        # Create interpolated colormap
        positions = np.linspace(0, 255, len(colors))
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        
        for i in range(256):
            # Find surrounding colors
            idx = np.searchsorted(positions, i)
            if idx == 0:
                colormap[i, 0] = colors[0]
            elif idx >= len(colors):
                colormap[i, 0] = colors[-1]
            else:
                # Sharp transition - use nearest color
                if i - positions[idx-1] < positions[idx] - i:
                    colormap[i, 0] = colors[idx-1]
                else:
                    colormap[i, 0] = colors[idx]
        
        return colormap
    
    def _calculate_features(self):
        """Calculate comprehensive feature maps"""
        features = {}
        
        # 1. Gradient calculations with multiple operators
        print("   - Calculating multi-operator gradients...")
        
        # Sobel gradients
        sobel_x = cv2.Sobel(self.denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.denoised, cv2.CV_64F, 0, 1, ksize=3)
        features['sobel_magnitude'] = np.sqrt(sobel_x**2 + sobel_y**2)
        features['sobel_direction'] = np.arctan2(sobel_y, sobel_x)
        
        # Scharr gradients (more accurate)
        scharr_x = cv2.Scharr(self.denoised, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(self.denoised, cv2.CV_64F, 0, 1)
        features['scharr_magnitude'] = np.sqrt(scharr_x**2 + scharr_y**2)
        
        # Prewitt gradients
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(self.denoised, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(self.denoised, cv2.CV_64F, kernel_y)
        features['prewitt_magnitude'] = np.sqrt(prewitt_x**2 + prewitt_y**2)
        
        # Combined gradient (weighted average)
        features['combined_gradient'] = (
            0.4 * features['sobel_magnitude'] + 
            0.4 * features['scharr_magnitude'] + 
            0.2 * features['prewitt_magnitude']
        )
        
        # 2. Local variance calculation
        print("   - Calculating multi-scale local variance...")
        window_sizes = [3, 5, 7, 9]
        variances = []
        
        for window in window_sizes:
            # Calculate local mean
            kernel = np.ones((window, window)) / (window * window)
            local_mean = cv2.filter2D(self.denoised.astype(np.float32), -1, kernel)
            
            # Calculate local variance
            local_sq_mean = cv2.filter2D(self.denoised.astype(np.float32)**2, -1, kernel)
            variance = local_sq_mean - local_mean**2
            variance[variance < 0] = 0  # Handle numerical errors
            
            variances.append(variance)
        
        # Multi-scale variance combination
        features['local_variance'] = np.mean(variances, axis=0)
        features['variance_gradient'] = np.gradient(features['local_variance'])[0]
        
        # 3. Texture analysis
        print("   - Performing texture analysis...")
        
        # Local Binary Patterns
        from skimage.feature import local_binary_pattern
        radius = 3
        n_points = 8 * radius
        features['lbp'] = local_binary_pattern(self.denoised, n_points, radius, method='uniform')
        
        # Gabor filters for texture
        features['gabor_responses'] = self._apply_gabor_filters(self.denoised)
        
        # 4. Edge detection with multiple methods
        print("   - Applying multi-method edge detection...")
        
        # Canny edge detection with automatic thresholds
        sigma = 1.4
        v = np.median(self.denoised)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        features['canny_edges'] = cv2.Canny(self.denoised, lower, upper)
        
        # Laplacian of Gaussian (LoG)
        features['log_edges'] = cv2.Laplacian(
            cv2.GaussianBlur(self.denoised, (5, 5), 1.4), 
            cv2.CV_64F
        )
        
        # 5. Radial analysis preparation
        print("   - Preparing radial analysis features...")
        center_y, center_x = self.height // 2, self.width // 2
        Y, X = np.ogrid[:self.height, :self.width]
        features['distance_map'] = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        features['angle_map'] = np.arctan2(Y - center_y, X - center_x)
        
        return features
    
    def _apply_gabor_filters(self, img):
        """Apply bank of Gabor filters for texture analysis"""
        # Gabor filter parameters
        ksize = 31
        sigma = 4.0
        theta_values = np.arange(0, np.pi, np.pi/8)
        lamda = 10.0
        gamma = 0.5
        
        responses = []
        
        for theta in theta_values:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        # Combine responses
        gabor_magnitude = np.sqrt(np.sum([r**2 for r in responses], axis=0))
        
        return gabor_magnitude
    
    def stage2_geometric_localization(self):
        """
        Stage 2: Geometric Localization and Iterative Refinement
        Implements Hough circles, iterative RANSAC, and least squares fitting
        """
        print("\n=== STAGE 2: Geometric Localization and Iterative Refinement ===")
        
        # 2.1 Initial estimation with Hough Circles
        print("2.1 Applying multi-resolution Hough Circle Transform...")
        initial_circles = self._multi_scale_hough_circles()
        
        # 2.2 Edge point extraction for fitting
        print("2.2 Extracting high-confidence edge points...")
        edge_points = self._extract_edge_points()
        
        # 2.3 Radial profile analysis
        print("2.3 Analyzing radial intensity profiles...")
        radial_profiles = self._analyze_radial_profiles(initial_circles)
        
        # 2.4 Iterative RANSAC and Least Squares refinement
        print("2.4 Performing iterative RANSAC with least squares refinement...")
        refined_circles = self._iterative_ransac_refinement(
            edge_points, initial_circles, radial_profiles
        )
        
        # 2.5 Validate and finalize geometric models
        print("2.5 Validating geometric models...")
        self.geometric_models = self._validate_geometric_models(refined_circles)
        
        # Visualize Stage 2 results
        self._visualize_stage2()
    
    def _multi_scale_hough_circles(self):
        """Apply Hough Circle Transform at multiple scales"""
        circles_all_scales = []
        
        # Multi-scale parameters
        scales = [1.0, 0.8, 0.6]
        
        for scale in scales:
            # Resize image
            scaled_img = cv2.resize(
                self.features['canny_edges'], 
                None, 
                fx=scale, 
                fy=scale, 
                interpolation=cv2.INTER_AREA
            )
            
            # Define radius range based on expected fiber dimensions
            min_radius = int(10 * scale)
            max_radius = int(min(scaled_img.shape) // 2 * scale)
            
            # Apply Hough transform
            circles = cv2.HoughCircles(
                scaled_img,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            if circles is not None:
                # Scale circles back to original size
                circles = circles[0] / scale
                circles_all_scales.extend(circles)
        
        # Cluster and merge similar circles
        if circles_all_scales:
            merged_circles = self._merge_similar_circles(np.array(circles_all_scales))
            return merged_circles
        else:
            # Fallback: estimate from image center
            center_x, center_y = self.width // 2, self.height // 2
            estimated_radius = min(self.width, self.height) // 4
            return np.array([[center_x, center_y, estimated_radius]])
    
    def _merge_similar_circles(self, circles, tolerance=10):
        """Merge circles that are similar in position and radius"""
        merged = []
        used = np.zeros(len(circles), dtype=bool)
        
        for i in range(len(circles)):
            if used[i]:
                continue
            
            similar_circles = [circles[i]]
            used[i] = True
            
            for j in range(i+1, len(circles)):
                if used[j]:
                    continue
                
                # Check if circles are similar
                dist = np.sqrt((circles[i][0] - circles[j][0])**2 + 
                              (circles[i][1] - circles[j][1])**2)
                radius_diff = abs(circles[i][2] - circles[j][2])
                
                if dist < tolerance and radius_diff < tolerance:
                    similar_circles.append(circles[j])
                    used[j] = True
            
            # Average similar circles
            avg_circle = np.mean(similar_circles, axis=0)
            merged.append(avg_circle)
        
        return np.array(merged)
    
    def _extract_edge_points(self):
        """Extract high-confidence edge points for geometric fitting"""
        # Combine multiple edge detection methods
        combined_edges = (
            self.features['canny_edges'] +
            (self.features['combined_gradient'] > np.percentile(self.features['combined_gradient'], 90)) * 255 +
            (np.abs(self.features['log_edges']) > np.percentile(np.abs(self.features['log_edges']), 90)) * 255
        )
        
        # Threshold and thin edges
        combined_edges = (combined_edges > 128).astype(np.uint8)
        thinned_edges = cv2.ximgproc.thinning(combined_edges)
        
        # Extract point coordinates
        edge_points = np.column_stack(np.where(thinned_edges > 0))
        edge_points = edge_points[:, [1, 0]]  # Convert to (x, y) format
        
        # Filter points by gradient strength
        gradient_strength = []
        for point in edge_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                strength = self.features['combined_gradient'][y, x]
                gradient_strength.append(strength)
            else:
                gradient_strength.append(0)
        
        gradient_strength = np.array(gradient_strength)
        threshold = np.percentile(gradient_strength, 70)
        strong_edge_points = edge_points[gradient_strength > threshold]
        
        return strong_edge_points
    
    def _analyze_radial_profiles(self, initial_circles):
        """Analyze radial intensity profiles to identify boundaries"""
        profiles = []
        
        for circle in initial_circles:
            cx, cy, r = circle
            
            # Sample along multiple radial lines
            num_angles = 360
            angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
            
            radial_data = []
            
            for angle in angles:
                # Sample points along this radial line
                max_radius = int(min(
                    cx, cy, 
                    self.width - cx, 
                    self.height - cy,
                    r * 2  # Look up to 2x the initial radius
                ))
                
                radii = np.arange(0, max_radius)
                x_coords = cx + radii * np.cos(angle)
                y_coords = cy + radii * np.sin(angle)
                
                # Sample intensities
                intensities = []
                for x, y in zip(x_coords, y_coords):
                    if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                        intensity = self.denoised[int(y), int(x)]
                        intensities.append(intensity)
                
                if intensities:
                    radial_data.append({
                        'angle': angle,
                        'radii': radii[:len(intensities)],
                        'intensities': np.array(intensities),
                        'gradient': np.gradient(intensities)
                    })
            
            profiles.append({
                'center': (cx, cy),
                'initial_radius': r,
                'radial_data': radial_data
            })
        
        return profiles
    
    def _iterative_ransac_refinement(self, edge_points, initial_circles, radial_profiles):
        """Iterative RANSAC with least squares refinement"""
        refined_circles = []
        
        for i, (initial_circle, profile) in enumerate(zip(initial_circles, radial_profiles)):
            print(f"   Refining circle {i+1}/{len(initial_circles)}...")
            
            # Multiple RANSAC iterations
            num_iterations = 10
            ransac_results = []
            
            for iteration in range(num_iterations):
                # RANSAC circle fitting
                ransac_circle = self._ransac_circle_fit(
                    edge_points, 
                    initial_guess=initial_circle,
                    max_iterations=1000,
                    threshold=5.0
                )
                
                if ransac_circle is not None:
                    # Least squares refinement
                    refined = self._least_squares_circle_fit(
                        edge_points, 
                        ransac_circle,
                        radial_profile=profile
                    )
                    
                    ransac_results.append(refined)
            
            # Analyze consistency of results
            if ransac_results:
                # Calculate median circle parameters
                centers = np.array([[c[0], c[1]] for c in ransac_results])
                radii = np.array([c[2] for c in ransac_results])
                
                median_center = np.median(centers, axis=0)
                median_radius = np.median(radii)
                
                # Calculate consistency scores
                center_std = np.std(centers, axis=0)
                radius_std = np.std(radii)
                
                # Final refinement with all inliers
                final_circle = self._final_circle_refinement(
                    edge_points,
                    [median_center[0], median_center[1], median_radius],
                    center_std,
                    radius_std
                )
                
                refined_circles.append({
                    'circle': final_circle,
                    'confidence': 1.0 / (1.0 + np.mean(center_std) + radius_std),
                    'iterations': len(ransac_results)
                })
        
        return refined_circles
    
    def _ransac_circle_fit(self, points, initial_guess=None, max_iterations=1000, threshold=5.0):
        """RANSAC circle fitting"""
        if len(points) < 3:
            return None
        
        best_circle = None
        best_inliers = 0
        
        for _ in range(max_iterations):
            # Random sample 3 points
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_idx]
            
            # Fit circle to 3 points
            circle = self._fit_circle_3points(sample_points)
            
            if circle is None:
                continue
            
            # Count inliers
            distances = np.sqrt((points[:, 0] - circle[0])**2 + 
                               (points[:, 1] - circle[1])**2)
            inliers = np.abs(distances - circle[2]) < threshold
            num_inliers = np.sum(inliers)
            
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_circle = circle
        
        return best_circle
    
    def _fit_circle_3points(self, points):
        """Fit circle through 3 points"""
        p1, p2, p3 = points
        
        # Convert to homogeneous coordinates
        temp = p2[0]**2 + p2[1]**2
        bc = (p1[0]**2 + p1[1]**2 - temp) / 2
        cd = (temp - p3[0]**2 - p3[1]**2) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
        
        if abs(det) < 1e-10:
            return None
        
        # Center
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        
        # Radius
        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        
        return [cx, cy, radius]
    
    def _least_squares_circle_fit(self, points, initial_circle, radial_profile=None):
        """Least squares circle fitting with radial profile guidance"""
        def residuals(params, points):
            cx, cy, r = params
            distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
            return distances - r
        
        # Filter points near the initial circle
        distances = np.sqrt((points[:, 0] - initial_circle[0])**2 + 
                           (points[:, 1] - initial_circle[1])**2)
        near_points = points[np.abs(distances - initial_circle[2]) < initial_circle[2] * 0.3]
        
        if len(near_points) < 10:
            return initial_circle
        
        # Optimize
        result = least_squares(
            residuals, 
            initial_circle, 
            args=(near_points,),
            method='lm'
        )
        
        return result.x
    
    def _final_circle_refinement(self, points, circle, center_std, radius_std):
        """Final refinement using robust statistics"""
        cx, cy, r = circle
        
        # Get points near the circle
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        circle_distances = np.abs(distances - r)
        
        # Use robust threshold based on MAD (Median Absolute Deviation)
        mad = np.median(np.abs(circle_distances - np.median(circle_distances)))
        threshold = 3 * mad
        
        inlier_points = points[circle_distances < threshold]
        
        if len(inlier_points) > 10:
            # Final least squares fit on clean inliers
            def residuals(params, points):
                cx, cy, r = params
                distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
                return distances - r
            
            result = least_squares(
                residuals, 
                circle, 
                args=(inlier_points,),
                method='lm'
            )
            
            return result.x
        
        return circle
    
    def _validate_geometric_models(self, refined_circles):
        """Validate and classify geometric models"""
        # Sort circles by radius
        sorted_circles = sorted(refined_circles, key=lambda x: x['circle'][2])
        
        validated_models = {
            'core': None,
            'cladding': None,
            'ferrule': None
        }
        
        # Typical fiber dimensions (in pixels, adjust based on magnification)
        # Core: 8-50 μm, Cladding: 125 μm, Ferrule: varies
        
        if len(sorted_circles) >= 1:
            # Smallest circle is likely the core
            validated_models['core'] = {
                'center': (sorted_circles[0]['circle'][0], sorted_circles[0]['circle'][1]),
                'radius': sorted_circles[0]['circle'][2],
                'confidence': sorted_circles[0]['confidence']
            }
        
        if len(sorted_circles) >= 2:
            # Second circle is likely the cladding
            validated_models['cladding'] = {
                'center': (sorted_circles[1]['circle'][0], sorted_circles[1]['circle'][1]),
                'radius': sorted_circles[1]['circle'][2],
                'confidence': sorted_circles[1]['confidence']
            }
        
        # Validate concentricity
        if validated_models['core'] and validated_models['cladding']:
            core_center = np.array(validated_models['core']['center'])
            clad_center = np.array(validated_models['cladding']['center'])
            
            offset = np.linalg.norm(core_center - clad_center)
            validated_models['concentricity_offset'] = offset
            
            # Use average center for better accuracy
            avg_center = (core_center + clad_center) / 2
            validated_models['core']['center'] = tuple(avg_center)
            validated_models['cladding']['center'] = tuple(avg_center)
        
        return validated_models
    
    def stage3_region_separation(self):
        """
        Stage 3: Region Separation
        Create masks and separate regions based on validated geometric models
        """
        print("\n=== STAGE 3: Region Separation ===")
        
        # 3.1 Create region masks
        print("3.1 Creating region masks...")
        self.masks = self._create_region_masks()
        
        # 3.2 Extract regions
        print("3.2 Extracting separated regions...")
        self.separated_regions = self._extract_regions()
        
        # 3.3 Refine boundaries using active contours
        print("3.3 Refining boundaries with active contours...")
        self.refined_masks = self._refine_boundaries()
        
        # Visualize Stage 3 results
        self._visualize_stage3()
    
    def _create_region_masks(self):
        """Create masks for each region"""
        masks = {}
        
        # Initialize masks
        masks['core'] = np.zeros((self.height, self.width), dtype=np.uint8)
        masks['cladding'] = np.zeros((self.height, self.width), dtype=np.uint8)
        masks['ferrule'] = np.zeros((self.height, self.width), dtype=np.uint8)
        
        if self.geometric_models['core']:
            # Core mask
            cv2.circle(
                masks['core'],
                (int(self.geometric_models['core']['center'][0]),
                 int(self.geometric_models['core']['center'][1])),
                int(self.geometric_models['core']['radius']),
                255, -1
            )
        
        if self.geometric_models['cladding']:
            # Cladding mask (annulus)
            cv2.circle(
                masks['cladding'],
                (int(self.geometric_models['cladding']['center'][0]),
                 int(self.geometric_models['cladding']['center'][1])),
                int(self.geometric_models['cladding']['radius']),
                255, -1
            )
            
            # Subtract core from cladding
            if self.geometric_models['core']:
                cv2.circle(
                    masks['cladding'],
                    (int(self.geometric_models['core']['center'][0]),
                     int(self.geometric_models['core']['center'][1])),
                    int(self.geometric_models['core']['radius']),
                    0, -1
                )
        
        # Ferrule mask (everything outside cladding)
        if self.geometric_models['cladding']:
            masks['ferrule'] = 255 - masks['cladding'] - masks['core']
        
        return masks
    
    def _extract_regions(self):
        """Extract image regions using masks"""
        regions = {}
        
        for region_name, mask in self.masks.items():
            # Apply mask to original image
            region = cv2.bitwise_and(self.original, self.original, mask=mask)
            regions[region_name] = region
        
        return regions
    
    def _refine_boundaries(self):
        """Refine boundaries using active contours (snakes)"""
        refined_masks = {}
        
        for region_name, mask in self.masks.items():
            if np.any(mask):
                # Find initial contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if contours:
                    # Use the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create edge map for active contour
                    edge_map = self.features['combined_gradient']
                    edge_map = cv2.GaussianBlur(edge_map, (5, 5), 1.0)
                    
                    # Apply morphological snakes (simplified active contour)
                    refined_mask = self._morphological_snakes(
                        mask, edge_map, iterations=20
                    )
                    
                    refined_masks[region_name] = refined_mask
                else:
                    refined_masks[region_name] = mask
            else:
                refined_masks[region_name] = mask
        
        return refined_masks
    
    def _morphological_snakes(self, initial_mask, edge_map, iterations=20):
        """Simplified morphological active contours"""
        mask = initial_mask.copy()
        
        # Normalize edge map
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
        
        for i in range(iterations):
            # Compute gradient of the mask
            grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3)))
            
            # Attraction to edges
            attraction = grad * edge_map
            
            # Update mask
            mask = mask + attraction * 0.1
            mask = np.clip(mask, 0, 255).astype(np.uint8)
            
            # Smooth
            mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
            mask = (mask > 128).astype(np.uint8) * 255
        
        return mask
    
    def stage4_artifact_removal(self):
        """
        Stage 4: Post-Separation Artifact Removal and Final Output
        Clean regions and produce final results
        """
        print("\n=== STAGE 4: Artifact Removal and Final Output ===")
        
        # 4.1 Advanced artifact removal for each region
        print("4.1 Applying advanced artifact removal...")
        self.cleaned_regions = {}
        
        # Core cleaning
        print("   - Cleaning core region...")
        self.cleaned_regions['core'] = self._clean_core_region(
            self.separated_regions['core'],
            self.refined_masks['core']
        )
        
        # Cladding cleaning
        print("   - Cleaning cladding region...")
        self.cleaned_regions['cladding'] = self._clean_cladding_region(
            self.separated_regions['cladding'],
            self.refined_masks['cladding']
        )
        
        # Ferrule cleaning
        print("   - Cleaning ferrule region...")
        self.cleaned_regions['ferrule'] = self._clean_ferrule_region(
            self.separated_regions['ferrule'],
            self.refined_masks['ferrule']
        )
        
        # 4.2 Final image reconstruction
        print("4.2 Reconstructing final image...")
        self.final_image = self._reconstruct_final_image()
        
        # 4.3 Generate comprehensive analysis report
        print("4.3 Generating analysis report...")
        self.analysis_report = self._generate_analysis_report()
        
        # Visualize final results
        self._visualize_final_results()
    
    def _clean_core_region(self, region, mask):
        """Advanced cleaning for core region"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Multi-level Otsu thresholding
        thresholds = threshold_multiotsu(gray[mask > 0], classes=3)
        
        # Create multi-level mask
        cleaned_mask = np.zeros_like(mask)
        
        # Keep only the brightest class (core material)
        brightest_threshold = thresholds[-1]
        bright_pixels = (gray > brightest_threshold) & (mask > 0)
        
        # Morphological operations to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_pixels = cv2.morphologyEx(bright_pixels.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        bright_pixels = cv2.morphologyEx(bright_pixels, cv2.MORPH_CLOSE, kernel)
        
        # Find largest connected component (main core)
        labels = measure.label(bright_pixels)
        if labels.max() > 0:
            largest_component = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            cleaned_mask = largest_component.astype(np.uint8) * 255
        
        # Apply cleaned mask
        cleaned_region = cv2.bitwise_and(region, region, mask=cleaned_mask)
        
        return cleaned_region
    
    def _clean_cladding_region(self, region, mask):
        """Advanced cleaning for cladding region"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for local variations
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Combine with original mask
        combined_mask = cv2.bitwise_and(adaptive_thresh, mask)
        
        # Remove noise while preserving structure
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Edge-preserving smoothing
        smoothed = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Reconstruct color image
        cleaned_region = region.copy()
        for i in range(3):
            channel = cleaned_region[:, :, i]
            channel[combined_mask == 0] = 0
            
            # Apply smoothing to valid pixels
            valid_pixels = channel[combined_mask > 0]
            if len(valid_pixels) > 0:
                mean_val = np.mean(valid_pixels)
                std_val = np.std(valid_pixels)
                
                # Remove outliers
                outlier_mask = np.abs(channel - mean_val) > 2 * std_val
                channel[outlier_mask & (combined_mask > 0)] = mean_val
        
        return cleaned_region
    
    def _clean_ferrule_region(self, region, mask):
        """Advanced cleaning for ferrule region"""
        # Simple cleaning for ferrule - mainly remove noise
        cleaned = region.copy()
        
        # Apply mask
        cleaned[mask == 0] = 0
        
        # Denoise
        cleaned = cv2.fastNlMeansDenoisingColored(cleaned, None, 10, 10, 7, 21)
        
        return cleaned
    
    def _reconstruct_final_image(self):
        """Reconstruct final image from cleaned regions"""
        # Start with black image
        final = np.zeros_like(self.original)
        
        # Add regions in order (ferrule, cladding, core)
        # This ensures proper layering
        for region_name in ['ferrule', 'cladding', 'core']:
            region = self.cleaned_regions[region_name]
            mask = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) > 0
            final[mask] = region[mask]
        
        return final
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'image_properties': {
                'width': self.width,
                'height': self.height,
                'channels': self.original.shape[2] if len(self.original.shape) > 2 else 1
            },
            'geometric_measurements': {},
            'quality_metrics': {},
            'statistics': {}
        }
        
        # Geometric measurements
        if self.geometric_models['core']:
            report['geometric_measurements']['core'] = {
                'center': self.geometric_models['core']['center'],
                'radius_pixels': self.geometric_models['core']['radius'],
                'diameter_pixels': self.geometric_models['core']['radius'] * 2,
                'area_pixels': np.pi * self.geometric_models['core']['radius']**2,
                'confidence': self.geometric_models['core']['confidence']
            }
        
        if self.geometric_models['cladding']:
            report['geometric_measurements']['cladding'] = {
                'center': self.geometric_models['cladding']['center'],
                'radius_pixels': self.geometric_models['cladding']['radius'],
                'diameter_pixels': self.geometric_models['cladding']['radius'] * 2,
                'area_pixels': np.pi * self.geometric_models['cladding']['radius']**2,
                'confidence': self.geometric_models['cladding']['confidence']
            }
        
        if 'concentricity_offset' in self.geometric_models:
            report['geometric_measurements']['concentricity_offset_pixels'] = \
                self.geometric_models['concentricity_offset']
        
        # Quality metrics
        report['quality_metrics']['edge_sharpness'] = self._calculate_edge_sharpness()
        report['quality_metrics']['circularity'] = self._calculate_circularity()
        report['quality_metrics']['surface_roughness'] = self._calculate_surface_roughness()
        
        # Statistical analysis
        for region_name in ['core', 'cladding', 'ferrule']:
            mask = self.refined_masks[region_name]
            if np.any(mask):
                gray = cv2.cvtColor(self.cleaned_regions[region_name], cv2.COLOR_BGR2GRAY)
                valid_pixels = gray[mask > 0]
                
                if len(valid_pixels) > 0:
                    report['statistics'][region_name] = {
                        'mean_intensity': float(np.mean(valid_pixels)),
                        'std_intensity': float(np.std(valid_pixels)),
                        'min_intensity': float(np.min(valid_pixels)),
                        'max_intensity': float(np.max(valid_pixels)),
                        'median_intensity': float(np.median(valid_pixels)),
                        'pixel_count': len(valid_pixels)
                    }
        
        return report
    
    def _calculate_edge_sharpness(self):
        """Calculate edge sharpness metric"""
        # Use gradient magnitude as sharpness indicator
        avg_gradient = np.mean(self.features['combined_gradient'])
        max_gradient = np.max(self.features['combined_gradient'])
        
        return {
            'average_gradient': float(avg_gradient),
            'maximum_gradient': float(max_gradient),
            'sharpness_score': float(avg_gradient / 255.0)  # Normalized score
        }
    
    def _calculate_circularity(self):
        """Calculate circularity of detected regions"""
        circularity_scores = {}
        
        for region_name in ['core', 'cladding']:
            if region_name in self.geometric_models and self.geometric_models[region_name]:
                # Find contour
                mask = self.refined_masks[region_name]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        circularity_scores[region_name] = float(circularity)
        
        return circularity_scores
    
    def _calculate_surface_roughness(self):
        """Calculate surface roughness of boundaries"""
        roughness_scores = {}
        
        for region_name in ['core', 'cladding']:
            if region_name in self.refined_masks:
                mask = self.refined_masks[region_name]
                
                # Extract boundary
                boundary = cv2.Canny(mask, 100, 200)
                
                if np.any(boundary):
                    # Calculate roughness as deviation from perfect circle
                    points = np.column_stack(np.where(boundary > 0))
                    
                    if len(points) > 10 and region_name in self.geometric_models:
                        center = self.geometric_models[region_name]['center']
                        radius = self.geometric_models[region_name]['radius']
                        
                        # Calculate radial distances
                        distances = np.sqrt((points[:, 1] - center[0])**2 + 
                                          (points[:, 0] - center[1])**2)
                        
                        # Roughness as standard deviation of distances
                        roughness = np.std(distances - radius)
                        roughness_scores[region_name] = float(roughness)
        
        return roughness_scores
    
    def _visualize_stage1(self):
        """Visualize Stage 1 results"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Stage 1: Image Preparation and Feature Enhancement', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Greyscale
        axes[0, 1].imshow(self.greyscale, cmap='gray')
        axes[0, 1].set_title('Greyscale')
        axes[0, 1].axis('off')
        
        # Illumination corrected
        axes[0, 2].imshow(self.illumination_corrected, cmap='gray')
        axes[0, 2].set_title('Illumination Corrected')
        axes[0, 2].axis('off')
        
        # Heatmap
        axes[0, 3].imshow(cv2.cvtColor(self.colored_heatmap, cv2.COLOR_BGR2RGB))
        axes[0, 3].set_title('Enhanced Heatmap')
        axes[0, 3].axis('off')
        
        # Combined gradient
        axes[1, 0].imshow(self.features['combined_gradient'], cmap='hot')
        axes[1, 0].set_title('Combined Gradient')
        axes[1, 0].axis('off')
        
        # Local variance
        axes[1, 1].imshow(self.features['local_variance'], cmap='viridis')
        axes[1, 1].set_title('Local Variance')
        axes[1, 1].axis('off')
        
        # Canny edges
        axes[1, 2].imshow(self.features['canny_edges'], cmap='gray')
        axes[1, 2].set_title('Canny Edges')
        axes[1, 2].axis('off')
        
        # Gabor response
        axes[1, 3].imshow(self.features['gabor_responses'], cmap='gray')
        axes[1, 3].set_title('Gabor Texture Response')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        self.visualizations.append(('stage1', fig))
    
    def _visualize_stage2(self):
        """Visualize Stage 2 results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Stage 2: Geometric Localization and Refinement', fontsize=16)
        
        # Draw circles on original
        img_circles = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB).copy()
        
        # Draw detected circles
        if self.geometric_models['core']:
            cv2.circle(img_circles,
                      (int(self.geometric_models['core']['center'][0]),
                       int(self.geometric_models['core']['center'][1])),
                      int(self.geometric_models['core']['radius']),
                      (255, 0, 0), 2)
        
        if self.geometric_models['cladding']:
            cv2.circle(img_circles,
                      (int(self.geometric_models['cladding']['center'][0]),
                       int(self.geometric_models['cladding']['center'][1])),
                      int(self.geometric_models['cladding']['radius']),
                      (0, 255, 0), 2)
        
        axes[0, 0].imshow(img_circles)
        axes[0, 0].set_title('Detected Circles')
        axes[0, 0].axis('off')
        
        # Edge points
        edge_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        edge_img[:, :] = [255, 255, 255]
        edge_points = self._extract_edge_points()
        for point in edge_points[::10]:  # Show every 10th point
            cv2.circle(edge_img, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
        
        axes[0, 1].imshow(edge_img)
        axes[0, 1].set_title('Edge Points for Fitting')
        axes[0, 1].axis('off')
        
        # Radial intensity profile
        if self.geometric_models['core']:
            center = self.geometric_models['core']['center']
            # Sample radial profile
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            
            ax = axes[0, 2]
            for angle in angles:
                max_r = min(center[0], center[1], self.width-center[0], self.height-center[1])
                radii = np.arange(0, int(max_r))
                x_coords = center[0] + radii * np.cos(angle)
                y_coords = center[1] + radii * np.sin(angle)
                
                intensities = []
                for x, y in zip(x_coords, y_coords):
                    if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                        intensities.append(self.denoised[int(y), int(x)])
                
                ax.plot(radii[:len(intensities)], intensities, alpha=0.5)
            
            ax.set_xlabel('Radius (pixels)')
            ax.set_ylabel('Intensity')
            ax.set_title('Radial Intensity Profiles')
            ax.grid(True)
        
        # Gradient along radial lines
        if self.geometric_models['core']:
            ax = axes[1, 0]
            center = self.geometric_models['core']['center']
            angle = 0  # Sample at 0 degrees
            
            max_r = min(center[0], center[1], self.width-center[0], self.height-center[1])
            radii = np.arange(0, int(max_r))
            x_coords = center[0] + radii * np.cos(angle)
            y_coords = center[1] + radii * np.sin(angle)
            
            gradients = []
            for x, y in zip(x_coords, y_coords):
                if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                    gradients.append(self.features['combined_gradient'][int(y), int(x)])
            
            ax.plot(radii[:len(gradients)], gradients)
            ax.set_xlabel('Radius (pixels)')
            ax.set_ylabel('Gradient Magnitude')
            ax.set_title('Radial Gradient Profile')
            ax.grid(True)
            
            # Mark detected boundaries
            if self.geometric_models['core']:
                ax.axvline(x=self.geometric_models['core']['radius'], 
                          color='r', linestyle='--', label='Core boundary')
            if self.geometric_models['cladding']:
                ax.axvline(x=self.geometric_models['cladding']['radius'], 
                          color='g', linestyle='--', label='Cladding boundary')
            ax.legend()
        
        # Confidence visualization
        ax = axes[1, 1]
        regions = []
        confidences = []
        
        if self.geometric_models['core']:
            regions.append('Core')
            confidences.append(self.geometric_models['core']['confidence'])
        if self.geometric_models['cladding']:
            regions.append('Cladding')
            confidences.append(self.geometric_models['cladding']['confidence'])
        
        if regions:
            ax.bar(regions, confidences)
            ax.set_ylabel('Confidence Score')
            ax.set_title('Detection Confidence')
            ax.set_ylim(0, 1)
        
        # Concentricity analysis
        ax = axes[1, 2]
        if 'concentricity_offset' in self.geometric_models:
            offset = self.geometric_models['concentricity_offset']
            ax.text(0.5, 0.5, f'Concentricity Offset:\n{offset:.2f} pixels', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Concentricity Analysis')
        ax.axis('off')
        
        plt.tight_layout()
        self.visualizations.append(('stage2', fig))
    
    def _visualize_stage3(self):
        """Visualize Stage 3 results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Stage 3: Region Separation', fontsize=16)
        
        # Original masks
        axes[0, 0].imshow(self.masks['core'], cmap='gray')
        axes[0, 0].set_title('Core Mask')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.masks['cladding'], cmap='gray')
        axes[0, 1].set_title('Cladding Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(self.masks['ferrule'], cmap='gray')
        axes[0, 2].set_title('Ferrule Mask')
        axes[0, 2].axis('off')
        
        # Separated regions
        axes[1, 0].imshow(cv2.cvtColor(self.separated_regions['core'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Separated Core')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(self.separated_regions['cladding'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Separated Cladding')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(self.separated_regions['ferrule'], cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Separated Ferrule')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        self.visualizations.append(('stage3', fig))
    
    def _visualize_final_results(self):
        """Visualize final results and analysis"""
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # Main results
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Original vs Final comparison
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(cv2.cvtColor(self.final_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Final Processed Image', fontsize=14)
        ax2.axis('off')
        
        # Cleaned regions
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(cv2.cvtColor(self.cleaned_regions['core'], cv2.COLOR_BGR2RGB))
        ax3.set_title('Cleaned Core')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(cv2.cvtColor(self.cleaned_regions['cladding'], cv2.COLOR_BGR2RGB))
        ax4.set_title('Cleaned Cladding')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.imshow(cv2.cvtColor(self.cleaned_regions['ferrule'], cv2.COLOR_BGR2RGB))
        ax5.set_title('Cleaned Ferrule')
        ax5.axis('off')
        
        # Intensity histogram
        ax6 = fig.add_subplot(gs[1, 3])
        for region_name, color in [('core', 'red'), ('cladding', 'green'), ('ferrule', 'blue')]:
            if region_name in self.analysis_report['statistics']:
                mask = self.refined_masks[region_name]
                gray = cv2.cvtColor(self.cleaned_regions[region_name], cv2.COLOR_BGR2GRAY)
                valid_pixels = gray[mask > 0]
                if len(valid_pixels) > 0:
                    ax6.hist(valid_pixels, bins=50, alpha=0.5, label=region_name, color=color)
        ax6.set_xlabel('Intensity')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Intensity Distribution by Region')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Measurements table
        ax7 = fig.add_subplot(gs[2, 0:2])
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create measurement table
        measurements = []
        if 'core' in self.analysis_report['geometric_measurements']:
            core_data = self.analysis_report['geometric_measurements']['core']
            measurements.append(['Core Diameter', f"{core_data['diameter_pixels']:.1f} px"])
            measurements.append(['Core Area', f"{core_data['area_pixels']:.1f} px²"])
        
        if 'cladding' in self.analysis_report['geometric_measurements']:
            clad_data = self.analysis_report['geometric_measurements']['cladding']
            measurements.append(['Cladding Diameter', f"{clad_data['diameter_pixels']:.1f} px"])
            measurements.append(['Cladding Area', f"{clad_data['area_pixels']:.1f} px²"])
        
        if 'concentricity_offset_pixels' in self.analysis_report['geometric_measurements']:
            offset = self.analysis_report['geometric_measurements']['concentricity_offset_pixels']
            measurements.append(['Concentricity Offset', f"{offset:.2f} px"])
        
        table = ax7.table(cellText=measurements, 
                         colLabels=['Measurement', 'Value'],
                         cellLoc='left',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax7.set_title('Geometric Measurements', fontsize=12, pad=20)
        
        # Quality metrics
        ax8 = fig.add_subplot(gs[2, 2:4])
        quality_data = []
        
        if 'edge_sharpness' in self.analysis_report['quality_metrics']:
            sharpness = self.analysis_report['quality_metrics']['edge_sharpness']['sharpness_score']
            quality_data.append(['Edge Sharpness', f"{sharpness:.3f}"])
        
        if 'circularity' in self.analysis_report['quality_metrics']:
            circ = self.analysis_report['quality_metrics']['circularity']
            if 'core' in circ:
                quality_data.append(['Core Circularity', f"{circ['core']:.3f}"])
            if 'cladding' in circ:
                quality_data.append(['Cladding Circularity', f"{circ['cladding']:.3f}"])
        
        if 'surface_roughness' in self.analysis_report['quality_metrics']:
            rough = self.analysis_report['quality_metrics']['surface_roughness']
            if 'core' in rough:
                quality_data.append(['Core Roughness', f"{rough['core']:.2f} px"])
            if 'cladding' in rough:
                quality_data.append(['Cladding Roughness', f"{rough['cladding']:.2f} px"])
        
        if quality_data:
            ax8.axis('tight')
            ax8.axis('off')
            table2 = ax8.table(cellText=quality_data,
                              colLabels=['Quality Metric', 'Value'],
                              cellLoc='left',
                              loc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1, 2)
            ax8.set_title('Quality Analysis', fontsize=12, pad=20)
        
        plt.suptitle('Fiber Optic End-Face Analysis Results', fontsize=16)
        plt.tight_layout()
        self.visualizations.append(('final_results', fig))
    
    def run_complete_analysis(self):
        """Run the complete fiber optic analysis pipeline"""
        print("Starting Fiber Optic End-Face Analysis...")
        print("=" * 60)
        
        # Execute all stages
        self.stage1_image_preparation()
        self.stage2_geometric_localization()
        self.stage3_region_separation()
        self.stage4_artifact_removal()
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        
        # Save results
        self.save_results()
        
        # Display all visualizations
        self.display_visualizations()
        
        return self.analysis_report
    
    def save_results(self, output_dir='fiber_analysis_results'):
        """Save all results and visualizations"""
        import os
        import json
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final image
        cv2.imwrite(os.path.join(output_dir, 'final_processed.png'), self.final_image)
        
        # Save cleaned regions
        for region_name, region_img in self.cleaned_regions.items():
            cv2.imwrite(os.path.join(output_dir, f'cleaned_{region_name}.png'), region_img)
        
        # Save masks
        for region_name, mask in self.refined_masks.items():
            cv2.imwrite(os.path.join(output_dir, f'mask_{region_name}.png'), mask)
        
        # Save analysis report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.analysis_report, f, indent=4)
        
        # Save visualizations
        for name, fig in self.visualizations:
            fig.savefig(os.path.join(output_dir, f'visualization_{name}.png'), 
                       dpi=150, bbox_inches='tight')
        
        print(f"\nResults saved to: {output_dir}/")
    
    def display_visualizations(self):
        """Display all visualizations"""
        for name, fig in self.visualizations:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your fiber optic image
    analyzer = FiberOpticAnalyzer('/home/jarvis/Downloads/gpt.png')
    
    # Run complete analysis
    report = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(json.dumps(report, indent=2))
