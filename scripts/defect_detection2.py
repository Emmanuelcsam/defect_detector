import cv2
import numpy as np
from scipy import ndimage, signal, stats
from scipy.ndimage import label, binary_erosion, binary_dilation
from scipy.fftpack import fft2, ifft2, fftshift
from skimage import morphology, feature, filters, measure, transform
from skimage.restoration import denoise_tv_chambolle
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pywt
from scipy.spatial import distance
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFiberDefectDetector:
    """
    Ultimate robust fiber optics end face defect detection system.
    Incorporates all methodologies from the provided documents.
    """
    
    def __init__(self, config=None):
        """Initialize the defect detector with configuration parameters."""
        self.config = config or self.get_default_config()
        self.defect_results = {}
        self.feature_vectors = {}
        
    def get_default_config(self):
        """Default configuration parameters from all documents."""
        return {
            # Preprocessing parameters
            'gaussian_sigma': 1.0,
            'anisotropic_iterations': 5,
            'anisotropic_kappa': 25,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            
            # Statistical parameters
            'zscore_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'mad_threshold': 3.0,
            'grubbs_alpha': 0.05,
            
            # Morphological parameters
            'morph_kernel_size': 3,
            'tophat_kernel_size': 15,
            
            # Detection thresholds
            'gradient_threshold': 0.1,
            'hessian_threshold': 0.01,
            'log_threshold': 0.02,
            'scratch_min_length': 10,
            'blob_min_area': 5,
            
            # DO2MR parameters (from paper)
            'do2mr_window_size': 5,
            'do2mr_gamma': 3.0,
            
            # LEI parameters (from paper)
            'lei_orientations': 12,
            'lei_line_length': 15,
            'lei_line_gap': 3,
            
            # Advanced parameters
            'wavelet_family': 'db4',
            'wavelet_level': 3,
            'gabor_frequencies': [0.05, 0.1, 0.2],
            'gabor_orientations': [0, 45, 90, 135],
            'hough_threshold': 50,
            'radon_angles': np.linspace(0, 180, 180),
            
            # Machine learning parameters
            'isolation_contamination': 0.1,
            'dbscan_eps': 0.5,
            'dbscan_min_samples': 5,
            'pca_components': 10
        }
    
    def analyze_defects(self, image, mask, region_type='inner'):
        """
        Main method to analyze defects in a specific region.
        
        Args:
            image: Original image
            mask: Binary mask for the region to analyze
            region_type: 'inner', 'annulus', or 'outer'
        
        Returns:
            Dictionary containing all defect analysis results
        """
        # Apply mask to get region of interest
        roi = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        
        # Store original for reference
        self.original_roi = roi.copy()
        self.mask = mask
        self.region_type = region_type
        
        # 1. Preprocessing pipeline
        preprocessed = self.comprehensive_preprocessing(roi)
        
        # 2. Statistical defect detection
        statistical_defects = self.statistical_defect_detection(preprocessed, mask)
        
        # 3. Spatial and morphological analysis
        spatial_defects = self.spatial_morphological_analysis(preprocessed, mask)
        
        # 4. Frequency domain analysis
        frequency_defects = self.frequency_domain_analysis(preprocessed, mask)
        
        # 5. Gradient and edge-based detection
        edge_defects = self.gradient_edge_detection(preprocessed, mask)
        
        # 6. Advanced blob and scratch detection
        blob_defects = self.advanced_blob_detection(preprocessed, mask)
        scratch_defects = self.advanced_scratch_detection(preprocessed, mask)
        
        # 7. DO2MR method from paper
        do2mr_defects = self.do2mr_detection(preprocessed, mask)
        
        # 8. LEI method from paper
        lei_defects = self.lei_scratch_detection(preprocessed, mask)
        
        # 9. Machine learning based detection
        ml_defects = self.machine_learning_detection(preprocessed, mask)
        
        # 10. Combine all detections
        combined_defects = self.combine_detections({
            'statistical': statistical_defects,
            'spatial': spatial_defects,
            'frequency': frequency_defects,
            'edge': edge_defects,
            'blob': blob_defects,
            'scratch': scratch_defects,
            'do2mr': do2mr_defects,
            'lei': lei_defects,
            'ml': ml_defects
        })
        
        # 11. Feature extraction and characterization
        defect_features = self.extract_defect_features(combined_defects, preprocessed)
        
        # 12. Quality metrics
        quality_metrics = self.calculate_quality_metrics(preprocessed, combined_defects, mask)
        
        # Compile results
        self.defect_results = {
            'region_type': region_type,
            'defects': combined_defects,
            'features': defect_features,
            'quality_metrics': quality_metrics,
            'individual_detections': {
                'statistical': statistical_defects,
                'spatial': spatial_defects,
                'frequency': frequency_defects,
                'edge': edge_defects,
                'blob': blob_defects,
                'scratch': scratch_defects,
                'do2mr': do2mr_defects,
                'lei': lei_defects,
                'ml': ml_defects
            }
        }
        
        return self.defect_results
    
    def comprehensive_preprocessing(self, image):
        """Apply all preprocessing techniques from the documents."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Gaussian smoothing (from documents)
        gaussian = cv2.GaussianBlur(gray, (5, 5), self.config['gaussian_sigma'])
        
        # 2. Anisotropic diffusion (Perona-Malik from documents)
        anisotropic = self.perona_malik_diffusion(
            gaussian, 
            self.config['anisotropic_iterations'],
            self.config['anisotropic_kappa']
        )
        
        # 3. Bilateral filtering for edge preservation
        bilateral = cv2.bilateralFilter(
            anisotropic.astype(np.uint8),
            self.config['bilateral_d'],
            self.config['bilateral_sigma_color'],
            self.config['bilateral_sigma_space']
        )
        
        # 4. Histogram equalization (from Otsu paper)
        equalized = cv2.equalizeHist(bilateral)
        
        # 5. Coherence enhancing diffusion (from documents)
        coherence = self.coherence_enhancing_diffusion(equalized)
        
        # Store preprocessing stages
        self.preprocessing_stages = {
            'original': gray,
            'gaussian': gaussian,
            'anisotropic': anisotropic,
            'bilateral': bilateral,
            'equalized': equalized,
            'coherence': coherence
        }
        
        return coherence
    
    def perona_malik_diffusion(self, image, iterations, kappa):
        """Perona-Malik anisotropic diffusion from documents."""
        img = image.astype(np.float32)
        for _ in range(iterations):
            # Calculate gradients
            dx = np.gradient(img, axis=1)
            dy = np.gradient(img, axis=0)
            
            # Diffusion coefficients (c1 function from paper)
            c_x = np.exp(-(dx/kappa)**2)
            c_y = np.exp(-(dy/kappa)**2)
            
            # Update image
            img += 0.25 * (
                np.gradient(c_x * dx, axis=1) +
                np.gradient(c_y * dy, axis=0)
            )
        
        return img
    
    def coherence_enhancing_diffusion(self, image):
        """Coherence enhancing diffusion from documents."""
        # Calculate structure tensor
        Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Structure tensor components
        Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), 1.0)
        Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), 1.0)
        Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), 1.0)
        
        # Eigenvalues
        lambda1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
        lambda2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
        
        # Coherence measure
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
        
        # Apply coherence-based smoothing
        result = image.copy()
        mask = coherence > 0.5
        result[mask] = cv2.bilateralFilter(image, 5, 50, 50)[mask]
        
        return result
    
    def statistical_defect_detection(self, image, mask):
        """Implement all statistical methods from documents."""
        defects = np.zeros_like(mask, dtype=bool)
        pixels = image[mask > 0]
        
        if len(pixels) == 0:
            return defects
        
        # 1. Z-score method
        z_scores = np.abs(stats.zscore(pixels.flatten()))
        z_defects = z_scores > self.config['zscore_threshold']
        
        # 2. Modified Z-score using MAD
        median = np.median(pixels)
        mad = np.median(np.abs(pixels - median))
        modified_z = 0.6745 * (pixels - median) / (mad + 1e-10)
        mad_defects = np.abs(modified_z) > self.config['mad_threshold']
        
        # 3. IQR method
        q1, q3 = np.percentile(pixels, [25, 75])
        iqr = q3 - q1
        iqr_defects = (pixels < q1 - self.config['iqr_multiplier']*iqr) | \
                      (pixels > q3 + self.config['iqr_multiplier']*iqr)
        
        # 4. Grubbs test
        grubbs_defects = self.grubbs_test(pixels, self.config['grubbs_alpha'])
        
        # 5. Local Outlier Factor (LOF)
        lof_defects = self.local_outlier_factor(image, mask)
        
        # Combine all statistical detections
        statistical_mask = np.zeros_like(image, dtype=bool)
        valid_pixels = mask > 0
        
        # Map back to image space
        pixel_indices = np.where(valid_pixels)
        for i, (is_z, is_mad, is_iqr) in enumerate(zip(z_defects, mad_defects, iqr_defects)):
            if is_z or is_mad or is_iqr:
                y_idx = pixel_indices[0][i]
                x_idx = pixel_indices[1][i]
                if y_idx < statistical_mask.shape[0] and x_idx < statistical_mask.shape[1]:
                    statistical_mask[y_idx, x_idx] = True
        
        return statistical_mask | lof_defects | grubbs_defects
    
    def grubbs_test(self, data, alpha):
        """Grubbs test for outliers."""
        defects = np.zeros(len(data), dtype=bool)
        n = len(data)
        if n < 3:
            return defects
        
        mean = np.mean(data)
        std = np.std(data)
        
        # Calculate Grubbs statistic
        g = np.abs(data - mean) / (std + 1e-10)
        
        # Critical value approximation
        t_squared = stats.t.ppf(1 - alpha/(2*n), n-2)**2
        critical = ((n-1)/np.sqrt(n)) * np.sqrt(t_squared/(n-2+t_squared))
        
        defects = g > critical
        return defects
    
    def local_outlier_factor(self, image, mask, k=20):
        """Local Outlier Factor implementation."""
        # Get valid pixels
        valid_coords = np.column_stack(np.where(mask > 0))
        if len(valid_coords) < k + 1:
            return np.zeros_like(mask, dtype=bool)
        
        # Calculate distances
        distances = distance.cdist(valid_coords, valid_coords)
        
        # k-nearest neighbors
        k_distances = np.partition(distances, k, axis=1)[:, k]
        
        # Local reachability density
        lrd = np.zeros(len(valid_coords))
        for i in range(len(valid_coords)):
            neighbors = np.argsort(distances[i])[1:k+1]
            reach_dist = np.maximum(distances[i][neighbors], k_distances[neighbors])
            lrd[i] = k / (np.sum(reach_dist) + 1e-10)
        
        # LOF scores
        lof = np.zeros(len(valid_coords))
        for i in range(len(valid_coords)):
            neighbors = np.argsort(distances[i])[1:k+1]
            lof[i] = np.mean(lrd[neighbors]) / (lrd[i] + 1e-10)
        
        # Create defect mask
        defect_mask = np.zeros_like(mask, dtype=bool)
        outliers = lof > 2.0  # LOF > 2 indicates outlier
        
        for i, is_outlier in enumerate(outliers):
            if is_outlier:
                y, x = valid_coords[i]
                defect_mask[y, x] = True
        
        return defect_mask
    
    def spatial_morphological_analysis(self, image, mask):
        """Implement all morphological operations from documents."""
        # 1. Top-hat transform for bright defects
        kernel_tophat = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.config['tophat_kernel_size'], self.config['tophat_kernel_size'])
        )
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel_tophat)
        
        # 2. Bottom-hat for dark defects  
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel_tophat)
        
        # 3. Morphological gradient
        kernel_grad = np.ones((3, 3), np.uint8)
        morph_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel_grad)
        
        # 4. Local binary patterns
        lbp = self.local_binary_pattern(image)
        
        # 5. Local variance
        local_var = self.local_variance(image, window_size=5)
        
        # Threshold operations
        _, tophat_binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, blackhat_binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, gradient_binary = cv2.threshold(morph_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine morphological detections
        morph_defects = (tophat_binary > 0) | (blackhat_binary > 0) | (gradient_binary > 0)
        
        # Apply mask
        morph_defects = morph_defects & mask
        
        # Clean up small noise
        morph_defects = morphology.remove_small_objects(morph_defects, min_size=3)
        
        return morph_defects
    
    def local_binary_pattern(self, image, radius=1, n_points=8):
        """Local Binary Pattern implementation."""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ""
                
                for n in range(n_points):
                    theta = 2 * np.pi * n / n_points
                    x = int(round(i + radius * np.cos(theta)))
                    y = int(round(j - radius * np.sin(theta)))
                    
                    if image[x, y] >= center:
                        binary_string += "1"
                    else:
                        binary_string += "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def local_variance(self, image, window_size=5):
        """Calculate local variance."""
        # Use uniform filter for efficient computation
        mean = ndimage.uniform_filter(image.astype(float), size=window_size)
        sqr_mean = ndimage.uniform_filter(image.astype(float)**2, size=window_size)
        variance = sqr_mean - mean**2
        
        return variance
    
    def frequency_domain_analysis(self, image, mask):
        """Implement all frequency domain methods from documents."""
        # 1. FFT analysis
        f_transform = fft2(image)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 2. High-pass filtering to detect defects
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create high-pass filter
        mask_hp = np.ones((rows, cols), np.uint8)
        r = 30  # Radius for high-pass
        center = (crow, ccol)
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2
        mask_hp[mask_area] = 0
        
        # Apply high-pass filter
        f_shift_hp = f_shift * mask_hp
        f_ishift = np.fft.ifftshift(f_shift_hp)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        # 3. Wavelet decomposition
        wavelet_defects = self.wavelet_defect_detection(image)
        
        # 4. Gabor filter bank
        gabor_defects = self.gabor_filter_bank(image)
        
        # 5. Curvelet transform (simplified using ridgelet)
        curvelet_defects = self.simplified_curvelet(image)
        
        # Threshold high-pass result
        hp_normalized = cv2.normalize(np.abs(img_back), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, hp_binary = cv2.threshold(hp_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine frequency domain detections
        freq_defects = (hp_binary > 0) | wavelet_defects | gabor_defects | curvelet_defects
        
        # Apply mask
        freq_defects = freq_defects & mask
        
        return freq_defects
    
    def wavelet_defect_detection(self, image):
        """Wavelet-based defect detection."""
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(image, self.config['wavelet_family'], 
                               level=self.config['wavelet_level'])
        
        # Analyze detail coefficients
        defect_map = np.zeros_like(image, dtype=bool)
        
        for level in range(1, len(coeffs)):
            (cH, cV, cD) = coeffs[level]
            
            # Reconstruct from detail coefficients
            coeffs_thresh = [coeffs[0]] + [(np.zeros_like(c) if i != level else c) 
                                          for i, c in enumerate(coeffs[1:])]
            
            # Handle the tuple structure properly
            for i in range(1, len(coeffs_thresh)):
                if i != level:
                    coeffs_thresh[i] = (np.zeros_like(coeffs[i][0]), 
                                       np.zeros_like(coeffs[i][1]), 
                                       np.zeros_like(coeffs[i][2]))
            
            detail_recon = pywt.waverec2(coeffs_thresh, self.config['wavelet_family'])
            
            # Resize if necessary
            if detail_recon.shape != image.shape:
                detail_recon = cv2.resize(detail_recon, (image.shape[1], image.shape[0]))
            
            # Threshold
            _, binary = cv2.threshold(np.abs(detail_recon).astype(np.uint8), 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            defect_map |= binary > 0
        
        return defect_map
    
    def gabor_filter_bank(self, image):
        """Apply Gabor filter bank for defect detection."""
        defect_map = np.zeros_like(image, dtype=bool)
        
        for frequency in self.config['gabor_frequencies']:
            for theta in self.config['gabor_orientations']:
                theta_rad = theta * np.pi / 180
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (21, 21), 4.0, theta_rad, 10.0, frequency, 0, ktype=cv2.CV_32F
                )
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                
                # Threshold
                _, binary = cv2.threshold(
                    np.abs(filtered).astype(np.uint8), 0, 255, 
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                
                defect_map |= binary > 0
        
        return defect_map
    
    def simplified_curvelet(self, image):
        """Simplified curvelet using directional filters."""
        defect_map = np.zeros_like(image, dtype=bool)
        
        # Apply directional filters at multiple scales
        for scale in [3, 5, 7]:
            for angle in range(0, 180, 15):
                # Create oriented kernel
                kernel = self.create_oriented_kernel(scale, angle)
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                
                # Detect edges
                edges = cv2.Canny(np.abs(filtered).astype(np.uint8), 50, 150)
                
                defect_map |= edges > 0
        
        return defect_map
    
    def create_oriented_kernel(self, size, angle):
        """Create an oriented kernel for directional filtering."""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Create line kernel
        for i in range(size):
            x = int(center + (i - center) * np.cos(np.radians(angle)))
            y = int(center + (i - center) * np.sin(np.radians(angle)))
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize
        if np.sum(kernel) > 0:
            kernel /= np.sum(kernel)
        
        return kernel
    
    def gradient_edge_detection(self, image, mask):
        """Implement all gradient and edge detection methods."""
        # 1. Sobel operators
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_direction = np.arctan2(sobel_y, sobel_x)
        
        # 2. Prewitt operators  
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
        
        # 3. Canny edge detection
        canny_edges = cv2.Canny(image, 50, 150)
        
        # 4. Laplacian of Gaussian (LoG)
        gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
        log = cv2.Laplacian(gaussian, cv2.CV_64F)
        
        # 5. Structure tensor analysis
        structure_defects = self.structure_tensor_analysis(image)
        
        # 6. Phase congruency (simplified)
        phase_defects = self.phase_congruency_simplified(image)
        
        # Threshold gradient magnitudes
        sobel_binary = sobel_magnitude > np.percentile(sobel_magnitude, 95)
        prewitt_binary = prewitt_magnitude > np.percentile(prewitt_magnitude, 95)
        log_binary = np.abs(log) > np.percentile(np.abs(log), 95)
        
        # Combine edge detections
        edge_defects = sobel_binary | prewitt_binary | (canny_edges > 0) | \
                      log_binary | structure_defects | phase_defects
        
        # Apply mask
        edge_defects = edge_defects & mask
        
        return edge_defects
    
    def structure_tensor_analysis(self, image):
        """Structure tensor analysis for defect detection."""
        # Calculate gradients
        Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Structure tensor components
        Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), 1.5)
        Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), 1.5)
        Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), 1.5)
        
        # Eigenvalues
        lambda1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
        lambda2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
        
        # Coherence (for lines)
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
        
        # Corner measure (both eigenvalues large)
        corner_measure = lambda1 * lambda2 - 0.04 * (lambda1 + lambda2)**2
        
        # Detect features
        line_defects = coherence > 0.7  # High coherence indicates lines
        corner_defects = corner_measure > np.percentile(corner_measure, 95)
        
        return line_defects | corner_defects
    
    def phase_congruency_simplified(self, image):
        """Simplified phase congruency for edge detection."""
        # Use multiple scales of LoG
        defects = np.zeros_like(image, dtype=bool)
        
        for sigma in [1, 2, 3]:
            # LoG at different scales
            gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
            log = cv2.Laplacian(gaussian, cv2.CV_64F)
            
            # Zero crossings indicate edges
            zero_cross = self.zero_crossings(log)
            defects |= zero_cross
        
        return defects
    
    def zero_crossings(self, image):
        """Detect zero crossings in image."""
        # Pad image
        padded = np.pad(image, 1, mode='edge')
        
        # Check for sign changes
        zero_cross = np.zeros_like(image, dtype=bool)
        
        for i in range(1, padded.shape[0]-1):
            for j in range(1, padded.shape[1]-1):
                # Check 4-connected neighbors
                center = padded[i, j]
                if ((center * padded[i-1, j] < 0) or 
                    (center * padded[i+1, j] < 0) or
                    (center * padded[i, j-1] < 0) or
                    (center * padded[i, j+1] < 0)):
                    zero_cross[i-1, j-1] = True
        
        return zero_cross
    
    def advanced_blob_detection(self, image, mask):
        """Advanced blob detection using multiple methods."""
        # 1. Laplacian of Gaussian (LoG) blob detection
        blobs_log = []
        for sigma in np.linspace(2, 10, 5):
            # LoG filter
            log = ndimage.gaussian_laplace(image, sigma=sigma)
            log = log * sigma**2  # Scale normalization
            
            # Find local maxima
            local_max = (log == ndimage.maximum_filter(log, size=5))
            
            # Threshold
            threshold = np.percentile(np.abs(log[mask > 0]), 95)
            blobs = local_max & (np.abs(log) > threshold)
            
            # Store blob locations with scale
            y, x = np.where(blobs & mask)
            for yi, xi in zip(y, x):
                blobs_log.append((yi, xi, sigma))
        
        # 2. Determinant of Hessian (DoH)
        blobs_doh = self.determinant_of_hessian_blobs(image, mask)
        
        # 3. MSER (Maximally Stable Extremal Regions)
        mser_blobs = self.mser_detection(image, mask)
        
        # 4. Connected components on thresholded image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cc_blobs = self.connected_component_analysis(binary & mask)
        
        # Create blob mask
        blob_mask = np.zeros_like(image, dtype=bool)
        
        # Add LoG blobs
        for y, x, sigma in blobs_log:
            cv2.circle(blob_mask, (int(x), int(y)), int(sigma * np.sqrt(2)), True, -1)
        
        # Combine all blob detections
        blob_mask = blob_mask | blobs_doh | mser_blobs | cc_blobs
        
        return blob_mask & mask
    
    def determinant_of_hessian_blobs(self, image, mask):
        """Determinant of Hessian blob detection."""
        blob_mask = np.zeros_like(image, dtype=bool)
        
        for sigma in np.linspace(2, 10, 5):
            # Gaussian smoothing
            smooth = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Hessian components
            Ixx = cv2.Sobel(smooth, cv2.CV_64F, 2, 0)
            Iyy = cv2.Sobel(smooth, cv2.CV_64F, 0, 2)
            Ixy = cv2.Sobel(smooth, cv2.CV_64F, 1, 1)
            
            # Determinant of Hessian
            det_hessian = (Ixx * Iyy - Ixy**2) * sigma**4
            
            # Find local maxima
            local_max = (det_hessian == ndimage.maximum_filter(det_hessian, size=5))
            
            # Threshold
            threshold = np.percentile(det_hessian[mask > 0], 95)
            blobs = local_max & (det_hessian > threshold)
            
            blob_mask |= blobs
        
        return blob_mask & mask
    
    def mser_detection(self, image, mask):
        """MSER (Maximally Stable Extremal Regions) detection."""
        # Create MSER detector
        mser = cv2.MSER_create(
            _delta=5,
            _min_area=5,
            _max_area=500,
            _max_variation=0.25
        )
        
        # Detect regions
        regions, _ = mser.detectRegions(image)
        
        # Create mask from regions
        mser_mask = np.zeros_like(image, dtype=bool)
        
        for region in regions:
            # Create contour from region points
            contour = np.array(region)
            cv2.fillPoly(mser_mask, [contour], True)
        
        return mser_mask & mask
    
    def connected_component_analysis(self, binary_image):
        """Analyze connected components for blob detection."""
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_image.astype(np.uint8))
        
        # Analyze each component
        blob_mask = np.zeros_like(binary_image, dtype=bool)
        
        for label in range(1, num_labels):
            component = labels == label
            area = np.sum(component)
            
            # Filter by area
            if self.config['blob_min_area'] < area < 1000:
                # Additional shape analysis
                contours, _ = cv2.findContours(
                    component.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    contour = contours[0]
                    
                    # Circularity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                    
                    # Add if circular enough (blob-like)
                    if circularity > 0.5:
                        blob_mask |= component
        
        return blob_mask
    
    def advanced_scratch_detection(self, image, mask):
        """Advanced scratch detection using multiple methods."""
        # 1. Hough line detection
        hough_lines = self.hough_line_detection(image, mask)
        
        # 2. Radon transform
        radon_lines = self.radon_transform_lines(image, mask)
        
        # 3. Ridge detection (Frangi filter)
        ridge_lines = self.frangi_ridge_detection(image)
        
        # 4. Morphological line detection
        morph_lines = self.morphological_line_detection(image)
        
        # 5. Template matching for lines
        template_lines = self.line_template_matching(image)
        
        # Combine all line detections
        scratch_mask = hough_lines | radon_lines | ridge_lines | morph_lines | template_lines
        
        return scratch_mask & mask
    
    def hough_line_detection(self, image, mask):
        """Hough transform for line detection."""
        # Edge detection first
        edges = cv2.Canny(image, 50, 150)
        edges = edges & mask
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.config['hough_threshold'],
            minLineLength=self.config['scratch_min_length'],
            maxLineGap=5
        )
        
        # Create line mask
        line_mask = np.zeros_like(image, dtype=bool)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), True, 2)
        
        return line_mask
    
    def radon_transform_lines(self, image, mask):
        """Radon transform for line detection."""
        # Apply Radon transform
        theta = self.config['radon_angles']
        sinogram = transform.radon(image * mask, theta=theta)
        
        # Find peaks in sinogram
        peaks = feature.peak_local_max(
            sinogram, 
            min_distance=10,
            threshold_abs=np.percentile(sinogram, 95)
        )
        
        # Back-project strong lines
        line_mask = np.zeros_like(image, dtype=bool)
        
        for peak in peaks:
            angle_idx, rho_idx = peak
            angle = theta[angle_idx]
            
            # Create line at this angle and distance
            cos_angle = np.cos(np.radians(angle))
            sin_angle = np.sin(np.radians(angle))
            
            for x in range(image.shape[1]):
                y = int((rho_idx - x * cos_angle) / (sin_angle + 1e-10))
                if 0 <= y < image.shape[0]:
                    line_mask[y, x] = True
        
        return morphology.dilation(line_mask, morphology.disk(1)) & mask
    
    def frangi_ridge_detection(self, image):
        """Frangi vesselness filter for ridge/line detection."""
        # Multi-scale ridge detection
        ridge_map = np.zeros_like(image, dtype=float)
        
        for sigma in [1, 2, 3]:
            # Hessian at this scale
            gaussian = cv2.GaussianBlur(image.astype(float), (0, 0), sigma)
            
            Ixx = cv2.Sobel(gaussian, cv2.CV_64F, 2, 0)
            Iyy = cv2.Sobel(gaussian, cv2.CV_64F, 0, 2)
            Ixy = cv2.Sobel(gaussian, cv2.CV_64F, 1, 1)
            
            # Eigenvalues of Hessian
            lambda1 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
            lambda2 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
            
            # Frangi vesselness measure
            Rb = lambda1 / (lambda2 + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            beta = 0.5
            c = 0.5 * np.max(S)
            
            vesselness = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*c**2)))
            vesselness[lambda2 > 0] = 0  # Only consider dark lines
            
            ridge_map = np.maximum(ridge_map, vesselness)
        
        # Threshold
        threshold = np.percentile(ridge_map, 95)
        return ridge_map > threshold
    
    def morphological_line_detection(self, image):
        """Detect lines using morphological operations."""
        line_mask = np.zeros_like(image, dtype=bool)
        
        # Try different orientations
        for angle in range(0, 180, 15):
            # Create line structuring element
            length = 15
            se = self.create_line_strel(length, angle)
            
            # Top-hat with line SE
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)
            
            # Threshold
            _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            line_mask |= binary > 0
        
        return line_mask
    
    def create_line_strel(self, length, angle):
        """Create line structuring element."""
        # Create line SE
        angle_rad = np.radians(angle)
        
        # Calculate line endpoints
        dx = length * np.cos(angle_rad) / 2
        dy = length * np.sin(angle_rad) / 2
        
        # Create SE
        se_size = length + 2
        se = np.zeros((se_size, se_size), dtype=np.uint8)
        
        # Draw line
        center = se_size // 2
        cv2.line(
            se, 
            (int(center - dx), int(center - dy)),
            (int(center + dx), int(center + dy)),
            1, 1
        )
        
        return se
    
    def line_template_matching(self, image):
        """Template matching for line detection."""
        line_mask = np.zeros_like(image, dtype=bool)
        
        # Create line templates
        for length in [10, 15, 20]:
            for angle in range(0, 180, 30):
                # Create template
                template = self.create_line_template(length, angle)
                
                # Match template
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                
                # Find peaks
                peaks = feature.peak_local_max(
                    result,
                    min_distance=5,
                    threshold_abs=0.6
                )
                
                # Add detected lines
                for y, x in peaks:
                    # Draw line at detected position
                    angle_rad = np.radians(angle)
                    dx = int(length * np.cos(angle_rad) / 2)
                    dy = int(length * np.sin(angle_rad) / 2)
                    
                    cv2.line(
                        line_mask,
                        (x - dx, y - dy),
                        (x + dx, y + dy),
                        True, 2
                    )
        
        return line_mask
    
    def create_line_template(self, length, angle):
        """Create line template for matching."""
        size = length + 4
        template = np.zeros((size, size), dtype=np.uint8)
        
        center = size // 2
        angle_rad = np.radians(angle)
        
        dx = int(length * np.cos(angle_rad) / 2)
        dy = int(length * np.sin(angle_rad) / 2)
        
        cv2.line(
            template,
            (center - dx, center - dy),
            (center + dx, center + dy),
            255, 1
        )
        
        # Add some width
        template = cv2.dilate(template, np.ones((3, 3), np.uint8), iterations=1)
        
        return template
    
    def do2mr_detection(self, image, mask):
        """DO2MR (Difference of Min-Max Ranking) from paper."""
        window = self.config['do2mr_window_size']
        gamma = self.config['do2mr_gamma']
        
        # Pad image
        pad = window // 2
        padded = np.pad(image, pad, mode='reflect')
        
        # Min-max filtering
        min_filtered = np.zeros_like(image)
        max_filtered = np.zeros_like(image)
        
        for i in range(pad, padded.shape[0] - pad):
            for j in range(pad, padded.shape[1] - pad):
                window_region = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                min_filtered[i-pad, j-pad] = np.min(window_region)
                max_filtered[i-pad, j-pad] = np.max(window_region)
        
        # Residual
        residual = max_filtered - min_filtered
        
        # Smooth residual
        residual_smooth = cv2.medianBlur(residual.astype(np.uint8), 3)
        
        # Statistical thresholding
        mean = np.mean(residual_smooth[mask > 0])
        std = np.std(residual_smooth[mask > 0])
        
        threshold = mean + gamma * std
        defects = (residual_smooth > threshold) & mask
        
        # Morphological cleanup
        defects = morphology.opening(defects, morphology.disk(1))
        
        return defects
    
    def lei_scratch_detection(self, image, mask):
        """LEI (Linear Enhancement Inspector) from paper."""
        orientations = self.config['lei_orientations']
        line_length = self.config['lei_line_length']
        line_gap = self.config['lei_line_gap']
        
        # Histogram equalization first (as per paper)
        equalized = cv2.equalizeHist(image)
        
        scratch_strength = np.zeros_like(image, dtype=float)
        
        # For each orientation
        for i in range(orientations):
            angle = i * 180 / orientations
            angle_rad = np.radians(angle)
            
            # Create linear detector
            for y in range(line_gap, image.shape[0] - line_gap):
                for x in range(line_gap, image.shape[1] - line_gap):
                    if not mask[y, x]:
                        continue
                    
                    # Red branch (along line)
                    red_sum = 0
                    red_count = 0
                    
                    # Gray branches (parallel to line)
                    gray_sum = 0
                    gray_count = 0
                    
                    # Sample along line
                    for t in range(-line_length//2, line_length//2 + 1):
                        # Red branch point
                        rx = int(x + t * np.cos(angle_rad))
                        ry = int(y + t * np.sin(angle_rad))
                        
                        if 0 <= rx < image.shape[1] and 0 <= ry < image.shape[0]:
                            red_sum += equalized[ry, rx]
                            red_count += 1
                        
                        # Gray branch points (offset perpendicular to line)
                        for offset in [-line_gap, line_gap]:
                            gx = int(x + t * np.cos(angle_rad) + offset * np.sin(angle_rad))
                            gy = int(y + t * np.sin(angle_rad) - offset * np.cos(angle_rad))
                            
                            if 0 <= gx < image.shape[1] and 0 <= gy < image.shape[0]:
                                gray_sum += equalized[gy, gx]
                                gray_count += 1
                    
                    # Calculate strength
                    if red_count > 0 and gray_count > 0:
                        red_avg = red_sum / red_count
                        gray_avg = gray_sum / gray_count
                        strength = 2 * red_avg - gray_avg
                        scratch_strength[y, x] = max(scratch_strength[y, x], strength)
        
        # Threshold using sigma-based method
        mean = np.mean(scratch_strength[mask > 0])
        std = np.std(scratch_strength[mask > 0])
        threshold = mean + 2 * std
        
        scratches = (scratch_strength > threshold) & mask
        
        # Connect broken scratches
        scratches = morphology.closing(scratches, morphology.disk(2))
        
        return scratches
    
    def machine_learning_detection(self, image, mask):
        """Machine learning based anomaly detection."""
        # Extract features for each pixel
        features = self.extract_pixel_features(image)
        
        # Get valid pixel features
        valid_features = features[mask > 0]
        
        if len(valid_features) < 10:
            return np.zeros_like(mask, dtype=bool)
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.config['isolation_contamination'],
            random_state=42
        )
        iso_labels = iso_forest.fit_predict(valid_features)
        
        # 2. One-class SVM
        ocsvm = OneClassSVM(gamma='scale', nu=0.1)
        svm_labels = ocsvm.fit_predict(valid_features)
        
        # 3. DBSCAN clustering
        dbscan = DBSCAN(
            eps=self.config['dbscan_eps'],
            min_samples=self.config['dbscan_min_samples']
        )
        cluster_labels = dbscan.fit_predict(valid_features)
        
        # Create defect mask
        ml_defects = np.zeros_like(mask, dtype=bool)
        
        # Map predictions back to image
        valid_indices = np.where(mask > 0)
        
        for i, (iso, svm, cluster) in enumerate(zip(iso_labels, svm_labels, cluster_labels)):
            # Anomaly if detected by any method
            if iso == -1 or svm == -1 or cluster == -1:
                y, x = valid_indices[0][i], valid_indices[1][i]
                ml_defects[y, x] = True
        
        return ml_defects
    
    def extract_pixel_features(self, image):
        """Extract features for each pixel."""
        features = []
        
        # Pad image
        padded = np.pad(image, 5, mode='reflect')
        
        for i in range(5, padded.shape[0] - 5):
            for j in range(5, padded.shape[1] - 5):
                # Local window
                window = padded[i-5:i+6, j-5:j+6]
                
                # Features
                pixel_features = [
                    padded[i, j],  # Intensity
                    np.mean(window),  # Local mean
                    np.std(window),   # Local std
                    np.max(window) - np.min(window),  # Local range
                    stats.skew(window.flatten()),  # Skewness
                    stats.kurtosis(window.flatten()),  # Kurtosis
                    np.percentile(window, 25),  # Q1
                    np.percentile(window, 75),  # Q3
                ]
                
                # Add gradient features
                dx = padded[i, j+1] - padded[i, j-1]
                dy = padded[i+1, j] - padded[i-1, j]
                pixel_features.extend([dx, dy, np.sqrt(dx**2 + dy**2)])
                
                features.append(pixel_features)
        
        return np.array(features).reshape(image.shape[0], image.shape[1], -1)
    
    def combine_detections(self, all_detections):
        """Combine all detection methods intelligently."""
        # Stack all detection masks
        detection_stack = np.stack([
            mask for mask in all_detections.values() 
            if isinstance(mask, np.ndarray)
        ], axis=0)
        
        # Voting - pixel is defect if detected by multiple methods
        vote_counts = np.sum(detection_stack, axis=0)
        
        # Adaptive threshold based on number of methods
        num_methods = len(detection_stack)
        threshold = max(2, num_methods // 3)  # At least 2 methods or 1/3 of methods
        
        combined = vote_counts >= threshold
        
        # Post-processing
        # Remove small objects
        combined = morphology.remove_small_objects(combined, min_size=5)
        
        # Fill small holes
        combined = morphology.remove_small_holes(combined, area_threshold=10)
        
        # Separate touching objects
        distance = ndimage.distance_transform_edt(combined)
        local_max = feature.peak_local_max(
            distance, 
            min_distance=3,
            indices=False
        )
        markers = ndimage.label(local_max)[0]
        combined = morphology.watershed(-distance, markers, mask=combined)
        
        return combined > 0
    
    def extract_defect_features(self, defect_mask, image):
        """Extract comprehensive features for each defect."""
        # Label individual defects
        labeled, num_defects = ndimage.label(defect_mask)
        
        features = []
        
        for i in range(1, num_defects + 1):
            defect = labeled == i
            
            # Basic measurements
            props = measure.regionprops(defect.astype(int), intensity_image=image)[0]
            
            # Geometric features
            geometric = {
                'area': props.area,
                'perimeter': props.perimeter,
                'circularity': 4 * np.pi * props.area / (props.perimeter**2 + 1e-10),
                'eccentricity': props.eccentricity,
                'solidity': props.solidity,
                'extent': props.extent,
                'major_axis': props.major_axis_length,
                'minor_axis': props.minor_axis_length,
                'orientation': props.orientation,
                'centroid': props.centroid,
                'bbox': props.bbox
            }
            
            # Intensity features
            intensity = {
                'mean_intensity': props.mean_intensity,
                'min_intensity': props.min_intensity,
                'max_intensity': props.max_intensity,
                'intensity_std': np.std(image[defect]),
                'intensity_skew': stats.skew(image[defect]),
                'intensity_kurtosis': stats.kurtosis(image[defect])
            }
            
            # Shape features
            if props.area > 10:
                # Hu moments
                moments = cv2.moments(defect.astype(np.uint8))
                hu_moments = cv2.HuMoments(moments).flatten()
            else:
                hu_moments = np.zeros(7)
            
            shape = {
                'hu_moments': hu_moments,
                'convex_area': props.convex_area,
                'filled_area': props.filled_area
            }
            
            # Classification
            defect_type = self.classify_defect(geometric, intensity)
            
            features.append({
                'id': i,
                'geometric': geometric,
                'intensity': intensity,
                'shape': shape,
                'type': defect_type
            })
        
        return features
    
    def classify_defect(self, geometric, intensity):
        """Classify defect type based on features."""
        # Simple rule-based classification
        if geometric['eccentricity'] > 0.9 and geometric['major_axis'] > geometric['minor_axis'] * 3:
            return 'scratch'
        elif geometric['circularity'] > 0.7:
            return 'dig'
        elif geometric['area'] < 10:
            return 'particle'
        elif intensity['mean_intensity'] > 200:
            return 'bright_spot'
        elif intensity['mean_intensity'] < 50:
            return 'dark_spot'
        else:
            return 'unknown'
    
    def calculate_quality_metrics(self, image, defect_mask, roi_mask):
        """Calculate comprehensive quality metrics."""
        # Basic statistics
        roi_pixels = image[roi_mask > 0]
        defect_pixels = image[defect_mask > 0] if np.any(defect_mask) else np.array([0])
        
        # Defect density
        defect_density = np.sum(defect_mask) / np.sum(roi_mask) if np.sum(roi_mask) > 0 else 0
        
        # Contrast metrics
        if len(defect_pixels) > 0 and len(roi_pixels) > 0:
            contrast = np.abs(np.mean(defect_pixels) - np.mean(roi_pixels)) / \
                      (np.mean(roi_pixels) + 1e-10)
        else:
            contrast = 0
        
        # Surface roughness (RMS)
        if len(roi_pixels) > 0:
            roughness = np.std(roi_pixels)
        else:
            roughness = 0
        
        # Signal-to-noise ratio
        if len(roi_pixels) > 0:
            signal = np.mean(roi_pixels)
            noise = np.std(roi_pixels)
            snr = signal / (noise + 1e-10)
        else:
            snr = 0
        
        # Uniformity
        if len(roi_pixels) > 0:
            uniformity = 1 - (np.std(roi_pixels) / (np.mean(roi_pixels) + 1e-10))
        else:
            uniformity = 0
        
        # Structural similarity to ideal (smooth) surface
        smooth = cv2.GaussianBlur(image, (15, 15), 5.0)
        
        # SSIM-like metric
        c1 = 0.01**2
        c2 = 0.03**2
        
        mu1 = np.mean(image[roi_mask > 0])
        mu2 = np.mean(smooth[roi_mask > 0])
        
        sigma1 = np.std(image[roi_mask > 0])
        sigma2 = np.std(smooth[roi_mask > 0])
        
        sigma12 = np.mean((image[roi_mask > 0] - mu1) * (smooth[roi_mask > 0] - mu2))
        
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        
        return {
            'defect_density': defect_density,
            'defect_count': len(measure.regionprops(label(defect_mask)[0])),
            'total_defect_area': np.sum(defect_mask),
            'contrast': contrast,
            'roughness': roughness,
            'snr': snr,
            'uniformity': uniformity,
            'structural_similarity': ssim,
            'mean_intensity': np.mean(roi_pixels) if len(roi_pixels) > 0 else 0,
            'std_intensity': np.std(roi_pixels) if len(roi_pixels) > 0 else 0
        }
    
    def visualize_results(self, save_path=None):
        """Visualize all detection results."""
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()
        
        # Original and preprocessed
        axes[0].imshow(self.original_roi, cmap='gray')
        axes[0].set_title('Original ROI')
        
        axes[1].imshow(self.preprocessing_stages['coherence'], cmap='gray')
        axes[1].set_title('Preprocessed (Coherence Enhanced)')
        
        # Individual detection results
        detections = self.defect_results['individual_detections']
        titles = ['Statistical', 'Spatial', 'Frequency', 'Edge', 
                 'Blob', 'Scratch', 'DO2MR', 'LEI', 'ML']
        
        for i, (key, title) in enumerate(zip(detections.keys(), titles)):
            axes[i+2].imshow(detections[key], cmap='hot')
            axes[i+2].set_title(f'{title} Detection')
        
        # Combined result
        axes[11].imshow(self.defect_results['defects'], cmap='hot')
        axes[11].set_title('Combined Detection')
        
        # Overlay on original
        overlay = self.original_roi.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        # Color defects in red
        overlay[self.defect_results['defects']] = [255, 0, 0]
        
        axes[12].imshow(overlay)
        axes[12].set_title('Defects Overlay')
        
        # Feature visualization
        if self.defect_results['features']:
            # Plot defect size distribution
            areas = [f['geometric']['area'] for f in self.defect_results['features']]
            axes[13].hist(areas, bins=20)
            axes[13].set_title('Defect Size Distribution')
            axes[13].set_xlabel('Area (pixels)')
            
            # Plot defect types
            types = [f['type'] for f in self.defect_results['features']]
            type_counts = {t: types.count(t) for t in set(types)}
            axes[14].bar(type_counts.keys(), type_counts.values())
            axes[14].set_title('Defect Types')
            axes[14].tick_params(axis='x', rotation=45)
        
        # Quality metrics
        metrics_text = '\n'.join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in self.defect_results['quality_metrics'].items()
        ])
        axes[15].text(0.1, 0.5, metrics_text, transform=axes[15].transAxes,
                     fontsize=10, verticalalignment='center')
        axes[15].set_title('Quality Metrics')
        axes[15].axis('off')
        
        # Remove axis ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


# Example usage
if __name__ == "__main__":
    import os
    import sys
    
    # Check if we have the outputs from mask_separation.py
    required_files = [
        'cleaned_image.png',
        'inner_white_mask.png', 
        'black_mask.png',
        'outside_mask.png',
        'white_region_original.png',
        'black_region_original.png',
        'outside_region_original.png'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: The following files from mask_separation.py are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run mask_separation.py first to generate these files.")
        sys.exit(1)
    
    # Load the cleaned image
    image = cv2.imread('cleaned_image.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load cleaned_image.png")
        sys.exit(1)
    
    # Load the masks
    inner_mask = cv2.imread('inner_white_mask.png', cv2.IMREAD_GRAYSCALE)
    annulus_mask = cv2.imread('black_mask.png', cv2.IMREAD_GRAYSCALE)
    outside_mask = cv2.imread('outside_mask.png', cv2.IMREAD_GRAYSCALE)
    
    # Convert masks to boolean
    inner_mask = inner_mask > 0 if inner_mask is not None else None
    annulus_mask = annulus_mask > 0 if annulus_mask is not None else None
    outside_mask = outside_mask > 0 if outside_mask is not None else None
    
    # Initialize detector
    detector = ComprehensiveFiberDefectDetector()
    
    # Analyze each region
    regions = [
        ('inner', inner_mask, 'white_region_original.png'),
        ('annulus', annulus_mask, 'black_region_original.png'),
        ('outside', outside_mask, 'outside_region_original.png')
    ]
    
    all_results = {}
    
    for region_name, mask, region_image_path in regions:
        if mask is None:
            print(f"Skipping {region_name} region - mask not found")
            continue
            
        # Load the specific region image
        region_image = cv2.imread(region_image_path, cv2.IMREAD_GRAYSCALE)
        if region_image is None:
            print(f"Warning: Could not load {region_image_path}, using cleaned image instead")
            region_image = image
        
        print(f"\n{'='*50}")
        print(f"Analyzing {region_name.upper()} region...")
        print(f"{'='*50}")
        
        # Analyze defects
        results = detector.analyze_defects(region_image, mask, region_type=region_name)
        all_results[region_name] = results
        
        # Print summary
        print(f"\nRegion Type: {results['region_type']}")
        print(f"Total Defects Found: {results['quality_metrics']['defect_count']}")
        print(f"Defect Density: {results['quality_metrics']['defect_density']:.4f}")
        print(f"Total Defect Area: {results['quality_metrics']['total_defect_area']} pixels")
        print(f"Surface Roughness: {results['quality_metrics']['roughness']:.2f}")
        print(f"Signal-to-Noise Ratio: {results['quality_metrics']['snr']:.2f}")
        print(f"Uniformity: {results['quality_metrics']['uniformity']:.4f}")
        
        # Print defect types if any found
        if results['features']:
            print(f"\nDefect Types Found:")
            defect_types = {}
            for feature in results['features']:
                dtype = feature['type']
                defect_types[dtype] = defect_types.get(dtype, 0) + 1
            
            for dtype, count in defect_types.items():
                print(f"  - {dtype}: {count}")
        
        # Visualize results for this region
        save_path = f'defect_analysis_{region_name}.png'
        detector.visualize_results(save_path)
        print(f"\nVisualization saved to: {save_path}")
    
    # Summary across all regions
    print(f"\n{'='*50}")
    print("OVERALL SUMMARY")
    print(f"{'='*50}")
    
    total_defects = sum(r['quality_metrics']['defect_count'] for r in all_results.values())
    total_area = sum(r['quality_metrics']['total_defect_area'] for r in all_results.values())
    
    print(f"Total defects across all regions: {total_defects}")
    print(f"Total defect area: {total_area} pixels")
    
    # Create a combined report
    print("\nDetailed Report by Region:")
    for region, results in all_results.items():
        print(f"\n{region.upper()}:")
        print(f"  - Defects: {results['quality_metrics']['defect_count']}")
        print(f"  - Density: {results['quality_metrics']['defect_density']:.4f}")
        print(f"  - Mean Intensity: {results['quality_metrics']['mean_intensity']:.2f}")
        print(f"  - Std Intensity: {results['quality_metrics']['std_intensity']:.2f}")
    
    print("\nDefect detection analysis complete!")