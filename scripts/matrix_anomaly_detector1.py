#!/usr/bin/env python3
"""
Comprehensive Matrix Anomaly Detection System for Fiber Optic Defect Analysis
Implements exhaustive mathematical, statistical, and information-theoretic methods
for comparing image-derived matrices to detect scratches, digs, blobs, and other anomalies.
"""

import cv2
import numpy as np
import json
import os
import pickle
from pathlib import Path
from scipy import stats, signal, ndimage
from scipy.spatial.distance import cdist
from scipy.stats import entropy, wasserstein_distance
from skimage.metrics import structural_similarity
from skimage.morphology import disk, square, opening, closing, white_tophat, black_tophat
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMatrixAnalyzer:
    """
    Implements exhaustive matrix comparison methods for anomaly detection
    """
    
    def __init__(self, learning_mode=True):
        self.learning_mode = learning_mode
        self.knowledge_base = {
            'reference_statistics': {},
            'anomaly_patterns': {},
            'golden_standards': {},
            'learned_thresholds': {},
            'historical_analyses': []
        }
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load previously learned patterns and thresholds"""
        kb_path = 'anomaly_knowledge_base.pkl'
        if os.path.exists(kb_path):
            with open(kb_path, 'rb') as f:
                self.knowledge_base = pickle.load(f)
            print(f"Loaded knowledge base with {len(self.knowledge_base['historical_analyses'])} historical analyses")
    
    def save_knowledge_base(self):
        """Save learned patterns and thresholds"""
        with open('anomaly_knowledge_base.pkl', 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        print("Knowledge base saved")
    
    def load_json_matrix(self, json_path):
        """Load matrix data from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract image dimensions
        width = data['image_dimensions']['width']
        height = data['image_dimensions']['height']
        
        # Create BGR matrix
        matrix = np.zeros((height, width, 3), dtype=np.uint8)
        
        for pixel in data['pixels']:
            x = pixel['coordinates']['x']
            y = pixel['coordinates']['y']
            bgr = pixel['bgr_intensity']
            matrix[y, x] = bgr
            
        return matrix, data
    
    def compute_matrix_features(self, matrix):
        """Extract comprehensive features from a matrix"""
        features = {}
        
        # Convert to grayscale for some analyses
        gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        
        # 1. Mathematical Norm-Based Features
        features['frobenius_norm'] = np.linalg.norm(matrix, 'fro')
        features['l1_norm'] = np.sum(np.abs(matrix))
        features['linf_norm'] = np.max(np.abs(matrix))
        
        # 2. Statistical Moments
        features['mean'] = np.mean(matrix, axis=(0,1))
        features['std'] = np.std(matrix, axis=(0,1))
        features['skewness'] = stats.skew(gray.flatten())
        features['kurtosis'] = stats.kurtosis(gray.flatten())
        
        # 3. Entropy Measures
        hist, _ = np.histogram(gray, bins=256, range=(0,256))
        hist = hist / hist.sum()
        features['shannon_entropy'] = entropy(hist)
        features['normalized_entropy'] = features['shannon_entropy'] / np.log2(256)
        
        # 4. Gradient Features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = np.mean(grad_mag)
        features['gradient_std'] = np.std(grad_mag)
        
        # 5. Frequency Domain Features (FFT)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        features['fft_energy'] = np.sum(magnitude_spectrum**2)
        
        # 6. Texture Features (GLCM)
        glcm_features = self.compute_glcm_features(gray)
        features.update(glcm_features)
        
        # 7. Morphological Features
        morph_features = self.compute_morphological_features(gray)
        features.update(morph_features)
        
        # 8. Hu Moments (rotation invariant)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments)
        features['hu_moments'] = hu_moments.flatten()
        
        return features
    
    def compute_glcm_features(self, gray):
        """Compute Gray Level Co-occurrence Matrix features"""
        # Simplified GLCM computation
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Quantize to reduce computation
        gray_quantized = (gray // 16).astype(np.uint8)
        
        features = {}
        for d in distances:
            for a_idx, a in enumerate(angles):
                # Create co-occurrence matrix
                glcm = np.zeros((16, 16))
                dx = int(np.round(d * np.cos(a)))
                dy = int(np.round(d * np.sin(a)))
                
                for i in range(gray_quantized.shape[0] - abs(dy)):
                    for j in range(gray_quantized.shape[1] - abs(dx)):
                        i1, j1 = i, j
                        i2, j2 = i + dy, j + dx
                        if 0 <= i2 < gray_quantized.shape[0] and 0 <= j2 < gray_quantized.shape[1]:
                            glcm[gray_quantized[i1, j1], gray_quantized[i2, j2]] += 1
                
                # Normalize
                if glcm.sum() > 0:
                    glcm = glcm / glcm.sum()
                
                # Compute features
                i, j = np.mgrid[0:16, 0:16]
                features[f'glcm_contrast_d{d}_a{a_idx}'] = np.sum((i - j)**2 * glcm)
                features[f'glcm_homogeneity_d{d}_a{a_idx}'] = np.sum(glcm / (1 + (i - j)**2))
                features[f'glcm_energy_d{d}_a{a_idx}'] = np.sum(glcm**2)
                
        return features
    
    def compute_morphological_features(self, gray):
        """Compute morphological features for defect detection"""
        features = {}
        
        # Multi-scale top-hat transforms
        for size in [3, 5, 7]:
            se = disk(size)
            wth = white_tophat(gray, se)  # Bright defects
            bth = black_tophat(gray, se)  # Dark defects
            
            features[f'white_tophat_{size}_mean'] = np.mean(wth)
            features[f'white_tophat_{size}_max'] = np.max(wth)
            features[f'black_tophat_{size}_mean'] = np.mean(bth)
            features[f'black_tophat_{size}_max'] = np.max(bth)
        
        return features
    
    def compare_matrices_comprehensive(self, matrix1, matrix2):
        """Comprehensive comparison using multiple methods"""
        results = {}
        
        # Ensure same dimensions
        if matrix1.shape != matrix2.shape:
            matrix1, matrix2 = self.align_matrices(matrix1, matrix2)
        
        gray1 = cv2.cvtColor(matrix1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(matrix2, cv2.COLOR_BGR2GRAY)
        
        # 1. Direct Pixel Differences
        results['mse'] = np.mean((matrix1.astype(float) - matrix2.astype(float))**2)
        results['mae'] = np.mean(np.abs(matrix1.astype(float) - matrix2.astype(float)))
        results['max_diff'] = np.max(np.abs(matrix1.astype(float) - matrix2.astype(float)))
        results['psnr'] = 10 * np.log10(255**2 / (results['mse'] + 1e-10))
        
        # 2. Structural Similarity (SSIM)
        results['ssim'], ssim_map = structural_similarity(gray1, gray2, full=True)
        results['ssim_map'] = ssim_map
        
        # SSIM Components
        win_size = 7
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Compute component maps
        results['luminance_map'], results['contrast_map'], results['structure_map'] = \
            self.compute_ssim_components(gray1, gray2, win_size, C1, C2)
        
        # 3. Correlation Measures
        results['pearson_correlation'] = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
        
        # 4. Histogram Comparisons
        hist1, _ = np.histogram(gray1, bins=256, range=(0, 256))
        hist2, _ = np.histogram(gray2, bins=256, range=(0, 256))
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        results['hist_correlation'] = np.corrcoef(hist1, hist2)[0, 1]
        results['chi_square'] = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
        results['bhattacharyya'] = -np.log(np.sum(np.sqrt(hist1 * hist2)) + 1e-10)
        
        # 5. Information-Theoretic Measures
        results['mutual_information'] = self.compute_mutual_information(gray1, gray2)
        results['kl_divergence'] = entropy(hist1, hist2)
        results['js_divergence'] = self.compute_js_divergence(hist1, hist2)
        results['emd'] = wasserstein_distance(np.arange(256), np.arange(256), hist1, hist2)
        
        # 6. Frequency Domain Analysis
        results['fft_correlation'] = self.compute_fft_correlation(gray1, gray2)
        
        # 7. Gradient Comparison
        grad1 = np.gradient(gray1.astype(float))
        grad2 = np.gradient(gray2.astype(float))
        results['gradient_mse'] = np.mean((grad1[0] - grad2[0])**2 + (grad1[1] - grad2[1])**2)
        
        # 8. Local Binary Pattern Comparison
        lbp1 = local_binary_pattern(gray1, 8, 1, method='uniform')
        lbp2 = local_binary_pattern(gray2, 8, 1, method='uniform')
        results['lbp_similarity'] = np.corrcoef(lbp1.flatten(), lbp2.flatten())[0, 1]
        
        return results
    
    def compute_ssim_components(self, img1, img2, win_size, C1, C2):
        """Compute individual SSIM components"""
        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
        
        # Luminance comparison
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        
        # Contrast comparison
        contrast = (2 * np.sqrt(sigma1_sq * sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
        
        # Structure comparison
        structure = (sigma12 + C2/2) / (np.sqrt(sigma1_sq * sigma2_sq) + C2/2)
        
        return luminance, contrast, structure
    
    def compute_mutual_information(self, img1, img2):
        """Compute mutual information between two images"""
        hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        return mi
    
    def compute_js_divergence(self, p, q):
        """Compute Jensen-Shannon divergence"""
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    
    def compute_fft_correlation(self, img1, img2):
        """Compute correlation in frequency domain"""
        f1 = np.fft.fft2(img1)
        f2 = np.fft.fft2(img2)
        mag1 = np.abs(f1)
        mag2 = np.abs(f2)
        return np.corrcoef(mag1.flatten(), mag2.flatten())[0, 1]
    
    def align_matrices(self, mat1, mat2):
        """Align matrices of different sizes"""
        h1, w1 = mat1.shape[:2]
        h2, w2 = mat2.shape[:2]
        
        if h1 * w1 > h2 * w2:
            # Resize mat2 to match mat1
            mat2 = cv2.resize(mat2, (w1, h1), interpolation=cv2.INTER_CUBIC)
        else:
            # Resize mat1 to match mat2
            mat1 = cv2.resize(mat1, (w2, h2), interpolation=cv2.INTER_CUBIC)
            
        return mat1, mat2
    
    def detect_anomalies_local(self, matrix, reference_features, window_size=32, stride=16):
        """Detect local anomalies using sliding window"""
        anomaly_map = np.zeros(matrix.shape[:2])
        gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                window = gray[y:y+window_size, x:x+window_size]
                
                # Compute local statistics
                local_mean = np.mean(window)
                local_std = np.std(window)
                local_entropy = entropy(np.histogram(window, bins=32)[0])
                
                # Compare with reference statistics
                z_score_mean = abs(local_mean - reference_features['mean_local']) / (reference_features['std_local'] + 1e-6)
                z_score_std = abs(local_std - reference_features['std_std_local']) / (reference_features['std_std_local'] + 1e-6)
                
                # Compute anomaly score
                anomaly_score = max(z_score_mean, z_score_std)
                
                # Update anomaly map
                anomaly_map[y:y+window_size, x:x+window_size] = np.maximum(
                    anomaly_map[y:y+window_size, x:x+window_size],
                    anomaly_score
                )
        
        return anomaly_map
    
    def detect_specific_defects(self, matrix):
        """Detect specific types of defects (scratches, digs, blobs)"""
        gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        defects = {
            'scratches': [],
            'digs': [],
            'blobs': []
        }
        
        # 1. Scratch Detection (linear features)
        # Use Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            defects['scratches'] = lines
        
        # 2. Dig Detection (small circular features)
        # Use blob detection with appropriate parameters
        wth = white_tophat(gray, disk(5))
        bth = black_tophat(gray, disk(5))
        
        # Threshold to find potential digs
        _, dig_mask = cv2.threshold(bth, np.percentile(bth, 95), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(dig_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # Small area constraint for digs
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    defects['digs'].append((cx, cy, area))
        
        # 3. Blob Detection (larger area features)
        # Use morphological operations
        _, blob_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, disk(7))
        
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Larger area for blobs
                defects['blobs'].append(contour)
        
        return defects
    
    def create_golden_standard(self, matrices_data):
        """Create golden standard from multiple reference matrices using DISTATIS"""
        # Extract features from all matrices
        all_features = []
        for matrix, _ in matrices_data:
            features = self.compute_matrix_features(matrix)
            all_features.append(features)
        
        # Create consensus features
        golden_features = {}
        
        # For numerical features, compute robust statistics
        numerical_keys = [k for k in all_features[0].keys() if isinstance(all_features[0][k], (int, float, np.number))]
        
        for key in numerical_keys:
            values = [f[key] for f in all_features]
            golden_features[f'{key}_mean'] = np.mean(values)
            golden_features[f'{key}_std'] = np.std(values)
            golden_features[f'{key}_median'] = np.median(values)
            
            # Robust statistics using Minimum Covariance Determinant
            if len(values) > 10:
                values_array = np.array(values).reshape(-1, 1)
                mcd = MinCovDet().fit(values_array)
                golden_features[f'{key}_robust_mean'] = mcd.location_[0]
                golden_features[f'{key}_robust_std'] = np.sqrt(mcd.covariance_[0, 0])
        
        # For local statistics
        local_means = []
        local_stds = []
        
        for matrix, _ in matrices_data:
            gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
            # Compute local statistics in windows
            for y in range(0, gray.shape[0] - 32, 16):
                for x in range(0, gray.shape[1] - 32, 16):
                    window = gray[y:y+32, x:x+32]
                    local_means.append(np.mean(window))
                    local_stds.append(np.std(window))
        
        golden_features['mean_local'] = np.mean(local_means)
        golden_features['std_local'] = np.std(local_means)
        golden_features['std_std_local'] = np.std(local_stds)
        
        return golden_features
    
    def analyze_all_matrices(self, directory_path):
        """Analyze all JSON matrices in a directory"""
        json_files = list(Path(directory_path).glob('*.json'))
        print(f"Found {len(json_files)} JSON files to analyze")
        
        # Load all matrices
        matrices_data = []
        for json_file in json_files:
            try:
                matrix, data = self.load_json_matrix(json_file)
                matrices_data.append((matrix, data))
                print(f"Loaded: {json_file.name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        # Create golden standard
        print("\nCreating golden standard from reference matrices...")
        golden_standard = self.create_golden_standard(matrices_data)
        self.knowledge_base['golden_standards']['current'] = golden_standard
        
        # Compute pairwise comparisons
        print("\nPerforming comprehensive pairwise comparisons...")
        comparison_results = []
        
        for i in range(len(matrices_data)):
            for j in range(i + 1, len(matrices_data)):
                results = self.compare_matrices_comprehensive(
                    matrices_data[i][0], matrices_data[j][0]
                )
                results['pair'] = (i, j)
                comparison_results.append(results)
        
        # Analyze comparison results to establish thresholds
        self.analyze_comparison_results(comparison_results)
        
        # Store reference statistics
        self.knowledge_base['reference_statistics'] = {
            'num_samples': len(matrices_data),
            'golden_standard': golden_standard,
            'comparison_stats': self.compute_comparison_statistics(comparison_results)
        }
        
        # Save knowledge base
        if self.learning_mode:
            self.save_knowledge_base()
        
        return golden_standard, comparison_results
    
    def analyze_comparison_results(self, comparison_results):
        """Analyze comparison results to establish anomaly thresholds"""
        # Extract metrics
        metrics = {}
        for metric in ['mse', 'mae', 'ssim', 'mutual_information', 'kl_divergence']:
            values = [r[metric] for r in comparison_results if metric in r]
            if values:
                metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'percentiles': np.percentile(values, [25, 50, 75, 90, 95, 99])
                }
        
        # Set adaptive thresholds
        self.knowledge_base['learned_thresholds'] = {
            'mse_threshold': metrics['mse']['percentiles'][4],  # 95th percentile
            'ssim_threshold': metrics['ssim']['percentiles'][1],  # 25th percentile (lower is worse)
            'mi_threshold': metrics['mutual_information']['percentiles'][1],  # 25th percentile
        }
        
        print("\nLearned thresholds:")
        for key, value in self.knowledge_base['learned_thresholds'].items():
            print(f"  {key}: {value:.4f}")
    
    def compute_comparison_statistics(self, comparison_results):
        """Compute statistics from comparison results"""
        stats = {}
        
        # Group by metric type
        metric_groups = {
            'similarity': ['ssim', 'pearson_correlation', 'mutual_information'],
            'difference': ['mse', 'mae', 'max_diff'],
            'distribution': ['kl_divergence', 'js_divergence', 'emd']
        }
        
        for group_name, metrics in metric_groups.items():
            stats[group_name] = {}
            for metric in metrics:
                values = [r[metric] for r in comparison_results if metric in r]
                if values:
                    stats[group_name][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'range': (np.min(values), np.max(values))
                    }
        
        return stats
    
    def analyze_test_image(self, test_path):
        """Analyze a test image/JSON for anomalies"""
        print(f"\nAnalyzing test image: {test_path}")
        
        # Load test data
        if test_path.endswith('.json'):
            test_matrix, test_data = self.load_json_matrix(test_path)
        else:
            # Load image directly
            test_matrix = cv2.imread(test_path)
            if test_matrix is None:
                raise ValueError(f"Cannot load image: {test_path}")
            test_data = {'filename': os.path.basename(test_path)}
        
        # Extract features
        test_features = self.compute_matrix_features(test_matrix)
        
        # Compare with golden standard
        golden = self.knowledge_base['golden_standards'].get('current', {})
        if not golden:
            print("Warning: No golden standard available. Run analyze_all_matrices first.")
            return None
        
        # Compute anomaly scores
        anomaly_scores = {}
        
        # Global anomaly detection
        for key in test_features:
            if isinstance(test_features[key], (int, float, np.number)):
                if f'{key}_robust_mean' in golden and f'{key}_robust_std' in golden:
                    z_score = abs(test_features[key] - golden[f'{key}_robust_mean']) / (golden[f'{key}_robust_std'] + 1e-6)
                    anomaly_scores[key] = z_score
        
        # Local anomaly detection
        anomaly_map = self.detect_anomalies_local(test_matrix, golden)
        
        # Specific defect detection
        specific_defects = self.detect_specific_defects(test_matrix)
        
        # Create visualization
        result_image = self.create_anomaly_visualization(
            test_matrix, anomaly_map, specific_defects, anomaly_scores
        )
        
        # Save results
        output_path = f"anomaly_result_{test_data.get('filename', 'output')}.png"
        cv2.imwrite(output_path, result_image)
        print(f"Results saved to: {output_path}")
        
        # Update knowledge base with this analysis
        if self.learning_mode:
            self.knowledge_base['historical_analyses'].append({
                'filename': test_data.get('filename'),
                'anomaly_scores': anomaly_scores,
                'defect_counts': {k: len(v) for k, v in specific_defects.items()},
                'max_anomaly_score': np.max(anomaly_map)
            })
            self.save_knowledge_base()
        
        return {
            'anomaly_map': anomaly_map,
            'specific_defects': specific_defects,
            'anomaly_scores': anomaly_scores,
            'result_image': result_image
        }
    
    def create_anomaly_visualization(self, original, anomaly_map, defects, scores):
        """Create visualization with anomalies highlighted in blue"""
        # Create overlay
        overlay = original.copy()
        h, w = original.shape[:2]
        
        # Normalize anomaly map
        if np.max(anomaly_map) > 0:
            anomaly_norm = (anomaly_map / np.max(anomaly_map) * 255).astype(np.uint8)
        else:
            anomaly_norm = anomaly_map.astype(np.uint8)
        
        # Create blue mask for anomalies
        blue_mask = np.zeros_like(overlay)
        threshold = np.percentile(anomaly_norm[anomaly_norm > 0], 75) if np.any(anomaly_norm > 0) else 128
        blue_mask[:, :, 0] = 255  # Blue channel
        blue_mask[:, :, 1] = 0    # Green channel
        blue_mask[:, :, 2] = 0    # Red channel
        
        # Apply mask where anomalies exceed threshold
        anomaly_binary = anomaly_norm > threshold
        overlay[anomaly_binary] = cv2.addWeighted(
            overlay[anomaly_binary], 0.5,
            blue_mask[anomaly_binary], 0.5, 0
        )
        
        # Draw specific defects
        # Scratches - bright blue lines
        if defects['scratches'] is not None:
            for line in defects['scratches']:
                x1, y1, x2, y2 = line[0]
                cv2.line(overlay, (x1, y1), (x2, y2), (255, 128, 0), 2)
        
        # Digs - blue circles
        for cx, cy, area in defects['digs']:
            radius = int(np.sqrt(area / np.pi))
            cv2.circle(overlay, (cx, cy), radius, (255, 0, 0), 2)
        
        # Blobs - blue contours
        for contour in defects['blobs']:
            cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)
        
        # Add text with anomaly statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        
        # Find top anomalies
        top_anomalies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        cv2.putText(overlay, "Top Anomaly Scores:", (10, y_pos), font, 0.7, (255, 255, 255), 2)
        y_pos += 30
        
        for feature, score in top_anomalies:
            if score > 3:  # Z-score > 3 is significant
                cv2.putText(overlay, f"{feature}: {score:.2f}", (10, y_pos), font, 0.5, (0, 0, 255), 1)
                y_pos += 20
        
        # Add defect counts
        y_pos += 10
        cv2.putText(overlay, f"Scratches: {len(defects['scratches']) if defects['scratches'] is not None else 0}", 
                   (10, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(overlay, f"Digs: {len(defects['digs'])}", (10, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(overlay, f"Blobs: {len(defects['blobs'])}", (10, y_pos), font, 0.5, (255, 255, 255), 1)
        
        return overlay


def main():
    """Main execution function"""
    analyzer = ComprehensiveMatrixAnalyzer(learning_mode=True)
    
    print("=== Comprehensive Matrix Anomaly Detection System ===")
    print("This system implements exhaustive mathematical, statistical, and")
    print("information-theoretic methods for fiber optic defect detection.\n")
    
    # Step 1: Analyze reference matrices
    ref_dir = input("Enter path to directory containing reference JSON matrices: ").strip()
    
    if os.path.isdir(ref_dir):
        golden_standard, comparisons = analyzer.analyze_all_matrices(ref_dir)
        print(f"\nAnalysis complete. Created golden standard from {len(os.listdir(ref_dir))} matrices.")
    else:
        print(f"Error: Directory '{ref_dir}' not found.")
        return
    
    # Step 2: Analyze test image
    while True:
        print("\n" + "="*50)
        test_path = input("Enter path to test image or JSON file (or 'quit' to exit): ").strip()
        
        if test_path.lower() == 'quit':
            break
            
        if os.path.exists(test_path):
            try:
                results = analyzer.analyze_test_image(test_path)
                
                if results:
                    print(f"\nAnalysis complete!")
                    print(f"Maximum anomaly score: {np.max(results['anomaly_map']):.2f}")
                    print(f"Detected defects:")
                    for defect_type, defect_list in results['specific_defects'].items():
                        count = len(defect_list) if isinstance(defect_list, list) else 0
                        print(f"  - {defect_type}: {count}")
                    print(f"\nVisualization saved with anomalies highlighted in blue.")
                    
            except Exception as e:
                print(f"Error analyzing {test_path}: {e}")
        else:
            print(f"Error: File '{test_path}' not found.")
    
    print("\nThank you for using the Comprehensive Matrix Anomaly Detection System!")


if __name__ == "__main__":
    main()
