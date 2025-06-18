#!/usr/bin/env python3
"""
Comprehensive Matrix Analysis and Anomaly Detection System
Based on mathematical methods from multiple research papers on matrix comparison,
defect detection, and image analysis.

Updated to work with JSON files created by multi_matrix.py with format:
{
    "filename": "original_image.jpg",
    "image_dimensions": {"width": W, "height": H, "channels": C},
    "pixels": [
        {"coordinates": {"x": X, "y": Y}, "bgr_intensity": [B, G, R]},
        ...
    ]
}

Features:
- Comprehensive feature extraction (Statistical, Fourier, Wavelet, LBP, GLCM, etc.)
- Multiple comparison metrics (Euclidean, Cosine, KL/JS divergence, etc.)
- Anomaly detection and classification (scratches, blobs, wire defects, etc.)
- Learning system that improves with more data
- Visual output with anomalies highlighted in blue
- Support for both JSON matrix files and regular images
"""

import json
import os
import numpy as np
import cv2
from scipy import stats, signal, ndimage
from scipy.spatial.distance import cdist
from scipy.stats import entropy, ks_2samp, chi2
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
import pywt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMatrixAnalyzer:
    """
    Comprehensive system for analyzing matrices and detecting anomalies
    using multiple mathematical and statistical methods.
    """
    
    def __init__(self, learning_db_path="matrix_analysis_db.pkl"):
        self.learning_db_path = learning_db_path
        self.matrix_database = []
        self.feature_database = []
        self.anomaly_patterns = {}
        self.comparison_results = {}
        self.learned_thresholds = {}
        self.current_image_metadata = None
        
        # Load existing learning database if available
        self.load_learning_db()
        
    def load_learning_db(self):
        """Load previously learned patterns and thresholds"""
        if os.path.exists(self.learning_db_path):
            with open(self.learning_db_path, 'rb') as f:
                data = pickle.load(f)
                self.anomaly_patterns = data.get('anomaly_patterns', {})
                self.learned_thresholds = data.get('learned_thresholds', {})
                print(f"Loaded learning database with {len(self.anomaly_patterns)} patterns")
    
    def save_learning_db(self):
        """Save learned patterns and thresholds"""
        data = {
            'anomaly_patterns': self.anomaly_patterns,
            'learned_thresholds': self.learned_thresholds,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.learning_db_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved learning database")
    
    def load_json_matrix(self, filepath):
        """Load matrix from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle the format from multi_matrix.py
        if isinstance(data, dict) and 'pixels' in data:
            # Extract image dimensions
            width = data['image_dimensions']['width']
            height = data['image_dimensions']['height']
            
            # Create matrix in format [x, y, B, G, R]
            matrix = []
            for pixel_data in data['pixels']:
                x = pixel_data['coordinates']['x']
                y = pixel_data['coordinates']['y']
                bgr = pixel_data['bgr_intensity']
                matrix.append([x, y, bgr[0], bgr[1], bgr[2]])
            
            matrix = np.array(matrix)
            
            # Store metadata for later use
            self.current_image_metadata = {
                'filename': data.get('filename', 'unknown'),
                'width': width,
                'height': height,
                'channels': data['image_dimensions']['channels']
            }
        else:
            # Handle other formats (backward compatibility)
            if isinstance(data, dict):
                if 'matrix' in data:
                    matrix_data = data['matrix']
                else:
                    matrix_data = data
            else:
                matrix_data = data
                
            matrix = np.array(matrix_data)
            self.current_image_metadata = None
            
        return matrix
    
    def extract_comprehensive_features(self, matrix):
        """Extract comprehensive features from matrix using multiple methods"""
        features = {}
        
        # Check if matrix needs to be converted to image
        if isinstance(matrix, np.ndarray) and len(matrix.shape) == 2 and matrix.shape[1] == 5:
            # Matrix is in format [x, y, B, G, R]
            img = self.matrix_to_image(matrix)
        else:
            # Already an image
            img = matrix
            
        # Ensure image is in correct format
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # 1. Statistical Features
        features['statistical'] = self.extract_statistical_features(gray)
        
        # 2. Matrix Norms
        features['norms'] = self.extract_matrix_norms(gray)
        
        # 3. SSIM Components (will use for comparison)
        features['texture'] = self.extract_texture_features(gray)
        
        # 4. Fourier Features
        features['fourier'] = self.extract_fourier_features(gray)
        
        # 5. Wavelet Features
        features['wavelet'] = self.extract_wavelet_features(gray)
        
        # 6. LBP Features
        features['lbp'] = self.extract_lbp_features(gray)
        
        # 7. GLCM Features
        features['glcm'] = self.extract_glcm_features(gray)
        
        # 8. SVD Features
        features['svd'] = self.extract_svd_features(gray)
        
        # 9. Entropy-based Features
        features['entropy'] = self.extract_entropy_features(gray)
        
        # 10. Morphological Features
        features['morphological'] = self.extract_morphological_features(gray)
        
        return features
    
    def matrix_to_image(self, matrix):
        """Convert coordinate matrix to image"""
        # If we have metadata from the JSON file, use it
        if hasattr(self, 'current_image_metadata') and self.current_image_metadata:
            width = self.current_image_metadata['width']
            height = self.current_image_metadata['height']
            
            # Create empty image
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Fill image efficiently using matrix data
            # Matrix format: [x, y, B, G, R]
            for row in matrix:
                x, y = int(row[0]), int(row[1])
                bgr = row[2:5].astype(np.uint8)
                img[y, x] = bgr
        else:
            # Fallback method for matrices without metadata
            # Extract coordinates and colors
            coords = matrix[:, :2].astype(int)
            colors = matrix[:, 2:5].astype(np.uint8)
            
            # Find image dimensions
            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)
            
            # Create image
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Fill image
            for i, (x, y) in enumerate(coords):
                img[y - min_y, x - min_x] = colors[i]
                
        return img
    
    def extract_statistical_features(self, img):
        """Extract statistical features"""
        return {
            'mean': np.mean(img),
            'std': np.std(img),
            'variance': np.var(img),
            'skewness': stats.skew(img.flatten()),
            'kurtosis': stats.kurtosis(img.flatten()),
            'min': np.min(img),
            'max': np.max(img),
            'median': np.median(img),
            'mad': np.median(np.abs(img - np.median(img))),  # Median absolute deviation
            'iqr': np.percentile(img, 75) - np.percentile(img, 25),  # Interquartile range
            'energy': np.sum(img**2),
            'entropy': entropy(np.histogram(img, bins=256)[0] + 1e-10)
        }
    
    def extract_matrix_norms(self, img):
        """Extract various matrix norms"""
        return {
            'frobenius': np.linalg.norm(img, 'fro'),
            'l1': np.linalg.norm(img, 1),
            'l2': np.linalg.norm(img, 2),
            'linf': np.linalg.norm(img, np.inf),
            'nuclear': np.linalg.norm(img, 'nuc'),
            'trace': np.trace(img)
        }
    
    def extract_texture_features(self, img):
        """Extract texture-based features"""
        # Gradient features
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Laplacian
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        
        return {
            'gradient_mean': np.mean(gradient_mag),
            'gradient_std': np.std(gradient_mag),
            'gradient_max': np.max(gradient_mag),
            'laplacian_mean': np.mean(np.abs(laplacian)),
            'laplacian_std': np.std(laplacian),
            'edge_density': np.sum(gradient_mag > np.mean(gradient_mag)) / gradient_mag.size
        }
    
    def extract_fourier_features(self, img):
        """Extract Fourier transform features"""
        # 2D FFT
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        power_spectrum = magnitude_spectrum**2
        
        # Radial average of power spectrum
        center = np.array(power_spectrum.shape) // 2
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Binned radial profile
        radial_profile = ndimage.mean(power_spectrum, labels=r, index=np.arange(1, r.max()))
        
        return {
            'spectral_centroid': np.sum(np.arange(len(radial_profile)) * radial_profile) / np.sum(radial_profile),
            'spectral_bandwidth': np.sqrt(np.sum((np.arange(len(radial_profile)) - np.sum(np.arange(len(radial_profile)) * radial_profile) / np.sum(radial_profile))**2 * radial_profile) / np.sum(radial_profile)),
            'spectral_flatness': stats.gmean(radial_profile + 1e-10) / (np.mean(radial_profile) + 1e-10),
            'spectral_rolloff': np.where(np.cumsum(radial_profile) > 0.85 * np.sum(radial_profile))[0][0] if len(np.where(np.cumsum(radial_profile) > 0.85 * np.sum(radial_profile))[0]) > 0 else 0,
            'dc_component': magnitude_spectrum[center[0], center[1]],
            'total_power': np.sum(power_spectrum)
        }
    
    def extract_wavelet_features(self, img):
        """Extract wavelet transform features"""
        # 2D Discrete Wavelet Transform
        coeffs = pywt.dwt2(img, 'db4')
        cA, (cH, cV, cD) = coeffs
        
        features = {}
        
        # Features from each subband
        for name, coeff in [('approx', cA), ('horizontal', cH), ('vertical', cV), ('diagonal', cD)]:
            features[f'wavelet_{name}_mean'] = np.mean(np.abs(coeff))
            features[f'wavelet_{name}_std'] = np.std(coeff)
            features[f'wavelet_{name}_energy'] = np.sum(coeff**2)
            features[f'wavelet_{name}_entropy'] = entropy(np.histogram(coeff.flatten(), bins=50)[0] + 1e-10)
            
        return features
    
    def extract_lbp_features(self, img):
        """Extract Local Binary Pattern features"""
        features = {}
        
        # LBP with different radii
        for radius in [1, 2, 3]:
            n_points = 8 * radius
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            
            # Histogram of LBP
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-10)
            
            features[f'lbp_r{radius}_mean'] = np.mean(lbp)
            features[f'lbp_r{radius}_std'] = np.std(lbp)
            features[f'lbp_r{radius}_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
            
        return features
    
    def extract_glcm_features(self, img):
        """Extract Gray-Level Co-occurrence Matrix features"""
        # Compute GLCM
        # Normalize image to 8 levels for faster computation
        img_norm = ((img - img.min()) * 7 / (img.max() - img.min() + 1e-10)).astype(np.uint8)
        
        # Compute GLCM for different angles and distances
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm_features = {}
        
        for d in distances:
            for a_idx, a in enumerate(angles):
                # Simple GLCM computation
                glcm = self.compute_glcm(img_norm, d, a)
                
                # Normalize
                glcm = glcm / (glcm.sum() + 1e-10)
                
                # GLCM features
                i, j = np.meshgrid(np.arange(8), np.arange(8))
                
                contrast = np.sum((i - j)**2 * glcm)
                homogeneity = np.sum(glcm / (1 + (i - j)**2))
                energy = np.sum(glcm**2)
                correlation = np.sum((i - i.mean()) * (j - j.mean()) * glcm) / (np.sqrt(np.sum((i - i.mean())**2 * glcm) * np.sum((j - j.mean())**2 * glcm)) + 1e-10)
                
                glcm_features[f'glcm_d{d}_a{a_idx}_contrast'] = contrast
                glcm_features[f'glcm_d{d}_a{a_idx}_homogeneity'] = homogeneity
                glcm_features[f'glcm_d{d}_a{a_idx}_energy'] = energy
                glcm_features[f'glcm_d{d}_a{a_idx}_correlation'] = correlation
                
        return glcm_features
    
    def compute_glcm(self, img, distance, angle):
        """Compute GLCM for given distance and angle"""
        # Simple GLCM implementation
        levels = 8
        glcm = np.zeros((levels, levels))
        
        # Determine offset based on angle
        if angle == 0:
            offset = (0, distance)
        elif angle == np.pi/4:
            offset = (-distance, distance)
        elif angle == np.pi/2:
            offset = (-distance, 0)
        else:  # 3*pi/4
            offset = (-distance, -distance)
            
        # Compute co-occurrences
        rows, cols = img.shape
        for i in range(max(0, -offset[0]), min(rows, rows - offset[0])):
            for j in range(max(0, -offset[1]), min(cols, cols - offset[1])):
                glcm[img[i, j], img[i + offset[0], j + offset[1]]] += 1
                
        return glcm
    
    def extract_svd_features(self, img):
        """Extract Singular Value Decomposition features"""
        # Compute SVD
        U, s, Vt = np.linalg.svd(img, full_matrices=False)
        
        # Normalize singular values
        s_norm = s / (s.sum() + 1e-10)
        
        return {
            'svd_largest': s[0] if len(s) > 0 else 0,
            'svd_second_largest': s[1] if len(s) > 1 else 0,
            'svd_ratio_1_2': s[0] / (s[1] + 1e-10) if len(s) > 1 else 0,
            'svd_sum_top10': np.sum(s[:10]),
            'svd_sum_ratio_top10': np.sum(s[:10]) / (np.sum(s) + 1e-10),
            'svd_entropy': -np.sum(s_norm * np.log2(s_norm + 1e-10)),
            'effective_rank': np.exp(-np.sum(s_norm * np.log(s_norm + 1e-10)))
        }
    
    def extract_entropy_features(self, img):
        """Extract entropy-based features"""
        # Histogram
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        
        # Shannon entropy
        shannon_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Renyi entropy (alpha = 2)
        renyi_entropy = -np.log2(np.sum(hist**2) + 1e-10)
        
        # Tsallis entropy (q = 2)
        tsallis_entropy = (1 - np.sum(hist**2)) / 1
        
        # Local entropy
        local_entropy = ndimage.generic_filter(img, lambda x: entropy(np.histogram(x, bins=10)[0] + 1e-10), size=5)
        
        return {
            'shannon_entropy': shannon_entropy,
            'renyi_entropy': renyi_entropy,
            'tsallis_entropy': tsallis_entropy,
            'local_entropy_mean': np.mean(local_entropy),
            'local_entropy_std': np.std(local_entropy),
            'local_entropy_max': np.max(local_entropy)
        }
    
    def extract_morphological_features(self, img):
        """Extract morphological features"""
        # Binary image for morphological operations
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(binary, kernel, iterations=1)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
        
        return {
            'morph_area_ratio': np.sum(binary) / binary.size,
            'morph_perimeter': np.sum(gradient) / 255,
            'morph_compactness': 4 * np.pi * np.sum(binary) / (np.sum(gradient)**2 + 1e-10),
            'morph_tophat_mean': np.mean(tophat),
            'morph_blackhat_mean': np.mean(blackhat),
            'morph_opening_diff': np.sum(np.abs(binary - opening)),
            'morph_closing_diff': np.sum(np.abs(binary - closing))
        }
    
    def compare_matrices(self, features1, features2):
        """Compare two feature sets using multiple metrics"""
        comparison = {}
        
        # 1. Statistical comparison
        stat_features1 = self.flatten_features(features1)
        stat_features2 = self.flatten_features(features2)
        
        # Euclidean distance
        comparison['euclidean_distance'] = np.linalg.norm(stat_features1 - stat_features2)
        
        # Cosine similarity
        comparison['cosine_similarity'] = np.dot(stat_features1, stat_features2) / (np.linalg.norm(stat_features1) * np.linalg.norm(stat_features2) + 1e-10)
        
        # Manhattan distance
        comparison['manhattan_distance'] = np.sum(np.abs(stat_features1 - stat_features2))
        
        # Mahalanobis distance (simplified - assuming identity covariance)
        comparison['mahalanobis_distance'] = np.sqrt(np.sum((stat_features1 - stat_features2)**2))
        
        # 2. Distribution comparison
        # KL divergence for histogram-like features
        hist1 = self.get_histogram_features(features1)
        hist2 = self.get_histogram_features(features2)
        
        # Normalize
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        # KL divergence
        comparison['kl_divergence'] = np.sum(hist1 * np.log((hist1 + 1e-10) / (hist2 + 1e-10)))
        
        # JS divergence
        m = 0.5 * (hist1 + hist2)
        comparison['js_divergence'] = 0.5 * np.sum(hist1 * np.log((hist1 + 1e-10) / (m + 1e-10))) + 0.5 * np.sum(hist2 * np.log((hist2 + 1e-10) / (m + 1e-10)))
        
        # Chi-square distance
        comparison['chi_square'] = 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
        
        # Earth Mover's Distance (simplified)
        comparison['emd'] = np.sum(np.abs(np.cumsum(hist1) - np.cumsum(hist2)))
        
        return comparison
    
    def flatten_features(self, features):
        """Flatten nested feature dictionary into a single vector"""
        flat_features = []
        
        for category, cat_features in features.items():
            if isinstance(cat_features, dict):
                for key, value in cat_features.items():
                    if isinstance(value, (int, float, np.number)):
                        flat_features.append(value)
            elif isinstance(cat_features, (int, float, np.number)):
                flat_features.append(cat_features)
                
        return np.array(flat_features)
    
    def get_histogram_features(self, features):
        """Extract histogram-like features for distribution comparison"""
        hist_features = []
        
        # Extract specific features that represent distributions
        for key in ['lbp', 'wavelet', 'glcm']:
            if key in features:
                for k, v in features[key].items():
                    if 'entropy' in k or 'energy' in k:
                        hist_features.append(v)
                        
        return np.array(hist_features) if hist_features else np.array([0])
    
    def detect_anomalies(self, test_features, reference_features_list):
        """Detect anomalies by comparing test features against reference features"""
        anomalies = {
            'scores': {},
            'detected': False,
            'confidence': 0.0,
            'anomaly_regions': []
        }
        
        # Compare against each reference
        comparison_scores = []
        
        for ref_features in reference_features_list:
            comparison = self.compare_matrices(test_features, ref_features)
            
            # Compute anomaly score (higher = more anomalous)
            anomaly_score = (
                comparison['euclidean_distance'] * 0.2 +
                (1 - comparison['cosine_similarity']) * 0.2 +
                comparison['manhattan_distance'] * 0.1 +
                comparison['kl_divergence'] * 0.2 +
                comparison['js_divergence'] * 0.2 +
                comparison['chi_square'] * 0.1
            )
            
            comparison_scores.append(anomaly_score)
        
        # Statistical analysis of scores
        scores_array = np.array(comparison_scores)
        anomalies['scores'] = {
            'mean': np.mean(scores_array),
            'std': np.std(scores_array),
            'min': np.min(scores_array),
            'max': np.max(scores_array),
            'median': np.median(scores_array)
        }
        
        # Determine if anomalous
        # Use adaptive threshold based on learned patterns or default
        if 'anomaly_threshold' in self.learned_thresholds:
            threshold = self.learned_thresholds['anomaly_threshold']
        else:
            # Default: mean + 2 * std
            threshold = np.mean(scores_array) + 2 * np.std(scores_array)
        
        anomalies['detected'] = np.max(scores_array) > threshold
        anomalies['confidence'] = min(1.0, np.max(scores_array) / threshold) if threshold > 0 else 0.0
        
        # Identify specific anomaly types
        if anomalies['detected']:
            anomaly_types = self.classify_anomaly_type(test_features, reference_features_list)
            anomalies['anomaly_types'] = anomaly_types
            
        return anomalies
    
    def classify_anomaly_type(self, test_features, reference_features_list):
        """Classify the type of anomaly based on feature patterns"""
        anomaly_types = []
        
        # Average reference features
        avg_ref_features = {}
        for key in test_features.keys():
            if key in ['statistical', 'texture', 'fourier', 'wavelet']:
                avg_ref_features[key] = {}
                for subkey in test_features[key].keys():
                    values = [ref[key][subkey] for ref in reference_features_list if key in ref and subkey in ref[key]]
                    if values:
                        avg_ref_features[key][subkey] = np.mean(values)
        
        # Check for specific anomaly patterns
        
        # 1. Texture anomaly (scratch/dig pattern)
        if 'texture' in test_features and 'texture' in avg_ref_features:
            gradient_diff = abs(test_features['texture']['gradient_mean'] - avg_ref_features['texture']['gradient_mean'])
            edge_diff = abs(test_features['texture']['edge_density'] - avg_ref_features['texture']['edge_density'])
            
            if gradient_diff > avg_ref_features['texture']['gradient_mean'] * 0.5:
                anomaly_types.append({
                    'type': 'scratch_or_dig',
                    'confidence': min(1.0, gradient_diff / avg_ref_features['texture']['gradient_mean']),
                    'features': ['high_gradient', 'texture_disruption']
                })
        
        # 2. Blob/stain pattern
        if 'statistical' in test_features and 'statistical' in avg_ref_features:
            mean_diff = abs(test_features['statistical']['mean'] - avg_ref_features['statistical']['mean'])
            std_diff = abs(test_features['statistical']['std'] - avg_ref_features['statistical']['std'])
            
            if mean_diff > avg_ref_features['statistical']['std']:
                anomaly_types.append({
                    'type': 'blob_or_stain',
                    'confidence': min(1.0, mean_diff / avg_ref_features['statistical']['std']),
                    'features': ['intensity_deviation', 'local_uniformity_change']
                })
        
        # 3. Periodic pattern disruption (wire defect)
        if 'fourier' in test_features and 'fourier' in avg_ref_features:
            spectral_diff = abs(test_features['fourier']['spectral_centroid'] - avg_ref_features['fourier']['spectral_centroid'])
            
            if spectral_diff > avg_ref_features['fourier']['spectral_centroid'] * 0.3:
                anomaly_types.append({
                    'type': 'wire_defect',
                    'confidence': min(1.0, spectral_diff / avg_ref_features['fourier']['spectral_centroid']),
                    'features': ['frequency_disruption', 'periodic_pattern_break']
                })
        
        # 4. Fine detail anomaly
        if 'wavelet' in test_features and 'wavelet' in avg_ref_features:
            detail_energy_diff = 0
            for subband in ['horizontal', 'vertical', 'diagonal']:
                key = f'wavelet_{subband}_energy'
                if key in test_features['wavelet'] and key in avg_ref_features['wavelet']:
                    detail_energy_diff += abs(test_features['wavelet'][key] - avg_ref_features['wavelet'][key])
            
            if detail_energy_diff > sum(avg_ref_features['wavelet'][f'wavelet_{s}_energy'] for s in ['horizontal', 'vertical', 'diagonal']) * 0.5:
                anomaly_types.append({
                    'type': 'fine_detail_anomaly',
                    'confidence': min(1.0, detail_energy_diff / sum(avg_ref_features['wavelet'][f'wavelet_{s}_energy'] for s in ['horizontal', 'vertical', 'diagonal'])),
                    'features': ['wavelet_coefficient_deviation', 'multi_scale_anomaly']
                })
        
        return anomaly_types
    
    def localize_anomalies(self, img, anomaly_info):
        """Localize anomalies in the image using sliding window approach"""
        # Ensure we're working with the right format
        if len(img.shape) == 2:
            gray = img
            height, width = img.shape
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
        anomaly_map = np.zeros((height, width), dtype=np.float32)
            
        # Sliding window parameters
        window_sizes = [32, 64, 128]
        stride = 16
        
        for window_size in window_sizes:
            for y in range(0, height - window_size + 1, stride):
                for x in range(0, width - window_size + 1, stride):
                    # Extract window
                    window = gray[y:y+window_size, x:x+window_size]
                    
                    # Compute local anomaly score
                    local_score = self.compute_local_anomaly_score(window, anomaly_info)
                    
                    # Update anomaly map
                    anomaly_map[y:y+window_size, x:x+window_size] = np.maximum(
                        anomaly_map[y:y+window_size, x:x+window_size],
                        local_score
                    )
        
        # Post-process anomaly map
        anomaly_map = cv2.GaussianBlur(anomaly_map, (21, 21), 0)
        
        # Threshold to get binary mask
        threshold = np.percentile(anomaly_map, 95)
        anomaly_mask = anomaly_map > threshold
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            anomaly_mask.astype(np.uint8), connectivity=8
        )
        
        # Extract anomaly regions
        anomaly_regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 100:  # Filter small regions
                confidence = np.mean(anomaly_map[labels == i])
                anomaly_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'centroid': centroids[i].tolist()
                })
        
        return anomaly_regions, anomaly_map
    
    def compute_local_anomaly_score(self, window, anomaly_info):
        """Compute anomaly score for a local window"""
        # Simple scoring based on local statistics
        score = 0.0
        
        # Check for high gradient (scratch/dig)
        gradient = np.std(window)
        if gradient > 30:
            score += 0.3
            
        # Check for uniform regions (blob/stain)
        if np.std(window) < 10 and abs(np.mean(window) - 128) > 50:
            score += 0.3
            
        # Check for texture disruption
        lbp = local_binary_pattern(window, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
        lbp_entropy = entropy(lbp_hist + 1e-10)
        
        if lbp_entropy < 1.5 or lbp_entropy > 3.0:
            score += 0.2
            
        # Edge density
        edges = cv2.Canny(window.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        if edge_density > 0.3:
            score += 0.2
            
        return min(1.0, score)
    
    def visualize_anomalies(self, img_path, anomaly_regions, anomaly_map):
        """Visualize detected anomalies on the image"""
        # Load image
        if img_path.endswith('.json'):
            matrix = self.load_json_matrix(img_path)
            img = self.matrix_to_image(matrix)
            # Ensure BGR to RGB conversion for display
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get original filename for title
            if self.current_image_metadata:
                original_name = self.current_image_metadata.get('filename', 'JSON Image')
            else:
                original_name = os.path.basename(img_path)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_name = os.path.basename(img_path)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f'Original Image\n({original_name})')
        axes[0].axis('off')
        
        # Anomaly heatmap
        axes[1].imshow(img)
        axes[1].imshow(anomaly_map, cmap='hot', alpha=0.5)
        axes[1].set_title('Anomaly Heatmap')
        axes[1].axis('off')
        
        # Detected regions
        axes[2].imshow(img)
        
        # Draw bounding boxes in blue
        for region in anomaly_regions:
            x, y, w, h = region['bbox']
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
            axes[2].add_patch(rect)
            
            # Add confidence score
            axes[2].text(x, y-5, f"{region['confidence']:.2f}", color='blue', fontsize=8, weight='bold')
        
        axes[2].set_title(f'Detected Anomalies ({len(anomaly_regions)} regions)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save result
        if img_path.endswith('.json'):
            # For JSON files, use a name that indicates it came from JSON
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = f"anomaly_detection_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            output_path = f"anomaly_detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results saved to: {output_path}")
        
        # If analyzing a JSON file, also save anomaly regions data
        if img_path.endswith('.json') and anomaly_regions:
            json_output = {
                'source_json': img_path,
                'original_image': self.current_image_metadata.get('filename', 'unknown') if self.current_image_metadata else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'anomaly_regions': anomaly_regions,
                'total_anomalies': len(anomaly_regions)
            }
            
            json_output_path = output_path.replace('.png', '_regions.json')
            with open(json_output_path, 'w') as f:
                json.dump(json_output, f, indent=2)
            print(f"Anomaly regions data saved to: {json_output_path}")
        
        return output_path
    
    def analyze_directory(self, directory_path):
        """Analyze all JSON files in a directory"""
        print(f"Analyzing directory: {directory_path}")
        
        # Find all JSON files
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {directory_path}")
            print("Please ensure the directory contains JSON files created by multi_matrix.py")
            return
            
        print(f"Found {len(json_files)} JSON files")
        
        # Process each file
        for i, json_file in enumerate(json_files):
            print(f"\nProcessing {i+1}/{len(json_files)}: {json_file}")
            
            filepath = os.path.join(directory_path, json_file)
            
            try:
                # Load matrix
                matrix = self.load_json_matrix(filepath)
                
                # Show metadata if available
                if self.current_image_metadata:
                    print(f"  - Original image: {self.current_image_metadata['filename']}")
                    print(f"  - Dimensions: {self.current_image_metadata['width']}x{self.current_image_metadata['height']}")
                    print(f"  - Matrix shape: {matrix.shape}")
                
                # Extract features
                print("  - Extracting features...")
                features = self.extract_comprehensive_features(matrix)
                
                # Store in database
                self.matrix_database.append({
                    'filename': json_file,
                    'original_filename': self.current_image_metadata.get('filename', 'unknown') if self.current_image_metadata else 'unknown',
                    'matrix': matrix,
                    'features': features,
                    'metadata': self.current_image_metadata.copy() if self.current_image_metadata else None
                })
                
                self.feature_database.append(features)
                print("  - Features extracted successfully")
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Compute pairwise comparisons
        if len(self.feature_database) > 1:
            print("\nComputing pairwise comparisons...")
            self.compute_all_comparisons()
        else:
            print("\nOnly one file analyzed, skipping pairwise comparisons.")
        
        # Find trends and patterns
        print("\nAnalyzing trends and patterns...")
        self.analyze_trends()
        
        # Update learning database
        self.update_learned_thresholds()
        self.save_learning_db()
        
        print("\nDirectory analysis complete!")
        print(f"Successfully processed {len(self.feature_database)} images")
        
    def compute_all_comparisons(self):
        """Compute all pairwise comparisons between matrices"""
        n = len(self.feature_database)
        total_comparisons = n * (n - 1) // 2
        
        print(f"Computing {total_comparisons} pairwise comparisons...")
        
        comparison_count = 0
        for i in range(n):
            for j in range(i+1, n):
                comparison = self.compare_matrices(
                    self.feature_database[i],
                    self.feature_database[j]
                )
                
                key = f"{i}_{j}"
                self.comparison_results[key] = comparison
                
                comparison_count += 1
                if comparison_count % 100 == 0:
                    print(f"  Progress: {comparison_count}/{total_comparisons} comparisons ({comparison_count/total_comparisons*100:.1f}%)")
        
        print(f"  Completed all {total_comparisons} comparisons")
    
    def analyze_trends(self):
        """Analyze trends within and across matrices"""
        # Global statistics
        all_features = self.feature_database
        
        if not all_features:
            print("No features to analyze")
            return
        
        # Compute feature statistics across all matrices
        feature_stats = {}
        
        # Get all feature keys
        sample_features = all_features[0]
        
        for category in sample_features:
            feature_stats[category] = {}
            
            if isinstance(sample_features[category], dict):
                for key in sample_features[category]:
                    values = [f[category][key] for f in all_features if category in f and key in f[category]]
                    
                    if values:
                        feature_stats[category][key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values)
                        }
        
        # Identify outliers
        outliers = []
        
        for i, features in enumerate(all_features):
            outlier_score = 0
            
            for category in features:
                if category in feature_stats and isinstance(features[category], dict):
                    for key in features[category]:
                        if key in feature_stats[category]:
                            value = features[category][key]
                            mean = feature_stats[category][key]['mean']
                            std = feature_stats[category][key]['std']
                            
                            # Z-score
                            if std > 0:
                                z_score = abs(value - mean) / std
                                if z_score > 3:
                                    outlier_score += 1
            
            if outlier_score > 5:
                outliers.append({
                    'index': i,
                    'filename': self.matrix_database[i]['filename'],
                    'original_filename': self.matrix_database[i].get('original_filename', 'unknown'),
                    'outlier_score': outlier_score
                })
        
        print(f"\nFound {len(outliers)} potential outliers")
        for outlier in outliers:
            orig_name = outlier['original_filename']
            print(f"  - {orig_name} (score = {outlier['outlier_score']})")
        
        # Store patterns
        self.anomaly_patterns['global_stats'] = feature_stats
        self.anomaly_patterns['outliers'] = outliers
    
    def update_learned_thresholds(self):
        """Update learned thresholds based on analyzed data"""
        if self.comparison_results:
            # Extract all anomaly scores
            scores = []
            
            for comp in self.comparison_results.values():
                anomaly_score = (
                    comp['euclidean_distance'] * 0.2 +
                    (1 - comp['cosine_similarity']) * 0.2 +
                    comp['manhattan_distance'] * 0.1 +
                    comp['kl_divergence'] * 0.2 +
                    comp['js_divergence'] * 0.2 +
                    comp['chi_square'] * 0.1
                )
                scores.append(anomaly_score)
            
            if scores:
                # Set threshold as mean + 2.5 * std
                self.learned_thresholds['anomaly_threshold'] = np.mean(scores) + 2.5 * np.std(scores)
                print(f"Updated anomaly threshold: {self.learned_thresholds['anomaly_threshold']:.4f}")
    
    def analyze_new_image(self, image_path):
        """Analyze a new image for anomalies"""
        print(f"\nAnalyzing new image: {image_path}")
        
        # Clear previous metadata
        self.current_image_metadata = None
        
        # Check if reference data exists
        if not self.feature_database:
            print("Error: No reference data available. Please analyze a directory first.")
            return
        
        # Load and process the new image
        if image_path.endswith('.json'):
            matrix = self.load_json_matrix(image_path)
            img = self.matrix_to_image(matrix)
            
            # Show loaded image info
            if self.current_image_metadata:
                print(f"  - Original image: {self.current_image_metadata['filename']}")
                print(f"  - Dimensions: {self.current_image_metadata['width']}x{self.current_image_metadata['height']}")
        else:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not load image from {image_path}")
                return
            print(f"  - Image dimensions: {img.shape[1]}x{img.shape[0]}")
        
        # Extract features
        print("Extracting features...")
        if image_path.endswith('.json'):
            # For JSON files, we already have the image
            test_features = self.extract_comprehensive_features(matrix)
        else:
            # For regular images, convert to grayscale first
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img
            test_features = self.extract_comprehensive_features(gray_img)
        
        # Detect anomalies
        print("Detecting anomalies...")
        anomaly_info = self.detect_anomalies(test_features, self.feature_database)
        
        print(f"\nAnomaly Detection Results:")
        print(f"  - Anomaly Detected: {anomaly_info['detected']}")
        print(f"  - Confidence: {anomaly_info['confidence']:.2%}")
        
        if anomaly_info['detected']:
            print(f"  - Anomaly Types:")
            for atype in anomaly_info.get('anomaly_types', []):
                print(f"    * {atype['type']} (confidence: {atype['confidence']:.2%})")
                print(f"      Features: {', '.join(atype['features'])}")
        
        # Localize anomalies
        print("\nLocalizing anomalies...")
        anomaly_regions, anomaly_map = self.localize_anomalies(img, anomaly_info)
        
        print(f"Found {len(anomaly_regions)} anomaly regions")
        
        # Visualize results
        output_path = self.visualize_anomalies(image_path, anomaly_regions, anomaly_map)
        
        # Store for learning
        self.store_analysis_result(image_path, test_features, anomaly_info, anomaly_regions)
        
        return {
            'anomaly_detected': anomaly_info['detected'],
            'confidence': anomaly_info['confidence'],
            'anomaly_types': anomaly_info.get('anomaly_types', []),
            'regions': anomaly_regions,
            'output_image': output_path
        }
    
    def store_analysis_result(self, image_path, features, anomaly_info, regions):
        """Store analysis results for future learning"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'features': features,
            'anomaly_info': anomaly_info,
            'regions': regions
        }
        
        # Add to learning database
        if 'analysis_history' not in self.anomaly_patterns:
            self.anomaly_patterns['analysis_history'] = []
        
        self.anomaly_patterns['analysis_history'].append(result)
        
        # Save updated database
        self.save_learning_db()


def main():
    """Main function to run the anomaly detection system"""
    print("=" * 80)
    print("COMPREHENSIVE MATRIX ANALYSIS AND ANOMALY DETECTION SYSTEM")
    print("=" * 80)
    print("\nThis system analyzes JSON files created by multi_matrix.py")
    print("Expected JSON format: {filename, image_dimensions, pixels: [{coordinates, bgr_intensity}]}")
    print("The system will learn normal patterns and detect anomalies in new images.\n")
    
    # Initialize analyzer
    analyzer = ComprehensiveMatrixAnalyzer()
    
    # Get directory path
    while True:
        directory_path = input("\nEnter the path to the directory containing JSON files: ").strip()
        
        # Clean up path (remove quotes if user pastes them)
        directory_path = directory_path.strip().strip('"\'')
        
        if os.path.isdir(directory_path):
            break
        else:
            print("Error: Invalid directory path. Please try again.")
    
    # Analyze directory
    analyzer.analyze_directory(directory_path)
    
    # Main loop for analyzing new images
    while True:
        print("\n" + "-" * 80)
        print("OPTIONS:")
        print("1. Analyze a new image/JSON file for anomalies")
        print("2. View analysis statistics")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # Get image path
            image_path = input("\nEnter the path to the image or JSON file to analyze: ").strip()
            
            # Clean up path (remove quotes if user pastes them)
            image_path = image_path.strip().strip('"\'')
            
            if os.path.isfile(image_path):
                result = analyzer.analyze_new_image(image_path)
                
                if result:
                    print("\n" + "=" * 50)
                    print("ANALYSIS COMPLETE")
                    print("=" * 50)
                    print(f"Output saved to: {result['output_image']}")
            else:
                print("Error: File not found.")
        
        elif choice == '2':
            # Show statistics
            print("\n" + "=" * 50)
            print("ANALYSIS STATISTICS")
            print("=" * 50)
            print(f"Total matrices analyzed: {len(analyzer.matrix_database)}")
            print(f"Total comparisons: {len(analyzer.comparison_results)}")
            print(f"Learned patterns: {len(analyzer.anomaly_patterns)}")
            
            if analyzer.matrix_database:
                print("\nAnalyzed images:")
                for i, entry in enumerate(analyzer.matrix_database[:10]):  # Show first 10
                    orig_name = entry.get('original_filename', entry['filename'])
                    print(f"  {i+1}. {orig_name}")
                if len(analyzer.matrix_database) > 10:
                    print(f"  ... and {len(analyzer.matrix_database) - 10} more")
            
            if 'outliers' in analyzer.anomaly_patterns:
                print(f"\nIdentified outliers: {len(analyzer.anomaly_patterns['outliers'])}")
                for outlier in analyzer.anomaly_patterns['outliers'][:5]:  # Show first 5
                    print(f"  - {outlier['filename']} (score: {outlier['outlier_score']})")
            
            if 'analysis_history' in analyzer.anomaly_patterns:
                print(f"\nPrevious analyses: {len(analyzer.anomaly_patterns['analysis_history'])}")
                
            if analyzer.learned_thresholds:
                print("\nLearned thresholds:")
                for key, value in analyzer.learned_thresholds.items():
                    print(f"  - {key}: {value:.4f}")
        
        elif choice == '3':
            print("\nExiting... Thank you for using the system!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
