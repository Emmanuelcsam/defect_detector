#!/usr/bin/env python3
"""
Ultra-Comprehensive Matrix Anomaly Detection System (v3.0 - Optimized)
======================================================================

This optimized version maintains all functionality while using only:
- json, os, cv2, matplotlib.pyplot, numpy

All statistical, feature extraction, and machine learning algorithms
have been reimplemented from scratch for maximum effectiveness.

Features:
- 100+ comprehensive features
- Multiple comparison metrics
- Advanced statistical modeling
- Local and global anomaly detection
- Learning system with JSON persistence
- SSIM-based structural comparison
- Specific defect detection
- Rich visualization with blue anomaly highlights
"""

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class UltraComprehensiveMatrixAnalyzer:
    """The ultimate matrix anomaly detection system - optimized version."""
    
    def __init__(self, knowledge_base_path="ultra_anomaly_kb.json"):
        self.knowledge_base_path = knowledge_base_path
        self.reference_model = {
            'features': [],
            'statistical_model': None,
            'archetype_image': None,
            'feature_names': [],
            'comparison_results': {},
            'learned_thresholds': {},
            'timestamp': None
        }
        self.current_metadata = None
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load previously saved knowledge base from JSON."""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    loaded_data = json.load(f)
                
                # Convert lists back to numpy arrays where needed
                if loaded_data.get('archetype_image'):
                    loaded_data['archetype_image'] = np.array(loaded_data['archetype_image'], dtype=np.uint8)
                
                if loaded_data.get('statistical_model'):
                    for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                        if key in loaded_data['statistical_model']:
                            loaded_data['statistical_model'][key] = np.array(loaded_data['statistical_model'][key])
                
                self.reference_model = loaded_data
                print(f"✓ Loaded knowledge base from {self.knowledge_base_path}")
            except Exception as e:
                print(f"⚠ Could not load knowledge base: {e}")
    
    def save_knowledge_base(self):
        """Save current knowledge base to JSON."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            save_data = self.reference_model.copy()
            
            if isinstance(save_data.get('archetype_image'), np.ndarray):
                save_data['archetype_image'] = save_data['archetype_image'].tolist()
            
            if save_data.get('statistical_model'):
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    if key in save_data['statistical_model'] and isinstance(save_data['statistical_model'][key], np.ndarray):
                        save_data['statistical_model'][key] = save_data['statistical_model'][key].tolist()
            
            # Remove comparison_scores if it's too large
            if 'comparison_scores' in save_data:
                del save_data['comparison_scores']
            
            save_data['timestamp'] = self._get_timestamp()
            
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"✓ Knowledge base saved to {self.knowledge_base_path}")
        except Exception as e:
            print(f"✗ Error saving knowledge base: {e}")
    
    def _get_timestamp(self):
        """Get current timestamp as string."""
        return "2024-01-01_00:00:00"  # Fixed timestamp since we can't use datetime
    
    # ==================== DATA LOADING ====================
    
    def load_image(self, path):
        """Load image from JSON or standard image file."""
        self.current_metadata = None
        
        if path.lower().endswith('.json'):
            return self._load_from_json(path)
        else:
            img = cv2.imread(path)
            if img is None:
                print(f"✗ Could not read image: {path}")
                return None
            self.current_metadata = {'filename': os.path.basename(path)}
            return img
    
    def _load_from_json(self, json_path):
        """Load matrix from JSON file with bounds checking."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            width = data['image_dimensions']['width']
            height = data['image_dimensions']['height']
            channels = data['image_dimensions'].get('channels', 3)
            
            # Create image array
            matrix = np.zeros((height, width, channels), dtype=np.uint8)
            
            # Track out-of-bounds pixels
            oob_count = 0
            
            for pixel in data['pixels']:
                x = pixel['coordinates']['x']
                y = pixel['coordinates']['y']
                
                # Bounds checking
                if 0 <= x < width and 0 <= y < height:
                    bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                    if isinstance(bgr, (int, float)):
                        bgr = [bgr] * 3
                    matrix[y, x] = bgr[:3]
                else:
                    oob_count += 1
            
            if oob_count > 0:
                print(f"⚠ Warning: Skipped {oob_count} out-of-bounds pixels")
            
            self.current_metadata = {
                'filename': data.get('filename', os.path.basename(json_path)),
                'width': width,
                'height': height,
                'channels': channels,
                'json_path': json_path
            }
            
            return matrix
            
        except Exception as e:
            print(f"✗ Error loading JSON {json_path}: {e}")
            return None
    
    # ==================== STATISTICAL FUNCTIONS ====================
    
    def _compute_skewness(self, data):
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_entropy(self, data, bins=256):
        """Compute Shannon entropy."""
        hist, _ = np.histogram(data, bins=bins, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def _compute_correlation(self, x, y):
        """Compute Pearson correlation coefficient."""
        if len(x) < 2:
            return 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.mean((x - x_mean) * (y - y_mean))
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)
    
    def _compute_spearman_correlation(self, x, y):
        """Compute Spearman rank correlation."""
        if len(x) < 2:
            return 0.0
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))
        return self._compute_correlation(rank_x, rank_y)
    
    # ==================== FEATURE EXTRACTION ====================
    
    def extract_ultra_comprehensive_features(self, image):
        """Extract 100+ features using all available methods."""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        print("  Extracting features: ", end='', flush=True)
        
        # Extract all feature categories
        feature_extractors = [
            ("Stats", self._extract_statistical_features),
            ("Norms", self._extract_matrix_norms),
            ("LBP", self._extract_lbp_features),
            ("GLCM", self._extract_glcm_features),
            ("FFT", self._extract_fourier_features),
            ("MultiScale", self._extract_multiscale_features),
            ("Morph", self._extract_morphological_features),
            ("Shape", self._extract_shape_features),
            ("SVD", self._extract_svd_features),
            ("Entropy", self._extract_entropy_features),
            ("Gradient", self._extract_gradient_features),
            ("Topology", self._extract_topological_proxy_features),
        ]
        
        for name, extractor in feature_extractors:
            print(name, end='', flush=True)
            try:
                features.update(extractor(gray))
            except Exception as e:
                print(f"(✗:{e})", end='', flush=True)
            if name != "Topology":
                print("→", end='', flush=True)
        
        print(" ✓")
        
        feature_names = sorted(features.keys())
        return features, feature_names
    
    def _extract_statistical_features(self, gray):
        """Extract comprehensive statistical features."""
        flat = gray.flatten()
        percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
        
        return {
            'stat_mean': float(np.mean(gray)),
            'stat_std': float(np.std(gray)),
            'stat_variance': float(np.var(gray)),
            'stat_skew': float(self._compute_skewness(flat)),
            'stat_kurtosis': float(self._compute_kurtosis(flat)),
            'stat_min': float(np.min(gray)),
            'stat_max': float(np.max(gray)),
            'stat_range': float(np.max(gray) - np.min(gray)),
            'stat_median': float(np.median(gray)),
            'stat_mad': float(np.median(np.abs(gray - np.median(gray)))),
            'stat_iqr': float(percentiles[3] - percentiles[1]),
            'stat_entropy': float(self._compute_entropy(gray)),
            'stat_energy': float(np.sum(gray**2)),
            'stat_p10': float(percentiles[0]),
            'stat_p25': float(percentiles[1]),
            'stat_p50': float(percentiles[2]),
            'stat_p75': float(percentiles[3]),
            'stat_p90': float(percentiles[4]),
        }
    
    def _extract_matrix_norms(self, gray):
        """Extract various matrix norms."""
        return {
            'norm_frobenius': float(np.linalg.norm(gray, 'fro')),
            'norm_l1': float(np.sum(np.abs(gray))),
            'norm_l2': float(np.sqrt(np.sum(gray**2))),
            'norm_linf': float(np.max(np.abs(gray))),
            'norm_nuclear': float(np.sum(np.linalg.svd(gray, compute_uv=False))),
            'norm_trace': float(np.trace(gray)),
        }
    
    def _extract_lbp_features(self, gray):
        """Extract Local Binary Pattern features using custom implementation."""
        features = {}
        
        for radius in [1, 2, 3, 5]:
            # Simplified LBP implementation
            lbp = np.zeros_like(gray, dtype=np.float32)
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    # Shift image
                    shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                    
                    # Compare with center
                    lbp += (shifted >= gray).astype(np.float32)
            
            features[f'lbp_r{radius}_mean'] = float(np.mean(lbp))
            features[f'lbp_r{radius}_std'] = float(np.std(lbp))
            features[f'lbp_r{radius}_entropy'] = float(self._compute_entropy(lbp))
            features[f'lbp_r{radius}_energy'] = float(np.sum(lbp**2) / lbp.size)
        
        return features
    
    def _extract_glcm_features(self, gray):
        """Extract Gray-Level Co-occurrence Matrix features using custom implementation."""
        # Quantize image for faster computation
        img_q = (gray // 32).astype(np.uint8)
        levels = 8
        
        features = {}
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]  # degrees
        
        for dist in distances:
            for angle in angles:
                # Create GLCM
                glcm = np.zeros((levels, levels), dtype=np.float32)
                
                # Determine offset based on angle
                if angle == 0:
                    dy, dx = 0, dist
                elif angle == 45:
                    dy, dx = -dist, dist
                elif angle == 90:
                    dy, dx = -dist, 0
                else:  # 135
                    dy, dx = -dist, -dist
                
                # Build GLCM
                rows, cols = img_q.shape
                for i in range(rows):
                    for j in range(cols):
                        if 0 <= i + dy < rows and 0 <= j + dx < cols:
                            glcm[img_q[i, j], img_q[i + dy, j + dx]] += 1
                
                # Normalize
                glcm = glcm / (glcm.sum() + 1e-10)
                
                # Compute properties
                # Contrast
                contrast = 0
                for i in range(levels):
                    for j in range(levels):
                        contrast += ((i - j) ** 2) * glcm[i, j]
                
                # Energy
                energy = np.sum(glcm ** 2)
                
                # Homogeneity
                homogeneity = 0
                for i in range(levels):
                    for j in range(levels):
                        homogeneity += glcm[i, j] / (1 + abs(i - j))
                
                # Store features
                features[f'glcm_d{dist}_a{angle}_contrast'] = float(contrast)
                features[f'glcm_d{dist}_a{angle}_energy'] = float(energy)
                features[f'glcm_d{dist}_a{angle}_homogeneity'] = float(homogeneity)
        
        return features
    
    def _extract_fourier_features(self, gray):
        """Extract 2D Fourier Transform features."""
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        power = magnitude**2
        phase = np.angle(fshift)
        
        # Radial profile
        center = np.array(power.shape) // 2
        y, x = np.ogrid[:power.shape[0], :power.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Compute radial average
        radial_prof = []
        for radius in range(1, min(center)):
            mask = (r >= radius - 1) & (r < radius)
            if mask.any():
                radial_prof.append(np.mean(power[mask]))
        
        radial_prof = np.array(radial_prof)
        
        if len(radial_prof) > 0:
            spectral_centroid = float(np.sum(np.arange(len(radial_prof)) * radial_prof) / (np.sum(radial_prof) + 1e-10))
            spectral_spread = float(np.sqrt(np.sum((np.arange(len(radial_prof)) - spectral_centroid)**2 * radial_prof) / (np.sum(radial_prof) + 1e-10)))
        else:
            spectral_centroid = 0.0
            spectral_spread = 0.0
        
        return {
            'fft_mean_magnitude': float(np.mean(magnitude)),
            'fft_std_magnitude': float(np.std(magnitude)),
            'fft_max_magnitude': float(np.max(magnitude)),
            'fft_total_power': float(np.sum(power)),
            'fft_dc_component': float(magnitude[center[0], center[1]]),
            'fft_mean_phase': float(np.mean(phase)),
            'fft_std_phase': float(np.std(phase)),
            'fft_spectral_centroid': spectral_centroid,
            'fft_spectral_spread': spectral_spread,
        }
    
    def _extract_multiscale_features(self, gray):
        """Extract multi-scale features using Gaussian pyramids (replaces wavelets)."""
        features = {}
        
        # Create Gaussian pyramid
        pyramid = [gray]
        for i in range(3):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        
        # Compute features at each scale
        for level, img in enumerate(pyramid):
            # Basic statistics
            features[f'pyramid_L{level}_mean'] = float(np.mean(img))
            features[f'pyramid_L{level}_std'] = float(np.std(img))
            features[f'pyramid_L{level}_energy'] = float(np.sum(img**2))
            
            # Difference between scales (similar to wavelet details)
            if level > 0:
                # Upsample previous level
                upsampled = cv2.pyrUp(img)
                # Resize to match previous level
                h, w = pyramid[level-1].shape
                upsampled = cv2.resize(upsampled, (w, h))
                
                # Compute difference (detail coefficients)
                detail = pyramid[level-1].astype(np.float32) - upsampled.astype(np.float32)
                
                features[f'pyramid_detail_L{level}_energy'] = float(np.sum(detail**2))
                features[f'pyramid_detail_L{level}_mean'] = float(np.mean(np.abs(detail)))
                features[f'pyramid_detail_L{level}_std'] = float(np.std(detail))
        
        # Laplacian pyramid features
        for level in range(2):
            # Approximate Laplacian
            laplacian = cv2.Laplacian(pyramid[level], cv2.CV_64F)
            features[f'laplacian_L{level}_energy'] = float(np.sum(laplacian**2))
            features[f'laplacian_L{level}_mean'] = float(np.mean(np.abs(laplacian)))
        
        return features
    
    def _extract_morphological_features(self, gray):
        """Extract morphological features."""
        features = {}
        
        # Multi-scale morphological operations
        for size in [3, 5, 7, 11]:
            # Create circular kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            
            # White and black tophat
            wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            features[f'morph_wth_{size}_mean'] = float(np.mean(wth))
            features[f'morph_wth_{size}_max'] = float(np.max(wth))
            features[f'morph_wth_{size}_sum'] = float(np.sum(wth))
            features[f'morph_bth_{size}_mean'] = float(np.mean(bth))
            features[f'morph_bth_{size}_max'] = float(np.max(bth))
            features[f'morph_bth_{size}_sum'] = float(np.sum(bth))
        
        # Binary morphology
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(binary, kernel, iterations=1)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        
        features['morph_binary_area_ratio'] = float(np.sum(binary) / binary.size)
        features['morph_gradient_sum'] = float(np.sum(gradient))
        features['morph_erosion_ratio'] = float(np.sum(erosion) / (np.sum(binary) + 1e-10))
        features['morph_dilation_ratio'] = float(np.sum(dilation) / (np.sum(binary) + 1e-10))
        
        return features
    
    def _extract_shape_features(self, gray):
        """Extract shape features using Hu moments."""
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        features = {}
        for i, hu in enumerate(hu_moments):
            # Log transform for scale invariance
            features[f'shape_hu_{i}'] = float(-np.sign(hu) * np.log10(abs(hu) + 1e-10))
        
        # Additional moment features
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            features['shape_centroid_x'] = float(cx / gray.shape[1])  # Normalize
            features['shape_centroid_y'] = float(cy / gray.shape[0])
        
        return features
    
    def _extract_svd_features(self, gray):
        """Extract Singular Value Decomposition features."""
        try:
            _, s, _ = np.linalg.svd(gray, full_matrices=False)
            s_norm = s / (np.sum(s) + 1e-10)
            
            # Cumulative energy
            cumsum = np.cumsum(s_norm)
            n_components_90 = np.argmax(cumsum >= 0.9) + 1
            n_components_95 = np.argmax(cumsum >= 0.95) + 1
            
            return {
                'svd_largest': float(s[0]) if len(s) > 0 else 0.0,
                'svd_top5_ratio': float(np.sum(s_norm[:5])) if len(s) >= 5 else float(np.sum(s_norm)),
                'svd_top10_ratio': float(np.sum(s_norm[:10])) if len(s) >= 10 else float(np.sum(s_norm)),
                'svd_entropy': float(self._compute_entropy(s_norm * 1000)),  # Scale for entropy
                'svd_energy_ratio': float(s[0] / (s[1] + 1e-10)) if len(s) > 1 else 0.0,
                'svd_n_components_90': float(n_components_90),
                'svd_n_components_95': float(n_components_95),
                'svd_effective_rank': float(np.exp(self._compute_entropy(s_norm * 1000))),
            }
        except:
            return {f'svd_{k}': 0.0 for k in ['largest', 'top5_ratio', 'top10_ratio', 'entropy', 
                                               'energy_ratio', 'n_components_90', 'n_components_95', 'effective_rank']}
    
    def _extract_entropy_features(self, gray):
        """Extract various entropy measures."""
        # Global histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist_norm = hist / (hist.sum() + 1e-10)
        
        # Shannon entropy
        shannon = self._compute_entropy(gray)
        
        # Renyi entropy (alpha = 2)
        renyi = -np.log2(np.sum(hist_norm**2) + 1e-10)
        
        # Tsallis entropy (q = 2)
        tsallis = (1 - np.sum(hist_norm**2)) / 1
        
        # Local entropy using convolution
        # Create local histogram approximation
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
        
        # Compute local variance as proxy for local entropy
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel)
        local_var = local_sq_mean - local_mean**2
        local_ent = np.log2(local_var + 1)  # Proxy for entropy
        
        return {
            'entropy_shannon': float(shannon),
            'entropy_renyi': float(renyi),
            'entropy_tsallis': float(tsallis),
            'entropy_local_mean': float(np.mean(local_ent)),
            'entropy_local_std': float(np.std(local_ent)),
            'entropy_local_max': float(np.max(local_ent)),
            'entropy_local_min': float(np.min(local_ent)),
        }
    
    def _extract_gradient_features(self, gray):
        """Extract gradient-based features."""
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and orientation
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_orient = np.arctan2(grad_y, grad_x)
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Canny edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        return {
            'gradient_magnitude_mean': float(np.mean(grad_mag)),
            'gradient_magnitude_std': float(np.std(grad_mag)),
            'gradient_magnitude_max': float(np.max(grad_mag)),
            'gradient_magnitude_sum': float(np.sum(grad_mag)),
            'gradient_orientation_mean': float(np.mean(grad_orient)),
            'gradient_orientation_std': float(np.std(grad_orient)),
            'laplacian_mean': float(np.mean(np.abs(laplacian))),
            'laplacian_std': float(np.std(laplacian)),
            'laplacian_sum': float(np.sum(np.abs(laplacian))),
            'edge_density': float(edge_density),
            'edge_count': float(np.sum(edges > 0)),
        }
    
    def _extract_topological_proxy_features(self, gray):
        """Extract topological proxy features using connected components analysis."""
        features = {}
        
        # Use percentile thresholds
        thresholds = np.percentile(gray, np.linspace(5, 95, 20))
        
        # Track connected components (proxy for Betti 0)
        n_components = []
        for t in thresholds:
            binary = (gray >= t).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            n_components.append(num_labels - 1)  # Subtract background
        
        # Track holes (proxy for Betti 1)
        n_holes = []
        for t in thresholds:
            binary = (gray < t).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            n_holes.append(num_labels - 1)  # Subtract background
        
        # Compute statistics
        if len(n_components) > 1:
            persistence_b0 = np.diff(n_components)
            features['topo_b0_max_components'] = float(np.max(n_components))
            features['topo_b0_mean_components'] = float(np.mean(n_components))
            features['topo_b0_persistence_sum'] = float(np.sum(np.abs(persistence_b0)))
            features['topo_b0_persistence_max'] = float(np.max(np.abs(persistence_b0)))
        else:
            features['topo_b0_max_components'] = float(n_components[0]) if n_components else 0.0
            features['topo_b0_mean_components'] = float(n_components[0]) if n_components else 0.0
            features['topo_b0_persistence_sum'] = 0.0
            features['topo_b0_persistence_max'] = 0.0
        
        if len(n_holes) > 1:
            persistence_b1 = np.diff(n_holes)
            features['topo_b1_max_holes'] = float(np.max(n_holes))
            features['topo_b1_mean_holes'] = float(np.mean(n_holes))
            features['topo_b1_persistence_sum'] = float(np.sum(np.abs(persistence_b1)))
            features['topo_b1_persistence_max'] = float(np.max(np.abs(persistence_b1)))
        else:
            features['topo_b1_max_holes'] = float(n_holes[0]) if n_holes else 0.0
            features['topo_b1_mean_holes'] = float(n_holes[0]) if n_holes else 0.0
            features['topo_b1_persistence_sum'] = 0.0
            features['topo_b1_persistence_max'] = 0.0
        
        return features
    
    # ==================== COMPARISON METHODS ====================
    
    def compute_exhaustive_comparison(self, features1, features2):
        """Compute all possible comparison metrics between two feature sets."""
        # Convert to vectors
        keys = sorted(set(features1.keys()) & set(features2.keys()))
        if not keys:
            return {
                'euclidean_distance': float('inf'),
                'manhattan_distance': float('inf'),
                'chebyshev_distance': float('inf'),
                'cosine_distance': 1.0,
                'pearson_correlation': 0.0,
                'spearman_correlation': 0.0,
                'ks_statistic': 1.0,
                'kl_divergence': float('inf'),
                'js_divergence': 1.0,
                'chi_square': float('inf'),
                'wasserstein_distance': float('inf'),
                'feature_ssim': 0.0,
            }
        
        vec1 = np.array([features1[k] for k in keys])
        vec2 = np.array([features2[k] for k in keys])
        
        # Handle edge cases
        if len(vec1) == 0 or len(vec2) == 0:
            return self.compute_exhaustive_comparison({}, {})
        
        # Normalize to avoid scale issues
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        vec1_norm = vec1 / (norm1 + 1e-10)
        vec2_norm = vec2 / (norm2 + 1e-10)
        
        comparison = {}
        
        # Distance metrics
        comparison['euclidean_distance'] = float(np.linalg.norm(vec1 - vec2))
        comparison['manhattan_distance'] = float(np.sum(np.abs(vec1 - vec2)))
        comparison['chebyshev_distance'] = float(np.max(np.abs(vec1 - vec2)))
        comparison['cosine_distance'] = float(1 - np.dot(vec1_norm, vec2_norm))
        
        # Correlation measures
        comparison['pearson_correlation'] = float(self._compute_correlation(vec1, vec2))
        comparison['spearman_correlation'] = float(self._compute_spearman_correlation(vec1, vec2))
        
        # Statistical tests
        comparison['ks_statistic'] = float(self._compute_ks_statistic(vec1, vec2))
        
        # Information theoretic measures
        bins = min(30, len(vec1) // 2)
        if bins > 2:
            # Create normalized histograms
            min_val = min(vec1.min(), vec2.min())
            max_val = max(vec1.max(), vec2.max())
            
            hist1, bin_edges = np.histogram(vec1, bins=bins, range=(min_val, max_val))
            hist2, _ = np.histogram(vec2, bins=bin_edges)
            
            hist1 = hist1 / (hist1.sum() + 1e-10)
            hist2 = hist2 / (hist2.sum() + 1e-10)
            
            # KL divergence
            kl_div = 0
            for i in range(len(hist1)):
                if hist1[i] > 0:
                    kl_div += hist1[i] * np.log((hist1[i] + 1e-10) / (hist2[i] + 1e-10))
            comparison['kl_divergence'] = float(kl_div)
            
            # JS divergence
            m = 0.5 * (hist1 + hist2)
            js_div = 0.5 * sum(hist1[i] * np.log((hist1[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist1)) if hist1[i] > 0)
            js_div += 0.5 * sum(hist2[i] * np.log((hist2[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist2)) if hist2[i] > 0)
            comparison['js_divergence'] = float(js_div)
            
            # Chi-square distance
            chi_sq = 0.5 * np.sum(np.where(hist1 + hist2 > 0, (hist1 - hist2)**2 / (hist1 + hist2 + 1e-10), 0))
            comparison['chi_square'] = float(chi_sq)
        else:
            comparison['kl_divergence'] = float('inf')
            comparison['js_divergence'] = 1.0
            comparison['chi_square'] = float('inf')
        
        # Wasserstein distance (1D approximation)
        comparison['wasserstein_distance'] = float(self._compute_wasserstein_distance(vec1, vec2))
        
        # Feature SSIM
        mean1, mean2 = np.mean(vec1), np.mean(vec2)
        comparison['feature_ssim'] = float((2 * mean1 * mean2 + 1e-10) / (mean1**2 + mean2**2 + 1e-10))
        
        return comparison
    
    def _compute_ks_statistic(self, x, y):
        """Compute Kolmogorov-Smirnov statistic."""
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        # Combine and sort
        combined = np.concatenate([x_sorted, y_sorted])
        combined_sorted = np.sort(combined)
        
        # Compute CDFs
        max_diff = 0
        for val in combined_sorted:
            cdf_x = np.sum(x_sorted <= val) / len(x_sorted)
            cdf_y = np.sum(y_sorted <= val) / len(y_sorted)
            max_diff = max(max_diff, abs(cdf_x - cdf_y))
        
        return max_diff
    
    def _compute_wasserstein_distance(self, x, y):
        """Compute 1D Wasserstein distance."""
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        # Interpolate to same size
        n = max(len(x_sorted), len(y_sorted))
        x_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x_sorted)), x_sorted)
        y_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y_sorted)), y_sorted)
        
        return np.mean(np.abs(x_interp - y_interp))
    
    def compute_image_structural_comparison(self, img1, img2):
        """Compute structural similarity between images."""
        # Ensure same size
        if img1.shape != img2.shape:
            h, w = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # SSIM implementation
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(img1.astype(float), -1, window)
        mu2 = cv2.filter2D(img2.astype(float), -1, window)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1.astype(float)**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2.astype(float)**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1.astype(float) * img2.astype(float), -1, window) - mu1_mu2
        
        # SSIM components
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast = (2 * np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2) / (sigma1_sq + sigma2_sq + C2)
        structure = (sigma12 + C2/2) / (np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2/2)
        
        ssim_map = luminance * contrast * structure
        ssim_index = np.mean(ssim_map)
        
        # Multi-scale SSIM
        ms_ssim_values = [ssim_index]
        for scale in [2, 4]:
            img1_scaled = cv2.resize(img1, (img1.shape[1]//scale, img1.shape[0]//scale))
            img2_scaled = cv2.resize(img2, (img2.shape[1]//scale, img2.shape[0]//scale))
            
            # Simplified SSIM for other scales
            diff = np.abs(img1_scaled.astype(float) - img2_scaled.astype(float))
            ms_ssim = 1 - np.mean(diff) / 255
            ms_ssim_values.append(ms_ssim)
        
        return {
            'ssim': float(ssim_index),
            'ssim_map': ssim_map,
            'ms_ssim': ms_ssim_values,
            'luminance_map': luminance,
            'contrast_map': contrast,
            'structure_map': structure,
            'mean_luminance': float(np.mean(luminance)),
            'mean_contrast': float(np.mean(contrast)),
            'mean_structure': float(np.mean(structure)),
        }
    
    # ==================== REFERENCE MODEL BUILDING ====================
    
    def build_comprehensive_reference_model(self, ref_dir):
        """Build an exhaustive reference model from a directory of JSON/image files."""
        print(f"\n{'='*70}")
        print(f"Building Comprehensive Reference Model from: {ref_dir}")
        print(f"{'='*70}")
        
        # Find all valid files
        valid_extensions = ['.json', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        all_files = []
        
        for filename in os.listdir(ref_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                all_files.append(os.path.join(ref_dir, filename))
        
        if not all_files:
            print(f"✗ No valid files found in {ref_dir}")
            return False
        
        print(f"✓ Found {len(all_files)} files to process")
        
        # Process each file
        all_features = []
        all_images = []
        feature_names = []
        
        print("\nProcessing files:")
        for i, file_path in enumerate(all_files, 1):
            print(f"\n[{i}/{len(all_files)}] {os.path.basename(file_path)}")
            
            # Load image
            image = self.load_image(file_path)
            if image is None:
                print(f"  ✗ Failed to load")
                continue
            
            # Convert to grayscale for storage
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Extract features
            features, f_names = self.extract_ultra_comprehensive_features(image)
            
            if not feature_names:
                feature_names = f_names
            
            # Store
            all_features.append(features)
            all_images.append(gray)
            
            print(f"  ✓ Processed: {len(features)} features extracted")
        
        if not all_features:
            print("✗ No features could be extracted from any file")
            return False
        
        # Check minimum requirement
        if len(all_features) < 2:
            print(f"✗ Error: At least 2 reference files are required, but only {len(all_features)} were successfully processed.")
            return False
        
        print(f"\n{'='*70}")
        print(f"Building Statistical Model...")
        print(f"{'='*70}")
        
        # Convert features to matrix
        feature_matrix = np.zeros((len(all_features), len(feature_names)))
        for i, features in enumerate(all_features):
            for j, fname in enumerate(feature_names):
                feature_matrix[i, j] = features.get(fname, 0)
        
        # Build statistical model
        mean_vector = np.mean(feature_matrix, axis=0)
        std_vector = np.std(feature_matrix, axis=0)
        median_vector = np.median(feature_matrix, axis=0)
        
        # Robust statistics using custom implementation
        print("Computing robust statistics...")
        robust_mean, robust_cov, robust_inv_cov = self._compute_robust_statistics(feature_matrix)
        
        # Create archetype image (median of all images)
        print("Creating archetype image...")
        target_shape = all_images[0].shape
        aligned_images = []
        for img in all_images:
            if img.shape != target_shape:
                img = cv2.resize(img, (target_shape[1], target_shape[0]))
            aligned_images.append(img)
        
        archetype_image = np.median(aligned_images, axis=0).astype(np.uint8)
        
        # Compute pairwise comparisons for threshold learning
        print("\nComputing pairwise comparisons for threshold learning...")
        n_comparisons = len(all_features) * (len(all_features) - 1) // 2
        print(f"Total comparisons to compute: {n_comparisons}")
        
        comparison_scores = []
        comparison_count = 0
        
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                comp = self.compute_exhaustive_comparison(all_features[i], all_features[j])
                
                # Compute anomaly score
                score = (comp['euclidean_distance'] * 0.2 +
                        comp['manhattan_distance'] * 0.1 +
                        comp['cosine_distance'] * 0.2 +
                        (1 - abs(comp['pearson_correlation'])) * 0.1 +
                        min(comp['kl_divergence'], 10.0) * 0.1 +
                        comp['js_divergence'] * 0.1 +
                        min(comp['chi_square'], 10.0) * 0.1 +
                        min(comp['wasserstein_distance'], 10.0) * 0.1)
                
                comparison_scores.append(score)
                comparison_count += 1
                
                if comparison_count % 100 == 0:
                    print(f"  Progress: {comparison_count}/{n_comparisons} ({comparison_count/n_comparisons*100:.1f}%)")
        
        # Learn thresholds
        scores_array = np.array(comparison_scores)
        
        if len(scores_array) > 0 and not np.all(np.isnan(scores_array)):
            valid_scores = scores_array[~np.isnan(scores_array)]
            if len(valid_scores) > 0:
                thresholds = {
                    'anomaly_mean': float(np.mean(valid_scores)),
                    'anomaly_std': float(np.std(valid_scores)),
                    'anomaly_p90': float(np.percentile(valid_scores, 90)),
                    'anomaly_p95': float(np.percentile(valid_scores, 95)),
                    'anomaly_p99': float(np.percentile(valid_scores, 99)),
                    'anomaly_threshold': float(max(np.mean(valid_scores) + 2.5 * np.std(valid_scores), 1e-6)),
                }
            else:
                thresholds = self._get_default_thresholds()
        else:
            thresholds = self._get_default_thresholds()
        
        # Store reference model
        self.reference_model = {
            'features': all_features,
            'feature_names': feature_names,
            'statistical_model': {
                'mean': mean_vector,
                'std': std_vector,
                'median': median_vector,
                'robust_mean': robust_mean,
                'robust_cov': robust_cov,
                'robust_inv_cov': robust_inv_cov,
                'n_samples': len(all_features),
            },
            'archetype_image': archetype_image,
            'learned_thresholds': thresholds,
            'timestamp': self._get_timestamp(),
        }
        
        # Save model
        self.save_knowledge_base()
        
        print(f"\n{'='*70}")
        print("✓ Reference Model Built Successfully!")
        print(f"  - Samples: {len(all_features)}")
        print(f"  - Features: {len(feature_names)}")
        print(f"  - Anomaly threshold: {thresholds['anomaly_threshold']:.4f}")
        print(f"{'='*70}\n")
        
        return True
    
    def _compute_robust_statistics(self, data):
        """Compute robust mean and covariance using custom implementation."""
        n_samples, n_features = data.shape
        
        # Use median as initial robust mean
        robust_mean = np.median(data, axis=0)
        
        # Compute MAD (Median Absolute Deviation) based covariance
        deviations = data - robust_mean
        mad = np.median(np.abs(deviations), axis=0)
        
        # Scale MAD to approximate standard deviation
        mad_scaled = mad * 1.4826
        
        # Compute robust covariance
        # Weight samples by their distance from median
        distances = np.sqrt(np.sum((deviations / (mad_scaled + 1e-10))**2, axis=1))
        weights = np.exp(-0.5 * distances)
        weights = weights / weights.sum()
        
        # Weighted covariance
        weighted_deviations = deviations * weights[:, np.newaxis]
        robust_cov = np.dot(weighted_deviations.T, deviations) / (weights.sum() - 1)
        
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        robust_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Compute pseudo-inverse
        robust_inv_cov = np.linalg.pinv(robust_cov + np.eye(n_features) * 1e-6)
        
        return robust_mean, robust_cov, robust_inv_cov
    
    def _get_default_thresholds(self):
        """Return default thresholds when learning fails."""
        return {
            'anomaly_mean': 1.0,
            'anomaly_std': 0.5,
            'anomaly_p90': 1.5,
            'anomaly_p95': 2.0,
            'anomaly_p99': 3.0,
            'anomaly_threshold': 2.5,
        }
    
    # ==================== ANOMALY DETECTION ====================
    
    def detect_anomalies_comprehensive(self, test_path):
        """Perform exhaustive anomaly detection on a test image."""
        print(f"\n{'='*70}")
        print(f"Analyzing: {test_path}")
        print(f"{'='*70}")
        
        # Check reference model
        if not self.reference_model.get('statistical_model'):
            print("✗ No reference model available. Build one first.")
            return None
        
        # Load test image
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        print(f"✓ Loaded image: {self.current_metadata}")
        
        # Convert to grayscale
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Extract features
        print("\nExtracting features from test image...")
        test_features, _ = self.extract_ultra_comprehensive_features(test_image)
        
        # --- Global Analysis ---
        print("\nPerforming global anomaly analysis...")
        
        # Get reference statistics
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        
        # Create feature vector
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        
        # Compute Mahalanobis distance
        diff = test_vector - stat_model['robust_mean']
        mahalanobis_dist = np.sqrt(diff.T @ stat_model['robust_inv_cov'] @ diff)
        
        # Compute Z-scores
        z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
        
        # Find most deviant features
        top_indices = np.argsort(z_scores)[::-1][:10]
        deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                           for i in top_indices]
        
        # --- Individual Comparisons ---
        print(f"\nComparing against {len(self.reference_model['features'])} reference samples...")
        
        individual_scores = []
        for i, ref_features in enumerate(self.reference_model['features']):
            comp = self.compute_exhaustive_comparison(test_features, ref_features)
            
            # Compute anomaly score
            score = (comp['euclidean_distance'] * 0.2 +
                    comp['manhattan_distance'] * 0.1 +
                    comp['cosine_distance'] * 0.2 +
                    (1 - abs(comp['pearson_correlation'])) * 0.1 +
                    min(comp['kl_divergence'], 10.0) * 0.1 +
                    comp['js_divergence'] * 0.1 +
                    min(comp['chi_square'], 10.0) * 0.1 +
                    min(comp['wasserstein_distance'], 10.0) * 0.1)
            
            individual_scores.append(score)
        
        # Statistics of individual comparisons
        scores_array = np.array(individual_scores)
        comparison_stats = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
        }
        
        # --- Structural Analysis ---
        print("\nPerforming structural analysis...")
        
        # Compare with archetype
        archetype = self.reference_model['archetype_image']
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
        else:
            test_gray_resized = test_gray
        
        structural_comp = self.compute_image_structural_comparison(test_gray_resized, archetype)
        
        # --- Local Anomaly Detection ---
        print("\nDetecting local anomalies...")
        
        # Sliding window analysis
        anomaly_map = self._compute_local_anomaly_map(test_gray_resized, archetype)
        
        # Find anomaly regions
        anomaly_regions = self._find_anomaly_regions(anomaly_map, test_gray.shape)
        
        # --- Specific Defect Detection ---
        print("\nDetecting specific defects...")
        specific_defects = self._detect_specific_defects(test_gray)
        
        # --- Determine Overall Status ---
        thresholds = self.reference_model['learned_thresholds']
        
        # Multiple criteria for anomaly detection
        is_anomalous = (
            mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6) or
            comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6) or
            structural_comp['ssim'] < 0.7 or
            len(anomaly_regions) > 3 or
            any(region['confidence'] > 0.8 for region in anomaly_regions)
        )
        
        # Overall confidence
        confidence = min(1.0, max(
            mahalanobis_dist / max(thresholds['anomaly_threshold'], 1e-6),
            comparison_stats['max'] / max(thresholds['anomaly_p95'], 1e-6),
            1 - structural_comp['ssim'],
            len(anomaly_regions) / 10
        ))
        
        print(f"\n{'='*70}")
        print("Analysis Complete!")
        print(f"{'='*70}\n")
        
        return {
            'test_image': test_image,
            'test_gray': test_gray,
            'test_features': test_features,
            'metadata': self.current_metadata,
            
            'global_analysis': {
                'mahalanobis_distance': float(mahalanobis_dist),
                'deviant_features': deviant_features,
                'comparison_stats': comparison_stats,
            },
            
            'structural_analysis': structural_comp,
            
            'local_analysis': {
                'anomaly_map': anomaly_map,
                'anomaly_regions': anomaly_regions,
            },
            
            'specific_defects': specific_defects,
            
            'verdict': {
                'is_anomalous': is_anomalous,
                'confidence': float(confidence),
                'criteria_triggered': {
                    'mahalanobis': mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6),
                    'comparison': comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6),
                    'structural': structural_comp['ssim'] < 0.7,
                    'local': len(anomaly_regions) > 3,
                }
            }
        }
    
    def _compute_local_anomaly_map(self, test_img, reference_img):
        """Compute local anomaly map using sliding window."""
        h, w = test_img.shape
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        
        window_sizes = [16, 32, 64]
        
        for win_size in window_sizes:
            stride = win_size // 2
            
            for y in range(0, h - win_size + 1, stride):
                for x in range(0, w - win_size + 1, stride):
                    # Extract windows
                    test_win = test_img[y:y+win_size, x:x+win_size]
                    ref_win = reference_img[y:y+win_size, x:x+win_size]
                    
                    # Compute local difference
                    diff = np.abs(test_win.astype(float) - ref_win.astype(float))
                    local_score = np.mean(diff) / 255.0
                    
                    # Simple SSIM approximation for window
                    mean_test = np.mean(test_win)
                    mean_ref = np.mean(ref_win)
                    var_test = np.var(test_win)
                    var_ref = np.var(ref_win)
                    cov = np.mean((test_win - mean_test) * (ref_win - mean_ref))
                    
                    c1 = 0.01**2 * 255**2
                    c2 = 0.03**2 * 255**2
                    
                    ssim_approx = ((2 * mean_test * mean_ref + c1) * (2 * cov + c2)) / \
                                  ((mean_test**2 + mean_ref**2 + c1) * (var_test + var_ref + c2))
                    
                    local_score = max(local_score, 1 - ssim_approx)
                    
                    # Update map
                    anomaly_map[y:y+win_size, x:x+win_size] = np.maximum(
                        anomaly_map[y:y+win_size, x:x+win_size],
                        local_score
                    )
        
        # Normalize and smooth
        anomaly_map = cv2.GaussianBlur(anomaly_map, (15, 15), 0)
        
        return anomaly_map
    
    def _find_anomaly_regions(self, anomaly_map, original_shape):
        """Find distinct anomaly regions from the anomaly map."""
        # Check if there are any positive values
        positive_values = anomaly_map[anomaly_map > 0]
        if positive_values.size == 0:
            return []
        
        # Threshold the map
        threshold = np.percentile(positive_values, 80)
        binary_map = (anomaly_map > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        
        regions = []
        h_scale = original_shape[0] / anomaly_map.shape[0]
        w_scale = original_shape[1] / anomaly_map.shape[1]
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Scale to original image size
            x_orig = int(x * w_scale)
            y_orig = int(y * h_scale)
            w_orig = int(w * w_scale)
            h_orig = int(h * h_scale)
            
            # Compute confidence
            region_mask = (labels == i)
            region_values = anomaly_map[region_mask]
            confidence = float(np.mean(region_values))
            
            if area > 20:  # Filter tiny regions
                regions.append({
                    'bbox': (x_orig, y_orig, w_orig, h_orig),
                    'area': int(area * h_scale * w_scale),
                    'confidence': confidence,
                    'centroid': (int(centroids[i][0] * w_scale), int(centroids[i][1] * h_scale)),
                    'max_intensity': float(np.max(region_values)),
                })
        
        # Sort by confidence
        regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return regions
    
    def _detect_specific_defects(self, gray):
        """Detect specific types of defects."""
        defects = {
            'scratches': [],
            'digs': [],
            'blobs': [],
            'edges': [],
        }
        
        # Scratch detection (linear features)
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                               minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 25:
                    defects['scratches'].append({
                        'line': (x1, y1, x2, y2),
                        'length': float(length),
                        'angle': float(np.arctan2(y2-y1, x2-x1) * 180 / np.pi),
                    })
        
        # Dig detection (small dark spots)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, dig_mask = cv2.threshold(bth, np.percentile(bth, 95), 255, cv2.THRESH_BINARY)
        
        dig_contours, _ = cv2.findContours(dig_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in dig_contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    defects['digs'].append({
                        'center': (cx, cy),
                        'area': float(area),
                        'contour': contour,
                    })
        
        # Blob detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        blob_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in blob_contours:
            area = cv2.contourArea(contour)
            if area > 100:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-10)
                
                defects['blobs'].append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': float(area),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                })
        
        # Edge irregularities
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        edge_thresh = np.percentile(grad_mag, 95)
        edge_mask = (grad_mag > edge_thresh).astype(np.uint8)
        
        edge_contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in edge_contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:
                defects['edges'].append({
                    'contour': contour,
                    'area': float(area),
                })
        
        return defects
    
    # ==================== VISUALIZATION ====================
    
    def visualize_comprehensive_results(self, results, output_path):
        """Create comprehensive visualization of all anomaly detection results."""
        fig = plt.figure(figsize=(24, 16))
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Get images
        test_img = results['test_image']
        if len(test_img.shape) == 3:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        else:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        
        archetype = self.reference_model['archetype_image']
        archetype_rgb = cv2.cvtColor(archetype, cv2.COLOR_GRAY2RGB)
        
        # Panel 1: Original Test Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(test_img_rgb)
        ax1.set_title('Test Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Panel 2: Reference Archetype
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(archetype_rgb)
        ax2.set_title('Reference Archetype', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Panel 3: SSIM Map
        ax3 = fig.add_subplot(gs[0, 2])
        ssim_map = results['structural_analysis']['ssim_map']
        im3 = ax3.imshow(ssim_map, cmap='RdYlBu', vmin=0, vmax=1)
        ax3.set_title(f'SSIM Map (Index: {results["structural_analysis"]["ssim"]:.3f})', 
                     fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Panel 4: Local Anomaly Heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        anomaly_map = results['local_analysis']['anomaly_map']
        
        # Resize anomaly map to match test image
        if anomaly_map.shape != test_img_rgb.shape[:2]:
            anomaly_map_resized = cv2.resize(anomaly_map, 
                                            (test_img_rgb.shape[1], test_img_rgb.shape[0]))
        else:
            anomaly_map_resized = anomaly_map
        
        ax4.imshow(test_img_rgb, alpha=0.7)
        im4 = ax4.imshow(anomaly_map_resized, cmap='hot', alpha=0.5, vmin=0)
        ax4.set_title('Local Anomaly Heatmap', fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # Panel 5: Detected Anomalies (Blue Highlights)
        ax5 = fig.add_subplot(gs[1, :2])
        overlay = test_img_rgb.copy()
        
        # Draw anomaly regions in blue
        for region in results['local_analysis']['anomaly_regions']:
            x, y, w, h = region['bbox']
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Fill with semi-transparent blue
            roi = overlay[y:y+h, x:x+w]
            blue_overlay = np.zeros_like(roi)
            blue_overlay[:, :] = [0, 0, 255]
            cv2.addWeighted(roi, 0.7, blue_overlay, 0.3, 0, roi)
            
            # Add confidence text
            cv2.putText(overlay, f'{region["confidence"]:.2f}', 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        ax5.imshow(overlay)
        ax5.set_title(f'Detected Anomalies ({len(results["local_analysis"]["anomaly_regions"])} regions)', 
                     fontsize=16, fontweight='bold', color='blue')
        ax5.axis('off')
        
        # Panel 6: Specific Defects
        ax6 = fig.add_subplot(gs[1, 2:])
        defect_overlay = test_img_rgb.copy()
        
        # Draw specific defects
        defects = results['specific_defects']
        
        # Scratches - cyan lines
        for scratch in defects['scratches']:
            x1, y1, x2, y2 = scratch['line']
            cv2.line(defect_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Digs - magenta circles
        for dig in defects['digs']:
            cx, cy = dig['center']
            radius = int(np.sqrt(dig['area'] / np.pi))
            cv2.circle(defect_overlay, (cx, cy), max(3, radius), (255, 0, 255), -1)
        
        # Blobs - yellow contours
        cv2.drawContours(defect_overlay, [b['contour'] for b in defects['blobs']], 
                        -1, (255, 255, 0), 2)
        
        # Edges - green contours
        cv2.drawContours(defect_overlay, [e['contour'] for e in defects['edges']], 
                        -1, (0, 255, 0), 1)
        
        ax6.imshow(defect_overlay)
        defect_counts = (f"Scratches: {len(defects['scratches'])}, " 
                        f"Digs: {len(defects['digs'])}, "
                        f"Blobs: {len(defects['blobs'])}, "
                        f"Edges: {len(defects['edges'])}")
        ax6.set_title(f'Specific Defects\n{defect_counts}', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        # Panel 7: Feature Deviation Chart
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Get top deviations
        deviations = results['global_analysis']['deviant_features'][:8]
        names = [d[0].replace('_', '\n') for d in deviations]
        z_scores = [d[1] for d in deviations]
        
        colors = ['red' if z > 3 else 'orange' if z > 2 else 'yellow' for z in z_scores]
        
        bars = ax7.barh(names, z_scores, color=colors)
        ax7.set_xlabel('Z-Score (Standard Deviations from Reference)', fontsize=12)
        ax7.set_title('Most Deviant Features', fontsize=14, fontweight='bold')
        ax7.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='2σ threshold')
        ax7.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3σ threshold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, z in zip(bars, z_scores):
            width = bar.get_width()
            ax7.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{z:.1f}', va='center', fontsize=10)
        
        # Panel 8: Analysis Summary
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        
        # Prepare summary text
        verdict = results['verdict']
        global_stats = results['global_analysis']
        structural = results['structural_analysis']
        
        summary_text = f"""COMPREHENSIVE ANALYSIS SUMMARY
        
Overall Verdict: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}
Confidence: {verdict['confidence']:.1%}

Global Analysis:
• Mahalanobis Distance: {global_stats['mahalanobis_distance']:.2f}
• Max Comparison Score: {global_stats['comparison_stats']['max']:.3f}
• Mean Comparison Score: {global_stats['comparison_stats']['mean']:.3f}

Structural Analysis:
• SSIM Index: {structural['ssim']:.3f}
• Mean Luminance Similarity: {structural['mean_luminance']:.3f}
• Mean Contrast Similarity: {structural['mean_contrast']:.3f}
• Mean Structure Similarity: {structural['mean_structure']:.3f}

Local Analysis:
• Anomaly Regions Found: {len(results['local_analysis']['anomaly_regions'])}
• Max Region Confidence: {max([r['confidence'] for r in results['local_analysis']['anomaly_regions']], default=0):.3f}

Criteria Triggered:
• Mahalanobis: {'✓' if verdict['criteria_triggered']['mahalanobis'] else '✗'}
• Comparison: {'✓' if verdict['criteria_triggered']['comparison'] else '✗'}
• Structural: {'✓' if verdict['criteria_triggered']['structural'] else '✗'}
• Local: {'✓' if verdict['criteria_triggered']['local'] else '✗'}"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Main title
        source_name = results['metadata'].get('filename', 'Unknown')
        fig.suptitle(f'Ultra-Comprehensive Anomaly Analysis\nTest: {source_name}', 
                    fontsize=20, fontweight='bold')
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Visualization saved to: {output_path}")
        
        # Also save a simplified version
        self._save_simple_anomaly_image(results, output_path.replace('.png', '_simple.png'))
    
    def _save_simple_anomaly_image(self, results, output_path):
        """Save a simple image with just anomalies highlighted in blue."""
        test_img = results['test_image'].copy()
        
        # Create blue overlay for anomaly regions
        for region in results['local_analysis']['anomaly_regions']:
            x, y, w, h = region['bbox']
            
            # Draw blue rectangle
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # Fill with semi-transparent blue
            overlay = test_img.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, test_img, 0.7, 0, test_img)
        
        # Draw specific defects
        defects = results['specific_defects']
        
        # All defects in blue
        for scratch in defects['scratches']:
            x1, y1, x2, y2 = scratch['line']
            cv2.line(test_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        for dig in defects['digs']:
            cx, cy = dig['center']
            radius = max(3, int(np.sqrt(dig['area'] / np.pi)))
            cv2.circle(test_img, (cx, cy), radius, (255, 0, 0), -1)
        
        cv2.drawContours(test_img, [b['contour'] for b in defects['blobs']], 
                        -1, (255, 0, 0), 2)
        
        # Add verdict text
        verdict = "ANOMALOUS" if results['verdict']['is_anomalous'] else "NORMAL"
        confidence = results['verdict']['confidence']
        
        cv2.putText(test_img, f"{verdict} ({confidence:.1%})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imwrite(output_path, test_img)
        print(f"✓ Simple anomaly image saved to: {output_path}")
    
    # ==================== REPORT GENERATION ====================
    
    def generate_detailed_report(self, results, output_path):
        """Generate a detailed text report of the analysis."""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ULTRA-COMPREHENSIVE ANOMALY DETECTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # File information
            f.write("FILE INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Test File: {results['metadata'].get('filename', 'Unknown')}\n")
            f.write(f"Analysis Date: {self._get_timestamp()}\n")
            f.write(f"Image Dimensions: {results['test_gray'].shape}\n")
            f.write("\n")
            
            # Overall verdict
            f.write("OVERALL VERDICT\n")
            f.write("-"*40 + "\n")
            verdict = results['verdict']
            f.write(f"Status: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}\n")
            f.write(f"Confidence: {verdict['confidence']:.1%}\n")
            f.write("\n")
            
            # Global analysis
            f.write("GLOBAL STATISTICAL ANALYSIS\n")
            f.write("-"*40 + "\n")
            global_stats = results['global_analysis']
            f.write(f"Mahalanobis Distance: {global_stats['mahalanobis_distance']:.4f}\n")
            f.write(f"Comparison Scores:\n")
            f.write(f"  - Mean: {global_stats['comparison_stats']['mean']:.4f}\n")
            f.write(f"  - Std: {global_stats['comparison_stats']['std']:.4f}\n")
            f.write(f"  - Min: {global_stats['comparison_stats']['min']:.4f}\n")
            f.write(f"  - Max: {global_stats['comparison_stats']['max']:.4f}\n")
            f.write("\n")
            
            # Top deviant features
            f.write("TOP DEVIANT FEATURES (Z-Score > 2)\n")
            f.write("-"*40 + "\n")
            for fname, z_score, test_val, ref_val in global_stats['deviant_features'][:10]:
                if z_score > 2:
                    f.write(f"{fname:30} Z={z_score:6.2f}  Test={test_val:10.4f}  Ref={ref_val:10.4f}\n")
            f.write("\n")
            
            # Structural analysis
            f.write("STRUCTURAL ANALYSIS\n")
            f.write("-"*40 + "\n")
            structural = results['structural_analysis']
            f.write(f"SSIM Index: {structural['ssim']:.4f}\n")
            f.write(f"Luminance Similarity: {structural['mean_luminance']:.4f}\n")
            f.write(f"Contrast Similarity: {structural['mean_contrast']:.4f}\n")
            f.write(f"Structure Similarity: {structural['mean_structure']:.4f}\n")
            f.write("\n")
            
            # Local anomalies
            f.write("LOCAL ANOMALY REGIONS\n")
            f.write("-"*40 + "\n")
            regions = results['local_analysis']['anomaly_regions']
            f.write(f"Total Regions Found: {len(regions)}\n")
            for i, region in enumerate(regions[:5], 1):
                f.write(f"\nRegion {i}:\n")
                f.write(f"  - Location: {region['bbox']}\n")
                f.write(f"  - Area: {region['area']} pixels\n")
                f.write(f"  - Confidence: {region['confidence']:.3f}\n")
                f.write(f"  - Centroid: {region['centroid']}\n")
            if len(regions) > 5:
                f.write(f"\n... and {len(regions) - 5} more regions\n")
            f.write("\n")
            
            # Specific defects
            f.write("SPECIFIC DEFECTS DETECTED\n")
            f.write("-"*40 + "\n")
            defects = results['specific_defects']
            f.write(f"Scratches: {len(defects['scratches'])}\n")
            f.write(f"Digs: {len(defects['digs'])}\n")
            f.write(f"Blobs: {len(defects['blobs'])}\n")
            f.write(f"Edge Irregularities: {len(defects['edges'])}\n")
            f.write("\n")
            
            # Criteria summary
            f.write("ANOMALY CRITERIA SUMMARY\n")
            f.write("-"*40 + "\n")
            criteria = verdict['criteria_triggered']
            f.write(f"Mahalanobis Threshold Exceeded: {'Yes' if criteria['mahalanobis'] else 'No'}\n")
            f.write(f"Comparison Threshold Exceeded: {'Yes' if criteria['comparison'] else 'No'}\n")
            f.write(f"Low Structural Similarity: {'Yes' if criteria['structural'] else 'No'}\n")
            f.write(f"Multiple Local Anomalies: {'Yes' if criteria['local'] else 'No'}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Detailed report saved to: {output_path}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("ULTRA-COMPREHENSIVE MATRIX ANOMALY DETECTION SYSTEM (v3.0 - Optimized)".center(80))
    print("="*80)
    print("\nThis optimized system performs exhaustive comparative analysis")
    print("using only: json, os, cv2, matplotlib.pyplot, and numpy")
    print("\nNOTE: Minimum 2 reference files required for statistical analysis.\n")
    
    analyzer = UltraComprehensiveMatrixAnalyzer()
    
    # Step 1: Get reference directory
    while True:
        ref_dir = input("Enter path to folder containing reference JSON/image files: ").strip()
        ref_dir = ref_dir.strip('"\'')  # Remove quotes if pasted
        
        if os.path.isdir(ref_dir):
            break
        else:
            print(f"✗ Directory not found: {ref_dir}")
            print("Please enter a valid directory path.\n")
    
    # Build reference model
    if not analyzer.build_comprehensive_reference_model(ref_dir):
        print("✗ Failed to build reference model. Exiting.")
        return
    
    # Step 2: Analyze test images
    while True:
        print("\n" + "-"*80)
        test_path = input("\nEnter path to test image/JSON file (or 'quit' to exit): ").strip()
        test_path = test_path.strip('"\'')  # Remove quotes
        
        if test_path.lower() == 'quit':
            break
        
        if not os.path.isfile(test_path):
            print(f"✗ File not found: {test_path}")
            continue
        
        # Perform analysis
        print("\nStarting analysis...")
        results = analyzer.detect_anomalies_comprehensive(test_path)
        
        if results:
            print(f"\n✓ Analysis completed!")
            
            # Print summary
            print("\n" + "="*70)
            print("ANALYSIS SUMMARY")
            print("="*70)
            
            verdict = results['verdict']
            print(f"Status: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}")
            print(f"Confidence: {verdict['confidence']:.1%}")
            print(f"Mahalanobis Distance: {results['global_analysis']['mahalanobis_distance']:.2f}")
            print(f"SSIM Score: {results['structural_analysis']['ssim']:.3f}")
            print(f"Anomaly Regions: {len(results['local_analysis']['anomaly_regions'])}")
            
            defects = results['specific_defects']
            print(f"Specific Defects: {len(defects['scratches'])} scratches, "
                  f"{len(defects['digs'])} digs, {len(defects['blobs'])} blobs")
            
            # Generate outputs
            base_name = os.path.splitext(os.path.basename(test_path))[0]
            
            # Visualization
            viz_path = f"anomaly_analysis_{base_name}.png"
            analyzer.visualize_comprehensive_results(results, viz_path)
            
            # Report
            report_path = f"anomaly_report_{base_name}.txt"
            analyzer.generate_detailed_report(results, report_path)
            
            print(f"\n✓ Results saved:")
            print(f"  - Visualization: {viz_path}")
            print(f"  - Simple image: {viz_path.replace('.png', '_simple.png')}")
            print(f"  - Report: {report_path}")
        else:
            print("✗ Analysis failed.")
    
    print("\n" + "="*80)
    print("Thank you for using the Ultra-Comprehensive Anomaly Detection System!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
