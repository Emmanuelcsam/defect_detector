#!/usr/bin/env python3
"""
Advanced Matrix Anomaly Detection System
-----------------------------------------

This script provides a comprehensive framework for detecting anomalies in image-derived
matrices, such as those from industrial inspection, medical imaging, or material science.
It combines and enhances traditional image analysis methods with advanced techniques inspired
by contemporary research, including Topological Data Analysis (TDA).

Key Features:
1.  Dual-Mode Operation:
    - Supervised Mode: Compares a test image against a "golden standard" derived from a
      directory of reference (normal) images to detect deviations.
    - Unsupervised Mode: Analyzes a single image for internal inconsistencies and outliers
      without needing a reference set. This makes the "golden standard" optional.

2.  Comprehensive Feature Engineering:
    - Statistical Moments: Mean, standard deviation, skewness, kurtosis.
    - Texture Analysis: Gray-Level Co-occurrence Matrix (GLCM), Local Binary Patterns (LBP).
    - Frequency Domain: 2D Fourier Transform features (e.g., spectral centroid, flatness).
    - Wavelet Analysis: Multi-resolution analysis using Discrete Wavelet Transform (DWT).
    - Morphological Properties: White/Black Top-hat transforms to find small bright/dark features.
    - Invariant Moments: Hu moments for rotation-invariant shape description.
    - Singular Value Decomposition (SVD): Features derived from the image's singular values.
    - NEW - Persistent Homology (Topological Data Analysis):
      Calculates topological features (Betti numbers) to quantify holes and connected components,
      which is highly effective for detecting structural defects like voids, cracks, or porosity.

3.  Flexible Input:
    - Supports JSON matrix files in the format produced by `multi_matrix.py`.
    - Supports standard image formats (e.g., PNG, JPG, BMP, TIFF).

4.  Advanced Anomaly Quantification:
    - In supervised mode, uses Mahalanobis distance for robust multivariate anomaly scoring.
    - In unsupervised mode, uses a sliding window to create a local anomaly map based on
      deviations from the image's global characteristics.
    - Employs the Wasserstein (Earth Mover's) distance for comparing feature distributions.

5.  Rich Visualization:
    - Generates a multi-panel plot showing the original image, a heatmap of the anomaly map,
      and an overlay of detected defects (scratches, blobs).

Dependencies:
- numpy
- opencv-python
- scikit-image
- scikit-learn
- scipy
- matplotlib
- pywavelets

You can install them using pip:
pip install numpy opencv-python scikit-image scikit-learn scipy matplotlib pywavelets
"""

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.ndimage import label, find_objects
from scipy.stats import wasserstein_distance
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import moments_hu
from skimage.morphology import disk, white_tophat, black_tophat

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=FutureWarning, module='skimage')


class UniversalMatrixAnalyzer:
    """
    A universal analyzer combining statistical, topological, and machine learning methods
    for both supervised and unsupervised anomaly detection in images.
    """

    def __init__(self, knowledge_base_path="analysis_knowledge_base.pkl"):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base: Dict[str, Any] = {
            'golden_standard_features': None,
            'feature_names': []
        }
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load a previously saved knowledge base."""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    self.knowledge_base = pickle.load(f)
                print(f"Successfully loaded knowledge base from {self.knowledge_base_path}")
            except Exception as e:
                print(f"Warning: Could not load knowledge base. Error: {e}")

    def save_knowledge_base(self):
        """Save the current knowledge base to a file."""
        try:
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.knowledge_base, f)
            print(f"Knowledge base saved to {self.knowledge_base_path}")
        except Exception as e:
            print(f"Error: Could not save knowledge base. Error: {e}")

    # --- Data Loading and Preprocessing ---

    def load_image(self, path: str) -> Optional[np.ndarray]:
        """Loads an image from either a JSON matrix file or a standard image file."""
        if path.lower().endswith('.json'):
            return self._load_from_json(path)
        elif any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
            img = cv2.imread(path)
            if img is None:
                print(f"Error: Could not read image file at {path}")
                return None
            return img
        else:
            print(f"Error: Unsupported file format for {path}")
            return None

    def _load_from_json(self, json_path: str) -> Optional[np.ndarray]:
        """Load matrix data from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            width = data['image_dimensions']['width']
            height = data['image_dimensions']['height']
            channels = data['image_dimensions'].get('channels', 3)
            
            matrix = np.zeros((height, width, channels), dtype=np.uint8)
            
            for pixel in data['pixels']:
                x = pixel['coordinates']['x']
                y = pixel['coordinates']['y']
                
                if 'bgr_intensity' in pixel:
                    bgr = pixel['bgr_intensity']
                    matrix[y, x] = bgr
                elif 'intensity' in pixel: # Grayscale support
                    intensity = pixel['intensity']
                    matrix[y, x] = [intensity] * 3

            return matrix
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Error reading or parsing JSON file {json_path}: {e}")
            return None

    def preprocess_image(self, matrix: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and normalize."""
        if len(matrix.shape) == 3 and matrix.shape[2] == 3:
            gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        else:
            gray = matrix.copy() # Assume it's already grayscale

        # Apply a light blur to reduce noise before feature extraction
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    # --- Comprehensive Feature Extraction ---

    def extract_all_features(self, gray_img: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
        """
        Extracts a comprehensive set of features from a grayscale image.
        Returns a dictionary of features and a list of their names.
        """
        features = {}
        
        # 1. Statistical Features
        features.update(self._extract_statistical_features(gray_img))
        
        # 2. Texture Features (LBP & GLCM)
        features.update(self._extract_lbp_features(gray_img))
        features.update(self._extract_glcm_features(gray_img))
        
        # 3. Frequency Domain Features (Fourier & Wavelet)
        features.update(self._extract_fourier_features(gray_img))
        features.update(self._extract_wavelet_features(gray_img, wavelet='haar', level=3))

        # 4. Morphological Features
        features.update(self._extract_morphological_features(gray_img))
        
        # 5. Shape and Invariant Features
        features.update(self._extract_shape_features(gray_img))
        
        # 6. SVD Features
        features.update(self._extract_svd_features(gray_img))

        # 7. NEW: Persistent Homology Features
        features.update(self._extract_persistent_homology_features(gray_img))

        feature_names = sorted(features.keys())
        return features, feature_names
        
    def _extract_statistical_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extracts first-order statistical features."""
        flat_gray = gray.flatten()
        return {
            'stat_mean': np.mean(gray),
            'stat_std': np.std(gray),
            'stat_skew': float(stats.skew(flat_gray)),
            'stat_kurtosis': float(stats.kurtosis(flat_gray)),
            'stat_entropy': float(stats.entropy(np.histogram(gray, bins=256)[0])),
        }

    def _extract_lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extracts Local Binary Pattern features at multiple radii."""
        features = {}
        for radius in [1, 3, 5]:
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp, density=True, bins=n_points + 2, range=(0, n_points + 2))
            features[f'lbp_r{radius}_entropy'] = stats.entropy(hist)
            features[f'lbp_r{radius}_mean'] = np.mean(lbp)
        return features

    def _extract_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extracts Gray-Level Co-occurrence Matrix features."""
        # Quantize image to 8 levels to speed up GLCM computation
        img_quantized = (gray // 32).astype(np.uint8)
        glcm = graycomatrix(img_quantized, distances=[1, 3], angles=[0, np.pi/2], levels=8, symmetric=True, normed=True)
        
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = {}
        for prop in props:
            prop_values = graycoprops(glcm, prop).flatten()
            for i, val in enumerate(prop_values):
                features[f'glcm_{prop}_{i}'] = val
        return features

    def _extract_fourier_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extracts features from the 2D Fourier Transform magnitude spectrum."""
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        return {
            'fft_mean': np.mean(magnitude_spectrum),
            'fft_std': np.std(magnitude_spectrum),
            'fft_energy': np.sum(magnitude_spectrum**2),
        }
    
    def _extract_wavelet_features(self, gray: np.ndarray, wavelet: str, level: int) -> Dict[str, float]:
        """Extracts energy features from wavelet decomposition."""
        coeffs = pywt.wavedec2(gray, wavelet, level=level)
        features = {}
        for i, detail_coeffs in enumerate(coeffs[1:]):
            # cA, (cH, cV, cD) = coeffs
            cH, cV, cD = detail_coeffs
            features[f'wavelet_L{i+1}_H_energy'] = np.sum(np.square(cH))
            features[f'wavelet_L{i+1}_V_energy'] = np.sum(np.square(cV))
            features[f'wavelet_L{i+1}_D_energy'] = np.sum(np.square(cD))
        return features

    def _extract_morphological_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extracts features using morphological top-hat transforms."""
        features = {}
        for size in [5, 11]:
            selem = disk(size)
            wth = white_tophat(gray, selem)
            bth = black_tophat(gray, selem)
            features[f'morph_wth_{size}_mean'] = np.mean(wth)
            features[f'morph_bth_{size}_mean'] = np.mean(bth)
            features[f'morph_wth_{size}_max'] = np.max(wth)
            features[f'morph_bth_{size}_max'] = np.max(bth)
        return features

    def _extract_shape_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extracts Hu Moments for shape description."""
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        return {f'shape_hu_{i}': m for i, m in enumerate(hu_moments)}
        
    def _extract_svd_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extracts features from Singular Value Decomposition."""
        _, s, _ = np.linalg.svd(gray, full_matrices=False)
        s_norm = s / np.sum(s)
        return {
            'svd_energy_top_5_ratio': np.sum(s_norm[:5]),
            'svd_entropy': stats.entropy(s_norm),
            'svd_largest_singular_value': s[0] if len(s) > 0 else 0,
        }

    def _extract_persistent_homology_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extracts simplified Persistent Homology features based on level-set filtration.
        This captures the lifespan of connected components (Betti-0) and holes (Betti-1).
        """
        features = {}
        all_persistences_b0 = []
        all_persistences_b1 = []

        # Use a smaller number of thresholds for efficiency
        thresholds = np.linspace(np.min(gray), np.max(gray), 32)
        
        # Track components across thresholds
        # `birth_levels` stores the birth threshold for each component label
        birth_levels_b0: Dict[int, float] = {}
        active_components_prev_step: Dict[int, List[int]] = {}

        for i, t in enumerate(thresholds):
            binary_img = gray >= t
            labeled_img, num_features = label(binary_img)
            
            # --- Betti 0: Connected Components ---
            current_labels = set(np.unique(labeled_img)) - {0}
            
            # New components born at this level
            new_labels = current_labels - set(birth_levels_b0.keys())
            for new_label in new_labels:
                birth_levels_b0[new_label] = t

            # Components that died (merged)
            if i > 0:
                prev_labels = set(active_components_prev_step.keys())
                disappeared_labels = prev_labels - current_labels
                for dead_label in disappeared_labels:
                    birth = birth_levels_b0.pop(dead_label, thresholds[i-1])
                    death = t
                    persistence = abs(birth - death)
                    all_persistences_b0.append(persistence)
            
            active_components_prev_step = {l:[] for l in current_labels}

        # --- Betti 1: Holes ---
        # A simplified way to track holes is to do the same for the inverted image
        inverted_gray = np.max(gray) - gray
        birth_levels_b1: Dict[int, float] = {}
        active_components_prev_step_b1: Dict[int, List[int]] = {}
        thresholds_inv = np.linspace(np.min(inverted_gray), np.max(inverted_gray), 32)

        for i, t in enumerate(thresholds_inv):
            binary_img = inverted_gray >= t
            labeled_img, _ = label(binary_img)
            current_labels = set(np.unique(labeled_img)) - {0}
            new_labels = current_labels - set(birth_levels_b1.keys())
            for new_label in new_labels:
                birth_levels_b1[new_label] = t
            if i > 0:
                prev_labels = set(active_components_prev_step_b1.keys())
                disappeared_labels = prev_labels - current_labels
                for dead_label in disappeared_labels:
                    birth = birth_levels_b1.pop(dead_label, thresholds_inv[i-1])
                    death = t
                    all_persistences_b1.append(abs(birth - death))
            active_components_prev_step_b1 = {l:[] for l in current_labels}

        # Calculate statistics from persistences
        if all_persistences_b0:
            features['ph_b0_max_persistence'] = np.max(all_persistences_b0)
            features['ph_b0_mean_persistence'] = np.mean(all_persistences_b0)
            features['ph_b0_sum_persistence'] = np.sum(all_persistences_b0)
        else: # Default values if no features found
            features.update({'ph_b0_max_persistence': 0, 'ph_b0_mean_persistence': 0, 'ph_b0_sum_persistence': 0})

        if all_persistences_b1:
            features['ph_b1_max_persistence'] = np.max(all_persistences_b1)
            features['ph_b1_mean_persistence'] = np.mean(all_persistences_b1)
            features['ph_b1_sum_persistence'] = np.sum(all_persistences_b1)
        else: # Default values if no features found
            features.update({'ph_b1_max_persistence': 0, 'ph_b1_mean_persistence': 0, 'ph_b1_sum_persistence': 0})

        return features

    # --- Supervised Mode: Golden Standard ---

    def create_golden_standard(self, ref_dir: str):
        """Creates a golden standard from a directory of reference images."""
        json_files = list(Path(ref_dir).glob('*.json'))
        img_files = list(Path(ref_dir).glob('*.[pP][nN][gG]')) + \
                    list(Path(ref_dir).glob('*.[jJ][pP][gG]')) + \
                    list(Path(ref_dir).glob('*.[bB][mM][pP]'))

        all_files = json_files + img_files
        if not all_files:
            print(f"Error: No reference images or JSON files found in {ref_dir}.")
            return

        print(f"Found {len(all_files)} reference files. Creating golden standard...")
        all_features_list = []
        feature_names = []

        for file_path in all_files:
            matrix = self.load_image(str(file_path))
            if matrix is not None:
                gray = self.preprocess_image(matrix)
                features, f_names = self.extract_all_features(gray)
                if not feature_names:
                    feature_names = f_names
                
                # Ensure consistent feature ordering
                feature_vector = [features[name] for name in feature_names]
                all_features_list.append(feature_vector)
        
        if not all_features_list:
            print("Could not extract features from any reference files.")
            return

        all_features_np = np.array(all_features_list)
        
        # Calculate mean and covariance for Mahalanobis distance
        mean_vector = np.mean(all_features_np, axis=0)
        # Add a small identity matrix for regularization to ensure invertibility
        covariance_matrix = np.cov(all_features_np, rowvar=False)
        inv_covariance_matrix = np.linalg.pinv(covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-6)

        self.knowledge_base['golden_standard_features'] = {
            'mean': mean_vector,
            'inv_cov': inv_covariance_matrix,
        }
        self.knowledge_base['feature_names'] = feature_names
        self.save_knowledge_base()
        print("Golden standard created and saved.")

    def analyze_with_standard(self, test_path: str) -> Optional[Dict[str, Any]]:
        """Analyzes a test image against the loaded golden standard."""
        if self.knowledge_base.get('golden_standard_features') is None:
            print("Error: No golden standard loaded. Please create one first.")
            return None

        test_matrix = self.load_image(test_path)
        if test_matrix is None:
            return None

        gray = self.preprocess_image(test_matrix)
        test_features, _ = self.extract_all_features(gray)
        
        feature_names = self.knowledge_base['feature_names']
        test_vector = np.array([test_features.get(name, 0) for name in feature_names])
        
        gs = self.knowledge_base['golden_standard_features']
        mean_vec = gs['mean']
        inv_cov = gs['inv_cov']
        
        # Calculate Mahalanobis distance as the anomaly score
        diff = test_vector - mean_vec
        anomaly_score = np.sqrt(diff.T @ inv_cov @ diff)

        # Also find the most deviant features
        z_scores = diff / (np.sqrt(np.diag(np.linalg.pinv(inv_cov))) + 1e-6)
        top_deviant_indices = np.argsort(np.abs(z_scores))[::-1][:5]
        top_deviant_features = {feature_names[i]: z_scores[i] for i in top_deviant_indices}

        analysis = self._perform_common_analysis(test_matrix, gray)
        analysis['mode'] = "Supervised"
        analysis['anomaly_score'] = anomaly_score
        analysis['deviant_features'] = top_deviant_features
        analysis['summary'] = f"Anomaly score (Mahalanobis distance) is {anomaly_score:.2f}. " \
                              f"Higher scores indicate greater deviation from the standard."

        return analysis

    # --- Unsupervised Mode: Single Image Analysis ---

    def analyze_unsupervised(self, test_path: str, window_size: int = 64, stride: int = 32) -> Optional[Dict[str, Any]]:
        """Performs unsupervised anomaly detection on a single image."""
        test_matrix = self.load_image(test_path)
        if test_matrix is None:
            return None

        gray = self.preprocess_image(test_matrix)
        h, w = gray.shape

        # 1. Compute global features for the entire image
        global_features, feature_names = self.extract_all_features(gray)
        global_vector = np.array([global_features[name] for name in feature_names])

        # 2. Slide a window and compute local features
        patch_features_list = []
        patch_coords = []
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                patch = gray[y:y + window_size, x:x + window_size]
                if patch.size == 0: continue
                
                local_features, _ = self.extract_all_features(patch)
                patch_vector = np.array([local_features.get(name, 0) for name in feature_names])
                patch_features_list.append(patch_vector)
                patch_coords.append((x, y))
        
        if not patch_features_list:
            print("Warning: Could not extract features from any patches.")
            return None

        # 3. Compute anomaly score for each patch
        anomaly_map = np.zeros_like(gray, dtype=float)
        all_patches_np = np.array(patch_features_list)
        
        # Use Wasserstein distance for a robust measure of difference
        global_dist, _ = np.histogram(global_vector, bins=256, density=True)

        for i, patch_vec in enumerate(all_patches_np):
            patch_dist, _ = np.histogram(patch_vec, bins=256, density=True)
            # score = np.linalg.norm(patch_vec - global_vector) # Simple Euclidean distance
            score = wasserstein_distance(global_dist, patch_dist)
            
            x, y = patch_coords[i]
            anomaly_map[y:y + window_size, x:x + window_size] += score
        
        # Normalize the anomaly map for visualization
        if np.max(anomaly_map) > 0:
            anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
        
        analysis = self._perform_common_analysis(test_matrix, gray)
        analysis['mode'] = "Unsupervised"
        analysis['anomaly_map'] = anomaly_map
        analysis['anomaly_score'] = np.max(anomaly_map) if anomaly_map.size > 0 else 0
        analysis['summary'] = f"Max local anomaly score is {analysis['anomaly_score']:.2f}. " \
                              f"Hotter areas in the map indicate regions most different from the image average."
        
        return analysis

    # --- Common Analysis and Visualization ---
    
    def _perform_common_analysis(self, original_matrix: np.ndarray, gray_matrix: np.ndarray) -> Dict[str, Any]:
        """Performs analysis steps common to both modes."""
        specific_defects = self._detect_specific_defects(gray_matrix)
        return {
            'original_image': cv2.cvtColor(original_matrix, cv2.COLOR_BGR2RGB) if len(original_matrix.shape) == 3 else cv2.cvtColor(original_matrix, cv2.COLOR_GRAY2RGB),
            'specific_defects': specific_defects,
        }

    def _detect_specific_defects(self, gray: np.ndarray) -> Dict[str, list]:
        """Detects specific defect types like scratches and blobs."""
        defects = {'scratches': [], 'blobs': []}
        
        # Scratch detection using Hough Transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            defects['scratches'] = [line[0] for line in lines]

        # Blob detection using thresholding and contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000: # Filter by area
                defects['blobs'].append(contour)
        
        return defects

    def visualize_results(self, analysis: Dict[str, Any], output_path: str):
        """Creates and saves a visualization of the analysis results."""
        original_img = analysis['original_image']
        anomaly_map = analysis.get('anomaly_map') # Only in unsupervised
        specific_defects = analysis['specific_defects']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Anomaly Detection Result ({analysis['mode']} Mode)", fontsize=16)

        # Panel 1: Original Image
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Panel 2: Anomaly Heatmap
        axes[1].imshow(original_img, alpha=0.6)
        if anomaly_map is not None:
            im = axes[1].imshow(anomaly_map, cmap='plasma', alpha=0.5)
            fig.colorbar(im, ax=axes[1], orientation='horizontal', pad=0.05)
            axes[1].set_title("Anomaly Heatmap")
        else:
             axes[1].set_title("Comparison Mode (No Heatmap)")
             axes[1].text(0.5, 0.5, f"Anomaly Score:\n{analysis.get('anomaly_score', 'N/A'):.2f}", 
                          ha='center', va='center', fontsize=14, color='red')
        axes[1].axis('off')

        # Panel 3: Detected Specific Defects
        overlay_img = original_img.copy()
        for x1, y1, x2, y2 in specific_defects.get('scratches', []):
            cv2.line(overlay_img, (x1, y1), (x2, y2), (0, 255, 255), 2) # Cyan lines
        
        cv2.drawContours(overlay_img, specific_defects.get('blobs', []), -1, (255, 0, 255), 2) # Magenta contours
        
        axes[2].imshow(overlay_img)
        axes[2].set_title("Detected Scratches & Blobs")
        axes[2].axis('off')
        
        # Add summary text
        summary_text = analysis.get('summary', '')
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, wrap=True)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path)
        plt.close()
        print(f"Analysis visualization saved to: {output_path}")

def main():
    """Main execution function with interactive user interface."""
    print("=" * 60)
    print(" Universal Matrix Anomaly Detection System ".center(60, "="))
    print("=" * 60)
    
    analyzer = UniversalMatrixAnalyzer()

    # --- Mode Selection ---
    mode = ""
    while mode not in ['1', '2']:
        print("\nSelect analysis mode:")
        print("1. Supervised (compare a test image to a 'golden standard')")
        print("2. Unsupervised (analyze a single image for internal anomalies)")
        mode = input("Enter choice (1 or 2): ").strip()

    if mode == '1':
        # --- Supervised Mode Workflow ---
        print("\n--- SUPERVISED MODE ---")
        
        # Check if golden standard needs to be created
        create_new_gs = 'y'
        if analyzer.knowledge_base.get('golden_standard_features') is not None:
            create_new_gs = input("A golden standard already exists. Create a new one? (y/n): ").strip().lower()

        if create_new_gs == 'y':
            ref_dir = input("Enter path to the directory with NORMAL reference images/JSONs: ").strip()
            if not os.path.isdir(ref_dir):
                print(f"Error: Directory '{ref_dir}' not found.")
                return
            analyzer.create_golden_standard(ref_dir)
        
        test_path = input("Enter path to the TEST image or JSON file to analyze: ").strip()
        if not os.path.isfile(test_path):
            print(f"Error: File '{test_path}' not found.")
            return

        print("\nAnalyzing test image against the golden standard...")
        results = analyzer.analyze_with_standard(test_path)

    else:
        # --- Unsupervised Mode Workflow ---
        print("\n--- UNSUPERVISED MODE ---")
        test_path = input("Enter path to the image or JSON file to analyze: ").strip()
        if not os.path.isfile(test_path):
            print(f"Error: File '{test_path}' not found.")
            return

        print("\nAnalyzing image for internal anomalies...")
        results = analyzer.analyze_unsupervised(test_path)

    # --- Display and Save Results ---
    if results:
        print("\n--- Analysis Report ---")
        print(f"Mode: {results['mode']}")
        print(f"Overall Anomaly Score: {results['anomaly_score']:.4f}")
        print(f"Summary: {results['summary']}")
        
        defects = results['specific_defects']
        print(f"Detected Specific Defects:")
        print(f"  - Scratches: {len(defects.get('scratches', []))}")
        print(f"  - Blobs: {len(defects.get('blobs', []))}")
        
        if 'deviant_features' in results:
            print("Top Deviant Features (Feature: Z-score):")
            for feat, score in results['deviant_features'].items():
                print(f"  - {feat}: {score:.2f}")

        output_filename = f"analysis_result_{Path(test_path).stem}.png"
        analyzer.visualize_results(results, output_filename)
    else:
        print("\nAnalysis could not be completed.")

if __name__ == "__main__":
    main()
