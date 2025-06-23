#!/usr/bin/env python3
"""
Advanced Comparative Matrix Anomaly Detection System
-----------------------------------------------------

This script provides a unified and powerful framework for detecting anomalies by comparing
a single test image against a reference set of images. It determines how a test image
statistically and structurally deviates from a given population, whether that population
is considered 'perfect' or 'imperfect'.

This system abandons the rigid supervised/unsupervised distinction in favor of a
single, more flexible comparative workflow.

Key Features:
1.  Comparative Analysis Core:
    - The system first builds a detailed statistical model from a directory of reference
      images (the "population").
    - It then analyzes a single test image against this population model to identify
      and quantify deviations.

2.  Comprehensive Feature Engineering:
    - Incorporates a wide array of features including statistical, texture (GLCM, LBP),
      frequency (Fourier, Wavelet), morphological, shape (Hu Moments), SVD, and
      Persistent Homology (Topological) features for a deep characterization of images.

3.  Advanced Anomaly Quantification:
    - Global Anomaly Score: Uses the Mahalanobis distance to provide a single, robust
      score measuring how much of a statistical outlier the test image is compared to the
      reference population.
    - Detailed Deviation Analysis: Calculates feature-wise Z-scores to pinpoint which
      specific characteristics (e.g., texture, topology) of the test image are most anomalous.
    - Local Anomaly Mapping: Generates a visual anomaly map by comparing the test image
      to a computed "archetype" (mean) of the reference set using the Structural
      Similarity Index (SSIM), highlighting regions of structural difference.

4.  Rich, Comparative Visualization:
    - Produces a multi-panel plot that includes:
        a) The test image.
        b) The reference population's "archetype" image for visual comparison.
        c) A heatmap of local structural anomalies (the SSIM difference map).
    - The visualization is designed to make the nature and location of imperfections clear.

Dependencies (install via pip):
- pip install numpy opencv-python scikit-image scikit-learn scipy matplotlib pywavelets
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
from scipy import stats
from scipy.ndimage import label
from skimage.metrics import structural_similarity
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import disk, white_tophat, black_tophat

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=FutureWarning, module='skimage')


class ComparativeMatrixAnalyzer:
    """
    A universal analyzer for comparing a test image to a reference population.
    """

    def __init__(self, knowledge_base_path="comparative_analysis_kb.pkl"):
        self.knowledge_base_path = knowledge_base_path
        self.reference_model: Dict[str, Any] = {
            'statistical_model': None,
            'archetype_image': None,
            'feature_names': [],
        }
        self.load_reference_model()

    def load_reference_model(self):
        """Load a previously saved reference model."""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    self.reference_model = pickle.load(f)
                print(f"Successfully loaded reference model from {self.knowledge_base_path}")
            except Exception as e:
                print(f"Warning: Could not load reference model. Error: {e}")

    def save_reference_model(self):
        """Save the current reference model to a file."""
        try:
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.reference_model, f)
            print(f"Reference model saved to {self.knowledge_base_path}")
        except Exception as e:
            print(f"Error: Could not save reference model. Error: {e}")

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
                x, y = pixel['coordinates']['x'], pixel['coordinates']['y']
                matrix[y, x] = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
            return matrix
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Error reading or parsing JSON file {json_path}: {e}")
            return None

    def preprocess_image(self, matrix: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and apply a light blur."""
        if len(matrix.shape) == 3 and matrix.shape[2] == 3:
            gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        else:
            gray = matrix.copy()
        return cv2.GaussianBlur(gray, (3, 3), 0)

    # --- Feature Extraction (identical to previous version) ---
    def extract_all_features(self, gray_img: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
        features = {}
        features.update(self._extract_statistical_features(gray_img))
        features.update(self._extract_lbp_features(gray_img))
        features.update(self._extract_glcm_features(gray_img))
        features.update(self._extract_fourier_features(gray_img))
        features.update(self._extract_wavelet_features(gray_img, 'haar', 3))
        features.update(self._extract_morphological_features(gray_img))
        features.update(self._extract_svd_features(gray_img))
        features.update(self._extract_persistent_homology_features(gray_img))
        feature_names = sorted(features.keys())
        return features, feature_names
        
    def _extract_statistical_features(self, gray: np.ndarray) -> Dict[str, float]:
        flat_gray = gray.flatten()
        return {'stat_mean': np.mean(gray), 'stat_std': np.std(gray), 'stat_skew': float(stats.skew(flat_gray)), 'stat_kurtosis': float(stats.kurtosis(flat_gray)), 'stat_entropy': float(stats.entropy(np.histogram(gray, bins=256)[0]))}
    def _extract_lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        features = {}
        for radius in [1, 3, 5]:
            n_points = 8 * radius; lbp = local_binary_pattern(gray, n_points, radius, method='uniform'); hist, _ = np.histogram(lbp, density=True, bins=n_points + 2, range=(0, n_points + 2)); features[f'lbp_r{radius}_entropy'] = stats.entropy(hist); features[f'lbp_r{radius}_mean'] = np.mean(lbp)
        return features
    def _extract_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        img_quantized = (gray // 32).astype(np.uint8)
        glcm = graycomatrix(img_quantized, distances=[1, 3], angles=[0, np.pi/2], levels=8, symmetric=True, normed=True)
        features = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            prop_values = graycoprops(glcm, prop).flatten()
            for i, val in enumerate(prop_values): features[f'glcm_{prop}_{i}'] = val
        return features
    def _extract_fourier_features(self, gray: np.ndarray) -> Dict[str, float]:
        f = np.fft.fft2(gray); fshift = np.fft.fftshift(f); magnitude_spectrum = np.log(np.abs(fshift) + 1)
        return {'fft_mean': np.mean(magnitude_spectrum), 'fft_std': np.std(magnitude_spectrum), 'fft_energy': np.sum(magnitude_spectrum**2)}
    def _extract_wavelet_features(self, gray: np.ndarray, wavelet: str, level: int) -> Dict[str, float]:
        coeffs = pywt.wavedec2(gray, wavelet, level=level); features = {}
        for i, detail_coeffs in enumerate(coeffs[1:]):
            cH, cV, cD = detail_coeffs
            features.update({f'wavelet_L{i+1}_H_energy': np.sum(np.square(cH)), f'wavelet_L{i+1}_V_energy': np.sum(np.square(cV)), f'wavelet_L{i+1}_D_energy': np.sum(np.square(cD))})
        return features
    def _extract_morphological_features(self, gray: np.ndarray) -> Dict[str, float]:
        features = {}
        for size in [5, 11]:
            selem = disk(size); wth = white_tophat(gray, selem); bth = black_tophat(gray, selem)
            features.update({f'morph_wth_{size}_mean': np.mean(wth), f'morph_bth_{size}_mean': np.mean(bth), f'morph_wth_{size}_max': np.max(wth), f'morph_bth_{size}_max': np.max(bth)})
        return features
    def _extract_svd_features(self, gray: np.ndarray) -> Dict[str, float]:
        _, s, _ = np.linalg.svd(gray, full_matrices=False); s_norm = s / (np.sum(s) + 1e-9)
        return {'svd_entropy': stats.entropy(s_norm), 'svd_energy_top_5_ratio': np.sum(s_norm[:5])}
    def _extract_persistent_homology_features(self, gray: np.ndarray) -> Dict[str, float]:
        features, all_persistences_b0, all_persistences_b1 = {}, [], []
        thresholds = np.linspace(np.min(gray), np.max(gray), 32)
        birth_levels_b0: Dict[int, float] = {}; active_components_prev_step: Dict[int, List[int]] = {}
        for i, t in enumerate(thresholds):
            labeled_img, _ = label(gray >= t); current_labels = set(np.unique(labeled_img)) - {0}
            for new_label in (current_labels - set(birth_levels_b0.keys())): birth_levels_b0[new_label] = t
            if i > 0:
                for dead_label in (set(active_components_prev_step.keys()) - current_labels):
                    birth = birth_levels_b0.pop(dead_label, thresholds[i-1]); all_persistences_b0.append(abs(birth - t))
            active_components_prev_step = {l:[] for l in current_labels}
        inverted_gray = np.max(gray) - gray; thresholds_inv = np.linspace(np.min(inverted_gray), np.max(inverted_gray), 32)
        birth_levels_b1: Dict[int, float] = {}; active_components_prev_step_b1: Dict[int, List[int]] = {}
        for i, t in enumerate(thresholds_inv):
            labeled_img, _ = label(inverted_gray >= t); current_labels = set(np.unique(labeled_img)) - {0}
            for new_label in (current_labels - set(birth_levels_b1.keys())): birth_levels_b1[new_label] = t
            if i > 0:
                for dead_label in (set(active_components_prev_step_b1.keys()) - current_labels):
                     birth = birth_levels_b1.pop(dead_label, thresholds_inv[i-1]); all_persistences_b1.append(abs(birth - t))
            active_components_prev_step_b1 = {l:[] for l in current_labels}
        features['ph_b0_max_persistence'] = np.max(all_persistences_b0) if all_persistences_b0 else 0
        features['ph_b1_max_persistence'] = np.max(all_persistences_b1) if all_persistences_b1 else 0
        return features

    # --- Core Comparative Analysis ---

    def build_reference_model(self, ref_dir: str):
        """Builds a statistical and visual model from a directory of reference images."""
        all_paths = [p for p in Path(ref_dir).glob('**/*') if p.is_file() and any(p.name.lower().endswith(ext) for ext in ['.json', '.png', '.jpg', '.jpeg', '.bmp'])]
        if not all_paths:
            print(f"Error: No reference images or JSON files found in {ref_dir}.")
            return

        print(f"Found {len(all_paths)} reference files. Building reference model...")
        all_features_list, all_gray_images, feature_names = [], [], []
        target_shape = None

        for path in all_paths:
            matrix = self.load_image(str(path))
            if matrix is not None:
                gray = self.preprocess_image(matrix)
                if target_shape is None: target_shape = gray.shape
                if gray.shape != target_shape: 
                    gray = cv2.resize(gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
                
                all_gray_images.append(gray)
                features, f_names = self.extract_all_features(gray)
                if not feature_names: feature_names = f_names
                all_features_list.append([features.get(name, 0) for name in feature_names])
        
        if not all_features_list:
            print("Could not build model. No valid reference files processed.")
            return

        # Build statistical model
        all_features_np = np.array(all_features_list)
        mean_vector = np.mean(all_features_np, axis=0)
        std_vector = np.std(all_features_np, axis=0)
        covariance_matrix = np.cov(all_features_np, rowvar=False)
        inv_covariance_matrix = np.linalg.pinv(covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-6)

        self.reference_model['statistical_model'] = {
            'mean': mean_vector,
            'std': std_vector,
            'inv_cov': inv_covariance_matrix,
        }
        
        # Build visual archetype
        self.reference_model['archetype_image'] = np.mean(all_gray_images, axis=0).astype(np.uint8)
        self.reference_model['feature_names'] = feature_names
        self.save_reference_model()
        print("Reference model built and saved successfully.")

    def compare_to_reference(self, test_path: str) -> Optional[Dict[str, Any]]:
        """Analyzes a test image by comparing it to the reference model."""
        if self.reference_model.get('statistical_model') is None:
            print("Error: No reference model loaded. Please build one first.")
            return None

        test_matrix = self.load_image(test_path)
        if test_matrix is None: return None

        gray_test = self.preprocess_image(test_matrix)
        
        # Resize test image to match archetype if necessary
        archetype_img = self.reference_model['archetype_image']
        if gray_test.shape != archetype_img.shape:
            gray_test = cv2.resize(gray_test, (archetype_img.shape[1], archetype_img.shape[0]), interpolation=cv2.INTER_AREA)
        
        # --- Global Anomaly Score ---
        test_features, _ = self.extract_all_features(gray_test)
        feature_names = self.reference_model['feature_names']
        test_vector = np.array([test_features.get(name, 0) for name in feature_names])
        
        stat_model = self.reference_model['statistical_model']
        mean_vec, std_vec, inv_cov = stat_model['mean'], stat_model['std'], stat_model['inv_cov']
        
        diff = test_vector - mean_vec
        global_anomaly_score = np.sqrt(diff.T @ inv_cov @ diff) # Mahalanobis distance

        # --- Detailed Feature Deviation (Z-scores) ---
        z_scores = diff / (std_vec + 1e-6)
        top_deviant_indices = np.argsort(np.abs(z_scores))[::-1][:5]
        deviant_features = {feature_names[i]: z_scores[i] for i in top_deviant_indices}

        # --- Local Anomaly Map (SSIM) ---
        ssim_score, diff_map = structural_similarity(gray_test, archetype_img, full=True)
        anomaly_map = 1 - diff_map # High values mean high structural difference

        # --- Specific Defect Detection ---
        specific_defects = self._detect_specific_defects(gray_test)
        
        summary = (f"Global Anomaly Score (Mahalanobis Distance): {global_anomaly_score:.2f}. "
                   f"This measures overall deviation from the reference set. Higher is more anomalous.\n"
                   f"Structural Similarity to Archetype (SSIM): {ssim_score:.2f}. "
                   f"Closer to 1 means more structurally similar.")
        
        return {
            'test_image': cv2.cvtColor(gray_test, cv2.COLOR_GRAY2RGB),
            'archetype_image': cv2.cvtColor(archetype_img, cv2.COLOR_GRAY2RGB),
            'anomaly_map': anomaly_map,
            'specific_defects': specific_defects,
            'global_anomaly_score': global_anomaly_score,
            'deviant_features': deviant_features,
            'ssim_score': ssim_score,
            'summary': summary
        }
        
    def _detect_specific_defects(self, gray: np.ndarray) -> Dict[str, list]:
        """Detects specific defect types like scratches and blobs."""
        defects = {'scratches': [], 'blobs': []}
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=25, maxLineGap=8)
        if lines is not None: defects['scratches'] = [line[0] for line in lines]
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if 30 < cv2.contourArea(c) < 10000: defects['blobs'].append(c)
        return defects

    def visualize_results(self, analysis: Dict[str, Any], output_path: str):
        """Creates and saves a visualization of the comparative analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle("Comparative Anomaly Analysis", fontsize=18)

        # Panel 1: Test Image with defect overlays
        test_img_overlay = analysis['test_image'].copy()
        for x1, y1, x2, y2 in analysis['specific_defects'].get('scratches', []):
            cv2.line(test_img_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.drawContours(test_img_overlay, analysis['specific_defects'].get('blobs', []), -1, (255, 0, 255), 2)
        axes[0].imshow(test_img_overlay)
        axes[0].set_title("Test Image with Detected Defects")
        axes[0].axis('off')
        
        # Panel 2: Reference Archetype Image
        axes[1].imshow(analysis['archetype_image'])
        axes[1].set_title("Reference Set 'Archetype' (Mean)")
        axes[1].axis('off')

        # Panel 3: Anomaly Heatmap (SSIM Difference)
        axes[2].imshow(analysis['test_image'], alpha=0.5)
        im = axes[2].imshow(analysis['anomaly_map'], cmap='magma', alpha=0.6)
        fig.colorbar(im, ax=axes[2], orientation='horizontal', pad=0.05, label="Structural Dissimilarity")
        axes[2].set_title("Local Anomaly Heatmap (SSIM)")
        axes[2].axis('off')
        
        # Add detailed summary text below plots
        fig.text(0.5, 0.1, analysis['summary'], ha='center', va='top', fontsize=12, wrap=True)
        
        deviants_str = "Top Deviant Features (Z-Score):\n" + "\n".join([f"- {k}: {v:.2f}" for k,v in analysis['deviant_features'].items()])
        fig.text(0.5, 0.0, deviants_str, ha='center', va='top', fontsize=10, wrap=True)

        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.savefig(output_path)
        plt.close()
        print(f"Analysis visualization saved to: {output_path}")

def main():
    """Main execution function with interactive user interface."""
    print("=" * 65)
    print(" Advanced Comparative Matrix Anomaly Detection System ".center(65, "="))
    print("=" * 65)
    
    analyzer = ComparativeMatrixAnalyzer()

    # --- Step 1: Build or Load Reference Model ---
    ref_dir = input("\nEnter path to the directory with REFERENCE images/JSONs: ").strip()
    if not os.path.isdir(ref_dir):
        print(f"Error: Directory '{ref_dir}' not found.")
        return
    analyzer.build_reference_model()

    # --- Step 2: Analyze a Test Image ---
    while True:
        print("\n" + "-"*65)
        test_path = input("Enter path to a TEST image/JSON (or type 'quit' to exit): ").strip()
        
        if test_path.lower() == 'quit':
            break
            
        if not os.path.isfile(test_path):
            print(f"Error: File '{test_path}' not found. Please try again.")
            continue

        print("\nComparing test image to the reference set...")
        results = analyzer.compare_to_reference(test_path)

        # --- Step 3: Display and Save Results ---
        if results:
            print("\n--- Analysis Report ---")
            print(results['summary'])
            print("\nTop 5 Most Deviant Features (Name: Z-score from reference set):")
            for feat, score in results['deviant_features'].items():
                print(f"  - {feat:<25}: {score:+.2f}")

            output_filename = f"comparative_analysis_{Path(test_path).stem}.png"
            analyzer.visualize_results(results, output_filename)
        else:
            print("\nAnalysis could not be completed for this file.")
            
    print("\nExiting program.")

if __name__ == "__main__":
    main()
