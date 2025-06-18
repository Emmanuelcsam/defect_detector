#!/usr/bin/env python3
"""
OmniFiberAnalyzer: Master System for Fiber Optic End-Face Analysis
==================================================================

This system unifies robust engineering from multiple sources:
- daniel.py: Robust mask generation and defect detection
- jill.py: Exhaustive ensemble methods
- jake.py: Statistical anomaly detection
- Research papers: Topological Data Analysis (TDA)

Author: Advanced Fiber Optics Analysis Team
Version: 1.1 - Interactive Mode
"""

import cv2
import numpy as np
import json
import os
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# Scientific computing imports
from scipy import ndimage, signal, stats
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import minimize
from skimage import morphology, measure, feature, segmentation
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import active_contour
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# TDA imports (GUDHI)
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    warnings.warn("GUDHI not available. Topological analysis will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


# ==================== PART 1: CORE ARCHITECTURAL BLUEPRINT ====================

@dataclass
class OmniConfig:
    """Unified configuration for all analysis parameters"""

    # --- General & Reporting Settings ---
    min_defect_area_px: int = 5
    pixels_per_micron: float = 0.5  # Example value
    output_dpi: int = 200
    save_intermediate_masks: bool = False
    generate_json_report: bool = True

    # --- Global Statistical Analysis Config (from jake.py) ---
    use_global_anomaly_analysis: bool = True
    knowledge_base_path: str = "ultra_anomaly_kb.json"
    anomaly_mahalanobis_threshold: float = 5.0
    anomaly_ssim_threshold: float = 0.85

    # --- Topological Data Analysis (TDA) Config (from Chung et al., 2024) ---
    use_topological_analysis: bool = True
    tda_library: str = 'gudhi'
    mf_threshold_range: Tuple[int, int] = (0, 255)
    mf_threshold_step: int = 8
    mf_opening_size_range: Tuple[int, int] = (0, 20)
    mf_opening_step: int = 1
    min_global_connectivity: float = 0.85  # Threshold for global Pass/Fail

    # --- Advanced Preprocessing Config (from jill.py & research) ---
    use_anisotropic_diffusion: bool = True
    use_coherence_enhancing_diffusion: bool = True
    gaussian_blur_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(3,3), (5,5), (7,7)])

    # --- Region Separation (Hybrid) ---
    primary_masking_method: str = 'ensemble'  # 'ensemble' or 'adaptive_contour'
    hough_dp_values: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    adaptive_threshold_block_size: int = 51

    # --- Exhaustive Detector Ensemble Config (All sources) ---
    enabled_detectors: List[str] = field(default_factory=lambda: [
        'do2mr', 'lei', 'log', 'doh', 'hessian_eigen', 'frangi',
        'structure_tensor', 'robust_pca', 'mser', 'lbp', 'watershed'
    ])
    ensemble_confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'do2mr': 1.0,
        'lei': 1.0,
        'log': 0.9,
        'doh': 0.85,
        'hessian_eigen': 0.9,
        'frangi': 0.9,
        'structure_tensor': 0.85,
        'robust_pca': 0.8,
        'mser': 0.8,
        'lbp': 0.75,
        'watershed': 0.85
    })
    ensemble_vote_threshold: float = 0.3
    min_methods_for_detection: int = 2

    # --- Advanced Segmentation Config (from research) ---
    segmentation_refinement_method: str = 'active_contour'  # 'morphology' or 'active_contour'
    active_contour_alpha: float = 0.015  # Elasticity
    active_contour_beta: float = 10.0   # Rigidity

    # --- False Positive Reduction Config (from jill.py) ---
    use_geometric_fp_reduction: bool = True
    use_contrast_fp_reduction: bool = True


@dataclass
class Defect:
    """Comprehensive feature vector for each detected anomaly"""

    # --- Core Geometric Properties (from daniel.py) ---
    defect_id: int
    defect_type: str  # e.g., 'Scratch', 'Dig', 'Contamination'
    zone: str  # 'Core', 'Cladding', 'Adhesive'
    severity: str  # 'Low', 'Medium', 'High', 'Critical'
    area_px: int
    eccentricity: float
    solidity: float
    circularity: float
    location_xy: Tuple[int, int]

    # --- Intensity & Contrast Properties (from daniel.py, intensity_map.py) ---
    mean_intensity: float
    contrast: float
    intensity_skewness: float
    intensity_kurtosis: float

    # --- Gradient Properties (from change_magnitude.py) ---
    mean_gradient: float
    max_gradient: float
    std_dev_gradient: float

    # --- Advanced Texture Properties (from research) ---
    glcm_contrast: float
    glcm_homogeneity: float
    glcm_energy: float

    # --- Advanced Structural Properties (from research) ---
    mean_hessian_eigen_ratio: float
    mean_coherence: float

    # --- Topological Data Analysis Properties (from Chung et al., 2024) ---
    tda_local_connectivity_score: float
    tda_betti_curve_signature: Any  # Store as a numpy array or list
    tda_size_distribution_signature: Any  # Store as a numpy array or list

    # --- Detection Provenance ---
    contributing_algorithms: List[str]


class UltraComprehensiveMatrixAnalyzer:
    """Statistical anomaly analyzer based on jake.py"""

    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = knowledge_base_path
        self.reference_model = None
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load the statistical reference model"""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    self.reference_model = json.load(f)
                logger.info(f"Loaded knowledge base from {self.knowledge_base_path}")
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")
                self.reference_model = None
        else:
            logger.warning(f"Knowledge base not found at {self.knowledge_base_path}")
            self.reference_model = None

    def detect_anomalies_comprehensive(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform statistical anomaly detection"""
        if self.reference_model is None:
            return {
                'verdict': 'UNKNOWN',
                'mahalanobis_distance': 0.0,
                'ssim_score': 1.0,
                'anomaly_map': np.zeros_like(image),
                'is_anomalous': False
            }

        # Extract features from test image
        features = self._extract_features(image)

        # Compare to reference model
        mahalanobis_dist = self._compute_mahalanobis_distance(features)
        ssim_score = self._compute_ssim(image)
        anomaly_map = self._compute_anomaly_map(image)

        # Determine verdict
        is_anomalous = (mahalanobis_dist > 5.0) or (ssim_score < 0.85)
        verdict = 'ANOMALOUS' if is_anomalous else 'NORMAL'

        return {
            'verdict': verdict,
            'mahalanobis_distance': mahalanobis_dist,
            'ssim_score': ssim_score,
            'anomaly_map': anomaly_map,
            'is_anomalous': is_anomalous
        }

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract statistical features from image"""
        features = []

        # Basic statistics
        features.extend([
            np.mean(image),
            np.std(image),
            stats.skew(image.flatten()),
            stats.kurtosis(image.flatten())
        ])

        # Gradient statistics
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.max(grad_mag)
        ])

        return np.array(features)

    def _compute_mahalanobis_distance(self, features: np.ndarray) -> float:
        """Compute Mahalanobis distance to reference model"""
        if 'mean_features' not in self.reference_model:
            return 0.0

        mean = np.array(self.reference_model['mean_features'])
        cov = np.array(self.reference_model['covariance_matrix'])

        diff = features - mean
        try:
            inv_cov = np.linalg.inv(cov + np.eye(len(cov)) * 1e-6)
            dist = np.sqrt(diff.T @ inv_cov @ diff)
            return float(dist)
        except:
            return 0.0

    def _compute_ssim(self, image: np.ndarray) -> float:
        """Compute SSIM to reference archetype"""
        if 'archetype_image' not in self.reference_model:
            return 1.0

        ref_image = np.array(self.reference_model['archetype_image'], dtype=np.uint8)

        # Resize if needed
        if image.shape != ref_image.shape:
            ref_image = cv2.resize(ref_image, (image.shape[1], image.shape[0]))

        # Simple SSIM implementation
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        mu1 = cv2.GaussianBlur(image.astype(float), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(ref_image.astype(float), (11, 11), 1.5)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(image.astype(float)**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(ref_image.astype(float)**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(image.astype(float) * ref_image.astype(float), (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    def _compute_anomaly_map(self, image: np.ndarray) -> np.ndarray:
        """Compute local anomaly heatmap"""
        # Simple local variance as anomaly measure
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        local_mean = cv2.filter2D(image.astype(float), -1, kernel)
        local_var = cv2.filter2D((image - local_mean)**2, -1, kernel)

        # Normalize
        anomaly_map = local_var / (np.max(local_var) + 1e-6)

        return anomaly_map


class TDA_Analyzer:
    """Topological Data Analysis based on Chung et al. (2024)"""

    def __init__(self, config: OmniConfig):
        self.config = config
        self.gudhi_available = GUDHI_AVAILABLE

    def analyze_region(self, image_region: np.ndarray) -> Dict[str, Any]:
        """Perform TDA analysis on image region"""
        if not self.gudhi_available:
            return {
                'normalized_betti_curves': [],
                'size_distributions': [],
                'connectivity_indices': []
            }

        # Create bifiltration
        bifiltration = self._create_bifiltration(image_region)

        # Compute persistence diagrams
        persistence_results = []
        for slice_params, binary_image in bifiltration:
            dgm = self._compute_persistence_diagrams_from_slice(binary_image)
            persistence_results.append((slice_params, dgm))

        # Extract features
        betti_curves = self._extract_betti_curves(persistence_results)
        size_distributions = self._extract_size_distributions(persistence_results)
        connectivity_indices = self._compute_connectivity_indices(persistence_results)

        return {
            'normalized_betti_curves': betti_curves,
            'size_distributions': size_distributions,
            'connectivity_indices': connectivity_indices
        }

    def _create_bifiltration(self, image_region: np.ndarray) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        """Create morphological multiparameter filtration"""
        bifiltration = []

        # Iterate through threshold values
        for threshold in range(self.config.mf_threshold_range[0], 
                               self.config.mf_threshold_range[1], 
                               self.config.mf_threshold_step):

            # Apply threshold
            _, binary = cv2.threshold(image_region, threshold, 255, cv2.THRESH_BINARY)

            # Iterate through opening sizes
            for opening_size in range(self.config.mf_opening_size_range[0],
                                      self.config.mf_opening_size_range[1],
                                      self.config.mf_opening_step):

                if opening_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                       (opening_size, opening_size))
                    filtered = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                else:
                    filtered = binary.copy()

                bifiltration.append(((threshold, opening_size), filtered))

        return bifiltration

    def _compute_persistence_diagrams_from_slice(self, binary_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute persistence diagram using GUDHI"""
        # Convert binary image to simplicial complex
        cubical_complex = gudhi.CubicalComplex(
            dimensions=binary_image.shape,
            top_dimensional_cells=binary_image.flatten()
        )

        # Compute persistence
        cubical_complex.compute_persistence()

        # Extract persistence diagrams for dimensions 0 and 1
        dgm_0 = np.array([p for p in cubical_complex.persistence() if p[0] == 0])
        dgm_1 = np.array([p for p in cubical_complex.persistence() if p[0] == 1])

        return {'dim_0': dgm_0, 'dim_1': dgm_1}

    def _extract_betti_curves(self, persistence_results: List[Tuple[Tuple[int, int], Dict]]) -> np.ndarray:
        """Extract normalized Betti curves"""
        betti_0_values = []
        betti_1_values = []

        for (threshold, opening), dgm in persistence_results:
            # Count connected components (Betti 0)
            if len(dgm['dim_0']) > 0:
                betti_0 = len(dgm['dim_0'])
            else:
                betti_0 = 0

            # Count holes (Betti 1)
            if len(dgm['dim_1']) > 0:
                betti_1 = len(dgm['dim_1'])
            else:
                betti_1 = 0

            betti_0_values.append(betti_0)
            betti_1_values.append(betti_1)

        # Normalize
        max_b0 = max(betti_0_values) if betti_0_values else 1
        max_b1 = max(betti_1_values) if betti_1_values else 1

        betti_0_normalized = [b / max_b0 for b in betti_0_values]
        betti_1_normalized = [b / max_b1 for b in betti_1_values]

        return np.array([betti_0_normalized, betti_1_normalized])

    def _extract_size_distributions(self, persistence_results: List[Tuple[Tuple[int, int], Dict]]) -> np.ndarray:
        """Extract size distribution features"""
        size_features = []

        for (threshold, opening), dgm in persistence_results:
            # Compute persistence lengths
            if len(dgm['dim_0']) > 0 and dgm['dim_0'].shape[1] >= 3:
                lifetimes_0 = dgm['dim_0'][:, 2] - dgm['dim_0'][:, 1]
                mean_lifetime_0 = np.mean(lifetimes_0)
                max_lifetime_0 = np.max(lifetimes_0)
            else:
                mean_lifetime_0 = 0
                max_lifetime_0 = 0

            size_features.append([mean_lifetime_0, max_lifetime_0])

        return np.array(size_features)

    def _compute_connectivity_indices(self, persistence_results: List[Tuple[Tuple[int, int], Dict]]) -> np.ndarray:
        """Compute connectivity indices"""
        connectivity_scores = []

        for (threshold, opening), dgm in persistence_results:
            # Simple connectivity score based on Betti numbers
            if len(dgm['dim_0']) > 0:
                n_components = len(dgm['dim_0'])
                n_holes = len(dgm['dim_1']) if len(dgm['dim_1']) > 0 else 0

                # Connectivity score: fewer components and holes = better connectivity
                score = 1.0 / (1 + n_components + 2 * n_holes)
            else:
                score = 0.0

            connectivity_scores.append(score)

        return np.array(connectivity_scores)


# ==================== MAIN ANALYZER CLASS ====================

class OmniFiberAnalyzer:
    """Master system for fiber optic end-face analysis"""

    def __init__(self, config: OmniConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize sub-analyzers
        self.statistical_analyzer = UltraComprehensiveMatrixAnalyzer(config.knowledge_base_path)
        self.tda_analyzer = TDA_Analyzer(config)

        # Storage for intermediate results
        self.intermediate_results = {}

    def analyze_end_face(self, image_path: str) -> Dict[str, Any]:
        """
        Main public method that executes the entire analysis pipeline

        Args:
            image_path: Path to the fiber optic image

        Returns:
            Comprehensive analysis results dictionary
        """
        self.logger.info(f"Starting OmniFiberAnalyzer on: {image_path}")
        start_time = time.time()

        # Step 1: Load and prepare image
        original_image, gray_image = self._load_and_prepare_image(image_path)

        # Step 2: Global statistical & topological analysis
        global_analysis_results = self._run_global_analysis(gray_image)

        # Step 3: Advanced preprocessing
        preprocessed_images = self._run_advanced_preprocessing(gray_image)

        # Step 4: Hybrid region separation
        fiber_properties, zone_masks = self._locate_fiber_and_define_zones(preprocessed_images)

        # Step 5: Exhaustive parallel detection
        all_raw_detections = self._run_all_detectors(preprocessed_images, zone_masks)

        # Step 6: Ensemble combination & filtering
        ensemble_masks = self._ensemble_detections(all_raw_detections, gray_image.shape)

        # Step 7: Advanced segmentation refinement
        refined_masks = self._refine_and_segment_defects(ensemble_masks, preprocessed_images)

        # Step 8: Meticulous defect characterization
        defects = self._characterize_defects_from_masks(refined_masks, original_image, 
                                                        gray_image, preprocessed_images, 
                                                        zone_masks)

        # Step 9: Final report and visualization
        final_report = self._generate_final_report(defects, global_analysis_results, 
                                                 fiber_properties, image_path, 
                                                 time.time() - start_time)

        # Generate visualization
        self._visualize_master_results(original_image, gray_image, zone_masks, 
                                       refined_masks, defects, global_analysis_results, 
                                       final_report, image_path)

        return final_report

    def _load_and_prepare_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1: Load and prepare image with robust handling

        Handles standard image formats and Jake's JSON pixel data format
        """
        self.logger.info("Step 1: Loading and preparing image")

        # Check if JSON format
        if image_path.endswith('.json'):
            original_image = self._load_json_image(image_path)
        else:
            # Load standard image format
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image from: {image_path}")

        # Convert to grayscale
        if len(original_image.shape) == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = original_image.copy()

        self.logger.info(f"Image loaded successfully. Shape: {original_image.shape}")

        return original_image, gray_image

    def _load_json_image(self, json_path: str) -> np.ndarray:
        """Load image from Jake's JSON pixel data format"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Determine image dimensions
        max_x = max(p['coordinates']['x'] for p in data['pixels']) + 1
        max_y = max(p['coordinates']['y'] for p in data['pixels']) + 1

        # Create image array
        if 'bgr_intensity' in data['pixels'][0]:
            image = np.zeros((max_y, max_x, 3), dtype=np.uint8)
            for pixel in data['pixels']:
                x, y = pixel['coordinates']['x'], pixel['coordinates']['y']
                image[y, x] = pixel['bgr_intensity']
        else:
            image = np.zeros((max_y, max_x), dtype=np.uint8)
            for pixel in data['pixels']:
                x, y = pixel['coordinates']['x'], pixel['coordinates']['y']
                image[y, x] = pixel['intensity']

        return image

    def _run_global_analysis(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """
        Step 2: Global statistical & topological analysis

        Fast holistic screening for immediate quality verdict
        """
        self.logger.info("Step 2: Running global statistical & topological analysis")

        results = {}

        # Statistical analysis
        if self.config.use_global_anomaly_analysis:
            stat_results = self.statistical_analyzer.detect_anomalies_comprehensive(gray_image)
            results['statistical'] = stat_results
            self.logger.info(f"Statistical verdict: {stat_results['verdict']}, "
                             f"Mahalanobis: {stat_results['mahalanobis_distance']:.2f}")

        # Topological analysis (will run on fiber region after masking)
        results['topological'] = {'pending': True}

        return results

    def _run_advanced_preprocessing(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Step 3: Advanced preprocessing

        Creates portfolio of task-specific images for optimal detector performance
        """
        self.logger.info("Step 3: Running advanced preprocessing")

        preprocessed = {
            'original_gray': gray_image.copy()
        }

        # Gaussian blurs
        for size in self.config.gaussian_blur_sizes:
            key = f'gaussian_{size[0]}'
            preprocessed[key] = cv2.GaussianBlur(gray_image, size, 0)

        # Anisotropic diffusion
        if self.config.use_anisotropic_diffusion:
            preprocessed['anisotropic'] = self._anisotropic_diffusion(gray_image)

        # Coherence-enhancing diffusion
        if self.config.use_coherence_enhancing_diffusion:
            preprocessed['coherence_enhanced'] = self._coherence_enhancing_diffusion(gray_image)

        # Change magnitude map
        preprocessed['change_magnitude'] = self._compute_change_magnitude(gray_image)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        preprocessed['clahe'] = clahe.apply(gray_image)

        # Additional preprocessing
        preprocessed['median'] = cv2.medianBlur(gray_image, 5)
        preprocessed['bilateral'] = cv2.bilateralFilter(gray_image, 9, 75, 75)

        self.logger.info(f"Generated {len(preprocessed)} preprocessed images")

        return preprocessed

    def _anisotropic_diffusion(self, image: np.ndarray, iterations: int = 10, 
                                 kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """Perona-Malik anisotropic diffusion"""
        img = image.copy().astype(np.float32)

        for _ in range(iterations):
            # Calculate gradients
            nablaE = np.roll(img, -1, axis=1) - img
            nablaW = np.roll(img, 1, axis=1) - img
            nablaN = np.roll(img, -1, axis=0) - img
            nablaS = np.roll(img, 1, axis=0) - img

            # Diffusion coefficients
            cE = 1.0 / (1.0 + (nablaE/kappa)**2)
            cW = 1.0 / (1.0 + (nablaW/kappa)**2)
            cN = 1.0 / (1.0 + (nablaN/kappa)**2)
            cS = 1.0 / (1.0 + (nablaS/kappa)**2)

            # Update
            img += gamma * (cE*nablaE + cW*nablaW + cN*nablaN + cS*nablaS)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _coherence_enhancing_diffusion(self, image: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Coherence-enhancing diffusion for linear structures"""
        img = image.copy().astype(np.float32)

        for _ in range(iterations):
            # Compute structure tensor
            Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

            # Structure tensor components
            Jxx = gaussian_filter(Ix * Ix, 1.0)
            Jxy = gaussian_filter(Ix * Iy, 1.0)
            Jyy = gaussian_filter(Iy * Iy, 1.0)

            # Apply diffusion
            img = gaussian_filter(img, 0.5)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _compute_change_magnitude(self, image: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude map"""
        # Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return magnitude.astype(np.uint8)

    def _locate_fiber_and_define_zones(self, preprocessed_images: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Step 4: Hybrid region separation

        Uses ensemble method with multi-stage fallback for robust fiber detection
        """
        self.logger.info("Step 4: Locating fiber and defining zones")

        # Try primary ensemble method
        if self.config.primary_masking_method == 'ensemble':
            fiber_props = self._ensemble_fiber_detection(preprocessed_images)
        else:
            fiber_props = None

        # Fallback strategies if primary fails
        if fiber_props is None or fiber_props['confidence'] < 0.5:
            self.logger.warning("Primary fiber detection failed, trying fallback methods")
            fiber_props = self._fallback_fiber_detection(preprocessed_images)

        # Create zone masks
        image_shape = preprocessed_images['original_gray'].shape
        zone_masks = self._create_zone_masks(image_shape, fiber_props)

        # Complete topological analysis now that we have fiber region
        if self.config.use_topological_analysis and 'topological' in self.intermediate_results:
            fiber_mask = zone_masks['fiber_complete']
            fiber_region = preprocessed_images['original_gray'][fiber_mask > 0]
            if fiber_region.size > 0:
                tda_results = self.tda_analyzer.analyze_region(fiber_region)
                connectivity_scores = tda_results['connectivity_indices']
                global_connectivity = np.mean(connectivity_scores) if len(connectivity_scores) > 0 else 1.0
                self.intermediate_results['topological']['global_connectivity'] = global_connectivity
                self.logger.info(f"Global connectivity score: {global_connectivity:.3f}")

        self.logger.info(f"Fiber detected at ({fiber_props['center'][0]}, {fiber_props['center'][1]}) "
                         f"with radius {fiber_props['radius']:.1f}px")

        return fiber_props, zone_masks

    def _ensemble_fiber_detection(self, preprocessed_images: Dict[str, np.ndarray]) -> Optional[Dict]:
        """Ensemble method for fiber center detection"""
        candidates = []

        # Run Hough circles on multiple preprocessed images
        for img_name in ['gaussian_5', 'bilateral', 'clahe', 'median']:
            if img_name not in preprocessed_images:
                continue

            img = preprocessed_images[img_name]

            for dp in self.config.hough_dp_values:
                circles = cv2.HoughCircles(
                    img, cv2.HOUGH_GRADIENT, dp=dp,
                    minDist=img.shape[0]//8,
                    param1=100, param2=30,
                    minRadius=img.shape[0]//10,
                    maxRadius=img.shape[0]//2
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0]:
                        candidates.append({
                            'center': (int(circle[0]), int(circle[1])),
                            'radius': float(circle[2])
                        })

        if not candidates:
            return None

        # Use median for robustness
        centers = np.array([c['center'] for c in candidates])
        radii = np.array([c['radius'] for c in candidates])

        best_center = (int(np.median(centers[:, 0])), int(np.median(centers[:, 1])))
        best_radius = float(np.median(radii))

        # Calculate confidence based on variance
        center_std = np.std(centers, axis=0).mean()
        radius_std = np.std(radii)
        confidence = 1.0 / (1 + center_std/10 + radius_std/20)

        return {
            'center': best_center,
            'radius': best_radius,
            'confidence': confidence
        }

    def _fallback_fiber_detection(self, preprocessed_images: Dict[str, np.ndarray]) -> Dict:
        """Fallback fiber detection methods"""
        img = preprocessed_images['original_gray']

        # Method 1: Largest component
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            return {
                'center': (int(x), int(y)),
                'radius': float(radius),
                'confidence': 0.7
            }

        # Method 2: Image center estimation
        h, w = img.shape
        return {
            'center': (w//2, h//2),
            'radius': min(h, w) // 4,
            'confidence': 0.3
        }

    def _create_zone_masks(self, image_shape: Tuple[int, int], fiber_props: Dict) -> Dict[str, np.ndarray]:
        """Create masks for different fiber zones"""
        h, w = image_shape
        y_coords, x_coords = np.ogrid[:h, :w]

        center = fiber_props['center']
        cladding_radius = fiber_props['radius']

        # Calculate zone radii
        core_radius = cladding_radius * 0.072  # Standard 9/125 ratio
        ferrule_radius = cladding_radius * 2.0
        adhesive_radius = ferrule_radius * 1.1

        # Distance from center
        dist_from_center = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)

        # Create masks
        masks = {
            'core': (dist_from_center <= core_radius).astype(np.uint8) * 255,
            'cladding': ((dist_from_center > core_radius) & 
                         (dist_from_center <= cladding_radius)).astype(np.uint8) * 255,
            'ferrule': ((dist_from_center > cladding_radius) & 
                        (dist_from_center <= ferrule_radius)).astype(np.uint8) * 255,
            'adhesive': ((dist_from_center > ferrule_radius) & 
                         (dist_from_center <= adhesive_radius)).astype(np.uint8) * 255,
            'fiber_complete': (dist_from_center <= cladding_radius).astype(np.uint8) * 255
        }

        return masks

    def _run_all_detectors(self, preprocessed_images: Dict[str, np.ndarray], 
                         zone_masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Step 5: Exhaustive parallel detection

        Runs all enabled detectors on appropriate preprocessed images
        """
        self.logger.info("Step 5: Running exhaustive parallel detection")

        # Detector registry mapping names to functions
        detector_registry = {
            'do2mr': self._detector_do2mr,
            'lei': self._detector_lei,
            'log': self._detector_log,
            'doh': self._detector_doh,
            'hessian_eigen': self._detector_hessian_eigen,
            'frangi': self._detector_frangi,
            'structure_tensor': self._detector_structure_tensor,
            'robust_pca': self._detector_robust_pca,
            'mser': self._detector_mser,
            'lbp': self._detector_lbp,
            'watershed': self._detector_watershed
        }

        # Map detectors to optimal preprocessed images
        detector_image_map = {
            'do2mr': 'clahe',
            'lei': 'coherence_enhanced',
            'log': 'gaussian_5',
            'doh': 'gaussian_5',
            'hessian_eigen': 'coherence_enhanced',
            'frangi': 'coherence_enhanced',
            'structure_tensor': 'anisotropic',
            'robust_pca': 'original_gray',
            'mser': 'clahe',
            'lbp': 'original_gray',
            'watershed': 'bilateral'
        }

        all_detections = {}

        # Run detectors for each zone
        for zone_name, zone_mask in zone_masks.items():
            if zone_name == 'fiber_complete':
                continue

            self.logger.info(f"Processing zone: {zone_name}")
            zone_detections = {}

            for detector_name in self.config.enabled_detectors:
                if detector_name in detector_registry:
                    # Get optimal image for this detector
                    img_key = detector_image_map.get(detector_name, 'original_gray')
                    if img_key in preprocessed_images:
                        img = preprocessed_images[img_key]
                    else:
                        img = preprocessed_images['original_gray']

                    # Run detector
                    try:
                        detector_func = detector_registry[detector_name]
                        detection_mask = detector_func(img, zone_mask)
                        zone_detections[detector_name] = detection_mask
                    except Exception as e:
                        self.logger.warning(f"Detector {detector_name} failed: {e}")
                        zone_detections[detector_name] = np.zeros_like(zone_mask)

            all_detections[zone_name] = zone_detections

        self.logger.info(f"Completed {len(self.config.enabled_detectors)} detectors on {len(zone_masks)-1} zones")

        return all_detections

    # Individual detector implementations
    def _detector_do2mr(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """DO2MR detector implementation"""
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        img_max = cv2.dilate(image, kernel)
        img_min = cv2.erode(image, kernel)
        residual = cv2.absdiff(img_max, img_min)

        # Threshold based on statistics
        masked_values = residual[zone_mask > 0]
        if len(masked_values) > 0:
            mean_val = np.mean(masked_values)
            std_val = np.std(masked_values)
            threshold = mean_val + 2.5 * std_val

            _, binary = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_and(binary, binary, mask=zone_mask)
        else:
            binary = np.zeros_like(zone_mask)

        return binary

    def _detector_lei(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """LEI scratch detector implementation"""
        scratch_strength = np.zeros_like(image, dtype=np.float32)

        kernel_length = 15
        for angle in range(0, 180, 15):
            # Create directional kernel
            kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
            cv2.line(kernel, (0, kernel_length//2), (kernel_length-1, kernel_length//2), 1.0, 1)

            # Rotate kernel
            M = cv2.getRotationMatrix2D((kernel_length//2, kernel_length//2), angle, 1.0)
            rotated_kernel = cv2.warpAffine(kernel, M, (kernel_length, kernel_length))

            if np.sum(rotated_kernel) > 0:
                rotated_kernel /= np.sum(rotated_kernel)

            # Apply filter
            response = cv2.filter2D(image.astype(np.float32), -1, rotated_kernel)
            scratch_strength = np.maximum(scratch_strength, response)

        # Threshold
        _, binary = cv2.threshold(scratch_strength.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _detector_log(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Laplacian of Gaussian detector"""
        blob_response = np.zeros_like(image, dtype=np.float32)

        for sigma in [2, 3, 4, 5]:
            # Apply LoG
            log = ndimage.gaussian_laplace(image.astype(np.float32), sigma)
            log *= sigma**2  # Scale normalization
            blob_response = np.maximum(blob_response, np.abs(log))

        # Threshold
        _, binary = cv2.threshold(blob_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _detector_doh(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Determinant of Hessian detector"""
        doh_response = np.zeros_like(image, dtype=np.float32)

        for sigma in [2, 4, 6]:
            smoothed = gaussian_filter(image.astype(np.float32), sigma)

            # Compute Hessian
            Hxx = gaussian_filter(smoothed, sigma, order=[0, 2])
            Hyy = gaussian_filter(smoothed, sigma, order=[2, 0])
            Hxy = gaussian_filter(smoothed, sigma, order=[1, 1])

            # Determinant
            det = Hxx * Hyy - Hxy**2
            det *= sigma**4  # Scale normalization

            doh_response = np.maximum(doh_response, np.abs(det))

        # Threshold
        _, binary = cv2.threshold(doh_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _detector_hessian_eigen(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Hessian eigenvalue ratio detector"""
        ridge_response = np.zeros_like(image, dtype=np.float32)

        for sigma in [1, 2, 3]:
            smoothed = gaussian_filter(image.astype(np.float32), sigma)

            # Compute Hessian
            Hxx = gaussian_filter(smoothed, sigma, order=[0, 2])
            Hyy = gaussian_filter(smoothed, sigma, order=[2, 0])
            Hxy = gaussian_filter(smoothed, sigma, order=[1, 1])

            # Eigenvalues
            trace = Hxx + Hyy
            det = Hxx * Hyy - Hxy**2
            discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))

            lambda1 = 0.5 * (trace + discriminant)
            lambda2 = 0.5 * (trace - discriminant)

            # Ridge measure
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)

            # Response
            response = np.exp(-Rb**2 / 2) * (lambda2 < 0).astype(float)
            ridge_response = np.maximum(ridge_response, response)

        # Threshold
        _, binary = cv2.threshold(ridge_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _detector_frangi(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Frangi vesselness filter"""
        vesselness = np.zeros_like(image, dtype=np.float32)

        for sigma in [1, 1.5, 2, 2.5]:
            smoothed = gaussian_filter(image.astype(np.float32), sigma)

            # Compute Hessian
            Hxx = gaussian_filter(smoothed, sigma, order=[0, 2])
            Hyy = gaussian_filter(smoothed, sigma, order=[2, 0])
            Hxy = gaussian_filter(smoothed, sigma, order=[1, 1])

            # Eigenvalues
            tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
            lambda1 = 0.5 * (Hxx + Hyy + tmp)
            lambda2 = 0.5 * (Hxx + Hyy - tmp)

            # Frangi measure
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)

            beta = 0.5
            gamma = 15

            v = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*gamma**2)))
            v[lambda2 > 0] = 0

            vesselness = np.maximum(vesselness, v)

        # Threshold
        _, binary = cv2.threshold(vesselness.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _detector_structure_tensor(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Structure tensor coherence detector"""
        # Compute gradients
        Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Structure tensor components
        Jxx = gaussian_filter(Ix * Ix, 2.0)
        Jxy = gaussian_filter(Ix * Iy, 2.0)
        Jyy = gaussian_filter(Iy * Iy, 2.0)

        # Eigenvalues
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))

        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)

        # Coherence
        coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10))**2

        # Threshold high coherence regions
        _, binary = cv2.threshold((coherence * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _detector_robust_pca(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Robust PCA detector (simplified)"""
        # Extract region of interest
        y_indices, x_indices = np.where(zone_mask > 0)
        if len(y_indices) == 0:
            return np.zeros_like(zone_mask)

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        roi = image[y_min:y_max+1, x_min:x_max+1]

        # Simple outlier detection using local statistics
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        local_mean = cv2.filter2D(roi.astype(float), -1, kernel)
        local_var = cv2.filter2D((roi - local_mean)**2, -1, kernel)

        # Z-score based outlier detection
        z_score = np.abs(roi - local_mean) / (np.sqrt(local_var) + 1e-6)

        # Create full-size mask
        outlier_mask = np.zeros_like(image)
        outlier_mask[y_min:y_max+1, x_min:x_max+1] = (z_score > 3).astype(np.uint8) * 255
        outlier_mask = cv2.bitwise_and(outlier_mask, outlier_mask, mask=zone_mask)

        return outlier_mask

    def _detector_mser(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """MSER detector"""
        mser = cv2.MSER_create(delta=5, min_area=5, max_area=500)
        regions, _ = mser.detectRegions(image)

        mask = np.zeros_like(image)
        for region in regions:
            cv2.fillPoly(mask, [region], 255)

        mask = cv2.bitwise_and(mask, mask, mask=zone_mask)
        return mask

    def _detector_lbp(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """LBP-based anomaly detector"""
        # Compute local texture variance
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        local_mean = cv2.filter2D(image.astype(float), -1, kernel)
        local_var = cv2.filter2D((image - local_mean)**2, -1, kernel)

        # Threshold high variance regions
        _, binary = cv2.threshold(local_var.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _detector_watershed(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Watershed segmentation detector"""
        # Simple gradient-based watershed
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))

        # Threshold
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=zone_mask)

        return binary

    def _ensemble_detections(self, all_detections: Dict[str, Dict[str, np.ndarray]], 
                           image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Step 6: Ensemble combination & filtering

        Fuses all detection results using weighted voting
        """
        self.logger.info("Step 6: Ensemble combination and filtering")

        h, w = image_shape
        ensemble_masks = {}

        # Process each zone
        for zone_name, zone_detections in all_detections.items():
            # Initialize vote accumulator
            vote_map = np.zeros((h, w), dtype=np.float32)

            # Accumulate weighted votes
            for detector_name, mask in zone_detections.items():
                if mask is not None and detector_name in self.config.ensemble_confidence_weights:
                    weight = self.config.ensemble_confidence_weights[detector_name]
                    vote_map += (mask > 0).astype(np.float32) * weight

            # Normalize votes
            max_possible_vote = sum(self.config.ensemble_confidence_weights.values())
            vote_map /= max_possible_vote

            # Apply threshold
            consensus_mask = (vote_map >= self.config.ensemble_vote_threshold).astype(np.uint8) * 255

            # Apply false positive reduction
            if self.config.use_geometric_fp_reduction or self.config.use_contrast_fp_reduction:
                consensus_mask = self._reduce_false_positives(consensus_mask, zone_name)

            ensemble_masks[zone_name] = consensus_mask

        self.logger.info(f"Generated ensemble masks for {len(ensemble_masks)} zones")

        return ensemble_masks

    def _reduce_false_positives(self, mask: np.ndarray, zone_name: str) -> np.ndarray:
        """Apply false positive reduction techniques"""
        refined = mask.copy()

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Remove too small components
            if area < self.config.min_defect_area_px:
                refined[labels == i] = 0
                continue

            # Geometric filtering for scratches
            if 'core' in zone_name or 'cladding' in zone_name:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

                # Check for line-like features
                if aspect_ratio > 3:
                    # Verify it's actually a line
                    component_mask = (labels == i).astype(np.uint8)
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours and len(contours[0]) >= 5:
                        # Fit line and check residuals
                        [vx, vy, x0, y0] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)

                        # Calculate mean distance from points to line
                        residuals = []
                        for point in contours[0]:
                            px, py = point[0]
                            dist = abs((py - y0) * vx - (px - x0) * vy) / np.sqrt(vx**2 + vy**2)
                            residuals.append(dist)

                        if np.mean(residuals) > 5:  # Not a clean line
                            refined[labels == i] = 0

        return refined

    def _refine_and_segment_defects(self, ensemble_masks: Dict[str, np.ndarray], 
                                    preprocessed_images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Step 7: Advanced segmentation refinement

        Uses active contours or morphological operations for precise boundaries
        """
        self.logger.info("Step 7: Refining and segmenting defects")

        refined_masks = {}

        for zone_name, mask in ensemble_masks.items():
            if self.config.segmentation_refinement_method == 'active_contour':
                # Use active contours for refinement
                refined = self._refine_with_active_contours(mask, preprocessed_images['original_gray'])
            else:
                # Use morphological refinement
                refined = self._refine_with_morphology(mask)

            refined_masks[zone_name] = refined

        self.logger.info("Segmentation refinement completed")

        return refined_masks

    def _refine_with_active_contours(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Refine mask using active contours"""
        # Find initial contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        refined_mask = np.zeros_like(mask)

        for contour in contours:
            if cv2.contourArea(contour) < self.config.min_defect_area_px:
                continue

            # Convert contour to points
            points = contour.squeeze()
            if len(points) < 10:
                # Too few points for active contour
                cv2.drawContours(refined_mask, [contour], -1, 255, -1)
                continue

            # Create initial snake
            snake = points.astype(float)

            # Apply active contour (simplified version)
            # In practice, you would use skimage.segmentation.active_contour
            # Here we just apply some smoothing
            for _ in range(5):
                # Simple smoothing iteration
                new_snake = snake.copy()
                for i in range(len(snake)):
                    prev_idx = (i - 1) % len(snake)
                    next_idx = (i + 1) % len(snake)
                    new_snake[i] = 0.7 * snake[i] + 0.15 * snake[prev_idx] + 0.15 * snake[next_idx]
                snake = new_snake

            # Draw refined contour
            cv2.drawContours(refined_mask, [snake.astype(np.int32)], -1, 255, -1)

        return refined_mask

    def _refine_with_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Refine mask using morphological operations"""
        # Opening to remove small objects
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Closing to fill small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)

        return refined

    def _characterize_defects_from_masks(self, refined_masks: Dict[str, np.ndarray], 
                                          original_image: np.ndarray, gray_image: np.ndarray,
                                          preprocessed_images: Dict[str, np.ndarray],
                                          zone_masks: Dict[str, np.ndarray]) -> List[Defect]:
        """
        Step 8: Meticulous defect characterization

        Extracts comprehensive feature vector for each defect
        """
        self.logger.info("Step 8: Characterizing defects from masks")

        all_defects = []
        defect_id = 0

        # Get change magnitude map
        change_magnitude = preprocessed_images.get('change_magnitude', gray_image)

        for zone_name, mask in refined_masks.items():
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, num_labels):
                defect_id += 1

                # Extract component mask
                component_mask = (labels == i).astype(np.uint8)

                # Get bounding box
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

                # Extract ROI
                roi_mask = component_mask[y:y+h, x:x+w]
                roi_gray = gray_image[y:y+h, x:x+w]
                roi_gradient = change_magnitude[y:y+h, x:x+w]

                # Calculate geometric properties
                area_px = stats[i, cv2.CC_STAT_AREA]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])

                # Get contour for shape analysis
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = contours[0] if contours else np.array([])

                # Calculate shape properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area_px / (perimeter**2) if perimeter > 0 else 0

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area_px / hull_area if hull_area > 0 else 0

                # Calculate eccentricity
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (_, _), (MA, ma), _ = ellipse
                    eccentricity = np.sqrt(1 - (ma/MA)**2) if MA > 0 else 0
                else:
                    eccentricity = 0

                # Calculate intensity properties
                pixels = roi_gray[roi_mask > 0]
                if len(pixels) > 0:
                    mean_intensity = np.mean(pixels)
                    intensity_skewness = stats.skew(pixels)
                    intensity_kurtosis = stats.kurtosis(pixels)

                    # Calculate contrast
                    dilated = cv2.dilate(roi_mask, np.ones((5,5), np.uint8))
                    surrounding_mask = dilated - roi_mask
                    if np.sum(surrounding_mask) > 0:
                        surrounding_pixels = roi_gray[surrounding_mask > 0]
                        contrast = abs(mean_intensity - np.mean(surrounding_pixels))
                    else:
                        contrast = 0
                else:
                    mean_intensity = 0
                    intensity_skewness = 0
                    intensity_kurtosis = 0
                    contrast = 0

                # Calculate gradient properties
                gradient_pixels = roi_gradient[roi_mask > 0]
                if len(gradient_pixels) > 0:
                    mean_gradient = np.mean(gradient_pixels)
                    max_gradient = np.max(gradient_pixels)
                    std_dev_gradient = np.std(gradient_pixels)
                else:
                    mean_gradient = 0
                    max_gradient = 0
                    std_dev_gradient = 0

                # Calculate texture properties (GLCM)
                glcm_props = self._compute_glcm_features(roi_gray, roi_mask)

                # Calculate structural properties
                struct_props = self._compute_structural_features(roi_gray, roi_mask)

                # Calculate topological properties
                tda_props = self._compute_local_tda_features(roi_gray, roi_mask)

                # Determine defect type and severity
                defect_type = self._classify_defect_type(eccentricity, circularity, area_px)
                severity = self._assess_severity(defect_type, area_px, contrast, zone_name)

                # Create defect object
                defect = Defect(
                    defect_id=defect_id,
                    defect_type=defect_type,
                    zone=zone_name,
                    severity=severity,
                    area_px=area_px,
                    eccentricity=eccentricity,
                    solidity=solidity,
                    circularity=circularity,
                    location_xy=(cx, cy),
                    mean_intensity=mean_intensity,
                    contrast=contrast,
                    intensity_skewness=intensity_skewness,
                    intensity_kurtosis=intensity_kurtosis,
                    mean_gradient=mean_gradient,
                    max_gradient=max_gradient,
                    std_dev_gradient=std_dev_gradient,
                    glcm_contrast=glcm_props['contrast'],
                    glcm_homogeneity=glcm_props['homogeneity'],
                    glcm_energy=glcm_props['energy'],
                    mean_hessian_eigen_ratio=struct_props['hessian_ratio'],
                    mean_coherence=struct_props['coherence'],
                    tda_local_connectivity_score=tda_props['connectivity'],
                    tda_betti_curve_signature=tda_props['betti_signature'],
                    tda_size_distribution_signature=tda_props['size_signature'],
                    contributing_algorithms=['ensemble']  # Could track individual contributors
                )

                all_defects.append(defect)

        self.logger.info(f"Characterized {len(all_defects)} defects")

        return all_defects

    def _compute_glcm_features(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Compute GLCM texture features"""
        # Simplified GLCM calculation
        if roi.size == 0 or np.sum(mask) == 0:
            return {'contrast': 0, 'homogeneity': 0, 'energy': 0}

        # Quantize image
        levels = 8
        roi_quantized = (roi // (256 // levels)).astype(np.uint8)

        # Create GLCM (simplified - just horizontal)
        glcm = np.zeros((levels, levels))

        rows, cols = roi.shape
        for i in range(rows):
            for j in range(cols-1):
                if mask[i, j] > 0 and mask[i, j+1] > 0:
                    glcm[roi_quantized[i, j], roi_quantized[i, j+1]] += 1

        # Normalize
        if glcm.sum() > 0:
            glcm /= glcm.sum()

        # Calculate properties
        contrast = 0
        homogeneity = 0
        energy = 0

        for i in range(levels):
            for j in range(levels):
                contrast += ((i - j)**2) * glcm[i, j]
                homogeneity += glcm[i, j] / (1 + abs(i - j))
                energy += glcm[i, j]**2

        return {
            'contrast': contrast,
            'homogeneity': homogeneity,
            'energy': energy
        }

    def _compute_structural_features(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Compute structural features"""
        if roi.size == 0 or np.sum(mask) == 0:
            return {'hessian_ratio': 0, 'coherence': 0}

        # Compute Hessian eigenvalue ratio
        smoothed = gaussian_filter(roi.astype(float), 1.0)

        Hxx = gaussian_filter(smoothed, 1.0, order=[0, 2])
        Hyy = gaussian_filter(smoothed, 1.0, order=[2, 0])
        Hxy = gaussian_filter(smoothed, 1.0, order=[1, 1])

        # Mean eigenvalue ratio over masked region
        ratios = []
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                if mask[i, j] > 0:
                    # Local Hessian
                    H = np.array([[Hxx[i, j], Hxy[i, j]], 
                                  [Hxy[i, j], Hyy[i, j]]])

                    eigvals = np.linalg.eigvalsh(H)
                    if len(eigvals) == 2 and abs(eigvals[1]) > 1e-6:
                        ratio = abs(eigvals[0]) / abs(eigvals[1])
                        ratios.append(ratio)

        mean_hessian_ratio = np.mean(ratios) if ratios else 0

        # Compute structure tensor coherence
        Ix = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        Jxx = gaussian_filter(Ix * Ix, 1.0)
        Jxy = gaussian_filter(Ix * Iy, 1.0)
        Jyy = gaussian_filter(Iy * Iy, 1.0)

        # Mean coherence over masked region
        coherences = []
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                if mask[i, j] > 0:
                    trace = Jxx[i, j] + Jyy[i, j]
                    det = Jxx[i, j] * Jyy[i, j] - Jxy[i, j]**2

                    if trace > 0:
                        discriminant = trace**2 - 4*det
                        if discriminant >= 0:
                            lambda1 = 0.5 * (trace + np.sqrt(discriminant))
                            lambda2 = 0.5 * (trace - np.sqrt(discriminant))

                            if lambda1 + lambda2 > 0:
                                coherence = ((lambda1 - lambda2) / (lambda1 + lambda2))**2
                                coherences.append(coherence)

        mean_coherence = np.mean(coherences) if coherences else 0

        return {
            'hessian_ratio': mean_hessian_ratio,
            'coherence': mean_coherence
        }

    def _compute_local_tda_features(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Compute local TDA features"""
        if not self.config.use_topological_analysis or not GUDHI_AVAILABLE:
            return {
                'connectivity': 0,
                'betti_signature': [],
                'size_signature': []
            }

        # Run TDA on ROI
        tda_results = self.tda_analyzer.analyze_region(roi)

        # Extract features
        connectivity_scores = tda_results['connectivity_indices']
        connectivity = np.mean(connectivity_scores) if len(connectivity_scores) > 0 else 0

        # Create signatures (simplified)
        betti_signature = tda_results['normalized_betti_curves'].flatten()[:10].tolist() if len(tda_results['normalized_betti_curves']) > 0 else []
        size_signature = tda_results['size_distributions'].flatten()[:10].tolist() if len(tda_results['size_distributions']) > 0 else []

        return {
            'connectivity': connectivity,
            'betti_signature': betti_signature,
            'size_signature': size_signature
        }

    def _classify_defect_type(self, eccentricity: float, circularity: float, area: int) -> str:
        """Classify defect type based on shape properties"""
        if eccentricity > 0.8 and circularity < 0.5:
            return 'Scratch'
        elif circularity > 0.7 and area < 100:
            return 'Dig'
        elif area > 500:
            return 'Contamination'
        else:
            return 'Defect'

    def _assess_severity(self, defect_type: str, area: int, contrast: float, zone: str) -> str:
        """Assess defect severity"""
        # Zone-specific thresholds
        if 'core' in zone:
            if area > 50 or contrast > 100:
                return 'Critical'
            elif area > 20 or contrast > 50:
                return 'High'
            elif area > 10:
                return 'Medium'
            else:
                return 'Low'
        elif 'cladding' in zone:
            if area > 200 or contrast > 150:
                return 'High'
            elif area > 50:
                return 'Medium'
            else:
                return 'Low'
        else:
            if area > 500:
                return 'Medium'
            else:
                return 'Low'

    def _generate_final_report(self, defects: List[Defect], global_analysis: Dict[str, Any],
                               fiber_properties: Dict[str, Any], image_path: str,
                               processing_time: float) -> Dict[str, Any]:
        """
        Step 9: Generate final report

        Combines all analysis results into comprehensive verdict
        """
        self.logger.info("Step 9: Generating final report")

        # Determine pass/fail status
        geometric_fail = any(d.severity == 'Critical' for d in defects)

        statistical_fail = False
        if 'statistical' in global_analysis:
            stat = global_analysis['statistical']
            statistical_fail = (stat['mahalanobis_distance'] > self.config.anomaly_mahalanobis_threshold or
                                stat['ssim_score'] < self.config.anomaly_ssim_threshold)

        topological_fail = False
        if 'topological' in self.intermediate_results:
            topo = self.intermediate_results['topological']
            if 'global_connectivity' in topo:
                topological_fail = topo['global_connectivity'] < self.config.min_global_connectivity

        overall_status = 'FAIL' if (geometric_fail or statistical_fail or topological_fail) else 'PASS'

        # Compile failure reasons
        failure_reasons = []
        if geometric_fail:
            critical_defects = [d for d in defects if d.severity == 'Critical']
            failure_reasons.append(f"Critical defects found: {len(critical_defects)}")
        if statistical_fail:
            failure_reasons.append("Statistical anomaly detected")
        if topological_fail:
            failure_reasons.append("Low topological connectivity")

        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'processing_time_seconds': processing_time,
            'overall_status': overall_status,
            'failure_reasons': failure_reasons,

            'fiber_properties': fiber_properties,

            'global_analysis': {
                'statistical': global_analysis.get('statistical', {}),
                'topological': self.intermediate_results.get('topological', {})
            },

            'defect_summary': {
                'total_defects': len(defects),
                'by_severity': {
                    'Critical': sum(1 for d in defects if d.severity == 'Critical'),
                    'High': sum(1 for d in defects if d.severity == 'High'),
                    'Medium': sum(1 for d in defects if d.severity == 'Medium'),
                    'Low': sum(1 for d in defects if d.severity == 'Low')
                },
                'by_type': {
                    'Scratch': sum(1 for d in defects if d.defect_type == 'Scratch'),
                    'Dig': sum(1 for d in defects if d.defect_type == 'Dig'),
                    'Contamination': sum(1 for d in defects if d.defect_type == 'Contamination'),
                    'Other': sum(1 for d in defects if d.defect_type not in ['Scratch', 'Dig', 'Contamination'])
                },
                'by_zone': {
                    'core': sum(1 for d in defects if d.zone == 'core'),
                    'cladding': sum(1 for d in defects if d.zone == 'cladding'),
                    'ferrule': sum(1 for d in defects if d.zone == 'ferrule'),
                    'adhesive': sum(1 for d in defects if d.zone == 'adhesive')
                }
            },

            'defects': [asdict(d) for d in defects]
        }

        # Save JSON report if configured
        if self.config.generate_json_report:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            json_path = os.path.join(output_dir, f"{base_name}_omni_report.json")
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Saved JSON report to: {json_path}")

        self.logger.info(f"Analysis complete. Status: {overall_status}")

        return report

    def _visualize_master_results(self, original_image: np.ndarray, gray_image: np.ndarray,
                                  zone_masks: Dict[str, np.ndarray], refined_masks: Dict[str, np.ndarray],
                                  defects: List[Defect], global_analysis: Dict[str, Any],
                                  final_report: Dict[str, Any], image_path: str):
        """Generate comprehensive visualization"""
        self.logger.info("Generating master visualization")

        # Create figure with GridSpec
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Convert to RGB for display
        if len(original_image.shape) == 3:
            display_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            display_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        # Panel 1: Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(display_image)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Panel 2: Zone masks
        ax2 = fig.add_subplot(gs[0, 1])
        zone_overlay = np.zeros_like(display_image)
        zone_colors = {
            'core': [255, 0, 0],
            'cladding': [0, 255, 0],
            'ferrule': [0, 0, 255],
            'adhesive': [255, 255, 0]
        }
        for zone_name, mask in zone_masks.items():
            if zone_name in zone_colors:
                zone_overlay[mask > 0] = zone_colors[zone_name]

        ax2.imshow(zone_overlay)
        ax2.set_title('Fiber Zones', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Create legend for zones
        patches = []
        for zone, color in zone_colors.items():
            patches.append(mpatches.Patch(color=np.array(color)/255, label=zone.capitalize()))
        ax2.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

        # Panel 3: Defect overlay
        ax3 = fig.add_subplot(gs[0, 2])
        defect_overlay = display_image.copy()

        # Draw defects with color coding by severity
        severity_colors = {
            'Critical': (255, 0, 0),
            'High': (255, 128, 0),
            'Medium': (255, 255, 0),
            'Low': (0, 255, 0)
        }

        for defect in defects:
            color = severity_colors.get(defect.severity, (255, 255, 255))
            cv2.circle(defect_overlay, defect.location_xy, 
                       max(3, int(np.sqrt(defect.area_px))), color, 2)
            cv2.putText(defect_overlay, str(defect.defect_id), 
                        (defect.location_xy[0] + 5, defect.location_xy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        ax3.imshow(defect_overlay)
        ax3.set_title(f'Detected Defects (n={len(defects)})', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Panel 4: Anomaly heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        if 'statistical' in global_analysis and 'anomaly_map' in global_analysis['statistical']:
            anomaly_map = global_analysis['statistical']['anomaly_map']
            im4 = ax4.imshow(anomaly_map, cmap='hot')
            ax4.set_title('Statistical Anomaly Map', fontsize=12, fontweight='bold')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
        else:
            ax4.text(0.5, 0.5, 'No anomaly map available', ha='center', va='center')
            ax4.set_title('Statistical Anomaly Map', fontsize=12, fontweight='bold')
        ax4.axis('off')

        # Panel 5: Detection masks combined
        ax5 = fig.add_subplot(gs[1, :2])
        combined_mask = np.zeros_like(gray_image)
        for zone_name, mask in refined_masks.items():
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        ax5.imshow(combined_mask, cmap='gray')
        ax5.set_title('Combined Detection Masks', fontsize=12, fontweight='bold')
        ax5.axis('off')

        # Panel 6: Defect statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        # Prepare statistics text
        stats_text = f"""Defect Statistics
{'='*25}
Total Defects: {final_report['defect_summary']['total_defects']}

By Severity:
  Critical: {final_report['defect_summary']['by_severity']['Critical']}
  High: {final_report['defect_summary']['by_severity']['High']}
  Medium: {final_report['defect_summary']['by_severity']['Medium']}
  Low: {final_report['defect_summary']['by_severity']['Low']}

By Type:
  Scratch: {final_report['defect_summary']['by_type']['Scratch']}
  Dig: {final_report['defect_summary']['by_type']['Dig']}
  Contamination: {final_report['defect_summary']['by_type']['Contamination']}

By Zone:
  Core: {final_report['defect_summary']['by_zone']['core']}
  Cladding: {final_report['defect_summary']['by_zone']['cladding']}
  Ferrule: {final_report['defect_summary']['by_zone']['ferrule']}"""

        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 7: Global analysis results
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.axis('off')

        # Prepare global analysis text
        global_text = f"""Global Analysis
{'='*25}
Statistical Analysis:"""

        if 'statistical' in global_analysis:
            stat = global_analysis['statistical']
            global_text += f"""
  Verdict: {stat.get('verdict', 'N/A')}
  Mahalanobis: {stat.get('mahalanobis_distance', 0):.2f}
  SSIM: {stat.get('ssim_score', 1):.3f}"""

        global_text += f"\n\nTopological Analysis:"
        if 'topological' in self.intermediate_results:
            topo = self.intermediate_results['topological']
            global_text += f"""
  Connectivity: {topo.get('global_connectivity', 0):.3f}"""

        global_text += f"""

Processing Time: {final_report['processing_time_seconds']:.2f}s"""

        ax7.text(0.05, 0.95, global_text, transform=ax7.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Panel 8: Final verdict
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis('off')

        # Prepare verdict text
        status = final_report['overall_status']
        status_color = 'green' if status == 'PASS' else 'red'

        verdict_text = f"FINAL VERDICT: {status}"
        if final_report['failure_reasons']:
            verdict_text += "\n\nFailure Reasons:"
            for reason in final_report['failure_reasons']:
                verdict_text += f"\n  • {reason}"

        ax8.text(0.5, 0.5, verdict_text, transform=ax8.transAxes,
                 fontsize=16, fontweight='bold', color=status_color,
                 ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor=status_color, linewidth=3))

        # Main title
        fig.suptitle('OmniFiberAnalyzer - Comprehensive End-Face Analysis', 
                     fontsize=18, fontweight='bold')

        # Save figure
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_omni_analysis.png")
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved visualization to: {output_path}")


# ==================== MAIN EXECUTION (MODIFIED) ====================

def main():
    """Main execution function with interactive menu."""
    
    # --- Configuration Setup ---
    config = OmniConfig()
    
    # --- Interactive Menu ---
    knowledge_base_path = None
    image_path = None

    while True:
        print("\n" + "="*50)
        print(" OmniFiberAnalyzer - Interactive Menu")
        print("="*50)
        print("Please select an option for the JSON dataset (knowledge base):")
        print("1. Select path to an existing JSON dataset file")
        print("2. Add to a JSON dataset file (select existing file)")
        print("3. Create a new JSON dataset file (specify new path)")
        print("="*50)
        
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1' or choice == '2':
            prompt = "Enter the path to the existing JSON dataset file: "
            knowledge_base_path = input(prompt)
            if os.path.exists(knowledge_base_path):
                print(f"Selected knowledge base: {knowledge_base_path}")
                break
            else:
                print(f"Error: File not found at '{knowledge_base_path}'. Please try again.")
        
        elif choice == '3':
            prompt = "Enter the path for the new JSON dataset file to be created: "
            knowledge_base_path = input(prompt)
            print(f"Path for new knowledge base set to: {knowledge_base_path}")
            print("(Note: The script will run without a reference model and will not create the file itself.)")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # --- Get Image Path ---
    while True:
        image_path = input("\nEnter the full path to the image you want to analyze: ")
        if os.path.exists(image_path):
            print(f"Image for analysis: {image_path}")
            break
        else:
            print(f"Error: Image file not found at '{image_path}'. Please try again.")

    # Apply user-defined paths to the configuration
    config.knowledge_base_path = knowledge_base_path
    
    # --- Run Analysis ---
    # Create analyzer
    analyzer = OmniFiberAnalyzer(config)
    
    try:
        results = analyzer.analyze_end_face(image_path)
        
        # --- Print Summary ---
        print("\n" + "="*60)
        print("OMNIFIBERANALYZER RESULTS")
        print("="*60)
        print(f"Status: {results['overall_status']}")
        print(f"Processing Time: {results['processing_time_seconds']:.2f} seconds")
        print(f"Total Defects: {results['defect_summary']['total_defects']}")
        
        if results['failure_reasons']:
            print("\nFailure Reasons:")
            for reason in results['failure_reasons']:
                print(f"  - {reason}")
        
        print("\nDefects by Severity:")
        for severity, count in results['defect_summary']['by_severity'].items():
            if count > 0:
                print(f"  {severity}: {count}")
        
        print("\nDefects by Zone:")
        for zone, count in results['defect_summary']['by_zone'].items():
            if count > 0:
                print(f"  {zone}: {count}")

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        
        print("\nResults saved to:")
        print(f"  - Visualization: {os.path.join(output_dir, f'{base_name}_omni_analysis.png')}")
        print(f"  - JSON Report:   {os.path.join(output_dir, f'{base_name}_omni_report.json')}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()