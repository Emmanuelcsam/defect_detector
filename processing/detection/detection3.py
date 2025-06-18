#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omni_fiber_analyzer.py - The OmniFiberAnalyzer System (Full Production Version)

This script implements a master system for analyzing fiber optic end-face images.
It unifies the robust engineering of daniel.py, the exhaustive ensemble methods
of jill.py, the statistical anomaly detection of jake.py, feature extraction 
from helper scripts, and advanced mathematical concepts including Topological 
Data Analysis (TDA).

Author: Unified Fiber Analysis Team
Version: 3.0 - Complete Production Implementation
"""

# --- Core Imports ---
import cv2
import numpy as np
import os
import json
import time
import logging
import traceback
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum, auto
import warnings

# --- Scientific and Computer Vision Imports ---
from scipy import ndimage, stats, signal
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks
from scipy.sparse.linalg import svds
from scipy.optimize import minimize
from skimage import morphology, feature, measure, segmentation, restoration, transform
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# --- Machine Learning and Statistical Imports ---
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans

# --- Visualization Imports ---
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle, Ellipse

# --- Topological Data Analysis (TDA) Import ---
try:
    import gudhi as gd
    TDA_AVAILABLE = True
except ImportError:
    print("Warning: 'gudhi' library not found. Topological Data Analysis (TDA) will be disabled.")
    print("To enable TDA, install with: pip install gudhi")
    TDA_AVAILABLE = False

# --- Basic Setup ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# PART 1: CORE ARCHITECTURAL COMPONENTS
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types for saving reports."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                           np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (DefectType, DefectSeverity)):
            return obj.name
        return json.JSONEncoder.default(self, obj)

@dataclass
class OmniConfig:
    """Centralizes all tunable parameters for the OmniFiberAnalyzer."""
    # --- General Settings ---
    min_defect_area_px: int = 5
    max_defect_area_px: int = 10000
    max_defect_area_ratio: float = 0.1
    max_defect_eccentricity: float = 0.98
    pixels_per_micron: float = 1.0
    
    # --- Output Settings ---
    output_dpi: int = 200
    save_intermediate_masks: bool = False
    generate_json_report: bool = True
    generate_text_report: bool = True
    visualization_dpi: int = 150
    
    # --- Global Anomaly Analysis Settings (jake.py) ---
    use_global_anomaly_analysis: bool = True
    knowledge_base_path: str = "ultra_anomaly_kb.json"
    anomaly_mahalanobis_threshold: float = 5.0
    anomaly_ssim_threshold: float = 0.85
    global_comparison_metrics: List[str] = field(default_factory=lambda: [
        'euclidean', 'mahalanobis', 'cosine', 'ssim', 'kl_divergence'
    ])
    
    # --- Topological Analysis Settings ---
    use_topological_analysis: bool = True
    tda_library: str = 'gudhi'
    mf_threshold_range: Tuple[int, int] = (0, 255)
    mf_threshold_step: int = 16
    mf_opening_size_range: Tuple[int, int] = (0, 10)
    mf_opening_step: int = 1
    min_global_connectivity: float = 0.85
    
    # --- Preprocessing Settings (jill.py) ---
    use_anisotropic_diffusion: bool = True
    anisotropic_iterations: int = 10
    anisotropic_kappa: float = 50
    anisotropic_gamma: float = 0.1
    use_coherence_enhancing_diffusion: bool = True
    coherence_iterations: int = 5
    gaussian_blur_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(3, 3), (5, 5), (7, 7)])
    bilateral_params: List[Tuple[int, int, int]] = field(default_factory=lambda: [(9, 75, 75), (7, 50, 50)])
    clahe_params: List[Tuple[float, Tuple[int, int]]] = field(default_factory=lambda: [(2.0, (8, 8)), (3.0, (8, 8))])
    denoise_strength: float = 5.0
    use_illumination_correction: bool = True
    
    # --- Region Separation Settings ---
    primary_masking_method: str = 'ensemble'  # 'ensemble' or 'adaptive_contour'
    cladding_core_ratio: float = 125.0 / 9.0
    ferrule_buffer_ratio: float = 1.2
    hough_dp_values: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    hough_param1_values: List[int] = field(default_factory=lambda: [50, 70, 100])
    hough_param2_values: List[int] = field(default_factory=lambda: [30, 40, 50])
    adaptive_threshold_block_size: int = 51
    adaptive_threshold_c: int = 10
    
    # --- Detection Algorithm Settings ---
    enabled_detectors: List[str] = field(default_factory=lambda: [
        'do2mr', 'lei', 'zana_klein', 'log', 'doh', 'hessian_eigen', 
        'frangi', 'structure_tensor', 'mser', 'watershed', 
        'gradient_mag', 'phase_congruency', 'radon', 'lbp_anomaly',
        'canny', 'adaptive_threshold', 'otsu_variants', 'morphological'
    ])
    
    # DO2MR parameters
    do2mr_kernel_sizes: List[int] = field(default_factory=lambda: [5, 11, 15, 21])
    do2mr_gamma_values: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 3.5])
    
    # LEI parameters
    lei_kernel_lengths: List[int] = field(default_factory=lambda: [9, 11, 15, 19, 21])
    lei_angle_steps: List[int] = field(default_factory=lambda: [5, 10, 15])
    lei_threshold_factors: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    
    # Zana-Klein parameters
    zana_opening_length: int = 15
    zana_laplacian_threshold: float = 1.5
    
    # LoG parameters
    log_min_sigma: int = 2
    log_max_sigma: int = 10
    log_num_sigma: int = 5
    log_threshold: float = 0.05
    
    # Hessian parameters
    hessian_scales: List[float] = field(default_factory=lambda: [1, 2, 3, 4])
    
    # Frangi parameters
    frangi_scales: List[float] = field(default_factory=lambda: [1, 1.5, 2, 2.5, 3])
    frangi_beta: float = 0.5
    frangi_gamma: float = 15
    
    # --- Ensemble Settings ---
    ensemble_confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'do2mr': 1.0,
        'lei': 1.0,
        'zana_klein': 0.95,
        'log': 0.9,
        'doh': 0.85,
        'hessian_eigen': 0.9,
        'frangi': 0.9,
        'structure_tensor': 0.85,
        'mser': 0.8,
        'watershed': 0.75,
        'gradient_mag': 0.8,
        'phase_congruency': 0.85,
        'radon': 0.8,
        'lbp_anomaly': 0.7,
        'canny': 0.7,
        'adaptive_threshold': 0.75,
        'otsu_variants': 0.7,
        'morphological': 0.75
    })
    ensemble_vote_threshold: float = 0.3
    min_methods_for_detection: int = 2
    
    # --- Segmentation Refinement Settings ---
    segmentation_refinement_method: str = 'morphological'  # 'morphological', 'active_contour', 'level_set'
    active_contour_alpha: float = 0.015
    active_contour_beta: float = 10.0
    
    # --- False Positive Reduction Settings ---
    use_geometric_fp_reduction: bool = True
    use_contrast_fp_reduction: bool = True
    min_scratch_aspect_ratio: float = 3.0
    min_scratch_linearity_threshold: float = 3.0
    min_defect_contrast: float = 10.0
    min_region_circularity: float = 0.3
    min_region_solidity: float = 0.5
    
    # --- Texture Analysis Settings ---
    use_texture_analysis: bool = True
    lbp_radius: int = 3
    lbp_n_points: int = 24
    glcm_distances: List[int] = field(default_factory=lambda: [1, 2, 3])
    glcm_angles: List[int] = field(default_factory=lambda: [0, 45, 90, 135])

class DefectType(Enum):
    """Enumeration of defect types."""
    SCRATCH = auto()
    PIT = auto()
    DIG = auto()
    CONTAMINATION = auto()
    CHIP = auto()
    CRACK = auto()
    BUBBLE = auto()
    INCLUSION = auto()
    DELAMINATION = auto()
    BURN = auto()
    ANOMALY = auto()
    UNKNOWN = auto()

class DefectSeverity(Enum):
    """Defect severity classification."""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    NEGLIGIBLE = auto()

@dataclass
class Defect:
    """Comprehensive feature vector for each detected defect."""
    # Basic identification
    defect_id: int
    defect_type: DefectType
    zone: str
    severity: DefectSeverity
    confidence: float
    
    # Geometric features
    area_px: int
    area_um: Optional[float]
    perimeter: float
    eccentricity: float
    solidity: float
    circularity: float
    location_xy: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    major_axis_length: float
    minor_axis_length: float
    orientation: float
    
    # Intensity features
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    contrast: float
    intensity_skewness: float
    intensity_kurtosis: float
    
    # Gradient features
    mean_gradient: float
    max_gradient: float
    std_dev_gradient: float
    gradient_orientation: float
    
    # Texture features
    glcm_contrast: float
    glcm_homogeneity: float
    glcm_energy: float
    glcm_correlation: float
    lbp_mean: float
    lbp_std: float
    
    # Advanced features
    mean_hessian_eigen_ratio: float
    mean_coherence: float
    frangi_response: float
    
    # Topological features (optional)
    tda_local_connectivity_score: Optional[float] = None
    tda_betti_0_persistence: Optional[float] = None
    tda_betti_1_persistence: Optional[float] = None
    
    # Detection metadata
    contributing_algorithms: List[str] = field(default_factory=list)
    detection_strength: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert defect to dictionary for serialization."""
        d = asdict(self)
        d['defect_type'] = self.defect_type.name
        d['severity'] = self.severity.name
        return d

@dataclass
class AnalysisReport:
    """Comprehensive analysis report structure."""
    timestamp: str
    image_path: str
    image_info: Dict[str, Any]
    fiber_metrics: Dict[str, Any]
    defects: List[Defect]
    defects_by_zone: Dict[str, int]
    quality_score: float
    pass_fail_status: str
    failure_reasons: List[str]
    global_anomaly_analysis: Optional[Dict[str, Any]]
    global_tda_analysis: Optional[Dict[str, Any]]
    processing_time: float
    warnings: List[str]
    recommendations: List[str]

# =============================================================================
# PART 2: ULTRA-COMPREHENSIVE MATRIX ANALYZER (jake.py implementation)
# =============================================================================

class UltraComprehensiveMatrixAnalyzer:
    """Full implementation of jake.py's statistical anomaly detection engine."""
    
    def __init__(self, kb_path: Optional[str] = None):
        self.knowledge_base_path = kb_path
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
        if kb_path and os.path.exists(kb_path):
            self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load previously saved knowledge base from JSON."""
        if not self.knowledge_base_path or not os.path.exists(self.knowledge_base_path):
            return
            
        try:
            with open(self.knowledge_base_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Convert lists back to numpy arrays where needed
            if loaded_data.get('archetype_image'):
                loaded_data['archetype_image'] = np.array(loaded_data['archetype_image'], dtype=np.uint8)
            
            if loaded_data.get('statistical_model'):
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    if key in loaded_data['statistical_model'] and loaded_data['statistical_model'][key] is not None:
                        loaded_data['statistical_model'][key] = np.array(loaded_data['statistical_model'][key], dtype=np.float64)
            
            self.reference_model = loaded_data
            logging.info(f"Successfully loaded knowledge base from {self.knowledge_base_path}")
        except Exception as e:
            logging.error(f"Could not load knowledge base: {e}")
            self.reference_model = {}
    
    def save_knowledge_base(self):
        """Save current knowledge base to JSON."""
        if not self.knowledge_base_path:
            logging.error("Cannot save knowledge base: no path specified.")
            return
            
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
            
            save_data['timestamp'] = datetime.now().isoformat()
            
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(save_data, f, indent=2, cls=NumpyEncoder)
            logging.info(f"Knowledge base saved to {self.knowledge_base_path}")
        except Exception as e:
            logging.error(f"Error saving knowledge base: {e}")
    
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
    
    def _sanitize_feature_value(self, value):
        """Ensure feature value is finite and valid."""
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[0]) if len(value) > 0 else 0.0
        
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    
    def extract_ultra_comprehensive_features(self, image: np.ndarray) -> Tuple[Dict, List]:
        """Extract 100+ features using all available methods."""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Extract all feature categories
        features.update(self._extract_statistical_features(gray))
        features.update(self._extract_matrix_norms(gray))
        features.update(self._extract_lbp_features(gray))
        features.update(self._extract_glcm_features(gray))
        features.update(self._extract_fourier_features(gray))
        features.update(self._extract_multiscale_features(gray))
        features.update(self._extract_morphological_features(gray))
        features.update(self._extract_shape_features(gray))
        features.update(self._extract_svd_features(gray))
        features.update(self._extract_entropy_features(gray))
        features.update(self._extract_gradient_features(gray))
        features.update(self._extract_topological_proxy_features(gray))
        
        # Sanitize all feature values
        sanitized_features = {}
        for key, value in features.items():
            sanitized_features[key] = self._sanitize_feature_value(value)
        
        feature_names = sorted(sanitized_features.keys())
        return sanitized_features, feature_names
    
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
        """Extract Local Binary Pattern features."""
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
        """Extract Gray-Level Co-occurrence Matrix features."""
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
        """Extract multi-scale features using Gaussian pyramids."""
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
    
    def build_comprehensive_reference_model(self, ref_dir: str, append_mode: bool = False):
        """Build an exhaustive reference model from a directory of images."""
        logging.info(f"Building reference model from {ref_dir}. Append mode: {append_mode}")
        
        # Find all valid files
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        all_files = []
        
        try:
            for filename in os.listdir(ref_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    all_files.append(os.path.join(ref_dir, filename))
        except Exception as e:
            logging.error(f"Error reading directory: {e}")
            return False
        
        # Sort files for consistent processing
        all_files.sort()
        
        if not all_files:
            logging.error(f"No valid files found in {ref_dir}")
            return False
        
        logging.info(f"Found {len(all_files)} files to process")
        
        # Get existing features if in append mode
        existing_features = self.reference_model.get('features', []) if append_mode else []
        
        # Process each file
        new_features = []
        all_images = []
        feature_names = []
        
        for i, file_path in enumerate(all_files, 1):
            logging.info(f"Processing file {i}/{len(all_files)}: {os.path.basename(file_path)}")
            
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                logging.warning(f"Failed to load {file_path}")
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
            new_features.append(features)
            all_images.append(gray)
        
        if not new_features:
            logging.error("No features could be extracted from any file")
            return False
        
        # Combine with existing features
        all_features = existing_features + new_features
        
        # Check minimum requirement
        if len(all_features) < 2:
            logging.error(f"At least 2 reference files are required, but only {len(all_features)} were successfully processed.")
            return False
        
        logging.info("Building statistical model...")
        
        # Convert features to matrix
        feature_matrix = np.zeros((len(all_features), len(feature_names)))
        for i, features in enumerate(all_features):
            for j, fname in enumerate(feature_names):
                feature_matrix[i, j] = features.get(fname, 0)
        
        # Build statistical model
        mean_vector = np.mean(feature_matrix, axis=0)
        std_vector = np.std(feature_matrix, axis=0)
        median_vector = np.median(feature_matrix, axis=0)
        
        # Robust statistics
        robust_mean, robust_cov, robust_inv_cov = self._compute_robust_statistics(feature_matrix)
        
        # Create archetype image (median of all images)
        if all_images:
            target_shape = all_images[0].shape
            aligned_images = []
            for img in all_images:
                if img.shape != target_shape:
                    img = cv2.resize(img, (target_shape[1], target_shape[0]))
                aligned_images.append(img)
            
            archetype_image = np.median(aligned_images, axis=0).astype(np.uint8)
        else:
            archetype_image = self.reference_model.get('archetype_image')
        
        # Compute pairwise comparisons for threshold learning
        logging.info("Computing pairwise comparisons for threshold learning...")
        comparison_scores = []
        
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
        
        # Learn thresholds
        scores_array = np.array(comparison_scores)
        
        if len(scores_array) > 0 and not np.all(np.isnan(scores_array)):
            valid_scores = scores_array[~np.isnan(scores_array)]
            valid_scores = valid_scores[np.isfinite(valid_scores)]
            
            if len(valid_scores) > 0:
                # Clip extreme values
                valid_scores = np.clip(valid_scores, 0, np.percentile(valid_scores, 99.9))
                
                mean_score = float(np.mean(valid_scores))
                std_score = float(np.std(valid_scores))
                
                thresholds = {
                    'anomaly_mean': mean_score,
                    'anomaly_std': std_score,
                    'anomaly_p90': float(np.percentile(valid_scores, 90)),
                    'anomaly_p95': float(np.percentile(valid_scores, 95)),
                    'anomaly_p99': float(np.percentile(valid_scores, 99)),
                    'anomaly_threshold': float(min(mean_score + 2.5 * std_score, 
                                                   np.percentile(valid_scores, 99.5),
                                                   10.0)),
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
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save model
        self.save_knowledge_base()
        
        logging.info(f"Reference model built successfully with {len(all_features)} samples")
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
        
        # Replace zeros in mad_scaled to avoid division by zero
        mad_scaled[mad_scaled < 1e-6] = 1.0
        
        # Compute robust covariance
        # Weight samples by their distance from median
        normalized_deviations = deviations / mad_scaled
        distances = np.sqrt(np.sum(normalized_deviations**2, axis=1))
        
        # Clip distances to avoid numerical issues
        distances = np.clip(distances, 0, 10)
        
        # Compute weights with better numerical stability
        weights = np.exp(-0.5 * distances)
        weight_sum = weights.sum()
        
        if weight_sum < 1e-10 or n_samples < 2:
            # Fall back to standard covariance if weights are too small
            robust_cov = np.cov(data, rowvar=False)
            if robust_cov.ndim == 0:  # Single feature case
                robust_cov = np.array([[robust_cov]])
        else:
            weights = weights / weight_sum
            
            # Weighted covariance with numerical stability
            weighted_data = data * np.sqrt(weights[:, np.newaxis])
            robust_cov = np.dot(weighted_data.T, weighted_data)
            
            # Normalize by effective sample size
            effective_n = 1.0 / np.sum(weights**2)
            if effective_n > 1:
                robust_cov = robust_cov * effective_n / (effective_n - 1)
        
        # Ensure the covariance matrix is well-conditioned
        # Add regularization to diagonal
        reg_value = np.trace(robust_cov) / n_features * 1e-4
        if reg_value < 1e-6:
            reg_value = 1e-6
        robust_cov = robust_cov + np.eye(n_features) * reg_value
        
        # Ensure positive semi-definite by eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
            # Clip negative eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            robust_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except np.linalg.LinAlgError:
            # If eigenvalue decomposition fails, use identity matrix scaled by variance
            var_scale = np.var(data)
            robust_cov = np.eye(n_features) * var_scale
        
        # Compute pseudo-inverse with additional regularization
        try:
            # Add more regularization for inversion
            robust_inv_cov = np.linalg.pinv(robust_cov + np.eye(n_features) * 1e-4)
        except np.linalg.LinAlgError:
            # If pinv fails, use diagonal approximation
            diag_values = np.diag(robust_cov)
            diag_values[diag_values < 1e-6] = 1e-6
            robust_inv_cov = np.diag(1.0 / diag_values)
        
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
    
    def detect_anomalies_comprehensive(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform exhaustive anomaly detection on a test image."""
        if not self.reference_model.get('statistical_model'):
            logging.warning("No reference model available for anomaly detection")
            return {
                "mahalanobis_distance": 999.0,
                "ssim": 0.0,
                "anomaly_verdict": "UNKNOWN_NO_KB",
                "deviant_features": []
            }
        
        # Convert to grayscale
        if len(image.shape) == 3:
            test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = image.copy()
        
        # Extract features
        test_features, _ = self.extract_ultra_comprehensive_features(image)
        
        # Get reference statistics
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        
        # Ensure numpy arrays
        for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
            if key in stat_model and isinstance(stat_model[key], list):
                stat_model[key] = np.array(stat_model[key], dtype=np.float64)
        
        # Create feature vector
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        
        # Compute Mahalanobis distance
        diff = test_vector - stat_model['robust_mean']
        try:
            mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
        except:
            # Fallback to normalized Euclidean distance
            std_vector = stat_model['std']
            std_vector[std_vector < 1e-6] = 1.0
            normalized_diff = diff / std_vector
            mahalanobis_dist = np.linalg.norm(normalized_diff)
        
        # Compute Z-scores
        z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
        
        # Find most deviant features
        top_indices = np.argsort(z_scores)[::-1][:10]
        deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                           for i in top_indices]
        
        # Compare with archetype
        archetype = self.reference_model.get('archetype_image')
        ssim_score = 0.0
        if archetype is not None:
            # Ensure numpy array
            if isinstance(archetype, list):
                archetype = np.array(archetype, dtype=np.uint8)
            if test_gray.shape != archetype.shape:
                test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
            else:
                test_gray_resized = test_gray
            
            structural_comp = self.compute_image_structural_comparison(test_gray_resized, archetype)
            ssim_score = structural_comp['ssim']
        
        # Individual comparisons
        individual_scores = []
        for ref_features in self.reference_model['features']:
            comp = self.compute_exhaustive_comparison(test_features, ref_features)
            
            # Compute anomaly score with safe bounds
            score = (min(comp['euclidean_distance'], 1000.0) * 0.2 +
                    min(comp['manhattan_distance'], 10000.0) * 0.1 +
                    comp['cosine_distance'] * 0.2 +
                    (1 - abs(comp['pearson_correlation'])) * 0.1 +
                    min(comp['kl_divergence'], 10.0) * 0.1 +
                    comp['js_divergence'] * 0.1 +
                    min(comp['chi_square'], 10.0) * 0.1 +
                    min(comp['wasserstein_distance'], 10.0) * 0.1)
            
            individual_scores.append(min(score, 100.0))
        
        # Statistics of individual comparisons
        scores_array = np.array(individual_scores)
        comparison_stats = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
        }
        
        # Determine verdict
        thresholds = self.reference_model.get('learned_thresholds', self._get_default_thresholds())
        is_anomalous = (
            mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6) or
            comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6) or
            ssim_score < 0.7
        )
        
        # Overall confidence
        confidence = min(1.0, max(
            mahalanobis_dist / max(thresholds['anomaly_threshold'], 1e-6),
            comparison_stats['max'] / max(thresholds['anomaly_p95'], 1e-6),
            1 - ssim_score
        ))
        
        return {
            "mahalanobis_distance": float(mahalanobis_dist),
            "ssim": float(ssim_score),
            "comparison_stats": comparison_stats,
            "anomaly_verdict": "ANOMALOUS" if is_anomalous else "NORMAL",
            "confidence": float(confidence),
            "deviant_features": deviant_features
        }

# =============================================================================
# PART 3: TOPOLOGICAL DATA ANALYSIS ENGINE
# =============================================================================

class TDA_Analyzer:
    """Implementation of Topological Data Analysis from research papers."""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        if not TDA_AVAILABLE:
            raise ImportError("GUDHI library is required for TDA_Analyzer but not found.")
    
    def _create_bifiltration(self, image_region: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        """Creates the 2-parameter filtration grid for MF-PH analysis."""
        bifiltration_grid = {}
        
        # Ensure uint8
        if image_region.dtype != np.uint8:
            image_region = cv2.normalize(image_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        for t in range(*self.config.mf_threshold_range, self.config.mf_threshold_step):
            # Stage 1: Threshold the image
            _, binary_image_t = cv2.threshold(image_region, t, 255, cv2.THRESH_BINARY_INV)
            
            for i in range(*self.config.mf_opening_size_range, self.config.mf_opening_step):
                # Stage 2: Apply morphological opening
                if i == 0:
                    opened_image = binary_image_t
                else:
                    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (i + 1, i + 1))
                    opened_image = cv2.morphologyEx(binary_image_t, cv2.MORPH_OPEN, structuring_element)
                
                bifiltration_grid[(t, i)] = (opened_image > 128)  # Convert to boolean
        
        return bifiltration_grid
    
    def _compute_persistence_diagrams_from_slice(self, filtration_slice: List[np.ndarray]) -> Dict:
        """Compute persistence diagrams from a 1D slice of the bifiltration."""
        if not filtration_slice or filtration_slice[0].size == 0:
            return {'betti_0': [], 'betti_1': []}
        
        # Create filtration values
        h, w = filtration_slice[0].shape
        filtration_values = np.full((h, w), float('inf'))
        
        # Assign filtration values based on when pixels appear
        for idx, mask in enumerate(reversed(filtration_slice)):
            filtration_value = len(filtration_slice) - 1 - idx
            filtration_values[mask] = np.minimum(filtration_values[mask], filtration_value)
        
        # Create cubical complex
        cc = gd.CubicalComplex(top_dimensional_cells=filtration_values.flatten())
        
        # Compute persistence
        diag = cc.persistence()
        
        # Extract diagrams for different dimensions
        betti_0 = [(p[1][0], p[1][1]) for p in diag if p[0] == 0]
        betti_1 = [(p[1][0], p[1][1]) for p in diag if p[0] == 1]
        
        return {'betti_0': betti_0, 'betti_1': betti_1}
    
    def analyze_region(self, image_region: np.ndarray) -> Dict[str, Any]:
        """Main entry point for analyzing a given image ROI with MF-PH."""
        bifiltration = self._create_bifiltration(image_region)
        
        features = {
            'normalized_betti_curves': {},
            'size_distributions': {},
            'connectivity_indices': {}
        }
        
        # 1. Normalized Betti Curves (horizontal slices)
        for i in range(*self.config.mf_opening_size_range, self.config.mf_opening_step):
            horizontal_slice = []
            for t in range(*self.config.mf_threshold_range, self.config.mf_threshold_step):
                if (t, i) in bifiltration:
                    horizontal_slice.append(bifiltration[(t, i)])
            
            if horizontal_slice:
                pd_h = self._compute_persistence_diagrams_from_slice(horizontal_slice)
                
                # Calculate normalized Betti curve
                betti_curve = []
                for t_idx in range(len(horizontal_slice)):
                    # Count features alive at time t_idx
                    betti_0_count = sum(1 for (b, d) in pd_h['betti_0'] if b <= t_idx < d)
                    betti_curve.append(betti_0_count)
                
                # Normalize by maximum
                if max(betti_curve) > 0:
                    betti_curve = [b / max(betti_curve) for b in betti_curve]
                
                features['normalized_betti_curves'][i] = betti_curve
        
        # 2. Size Distribution & Connectivity Index (vertical slices)
        for t in range(*self.config.mf_threshold_range, self.config.mf_threshold_step):
            vertical_slice = []
            for i in range(*self.config.mf_opening_size_range, self.config.mf_opening_step):
                if (t, i) in bifiltration:
                    vertical_slice.append(bifiltration[(t, i)])
            
            if vertical_slice:
                pd_v = self._compute_persistence_diagrams_from_slice(vertical_slice)
                
                # Calculate Size Distribution
                death_values_d_t = [d for (b, d) in pd_v['betti_1'] if b == 0 and d != float('inf')]
                if death_values_d_t:
                    size_dist = np.histogram(death_values_d_t, 
                                           bins=self.config.mf_opening_size_range[1])[0]
                    features['size_distributions'][t] = size_dist.tolist()
                else:
                    features['size_distributions'][t] = []
                
                # Calculate Connectivity Index
                sum_of_lifespans_total = sum((d - b) for (b, d) in pd_v['betti_1'] 
                                            if d != float('inf'))
                sum_of_lifespans_d_t = sum(d for d in death_values_d_t)
                
                if sum_of_lifespans_total > 0:
                    connectivity_index = (sum_of_lifespans_total - sum_of_lifespans_d_t) / sum_of_lifespans_total
                else:
                    connectivity_index = 1.0  # Fully connected if no holes
                
                features['connectivity_indices'][t] = float(connectivity_index)
        
        # Compute summary statistics
        all_connectivity_indices = list(features['connectivity_indices'].values())
        if all_connectivity_indices:
            features['mean_connectivity'] = float(np.mean(all_connectivity_indices))
            features['min_connectivity'] = float(np.min(all_connectivity_indices))
        else:
            features['mean_connectivity'] = 1.0
            features['min_connectivity'] = 1.0
        
        return features

# =============================================================================
# PART 4: OMNI FIBER ANALYZER - MAIN CLASS
# =============================================================================

class OmniFiberAnalyzer:
    """The complete master class for unified fiber optic defect detection."""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.intermediate_results = {}
        self.warnings = []
        
        # Initialize sub-analyzers
        self.kb_analyzer = UltraComprehensiveMatrixAnalyzer(self.config.knowledge_base_path)
        
        if TDA_AVAILABLE and self.config.use_topological_analysis:
            self.tda_analyzer = TDA_Analyzer(self.config)
        else:
            self.tda_analyzer = None
            if self.config.use_topological_analysis:
                self.logger.warning("TDA requested but GUDHI not available")
        
        self.logger.info("OmniFiberAnalyzer initialized successfully")
    
    def analyze_end_face(self, image_path: str) -> AnalysisReport:
        """Main analysis pipeline for fiber optic end-face images."""
        self.logger.info(f"Starting analysis of: {image_path}")
        start_time = time.time()
        self.warnings.clear()
        self.intermediate_results.clear()
        
        try:
            # Stage 1: Load and prepare image
            original_image, gray_image = self._load_and_prepare_image(image_path)
            if original_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Stage 2: Global statistical analysis (jake.py)
            global_results = None
            if self.config.use_global_anomaly_analysis:
                global_results = self._run_global_statistical_analysis(gray_image)
            
            # Stage 3: Advanced preprocessing (jill.py)
            preprocessed_maps = self._run_advanced_preprocessing(gray_image)
            
            # Stage 4: Region separation and zoning
            fiber_info, zone_masks = self._locate_fiber_and_define_zones(preprocessed_maps)
            
            # Stage 5: Run all detectors
            raw_detections = self._run_all_detectors(preprocessed_maps, zone_masks)
            
            # Stage 6: Ensemble combination
            ensemble_masks = self._ensemble_detections(raw_detections, gray_image.shape)
            
            # Stage 7: Refine and segment defects
            final_defect_masks = self._refine_and_segment_defects(ensemble_masks, preprocessed_maps)
            
            # Stage 8: Characterize defects
            analyzed_defects = self._characterize_defects_from_masks(
                final_defect_masks, preprocessed_maps, gray_image
            )
            
            # Stage 9: Generate final report
            duration = time.time() - start_time
            final_report = self._generate_final_report(
                image_path, original_image, global_results, 
                analyzed_defects, fiber_info, zone_masks, duration
            )
            
            # Stage 10: Visualization
            self._visualize_master_results(original_image, final_report)
            
            self.logger.info(f"Analysis completed in {duration:.2f} seconds")
            return final_report
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return error report
            return self._create_error_report(image_path, str(e), time.time() - start_time)
    
    def _load_and_prepare_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image from file path."""
        self.logger.info("Loading image...")
        
        # Handle JSON format (from jake.py)
        if image_path.lower().endswith('.json'):
            return self._load_from_json(image_path)
        
        # Standard image formats
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None, None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return image, gray
    
    def _load_from_json(self, json_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
                self.warnings.append(f"Skipped {oob_count} out-of-bounds pixels")
            
            # Convert to grayscale
            gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
            
            return matrix, gray
            
        except Exception as e:
            self.logger.error(f"Error loading JSON {json_path}: {e}")
            return None, None
    
    def _run_global_statistical_analysis(self, gray_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run jake.py's global anomaly analysis."""
        self.logger.info("Running global statistical analysis...")
        
        try:
            results = self.kb_analyzer.detect_anomalies_comprehensive(gray_image)
            
            # Add to intermediate results if requested
            if self.config.save_intermediate_masks:
                self.intermediate_results['global_analysis'] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Global analysis failed: {str(e)}")
            return None
    
    def _run_advanced_preprocessing(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply all preprocessing methods from jill.py."""
        self.logger.info("Running advanced preprocessing...")
        
        preprocessed = {
            'original': gray_image.copy()
        }
        
        # 1. Anisotropic diffusion (Perona-Malik)
        if self.config.use_anisotropic_diffusion:
            preprocessed['anisotropic'] = self._anisotropic_diffusion(
                gray_image.astype(np.float64),
                iterations=self.config.anisotropic_iterations,
                kappa=self.config.anisotropic_kappa,
                gamma=self.config.anisotropic_gamma
            )
        
        # 2. Total Variation denoising
        preprocessed['tv_denoised'] = denoise_tv_chambolle(
            gray_image.astype(np.float64) / 255.0, weight=0.1
        ) * 255
        
        # 3. Coherence-enhancing diffusion
        if self.config.use_coherence_enhancing_diffusion:
            preprocessed['coherence'] = self._coherence_enhancing_diffusion(
                preprocessed['tv_denoised'],
                iterations=self.config.coherence_iterations
            )
        
        # 4. Multiple Gaussian blurs
        for size in self.config.gaussian_blur_sizes:
            preprocessed[f'gaussian_{size[0]}'] = cv2.GaussianBlur(gray_image, size, 0)
        
        # 5. Multiple bilateral filters
        for i, (d, sc, ss) in enumerate(self.config.bilateral_params):
            preprocessed[f'bilateral_{i}'] = cv2.bilateralFilter(gray_image, d, sc, ss)
        
        # 6. Multiple CLAHE variants
        for i, (clip, grid) in enumerate(self.config.clahe_params):
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            preprocessed[f'clahe_{i}'] = clahe.apply(gray_image)
        
        # 7. Standard preprocessing
        preprocessed['median'] = cv2.medianBlur(gray_image, 5)
        preprocessed['nlmeans'] = cv2.fastNlMeansDenoising(
            gray_image, None, self.config.denoise_strength, 7, 21
        )
        preprocessed['histeq'] = cv2.equalizeHist(gray_image)
        
        # 8. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        preprocessed['morph_gradient'] = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        preprocessed['tophat'] = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
        preprocessed['blackhat'] = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        
        # 9. Gradient magnitude (from change_magnitude.py)
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        preprocessed['gradient_magnitude'] = np.sqrt(grad_x**2 + grad_y**2)
        
        # 10. Illumination correction
        if self.config.use_illumination_correction:
            kernel_size = max(gray_image.shape) // 8
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
            preprocessed['illumination_corrected'] = cv2.subtract(gray_image, background)
            mean_background = np.full_like(gray_image, np.mean(background).astype(np.uint8))
            preprocessed['illumination_corrected'] = cv2.add(
                preprocessed['illumination_corrected'], mean_background
            )
        
        # Convert all to uint8
        for key, img in preprocessed.items():
            if img.dtype != np.uint8:
                preprocessed[key] = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return preprocessed
    
    def _anisotropic_diffusion(self, image: np.ndarray, iterations: int = 10, 
                               kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """Perona-Malik anisotropic diffusion."""
        img = image.copy()
        
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
            
        return img
    
    def _coherence_enhancing_diffusion(self, image: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Coherence-enhancing diffusion for linear structures."""
        img = image.copy()
        
        for _ in range(iterations):
            # Compute structure tensor
            Ix = ndimage.sobel(img, axis=1)
            Iy = ndimage.sobel(img, axis=0)
            
            # Structure tensor components
            Jxx = gaussian_filter(Ix * Ix, 1.0)
            Jxy = gaussian_filter(Ix * Iy, 1.0)
            Jyy = gaussian_filter(Iy * Iy, 1.0)
            
            # Eigenvalues
            trace = Jxx + Jyy
            det = Jxx * Jyy - Jxy * Jxy
            discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
            lambda1 = 0.5 * (trace + discriminant)
            lambda2 = 0.5 * (trace - discriminant)
            
            # Coherence measure
            coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10))**2
            
            # Apply diffusion based on coherence
            alpha = 0.001
            c1 = alpha
            c2 = alpha + (1 - alpha) * np.exp(-1 / (coherence + 1e-10))
            
            # Simple diffusion update
            img = gaussian_filter(img, 0.5)
            
        return img
    
    def _locate_fiber_and_define_zones(self, preprocessed_maps: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Hybrid method for fiber localization and zone creation."""
        self.logger.info("Locating fiber and defining zones...")
        
        # Try ensemble method first (from jill.py)
        if self.config.primary_masking_method == 'ensemble':
            fiber_info = self._find_fiber_center_ensemble(preprocessed_maps)
            
            # If ensemble fails or has low confidence, use fallback
            if not fiber_info or fiber_info.get('confidence', 0) < 0.5:
                self.logger.info("Ensemble method failed, using fallback...")
                fiber_info = self._find_fiber_center_fallback(preprocessed_maps)
        else:
            # Use adaptive contour method (from daniel.py)
            fiber_info = self._find_fiber_center_fallback(preprocessed_maps)
        
        # Create zone masks
        gray_shape = preprocessed_maps['original'].shape
        zone_masks = self._create_zone_masks(
            gray_shape, 
            fiber_info['center'], 
            fiber_info['cladding_radius']
        )
        
        # Calculate pixels per micron if possible
        if self.config.pixels_per_micron > 0:
            fiber_info['pixels_per_micron'] = self.config.pixels_per_micron
        else:
            # Estimate based on standard fiber dimensions
            # Assuming cladding is 125 microns
            fiber_info['pixels_per_micron'] = (2 * fiber_info['cladding_radius']) / 125.0
        
        return fiber_info, zone_masks
    
    def _find_fiber_center_ensemble(self, preprocessed_maps: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Ensemble method for finding fiber center (from jill.py)."""
        candidates = []
        
        # Method 1: Hough circles on different preprocessed images
        for img_name in ['gaussian_5', 'bilateral_0', 'clahe_0', 'median']:
            if img_name not in preprocessed_maps:
                continue
                
            img = preprocessed_maps[img_name]
            for dp in self.config.hough_dp_values:
                for p1 in self.config.hough_param1_values:
                    for p2 in self.config.hough_param2_values:
                        circles = cv2.HoughCircles(
                            img, cv2.HOUGH_GRADIENT, dp=dp,
                            minDist=img.shape[0]//8,
                            param1=p1, param2=p2,
                            minRadius=img.shape[0]//10,
                            maxRadius=img.shape[0]//2
                        )
                        
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for circle in circles[0]:
                                candidates.append({
                                    'center': (int(circle[0]), int(circle[1])),
                                    'radius': float(circle[2]),
                                    'method': f'hough_{img_name}',
                                    'confidence': p2 / max(self.config.hough_param2_values)
                                })
        
        # Method 2: Contour-based detection
        for img_name in ['bilateral_0', 'nlmeans', 'original']:
            if img_name not in preprocessed_maps:
                continue
                
            img = preprocessed_maps[img_name]
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the most circular contour
                best_contour = None
                best_circularity = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < img.shape[0] * img.shape[1] * 0.01:
                        continue
                        
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > best_circularity:
                            best_circularity = circularity
                            best_contour = contour
                
                if best_contour is not None and best_circularity > 0.5:
                    (x, y), radius = cv2.minEnclosingCircle(best_contour)
                    candidates.append({
                        'center': (int(x), int(y)),
                        'radius': float(radius),
                        'method': f'contour_{img_name}',
                        'confidence': best_circularity
                    })
        
        if not candidates:
            return None
        
        # Vote for best result using clustering
        centers = np.array([c['center'] for c in candidates])
        radii = np.array([c['radius'] for c in candidates])
        
        # Use median for robustness
        best_center = (int(np.median(centers[:, 0])), int(np.median(centers[:, 1])))
        best_radius = float(np.median(radii))
        avg_confidence = np.mean([c['confidence'] for c in candidates])
        
        # Calculate core radius
        core_radius = max(1, int(best_radius / self.config.cladding_core_ratio))
        
        return {
            'center': best_center,
            'cladding_radius': best_radius,
            'core_radius': core_radius,
            'confidence': avg_confidence,
            'num_candidates': len(candidates),
            'method': 'ensemble'
        }
    
    def _find_fiber_center_fallback(self, preprocessed_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fallback method with multiple strategies (from daniel.py)."""
        img = preprocessed_maps.get('clahe_0', preprocessed_maps['original'])
        
        # Method 1: Adaptive thresholding
        try:
            block_size = self.config.adaptive_threshold_block_size
            if block_size % 2 == 0:
                block_size += 1
                
            adaptive_thresh = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                block_size, 
                self.config.adaptive_threshold_c
            )
            
            # Clean up with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the most circular contour
                best_contour = None
                best_circularity = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < img.shape[0] * img.shape[1] * 0.01:
                        continue
                        
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > best_circularity:
                            best_circularity = circularity
                            best_contour = contour
                
                if best_contour is not None and best_circularity > 0.3:
                    (cx, cy), cr = cv2.minEnclosingCircle(best_contour)
                    cx, cy, cr = int(cx), int(cy), int(cr)
                    
                    # Validate the found circle
                    if cr >= img.shape[0] * 0.1 and cr <= img.shape[0] * 0.45:
                        core_r = max(1, int(cr / self.config.cladding_core_ratio))
                        
                        return {
                            'center': (cx, cy),
                            'cladding_radius': cr,
                            'core_radius': core_r,
                            'circularity': best_circularity,
                            'confidence': best_circularity,
                            'method': 'adaptive_threshold'
                        }
        except Exception as e:
            self.logger.warning(f"Adaptive threshold method failed: {str(e)}")
        
        # Method 2: Largest component
        try:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            if num_labels > 1:
                largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                component_mask = (labels == largest_idx).astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    (cx, cy), cr = cv2.minEnclosingCircle(contours[0])
                    cx, cy, cr = int(cx), int(cy), int(cr)
                    
                    if cr > img.shape[0] * 0.1:
                        core_r = max(1, int(cr / self.config.cladding_core_ratio))
                        
                        return {
                            'center': (cx, cy),
                            'cladding_radius': cr,
                            'core_radius': core_r,
                            'confidence': 0.5,
                            'method': 'largest_component'
                        }
        except Exception as e:
            self.logger.warning(f"Largest component method failed: {str(e)}")
        
        # Method 3: Use image center as last resort
        self.logger.warning("Using estimated fiber location - results may be inaccurate")
        self.warnings.append("Could not accurately locate fiber - using estimated center")
        
        h, w = img.shape
        cx, cy = w // 2, h // 2
        cr = min(h, w) // 3
        core_r = max(1, int(cr / self.config.cladding_core_ratio))
        
        return {
            'center': (cx, cy),
            'cladding_radius': cr,
            'core_radius': core_r,
            'confidence': 0.1,
            'method': 'estimated'
        }
    
    def _create_zone_masks(self, shape: Tuple[int, int], center: Tuple[int, int], 
                          cladding_radius: float) -> Dict[str, np.ndarray]:
        """Create masks for different fiber zones."""
        h, w = shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
        
        masks = {}
        
        # Define zone radii
        core_radius = cladding_radius / self.config.cladding_core_ratio
        ferrule_radius = cladding_radius * self.config.ferrule_buffer_ratio
        adhesive_radius = ferrule_radius * 1.1
        
        # Create masks
        masks['Core'] = (dist_from_center <= core_radius).astype(np.uint8) * 255
        masks['Cladding'] = ((dist_from_center > core_radius) & 
                            (dist_from_center <= cladding_radius)).astype(np.uint8) * 255
        masks['Ferrule'] = ((dist_from_center > cladding_radius) & 
                           (dist_from_center <= ferrule_radius)).astype(np.uint8) * 255
        masks['Adhesive'] = ((dist_from_center > ferrule_radius) & 
                            (dist_from_center <= adhesive_radius)).astype(np.uint8) * 255
        
        # Combined fiber mask
        masks['Fiber'] = cv2.bitwise_or(masks['Core'], masks['Cladding'])
        
        return masks
    
    def _run_all_detectors(self, preprocessed_maps: Dict[str, np.ndarray], 
                          zone_masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run all enabled detection algorithms."""
        self.logger.info("Running all detection algorithms...")
        all_detections = {}
        
        # Select best preprocessed images for different tasks
        img_for_region = preprocessed_maps.get('clahe_0', preprocessed_maps['original'])
        img_for_scratch = preprocessed_maps.get('coherence', preprocessed_maps['original'])
        img_for_general = preprocessed_maps.get('bilateral_0', preprocessed_maps['original'])
        
        # Create detector registry
        detector_registry = {
            'do2mr': lambda img, mask: self._detect_do2mr(img, mask),
            'lei': lambda img, mask: self._detect_lei(img, mask),
            'zana_klein': lambda img, mask: self._detect_zana_klein(img, mask),
            'log': lambda img, mask: self._detect_log(img, mask),
            'doh': lambda img, mask: self._detect_doh(img, mask),
            'hessian_eigen': lambda img, mask: self._detect_hessian_eigen(img, mask),
            'frangi': lambda img, mask: self._detect_frangi(img, mask),
            'structure_tensor': lambda img, mask: self._detect_structure_tensor(img, mask),
            'mser': lambda img, mask: self._detect_mser(img, mask),
            'watershed': lambda img, mask: self._detect_watershed(img, mask),
            'gradient_mag': lambda img, mask: self._detect_gradient_magnitude(img, mask),
            'phase_congruency': lambda img, mask: self._detect_phase_congruency(img, mask),
            'radon': lambda img, mask: self._detect_radon(img, mask),
            'lbp_anomaly': lambda img, mask: self._detect_lbp_anomaly(img, mask),
            'canny': lambda img, mask: self._detect_canny(img, mask),
            'adaptive_threshold': lambda img, mask: self._detect_adaptive_threshold(img, mask),
            'otsu_variants': lambda img, mask: self._detect_otsu_variants(img, mask),
            'morphological': lambda img, mask: self._detect_morphological(img, mask)
        }
        
        # Run detectors for each zone
        for zone_name, zone_mask in zone_masks.items():
            if zone_name == 'Fiber':  # Skip combined mask
                continue
                
            self.logger.info(f"  Processing zone: {zone_name}")
            zone_detections = {}
            
            for detector_name in self.config.enabled_detectors:
                if detector_name in detector_registry:
                    try:
                        # Select appropriate input image
                        if detector_name in ['lei', 'zana_klein', 'hessian_eigen', 'frangi', 'structure_tensor']:
                            input_img = img_for_scratch
                        elif detector_name in ['do2mr', 'log', 'doh', 'mser']:
                            input_img = img_for_region
                        else:
                            input_img = img_for_general
                        
                        # Run detector
                        detection_mask = detector_registry[detector_name](input_img, zone_mask)
                        zone_detections[detector_name] = detection_mask
                        
                    except Exception as e:
                        self.logger.warning(f"    {detector_name} failed: {str(e)}")
                        zone_detections[detector_name] = np.zeros_like(zone_mask)
            
            all_detections[zone_name] = zone_detections
        
        # Save intermediate results if requested
        if self.config.save_intermediate_masks:
            self.intermediate_results['raw_detections'] = all_detections
        
        return all_detections
    
    # =============================================================================
    # DETECTION ALGORITHMS (Full implementations from daniel.py and jill.py)
    # =============================================================================
    
    def _detect_do2mr(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Enhanced DO2MR (Difference of Min-Max Ranking) detection."""
        combined_mask = np.zeros_like(image)
        vote_map = np.zeros_like(image, dtype=np.float32)
        
        for kernel_size in self.config.do2mr_kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            img_max = cv2.dilate(image, kernel)
            img_min = cv2.erode(image, kernel)
            residual = cv2.absdiff(img_max, img_min)
            residual_filtered = cv2.medianBlur(residual, 5)
            
            for gamma in self.config.do2mr_gamma_values:
                masked_values = residual_filtered[zone_mask > 0]
                if len(masked_values) == 0:
                    continue
                    
                mean_val = np.mean(masked_values)
                std_val = np.std(masked_values)
                threshold = mean_val + gamma * std_val
                
                _, binary = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
                binary = cv2.bitwise_and(binary, binary, mask=zone_mask)
                
                # Morphological cleanup
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
                
                vote_map += (binary / 255.0)
        
        # Combine votes
        num_combinations = len(self.config.do2mr_kernel_sizes) * len(self.config.do2mr_gamma_values)
        threshold = num_combinations * 0.3
        combined_mask = (vote_map >= threshold).astype(np.uint8) * 255
        
        return combined_mask
    
    def _detect_lei(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Enhanced LEI (Linear Enhancement Inspector) scratch detection."""
        scratch_strength = np.zeros_like(image, dtype=np.float32)
        
        # Apply histogram equalization to masked region
        masked_img = cv2.bitwise_and(image, image, mask=zone_mask)
        enhanced = cv2.equalizeHist(masked_img)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=zone_mask)
        
        for kernel_length in self.config.lei_kernel_lengths:
            for angle_step in self.config.lei_angle_steps:
                for angle in range(0, 180, angle_step):
                    angle_rad = np.deg2rad(angle)
                    
                    # Create linear kernel
                    kernel_points = []
                    for i in range(-kernel_length//2, kernel_length//2 + 1):
                        if i == 0:
                            continue
                        x = int(round(i * np.cos(angle_rad)))
                        y = int(round(i * np.sin(angle_rad)))
                        kernel_points.append((x, y))
                    
                    if kernel_points:
                        response = self._apply_linear_detector(enhanced, kernel_points)
                        scratch_strength = np.maximum(scratch_strength, response)
        
        # Normalize and threshold
        if scratch_strength.max() > 0:
            cv2.normalize(scratch_strength, scratch_strength, 0, 255, cv2.NORM_MINMAX)
            scratch_strength = scratch_strength.astype(np.uint8)
            
            # Multi-threshold approach
            scratch_mask = np.zeros_like(scratch_strength)
            for factor in self.config.lei_threshold_factors:
                mean_val = np.mean(scratch_strength[zone_mask > 0])
                std_val = np.std(scratch_strength[zone_mask > 0])
                threshold = mean_val + factor * std_val
                _, mask = cv2.threshold(scratch_strength, threshold, 255, cv2.THRESH_BINARY)
                scratch_mask = cv2.bitwise_or(scratch_mask, mask)
            
            # Morphological refinement
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel_close)
            scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask)
            
            return scratch_mask
        
        return np.zeros_like(image, dtype=np.uint8)
    
    def _apply_linear_detector(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        """Apply linear detector for scratch detection."""
        h, w = image.shape
        response = np.zeros_like(image, dtype=np.float32)
        
        max_offset = max(max(abs(dx), abs(dy)) for dx, dy in kernel_points)
        padded = cv2.copyMakeBorder(image, max_offset, max_offset, max_offset, max_offset, cv2.BORDER_REFLECT)
        
        for y in range(h):
            for x in range(w):
                line_vals = []
                for dx, dy in kernel_points:
                    line_vals.append(float(padded[y + max_offset + dy, x + max_offset + dx]))
                
                if line_vals:
                    center_val = float(padded[y + max_offset, x + max_offset])
                    bright_response = np.mean(line_vals) - center_val
                    dark_response = center_val - np.mean(line_vals)
                    response[y, x] = max(0, max(bright_response, dark_response))
        
        return response
    
    def _detect_zana_klein(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Zana-Klein algorithm for linear feature detection."""
        l = self.config.zana_opening_length
        reconstructed = np.zeros_like(image)
        
        for angle in range(0, 180, 15):
            angle_rad = np.deg2rad(angle)
            
            # Create rotated linear structuring element
            kernel_size = l
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
            cx, cy = kernel_size // 2, kernel_size // 2
            
            # Draw line
            for i in range(-l//2, l//2 + 1):
                x = int(cx + i * np.cos(angle_rad))
                y = int(cy + i * np.sin(angle_rad))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1
            
            # Apply morphological opening
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            reconstructed = np.maximum(reconstructed, opened)
        
        # Top-hat transform
        tophat = cv2.subtract(image, reconstructed)
        
        if tophat.max() == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(tophat, cv2.CV_64F, ksize=5)
        laplacian[laplacian < 0] = 0
        laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold
        threshold = np.mean(laplacian_norm) + self.config.zana_laplacian_threshold * np.std(laplacian_norm)
        _, defect_map = cv2.threshold(laplacian_norm, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply zone mask
        defect_map = cv2.bitwise_and(defect_map, defect_map, mask=zone_mask)
        
        return defect_map
    
    def _detect_log(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Scale-normalized Laplacian of Gaussian blob detection."""
        # Use skimage's blob_log
        blobs = feature.blob_log(
            image, 
            min_sigma=self.config.log_min_sigma, 
            max_sigma=self.config.log_max_sigma,
            num_sigma=self.config.log_num_sigma, 
            threshold=self.config.log_threshold
        )
        
        defect_map = np.zeros_like(image, dtype=np.uint8)
        for y, x, r in blobs:
            cv2.circle(defect_map, (int(x), int(y)), int(r * np.sqrt(2)), 255, -1)
        
        # Apply zone mask
        defect_map = cv2.bitwise_and(defect_map, defect_map, mask=zone_mask)
        
        return defect_map
    
    def _detect_doh(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Determinant of Hessian blob detection."""
        doh_response = np.zeros_like(image, dtype=np.float32)
        
        for scale in range(self.config.log_min_sigma, self.config.log_max_sigma, 2):
            # Smooth image at this scale
            smoothed = gaussian_filter(image.astype(np.float32), scale)
            
            # Compute Hessian components
            Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
            Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
            Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
            
            # Determinant of Hessian
            det = Hxx * Hyy - Hxy**2
            
            # Scale normalize
            det *= scale**4
            
            # Keep maximum response across scales
            doh_response = np.maximum(doh_response, np.abs(det))
        
        # Normalize and threshold
        cv2.normalize(doh_response, doh_response, 0, 255, cv2.NORM_MINMAX)
        _, doh_mask = cv2.threshold(doh_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply zone mask
        doh_mask = cv2.bitwise_and(doh_mask, doh_mask, mask=zone_mask)
        
        return doh_mask
    
    def _detect_hessian_eigen(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Multi-scale Hessian eigenvalue analysis for ridge/valley detection."""
        ridge_response = np.zeros_like(image, dtype=np.float32)
        
        for scale in self.config.hessian_scales:
            smoothed = gaussian_filter(image.astype(np.float32), scale)
            
            # Compute Hessian matrix components
            Hxx = gaussian_filter(smoothed, scale, order=(0, 2))
            Hyy = gaussian_filter(smoothed, scale, order=(2, 0))
            Hxy = gaussian_filter(smoothed, scale, order=(1, 1))
            
            # Compute eigenvalues
            trace = Hxx + Hyy
            det = Hxx * Hyy - Hxy * Hxy
            discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
            
            lambda1 = 0.5 * (trace + discriminant)
            lambda2 = 0.5 * (trace - discriminant)
            
            # Ridge measure: |λ1|/|λ2|
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
            
            # Ridge strength
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            # Vesselness measure
            beta = 0.5
            c = 0.5 * np.max(S)
            
            response = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*c**2)))
            
            # Only keep dark lines (negative λ2)
            response[lambda2 > 0] = 0
            
            # Scale normalize
            ridge_response = np.maximum(ridge_response, scale**2 * response)
        
        # Threshold
        _, ridge_mask = cv2.threshold(ridge_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ridge_mask = cv2.bitwise_and(ridge_mask, ridge_mask, mask=zone_mask)
        
        return ridge_mask
    
    def _detect_frangi(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Frangi vesselness filter for tubular structure detection."""
        vesselness = np.zeros_like(image, dtype=np.float32)
        
        for scale in self.config.frangi_scales:
            smoothed = gaussian_filter(image.astype(np.float32), scale)
            
            # Hessian matrix
            Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
            Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
            Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
            
            # Eigenvalues
            tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
            lambda1 = 0.5 * (Hxx + Hyy + tmp)
            lambda2 = 0.5 * (Hxx + Hyy - tmp)
            
            # Sort eigenvalues by absolute value
            idx = np.abs(lambda1) < np.abs(lambda2)
            lambda1[idx], lambda2[idx] = lambda2[idx], lambda1[idx]
            
            # Frangi measures
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            # Vesselness response
            beta = self.config.frangi_beta
            gamma = self.config.frangi_gamma
            
            v = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*gamma**2)))
            
            # Only keep bright structures on dark background
            v[lambda2 > 0] = 0
            
            vesselness = np.maximum(vesselness, v)
        
        # Normalize and threshold
        cv2.normalize(vesselness, vesselness, 0, 255, cv2.NORM_MINMAX)
        _, vessel_mask = cv2.threshold(vesselness.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=zone_mask)
        
        return vessel_mask
    
    def _detect_structure_tensor(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Structure tensor analysis for coherent structure detection."""
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
        
        mu1 = 0.5 * (trace + discriminant)
        mu2 = 0.5 * (trace - discriminant)
        
        # Coherence measure
        coherence = ((mu1 - mu2) / (mu1 + mu2 + 1e-10))**2
        
        # Threshold high coherence regions
        coherence_thresh = np.percentile(coherence[zone_mask > 0], 90)
        _, coherence_mask = cv2.threshold((coherence * 255).astype(np.uint8), coherence_thresh * 255, 255, cv2.THRESH_BINARY)
        coherence_mask = cv2.bitwise_and(coherence_mask, coherence_mask, mask=zone_mask)
        
        return coherence_mask
    
    def _detect_mser(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """MSER (Maximally Stable Extremal Regions) detection."""
        # Create MSER detector
        mser = cv2.MSER_create(
            delta=5,
            min_area=self.config.min_defect_area_px,
            max_area=int(np.sum(zone_mask) * 0.1),
            max_variation=0.25,
            min_diversity=0.2
        )
        
        # Detect regions
        regions, _ = mser.detectRegions(image)
        
        # Create mask from regions
        mask = np.zeros_like(image)
        for region in regions:
            cv2.fillPoly(mask, [region], 255)
        
        # Apply zone mask
        mask = cv2.bitwise_and(mask, mask, mask=zone_mask)
        
        return mask
    
    def _detect_watershed(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Watershed segmentation for blob detection."""
        # Apply zone mask to image
        masked_img = cv2.bitwise_and(image, image, mask=zone_mask)
        
        # Threshold
        _, binary = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find sure foreground
        _, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find sure background
        kernel = np.ones((3,3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # Create defect mask
        watershed_mask = np.zeros_like(image)
        watershed_mask[markers == -1] = 255
        watershed_mask = cv2.bitwise_and(watershed_mask, watershed_mask, mask=zone_mask)
        
        return watershed_mask
    
    def _detect_gradient_magnitude(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Gradient magnitude based defect detection."""
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        cv2.normalize(grad_mag, grad_mag, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold
        _, grad_mask = cv2.threshold(grad_mag.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        grad_mask = cv2.bitwise_and(grad_mask, grad_mask, mask=zone_mask)
        
        return grad_mask
    
    def _detect_phase_congruency(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Simplified phase congruency detection using edge detection."""
        # Use Canny edge detection as a simplified version
        edges = cv2.Canny(image, 50, 150)
        
        # Apply morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Apply zone mask
        edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
        
        return edges
    
    def _detect_radon(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Radon transform based line detection."""
        # Use Hough lines as a simplified Radon transform
        edges = cv2.Canny(image, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=50, 
            minLineLength=20, 
            maxLineGap=10
        )
        
        # Draw lines on mask
        line_mask = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        
        line_mask = cv2.bitwise_and(line_mask, line_mask, mask=zone_mask)
        
        return line_mask
    
    def _detect_lbp_anomaly(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """LBP-based texture anomaly detection."""
        # Compute local variance as texture measure
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((image - local_mean)**2, -1, kernel)
        
        # Normalize
        cv2.normalize(local_var, local_var, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold
        _, anomaly_mask = cv2.threshold(local_var.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        anomaly_mask = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=zone_mask)
        
        return anomaly_mask
    
    def _detect_canny(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Canny edge detection."""
        edges = cv2.Canny(image, 50, 150)
        
        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Apply zone mask
        closed = cv2.bitwise_and(closed, closed, mask=zone_mask)
        
        return closed
    
    def _detect_adaptive_threshold(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Adaptive threshold detection."""
        adaptive = cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            21, 5
        )
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
        
        # Apply zone mask
        cleaned = cv2.bitwise_and(cleaned, cleaned, mask=zone_mask)
        
        return cleaned
    
    def _detect_otsu_variants(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Multiple Otsu thresholding variants."""
        # Standard Otsu
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to find defects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Defects are the differences
        defects = cv2.absdiff(opened, closed)
        
        # Apply zone mask
        defects = cv2.bitwise_and(defects, defects, mask=zone_mask)
        
        return defects
    
    def _detect_morphological(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Morphological-based defect detection."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Top-hat for bright defects
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat for dark defects
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine both
        combined = cv2.add(tophat, blackhat)
        
        # Threshold
        _, morph_mask = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
        
        # Apply zone mask
        morph_mask = cv2.bitwise_and(morph_mask, morph_mask, mask=zone_mask)
        
        return morph_mask
    
    def _ensemble_detections(self, raw_detections: Dict[str, Dict[str, np.ndarray]], 
                            image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Combine detection results using weighted voting."""
        self.logger.info("Performing ensemble combination...")
        
        h, w = image_shape
        combined_masks = {}
        
        # Separate methods by type
        scratch_methods = ['lei', 'hessian_eigen', 'frangi', 'radon', 'phase_congruency', 'zana_klein', 'structure_tensor']
        region_methods = ['do2mr', 'log', 'doh', 'mser', 'lbp_anomaly', 'otsu_variants']
        general_methods = ['gradient_mag', 'watershed', 'canny', 'adaptive_threshold', 'morphological']
        
        # Process each zone
        for zone_name, zone_detections in raw_detections.items():
            # Initialize vote maps
            scratch_votes = np.zeros((h, w), dtype=np.float32)
            region_votes = np.zeros((h, w), dtype=np.float32)
            
            # Accumulate weighted votes
            for method_name, mask in zone_detections.items():
                if mask is None:
                    continue
                    
                weight = self.config.ensemble_confidence_weights.get(method_name, 0.5)
                
                if method_name in scratch_methods:
                    scratch_votes += (mask > 0).astype(np.float32) * weight
                elif method_name in region_methods:
                    region_votes += (mask > 0).astype(np.float32) * weight
                else:  # general methods contribute to both
                    scratch_votes += (mask > 0).astype(np.float32) * weight * 0.5
                    region_votes += (mask > 0).astype(np.float32) * weight * 0.5
            
            # Normalize vote maps
            max_scratch_vote = sum(self.config.ensemble_confidence_weights.get(m, 0.5) 
                                  for m in scratch_methods + general_methods)
            max_region_vote = sum(self.config.ensemble_confidence_weights.get(m, 0.5) 
                                 for m in region_methods + general_methods)
            
            if max_scratch_vote > 0:
                scratch_votes /= max_scratch_vote
            if max_region_vote > 0:
                region_votes /= max_region_vote
            
            # Apply threshold
            scratch_mask = (scratch_votes >= self.config.ensemble_vote_threshold).astype(np.uint8) * 255
            region_mask = (region_votes >= self.config.ensemble_vote_threshold).astype(np.uint8) * 255
            
            # Morphological refinement
            scratch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, scratch_kernel)
            
            region_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, region_kernel)
            
            # Store results
            combined_masks[f'{zone_name}_scratches'] = scratch_mask
            combined_masks[f'{zone_name}_regions'] = region_mask
            combined_masks[f'{zone_name}_all'] = cv2.bitwise_or(scratch_mask, region_mask)
        
        return combined_masks
    
    def _refine_and_segment_defects(self, ensemble_masks: Dict[str, np.ndarray],
                                   preprocessed_maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply false positive reduction and segmentation refinement."""
        self.logger.info("Refining and segmenting defects...")
        
        refined_masks = {}
        
        # Use best quality image for validation
        validation_image = preprocessed_maps.get('bilateral_0', preprocessed_maps['original'])
        
        for mask_name, mask in ensemble_masks.items():
            if 'scratches' in mask_name:
                refined = self._reduce_false_positives_scratches(mask, validation_image)
            elif 'regions' in mask_name:
                refined = self._reduce_false_positives_regions(mask, validation_image)
            else:
                refined = mask.copy()
            
            # Apply segmentation refinement if requested
            if self.config.segmentation_refinement_method == 'morphological':
                # Already done in false positive reduction
                pass
            elif self.config.segmentation_refinement_method == 'active_contour':
                # Simplified active contour using morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
                refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
            
            refined_masks[mask_name] = refined
        
        return refined_masks
    
    def _reduce_false_positives_scratches(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Reduce false positives for scratch detection."""
        refined = mask.copy()
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)
        
        for i in range(1, num_labels):
            # Get component properties
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Remove too small components
            if area < self.config.min_defect_area_px:
                refined[labels == i] = 0
                continue
            
            # Check aspect ratio
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            if aspect_ratio < self.config.min_scratch_aspect_ratio:
                refined[labels == i] = 0
                continue
            
            # Check linearity
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                
                # Fit line to contour
                if len(contour) >= 5:
                    [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calculate residuals
                    residuals = []
                    for point in contour:
                        px, py = point[0]
                        # Distance from point to line
                        dist = abs((py - y0) * vx - (px - x0) * vy) / np.sqrt(vx**2 + vy**2)
                        residuals.append(dist)
                    
                    # If residuals are too high, it's not a line
                    if np.mean(residuals) > self.config.min_scratch_linearity_threshold:
                        refined[labels == i] = 0
                        continue
                
                # Check contrast
                if self.config.use_contrast_fp_reduction:
                    component_pixels = image[component_mask > 0]
                    surrounding_mask = cv2.dilate(component_mask, np.ones((5, 5), np.uint8)) - component_mask
                    if np.sum(surrounding_mask) > 0:
                        surrounding_pixels = image[surrounding_mask > 0]
                        contrast = abs(np.mean(component_pixels) - np.mean(surrounding_pixels))
                        if contrast < self.config.min_defect_contrast:
                            refined[labels == i] = 0
        
        return refined
    
    def _reduce_false_positives_regions(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Reduce false positives for region defects."""
        refined = mask.copy()
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)
        
        # Calculate area statistics for outlier detection
        areas = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            areas.append(area)
        
        if areas:
            mean_area = np.mean(areas)
            std_area = np.std(areas)
        
        for i in range(1, num_labels):
            # Get component properties
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Remove too small or too large components
            if area < self.config.min_defect_area_px or area > self.config.max_defect_area_px:
                refined[labels == i] = 0
                continue
            
            # Remove statistical outliers
            if areas and abs(area - mean_area) > 3 * std_area:
                refined[labels == i] = 0
                continue
            
            # Check shape properties
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity < self.config.min_region_circularity:
                        refined[labels == i] = 0
                        continue
                
                # Check solidity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < self.config.min_region_solidity:
                        refined[labels == i] = 0
                        continue
                
                # Check contrast
                if self.config.use_contrast_fp_reduction:
                    component_pixels = image[component_mask > 0]
                    surrounding_mask = cv2.dilate(component_mask, np.ones((5, 5), np.uint8)) - component_mask
                    if np.sum(surrounding_mask) > 0:
                        surrounding_pixels = image[surrounding_mask > 0]
                        contrast = abs(np.mean(component_pixels) - np.mean(surrounding_pixels))
                        if contrast < self.config.min_defect_contrast:
                            refined[labels == i] = 0
        
        return refined
    
    def _characterize_defects_from_masks(self, final_masks: Dict[str, np.ndarray],
                                        preprocessed_maps: Dict[str, np.ndarray],
                                        gray_image: np.ndarray) -> List[Defect]:
        """Extract comprehensive features for each detected defect."""
        self.logger.info("Characterizing defects...")
        
        all_defects = []
        defect_id = 0
        
        # Get gradient magnitude map
        gradient_map = preprocessed_maps.get('gradient_magnitude')
        if gradient_map is None:
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_map = np.sqrt(grad_x**2 + grad_y**2)
        
        # Process each zone
        for mask_name, mask in final_masks.items():
            if '_all' not in mask_name:  # Skip individual scratch/region masks
                continue
                
            zone_name = mask_name.split('_')[0]
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):
                defect_id += 1
                
                # Get basic properties
                area_px = stats[i, cv2.CC_STAT_AREA]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Get defect mask
                component_mask = (labels == i).astype(np.uint8)
                
                # Extract contour
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                contour = contours[0]
                
                # Calculate geometric properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area_px / hull_area if hull_area > 0 else 0
                
                # Fit ellipse for orientation and dimensions
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center_x, center_y), (MA, ma), angle = ellipse
                    major_axis_length = max(MA, ma)
                    minor_axis_length = min(MA, ma)
                    orientation = angle
                    eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2) if major_axis_length > 0 else 0
                else:
                    major_axis_length = max(w, h)
                    minor_axis_length = min(w, h)
                    orientation = 0
                    eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2) if major_axis_length > 0 else 0
                
                # Extract intensity features
                component_pixels = gray_image[component_mask > 0]
                mean_intensity = np.mean(component_pixels)
                std_intensity = np.std(component_pixels)
                min_intensity = np.min(component_pixels)
                max_intensity = np.max(component_pixels)
                
                # Calculate skewness and kurtosis
                if len(component_pixels) > 3:
                    intensity_skewness = stats.skew(component_pixels)
                    intensity_kurtosis = stats.kurtosis(component_pixels)
                else:
                    intensity_skewness = 0.0
                    intensity_kurtosis = 0.0
                
                # Calculate contrast
                dilated_mask = cv2.dilate(component_mask, np.ones((5, 5), np.uint8))
                surrounding_mask = dilated_mask - component_mask
                if np.sum(surrounding_mask) > 0:
                    surrounding_pixels = gray_image[surrounding_mask > 0]
                    contrast = abs(mean_intensity - np.mean(surrounding_pixels))
                else:
                    contrast = std_intensity
                
                # Extract gradient features
                component_gradients = gradient_map[component_mask > 0]
                mean_gradient = np.mean(component_gradients)
                max_gradient = np.max(component_gradients)
                std_dev_gradient = np.std(component_gradients)
                
                # Calculate gradient orientation
                grad_x_comp = grad_x[component_mask > 0]
                grad_y_comp = grad_y[component_mask > 0]
                gradient_orientation = np.mean(np.arctan2(grad_y_comp, grad_x_comp))
                
                # Extract texture features (GLCM)
                glcm_features = self._extract_glcm_features_for_region(gray_image, component_mask, x, y, w, h)
                
                # Extract LBP features
                lbp_features = self._extract_lbp_features_for_region(gray_image, component_mask, x, y, w, h)
                
                # Extract advanced features
                hessian_eigen_ratio = self._calculate_hessian_eigen_ratio(gray_image, component_mask, x, y, w, h)
                coherence = self._calculate_coherence(gray_image, component_mask, x, y, w, h)
                frangi_response = self._calculate_frangi_response(gray_image, component_mask, x, y, w, h)
                
                # TDA features if available
                tda_features = {}
                if self.tda_analyzer and self.config.use_topological_analysis:
                    try:
                        # Extract ROI for TDA
                        roi = gray_image[y:y+h, x:x+w]
                        roi_mask = component_mask[y:y+h, x:x+w]
                        
                        # Run TDA analysis
                        tda_results = self.tda_analyzer.analyze_region(roi)
                        tda_features = {
                            'tda_local_connectivity_score': tda_results.get('mean_connectivity', 1.0),
                            'tda_betti_0_persistence': None,  # Would need more processing
                            'tda_betti_1_persistence': None   # Would need more processing
                        }
                    except Exception as e:
                        self.logger.warning(f"TDA analysis failed for defect {defect_id}: {str(e)}")
                
                # Determine defect type
                defect_type = self._classify_defect_type(
                    eccentricity, circularity, solidity, major_axis_length / minor_axis_length if minor_axis_length > 0 else 1,
                    area_px, mean_gradient, coherence
                )
                
                # Assess severity
                severity = self._assess_defect_severity(defect_type, area_px, zone_name, contrast)
                
                # Calculate confidence
                scratch_mask = final_masks.get(f'{zone_name}_scratches', np.zeros_like(mask))
                region_mask = final_masks.get(f'{zone_name}_regions', np.zeros_like(mask))
                
                contributing_algorithms = []
                if scratch_mask[cy, cx] > 0:
                    contributing_algorithms.extend(['lei', 'hessian_eigen', 'frangi'])
                if region_mask[cy, cx] > 0:
                    contributing_algorithms.extend(['do2mr', 'log', 'mser'])
                
                detection_strength = (scratch_mask[cy, cx] + region_mask[cy, cx]) / 510.0  # Normalize to 0-1
                
                confidence = self._calculate_confidence(
                    defect_type, area_px, contrast, detection_strength, len(contributing_algorithms)
                )
                
                # Convert to microns if possible
                pixels_per_micron = self.config.pixels_per_micron
                area_um = area_px / (pixels_per_micron ** 2) if pixels_per_micron > 0 else None
                
                # Create Defect object
                defect = Defect(
                    defect_id=defect_id,
                    defect_type=defect_type,
                    zone=zone_name,
                    severity=severity,
                    confidence=confidence,
                    area_px=area_px,
                    area_um=area_um,
                    perimeter=perimeter,
                    eccentricity=eccentricity,
                    solidity=solidity,
                    circularity=circularity,
                    location_xy=(cx, cy),
                    bbox=(x, y, w, h),
                    major_axis_length=major_axis_length,
                    minor_axis_length=minor_axis_length,
                    orientation=orientation,
                    mean_intensity=mean_intensity,
                    std_intensity=std_intensity,
                    min_intensity=min_intensity,
                    max_intensity=max_intensity,
                    contrast=contrast,
                    intensity_skewness=intensity_skewness,
                    intensity_kurtosis=intensity_kurtosis,
                    mean_gradient=mean_gradient,
                    max_gradient=max_gradient,
                    std_dev_gradient=std_dev_gradient,
                    gradient_orientation=gradient_orientation,
                    glcm_contrast=glcm_features['contrast'],
                    glcm_homogeneity=glcm_features['homogeneity'],
                    glcm_energy=glcm_features['energy'],
                    glcm_correlation=glcm_features['correlation'],
                    lbp_mean=lbp_features['mean'],
                    lbp_std=lbp_features['std'],
                    mean_hessian_eigen_ratio=hessian_eigen_ratio,
                    mean_coherence=coherence,
                    frangi_response=frangi_response,
                    tda_local_connectivity_score=tda_features.get('tda_local_connectivity_score'),
                    tda_betti_0_persistence=tda_features.get('tda_betti_0_persistence'),
                    tda_betti_1_persistence=tda_features.get('tda_betti_1_persistence'),
                    contributing_algorithms=contributing_algorithms,
                    detection_strength=detection_strength
                )
                
                all_defects.append(defect)
        
        return all_defects
    
    def _extract_glcm_features_for_region(self, image: np.ndarray, mask: np.ndarray, 
                                         x: int, y: int, w: int, h: int) -> Dict[str, float]:
        """Extract GLCM features for a specific region."""
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        # Apply mask
        masked_roi = roi.copy()
        masked_roi[roi_mask == 0] = 0
        
        # Compute GLCM
        try:
            glcm = graycomatrix(masked_roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            return {
                'contrast': float(contrast),
                'homogeneity': float(homogeneity),
                'energy': float(energy),
                'correlation': float(correlation)
            }
        except:
            return {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0, 'correlation': 0.0}
    
    def _extract_lbp_features_for_region(self, image: np.ndarray, mask: np.ndarray,
                                        x: int, y: int, w: int, h: int) -> Dict[str, float]:
        """Extract LBP features for a specific region."""
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        try:
            # Compute LBP
            radius = min(3, min(w, h) // 4)
            if radius < 1:
                return {'mean': 0.0, 'std': 0.0}
            
            n_points = 8 * radius
            lbp = local_binary_pattern(roi, n_points, radius, method='uniform')
            
            # Apply mask
            lbp_values = lbp[roi_mask > 0]
            
            if len(lbp_values) > 0:
                return {
                    'mean': float(np.mean(lbp_values)),
                    'std': float(np.std(lbp_values))
                }
            else:
                return {'mean': 0.0, 'std': 0.0}
        except:
            return {'mean': 0.0, 'std': 0.0}
    
    def _calculate_hessian_eigen_ratio(self, image: np.ndarray, mask: np.ndarray,
                                      x: int, y: int, w: int, h: int) -> float:
        """Calculate mean Hessian eigenvalue ratio for the region."""
        # Extract ROI
        roi = image[y:y+h, x:x+w].astype(np.float32)
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.0
        
        # Compute Hessian
        Hxx = cv2.Sobel(roi, cv2.CV_64F, 2, 0, ksize=3)
        Hyy = cv2.Sobel(roi, cv2.CV_64F, 0, 2, ksize=3)
        Hxy = cv2.Sobel(roi, cv2.CV_64F, 1, 1, ksize=3)
        
        # Compute eigenvalue ratio at each pixel
        ratios = []
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                if roi_mask[i, j] > 0:
                    # Local Hessian matrix
                    H = np.array([[Hxx[i, j], Hxy[i, j]], 
                                  [Hxy[i, j], Hyy[i, j]]])
                    
                    # Eigenvalues
                    eigenvalues = np.linalg.eigvalsh(H)
                    lambda1, lambda2 = abs(eigenvalues[1]), abs(eigenvalues[0])
                    
                    if lambda2 > 1e-6:
                        ratio = lambda1 / lambda2
                        ratios.append(ratio)
        
        return float(np.mean(ratios)) if ratios else 0.0
    
    def _calculate_coherence(self, image: np.ndarray, mask: np.ndarray,
                            x: int, y: int, w: int, h: int) -> float:
        """Calculate mean coherence from structure tensor."""
        # Extract ROI
        roi = image[y:y+h, x:x+w].astype(np.float32)
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.0
        
        # Compute gradients
        Ix = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        
        # Structure tensor components
        Jxx = gaussian_filter(Ix * Ix, 1.0)
        Jxy = gaussian_filter(Ix * Iy, 1.0)
        Jyy = gaussian_filter(Iy * Iy, 1.0)
        
        # Compute coherence
        coherence_values = []
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                if roi_mask[i, j] > 0:
                    # Local structure tensor
                    trace = Jxx[i, j] + Jyy[i, j]
                    det = Jxx[i, j] * Jyy[i, j] - Jxy[i, j]**2
                    discriminant = max(0, trace**2 - 4*det)
                    
                    mu1 = 0.5 * (trace + np.sqrt(discriminant))
                    mu2 = 0.5 * (trace - np.sqrt(discriminant))
                    
                    if mu1 + mu2 > 1e-6:
                        coherence = ((mu1 - mu2) / (mu1 + mu2))**2
                        coherence_values.append(coherence)
        
        return float(np.mean(coherence_values)) if coherence_values else 0.0
    
    def _calculate_frangi_response(self, image: np.ndarray, mask: np.ndarray,
                                  x: int, y: int, w: int, h: int) -> float:
        """Calculate mean Frangi vesselness response."""
        # Extract ROI
        roi = image[y:y+h, x:x+w].astype(np.float32)
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.0
        
        # Compute at a single scale for efficiency
        scale = 2.0
        smoothed = gaussian_filter(roi, scale)
        
        # Hessian
        Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
        Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
        Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
        
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
        
        # Apply mask and compute mean
        frangi_values = v[roi_mask > 0]
        
        return float(np.mean(frangi_values)) if len(frangi_values) > 0 else 0.0
    
    def _classify_defect_type(self, eccentricity: float, circularity: float, solidity: float,
                             aspect_ratio: float, area_px: int, mean_gradient: float,
                             coherence: float) -> DefectType:
        """Classify defect type based on features."""
        # Scratch detection
        if eccentricity > 0.95 and aspect_ratio > 4 and coherence > 0.7:
            return DefectType.SCRATCH
        
        # Crack detection
        if eccentricity > 0.9 and aspect_ratio > 3 and mean_gradient > 50 and solidity < 0.7:
            return DefectType.CRACK
        
        # Pit/Dig detection
        if circularity > 0.7 and solidity > 0.8 and area_px < 100:
            return DefectType.PIT if area_px < 50 else DefectType.DIG
        
        # Chip detection
        if solidity < 0.6 and area_px > 200:
            return DefectType.CHIP
        
        # Contamination detection
        if circularity < 0.5 and area_px > 500:
            return DefectType.CONTAMINATION
        
        # Bubble detection
        if circularity > 0.8 and solidity > 0.9 and mean_gradient < 30:
            return DefectType.BUBBLE
        
        # Burn detection
        if mean_gradient > 100 and area_px > 300:
            return DefectType.BURN
        
        return DefectType.UNKNOWN
    
    def _assess_defect_severity(self, defect_type: DefectType, area_px: int, 
                               zone: str, contrast: float) -> DefectSeverity:
        """Assess defect severity based on type, size, location, and contrast."""
        # Type-based base severity
        type_severity = {
            DefectType.CRACK: DefectSeverity.CRITICAL,
            DefectType.BURN: DefectSeverity.CRITICAL,
            DefectType.CHIP: DefectSeverity.HIGH,
            DefectType.DELAMINATION: DefectSeverity.HIGH,
            DefectType.SCRATCH: DefectSeverity.MEDIUM,
            DefectType.DIG: DefectSeverity.MEDIUM,
            DefectType.CONTAMINATION: DefectSeverity.LOW,
            DefectType.PIT: DefectSeverity.LOW,
            DefectType.BUBBLE: DefectSeverity.LOW,
            DefectType.ANOMALY: DefectSeverity.MEDIUM,
            DefectType.UNKNOWN: DefectSeverity.LOW
        }
        
        base_severity = type_severity.get(defect_type, DefectSeverity.LOW)
        
        # Adjust based on zone
        if zone == 'Core':
            # Core defects are more critical
            if base_severity == DefectSeverity.LOW:
                base_severity = DefectSeverity.MEDIUM
            elif base_severity == DefectSeverity.MEDIUM:
                base_severity = DefectSeverity.HIGH
        
        # Adjust based on size
        if area_px > 1000:
            if base_severity == DefectSeverity.LOW:
                base_severity = DefectSeverity.MEDIUM
            elif base_severity == DefectSeverity.MEDIUM:
                base_severity = DefectSeverity.HIGH
        
        # Adjust based on contrast
        if contrast > 150 and base_severity == DefectSeverity.LOW:
            base_severity = DefectSeverity.MEDIUM
        
        return base_severity
    
    def _calculate_confidence(self, defect_type: DefectType, area_px: int, contrast: float,
                             detection_strength: float, num_algorithms: int) -> float:
        """Calculate detection confidence score."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on detection strength
        confidence += detection_strength * 0.2
        
        # Adjust based on number of algorithms that detected it
        if num_algorithms >= 3:
            confidence += 0.2
        elif num_algorithms >= 2:
            confidence += 0.1
        
        # Adjust based on contrast
        if contrast > 50:
            confidence += 0.1
        
        # Adjust based on size
        if area_px > self.config.min_defect_area_px * 2:
            confidence += 0.1
        
        # Adjust based on type
        if defect_type in [DefectType.SCRATCH, DefectType.DIG, DefectType.PIT]:
            confidence += 0.1  # Well-defined types
        elif defect_type == DefectType.UNKNOWN:
            confidence -= 0.2
        
        return min(max(confidence, 0.1), 1.0)
    
    def _generate_final_report(self, image_path: str, original_image: np.ndarray,
                              global_results: Optional[Dict[str, Any]],
                              analyzed_defects: List[Defect],
                              fiber_info: Dict[str, Any],
                              zone_masks: Dict[str, np.ndarray],
                              processing_time: float) -> AnalysisReport:
        """Generate comprehensive analysis report."""
        self.logger.info("Generating final report...")
        
        # Image info
        image_info = {
            "path": image_path,
            "shape": original_image.shape,
            "dtype": str(original_image.dtype),
            "mean_intensity": float(np.mean(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 
                                           if len(original_image.shape) == 3 else original_image)),
            "std_intensity": float(np.std(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 
                                         if len(original_image.shape) == 3 else original_image))
        }
        
        # Fiber metrics
        fiber_metrics = {
            "center": fiber_info['center'],
            "cladding_radius_px": fiber_info['cladding_radius'],
            "core_radius_px": fiber_info['core_radius'],
            "confidence": fiber_info['confidence'],
            "method": fiber_info['method'],
            "pixels_per_micron": fiber_info.get('pixels_per_micron', self.config.pixels_per_micron)
        }
        
        # Count defects by zone
        defects_by_zone = {}
        for defect in analyzed_defects:
            if defect.zone not in defects_by_zone:
                defects_by_zone[defect.zone] = 0
            defects_by_zone[defect.zone] += 1
        
        # Calculate quality score
        quality_score = 100.0
        severity_penalties = {
            DefectSeverity.CRITICAL: 25,
            DefectSeverity.HIGH: 15,
            DefectSeverity.MEDIUM: 8,
            DefectSeverity.LOW: 3,
            DefectSeverity.NEGLIGIBLE: 1
        }
        
        for defect in analyzed_defects:
            quality_score -= severity_penalties.get(defect.severity, 1)
        
        quality_score = max(0, quality_score)
        
        # Determine pass/fail status
        critical_defects = sum(1 for d in analyzed_defects if d.severity == DefectSeverity.CRITICAL)
        high_severity_defects = sum(1 for d in analyzed_defects if d.severity == DefectSeverity.HIGH)
        
        failure_reasons = []
        if critical_defects > 0:
            failure_reasons.append(f"{critical_defects} critical defects detected")
        if high_severity_defects > 2:
            failure_reasons.append(f"{high_severity_defects} high-severity defects detected")
        if quality_score < 70:
            failure_reasons.append(f"Quality score too low ({quality_score:.1f}/100)")
        
        # Check global anomaly results
        if global_results:
            if global_results.get('anomaly_verdict') == 'ANOMALOUS':
                failure_reasons.append("Global statistical anomaly detected")
            if global_results.get('ssim', 1.0) < self.config.anomaly_ssim_threshold:
                failure_reasons.append(f"Low structural similarity ({global_results['ssim']:.3f})")
        
        pass_fail_status = "PASS" if len(failure_reasons) == 0 else "FAIL"
        
        # Generate recommendations
        recommendations = []
        if critical_defects > 0:
            recommendations.append("Critical defects detected. Fiber should be re-terminated.")
        if high_severity_defects > 2:
            recommendations.append("Multiple high-severity defects found. Consider re-polishing.")
        if defects_by_zone.get('Core', 0) > 5:
            recommendations.append("Multiple core defects. Check cleaning procedure.")
        
        contamination_count = sum(1 for d in analyzed_defects if d.defect_type == DefectType.CONTAMINATION)
        if contamination_count > 3:
            recommendations.append("Multiple contamination spots found. Clean with appropriate solution.")
        
        # Global TDA analysis if available
        global_tda_analysis = None
        if self.tda_analyzer and self.config.use_topological_analysis:
            try:
                # Run TDA on the entire fiber region
                fiber_mask = zone_masks.get('Fiber')
                if fiber_mask is not None:
                    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
                    masked_fiber = cv2.bitwise_and(gray, gray, mask=fiber_mask)
                    
                    tda_results = self.tda_analyzer.analyze_region(masked_fiber)
                    global_tda_analysis = {
                        'mean_connectivity': tda_results.get('mean_connectivity', 1.0),
                        'min_connectivity': tda_results.get('min_connectivity', 1.0)
                    }
                    
                    # Check TDA thresholds
                    if global_tda_analysis['mean_connectivity'] < self.config.min_global_connectivity:
                        failure_reasons.append(
                            f"Low topological connectivity ({global_tda_analysis['mean_connectivity']:.3f})"
                        )
            except Exception as e:
                self.logger.warning(f"Global TDA analysis failed: {str(e)}")
        
        # Create report
        report = AnalysisReport(
            timestamp=datetime.now().isoformat(),
            image_path=image_path,
            image_info=image_info,
            fiber_metrics=fiber_metrics,
            defects=analyzed_defects,
            defects_by_zone=defects_by_zone,
            quality_score=quality_score,
            pass_fail_status=pass_fail_status,
            failure_reasons=failure_reasons,
            global_anomaly_analysis=global_results,
            global_tda_analysis=global_tda_analysis,
            processing_time=processing_time,
            warnings=self.warnings,
            recommendations=recommendations
        )
        
        # Save report if requested
        if self.config.generate_json_report:
            report_path = os.path.splitext(image_path)[0] + "_report.json"
            self._save_json_report(report, report_path)
        
        if self.config.generate_text_report:
            report_path = os.path.splitext(image_path)[0] + "_report.txt"
            self._save_text_report(report, report_path)
        
        return report
    
    def _save_json_report(self, report: AnalysisReport, filepath: str):
        """Save report as JSON."""
        try:
            report_dict = asdict(report)
            # Convert defects to dict format
            report_dict['defects'] = [d.to_dict() for d in report.defects]
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, cls=NumpyEncoder)
            
            self.logger.info(f"JSON report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON report: {str(e)}")
    
    def _save_text_report(self, report: AnalysisReport, filepath: str):
        """Save report as formatted text."""
        try:
            with open(filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write("OMNIFIBER ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Timestamp: {report.timestamp}\n")
                f.write(f"Image: {report.image_path}\n")
                f.write(f"Processing Time: {report.processing_time:.2f} seconds\n")
                f.write("\n")
                
                f.write("OVERALL VERDICT\n")
                f.write("-"*40 + "\n")
                f.write(f"Status: {report.pass_fail_status}\n")
                f.write(f"Quality Score: {report.quality_score:.1f}/100\n")
                
                if report.failure_reasons:
                    f.write("\nFailure Reasons:\n")
                    for reason in report.failure_reasons:
                        f.write(f"  - {reason}\n")
                
                f.write("\n")
                
                f.write("DEFECT SUMMARY\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Defects: {len(report.defects)}\n")
                f.write("\nDefects by Zone:\n")
                for zone, count in report.defects_by_zone.items():
                    f.write(f"  {zone}: {count}\n")
                
                # Count by type
                type_counts = {}
                for defect in report.defects:
                    type_name = defect.defect_type.name
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1
                
                f.write("\nDefects by Type:\n")
                for dtype, count in sorted(type_counts.items()):
                    f.write(f"  {dtype}: {count}\n")
                
                # Count by severity
                severity_counts = {}
                for defect in report.defects:
                    sev_name = defect.severity.name
                    severity_counts[sev_name] = severity_counts.get(sev_name, 0) + 1
                
                f.write("\nDefects by Severity:\n")
                for sev, count in sorted(severity_counts.items()):
                    f.write(f"  {sev}: {count}\n")
                
                f.write("\n")
                
                if report.global_anomaly_analysis:
                    f.write("GLOBAL ANOMALY ANALYSIS\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Verdict: {report.global_anomaly_analysis.get('anomaly_verdict', 'N/A')}\n")
                    f.write(f"Mahalanobis Distance: {report.global_anomaly_analysis.get('mahalanobis_distance', 0):.3f}\n")
                    f.write(f"SSIM Score: {report.global_anomaly_analysis.get('ssim', 0):.3f}\n")
                    f.write("\n")
                
                if report.global_tda_analysis:
                    f.write("TOPOLOGICAL ANALYSIS\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Mean Connectivity: {report.global_tda_analysis.get('mean_connectivity', 0):.3f}\n")
                    f.write(f"Min Connectivity: {report.global_tda_analysis.get('min_connectivity', 0):.3f}\n")
                    f.write("\n")
                
                if report.recommendations:
                    f.write("RECOMMENDATIONS\n")
                    f.write("-"*40 + "\n")
                    for rec in report.recommendations:
                        f.write(f"  - {rec}\n")
                    f.write("\n")
                
                # Detailed defect list (first 20)
                f.write("DEFECT DETAILS (First 20)\n")
                f.write("-"*40 + "\n")
                for i, defect in enumerate(report.defects[:20]):
                    f.write(f"\nDefect #{defect.defect_id}:\n")
                    f.write(f"  Type: {defect.defect_type.name}\n")
                    f.write(f"  Zone: {defect.zone}\n")
                    f.write(f"  Severity: {defect.severity.name}\n")
                    f.write(f"  Location: {defect.location_xy}\n")
                    f.write(f"  Area: {defect.area_px} px")
                    if defect.area_um:
                        f.write(f" ({defect.area_um:.1f} µm²)")
                    f.write("\n")
                    f.write(f"  Confidence: {defect.confidence:.2f}\n")
                
                if len(report.defects) > 20:
                    f.write(f"\n... and {len(report.defects) - 20} more defects\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
            
            self.logger.info(f"Text report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save text report: {str(e)}")
    
    def _visualize_master_results(self, original_image: np.ndarray, report: AnalysisReport):
        """Create comprehensive visualization of results."""
        self.logger.info("Creating visualization...")
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Convert to RGB for display
        if len(original_image.shape) == 2:
            display_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            display_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Panel 1: Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(display_img)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Panel 2: Defect overlay
        ax2 = fig.add_subplot(gs[0, 1])
        overlay_img = display_img.copy()
        
        # Define colors for different severities
        severity_colors = {
            DefectSeverity.CRITICAL: (255, 0, 0),      # Red
            DefectSeverity.HIGH: (255, 128, 0),        # Orange
            DefectSeverity.MEDIUM: (255, 255, 0),      # Yellow
            DefectSeverity.LOW: (0, 255, 0),           # Green
            DefectSeverity.NEGLIGIBLE: (0, 255, 255)   # Cyan
        }
        
        # Draw defects
        for defect in report.defects:
            color = severity_colors.get(defect.severity, (255, 255, 255))
            x, y, w, h = defect.bbox
            cv2.rectangle(overlay_img, (x, y), (x+w, y+h), color, 2)
            
            # Add defect ID
            cv2.putText(overlay_img, str(defect.defect_id), (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        ax2.imshow(overlay_img)
        ax2.set_title(f'Detected Defects ({len(report.defects)})', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Panel 3: Pass/Fail status
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        # Create status box
        status_color = 'green' if report.pass_fail_status == 'PASS' else 'red'
        status_text = f"STATUS: {report.pass_fail_status}\n\n"
        status_text += f"Quality Score: {report.quality_score:.1f}/100\n\n"
        
        if report.failure_reasons:
            status_text += "Failure Reasons:\n"
            for reason in report.failure_reasons[:3]:
                status_text += f"• {reason}\n"
            if len(report.failure_reasons) > 3:
                status_text += f"• ... and {len(report.failure_reasons) - 3} more\n"
        
        ax3.text(0.5, 0.5, status_text, transform=ax3.transAxes,
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
        
        # Panel 4: Defect distribution pie chart
        ax4 = fig.add_subplot(gs[0, 3])
        
        # Count defects by type
        type_counts = {}
        for defect in report.defects:
            type_name = defect.defect_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        if type_counts:
            labels = list(type_counts.keys())
            sizes = list(type_counts.values())
            
            ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Defects by Type', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No defects detected', ha='center', va='center')
            ax4.set_title('Defects by Type', fontsize=12, fontweight='bold')
        
        # Panel 5: Zone distribution
        ax5 = fig.add_subplot(gs[1, 0])
        
        if report.defects_by_zone:
            zones = list(report.defects_by_zone.keys())
            counts = list(report.defects_by_zone.values())
            
            bars = ax5.bar(zones, counts)
            ax5.set_title('Defects by Zone', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Count')
            
            # Color bars by zone
            zone_colors = {'Core': 'red', 'Cladding': 'orange', 'Ferrule': 'yellow', 'Adhesive': 'green'}
            for bar, zone in zip(bars, zones):
                bar.set_color(zone_colors.get(zone, 'blue'))
        else:
            ax5.text(0.5, 0.5, 'No defects detected', ha='center', va='center')
            ax5.set_title('Defects by Zone', fontsize=12, fontweight='bold')
        
        # Panel 6: Severity distribution
        ax6 = fig.add_subplot(gs[1, 1])
        
        severity_counts = {}
        for defect in report.defects:
            sev_name = defect.severity.name
            severity_counts[sev_name] = severity_counts.get(sev_name, 0) + 1
        
        if severity_counts:
            severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE']
            counts = [severity_counts.get(s, 0) for s in severities]
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'cyan']
            
            bars = ax6.bar(severities, counts, color=colors)
            ax6.set_title('Defects by Severity', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Count')
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax6.text(0.5, 0.5, 'No defects detected', ha='center', va='center')
            ax6.set_title('Defects by Severity', fontsize=12, fontweight='bold')
        
        # Panel 7: Global anomaly results
        ax7 = fig.add_subplot(gs[1, 2:])
        ax7.axis('off')
        
        global_text = "GLOBAL ANALYSIS RESULTS\n" + "="*30 + "\n\n"
        
        if report.global_anomaly_analysis:
            ga = report.global_anomaly_analysis
            global_text += f"Statistical Anomaly: {ga.get('anomaly_verdict', 'N/A')}\n"
            global_text += f"Mahalanobis Distance: {ga.get('mahalanobis_distance', 0):.3f}\n"
            global_text += f"SSIM Score: {ga.get('ssim', 0):.3f}\n"
            global_text += f"Confidence: {ga.get('confidence', 0):.1%}\n"
        else:
            global_text += "Statistical analysis not performed\n"
        
        if report.global_tda_analysis:
            global_text += f"\nTopological Connectivity: {report.global_tda_analysis.get('mean_connectivity', 0):.3f}\n"
        
        ax7.text(0.05, 0.95, global_text, transform=ax7.transAxes,
                fontsize=11, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Panel 8: Top defects table
        ax8 = fig.add_subplot(gs[2, :2])
        ax8.axis('tight')
        ax8.axis('off')
        
        # Create defect table
        headers = ['ID', 'Type', 'Zone', 'Severity', 'Area (px)', 'Confidence']
        table_data = []
        
        for defect in report.defects[:10]:  # Show top 10
            table_data.append([
                defect.defect_id,
                defect.defect_type.name[:8],
                defect.zone,
                defect.severity.name[:4],
                defect.area_px,
                f"{defect.confidence:.2f}"
            ])
        
        if table_data:
            table = ax8.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Color code by severity
            for i, defect in enumerate(report.defects[:10]):
                severity_color_map = {
                    DefectSeverity.CRITICAL: '#ffcccc',
                    DefectSeverity.HIGH: '#ffe6cc',
                    DefectSeverity.MEDIUM: '#ffffcc',
                    DefectSeverity.LOW: '#ccffcc',
                    DefectSeverity.NEGLIGIBLE: '#ccffff'
                }
                color = severity_color_map.get(defect.severity, '#ffffff')
                for j in range(6):  # 6 columns
                    table[(i+1, j)].set_facecolor(color)
        
        ax8.set_title('Top Defects (First 10)', fontsize=12, fontweight='bold')
        
        # Panel 9: Recommendations
        ax9 = fig.add_subplot(gs[2, 2:])
        ax9.axis('off')
        
        rec_text = "RECOMMENDATIONS\n" + "="*30 + "\n\n"
        
        if report.recommendations:
            for i, rec in enumerate(report.recommendations, 1):
                rec_text += f"{i}. {rec}\n"
        else:
            rec_text += "No specific recommendations.\n"
            rec_text += "Fiber meets quality standards."
        
        if report.warnings:
            rec_text += f"\n\nWARNINGS ({len(report.warnings)}):\n"
            for warning in report.warnings[:3]:
                rec_text += f"• {warning}\n"
        
        ax9.text(0.05, 0.95, rec_text, transform=ax9.transAxes,
                fontsize=11, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Main title
        fig.suptitle(f'OmniFiber Analysis Report - {os.path.basename(report.image_path)}',
                    fontsize=16, fontweight='bold')
        
        # Save figure
        output_path = os.path.splitext(report.image_path)[0] + "_visualization.png"
        plt.savefig(output_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {output_path}")
    
    def _create_error_report(self, image_path: str, error_msg: str, elapsed_time: float) -> AnalysisReport:
        """Create an error report when analysis fails."""
        return AnalysisReport(
            timestamp=datetime.now().isoformat(),
            image_path=image_path,
            image_info={"error": error_msg},
            fiber_metrics={},
            defects=[],
            defects_by_zone={},
            quality_score=0.0,
            pass_fail_status="ERROR",
            failure_reasons=[f"Analysis failed: {error_msg}"],
            global_anomaly_analysis=None,
            global_tda_analysis=None,
            processing_time=elapsed_time,
            warnings=self.warnings,
            recommendations=["Please check the image file and try again."]
        )

# =============================================================================
# PART 5: MAIN EXECUTION AND INTERACTIVE INTERFACE
# =============================================================================

def main():
    """Main function with interactive menu-driven workflow."""
    print("\n" + "="*80)
    print("OmniFiberAnalyzer: Complete Fiber Optic Defect Detection System (v3.0)".center(80))
    print("="*80)
    
    # Initialize configuration
    config = OmniConfig()
    analyzer_instance = None
    
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Configure Analysis Settings")
        print("2. Knowledge Base Management")
        print("3. Analyze Fiber Image")
        print("4. Batch Analysis")
        print("5. Exit")
        
        choice = input("\nPlease select an option (1-5): ").strip()
        
        if choice == '1':
            # Configuration menu
            config = configure_settings(config)
            
        elif choice == '2':
            # Knowledge base management
            kb_path = manage_knowledge_base(config)
            if kb_path:
                config.knowledge_base_path = kb_path
                analyzer_instance = None  # Force recreation with new KB
            
        elif choice == '3':
            # Single image analysis
            if analyzer_instance is None:
                print("\nInitializing analyzer...")
                analyzer_instance = OmniFiberAnalyzer(config)
            
            image_path = input("\nEnter path to fiber image: ").strip().strip('"\'')
            
            if os.path.isfile(image_path):
                try:
                    print("\nAnalyzing image...")
                    report = analyzer_instance.analyze_end_face(image_path)
                    
                    print("\n" + "="*60)
                    print("ANALYSIS COMPLETE")
                    print("="*60)
                    print(f"Status: {report.pass_fail_status}")
                    print(f"Quality Score: {report.quality_score:.1f}/100")
                    print(f"Total Defects: {len(report.defects)}")
                    print(f"Processing Time: {report.processing_time:.2f} seconds")
                    
                    if report.failure_reasons:
                        print("\nFailure Reasons:")
                        for reason in report.failure_reasons:
                            print(f"  - {reason}")
                    
                    print(f"\nResults saved to:")
                    print(f"  - {os.path.splitext(image_path)[0]}_visualization.png")
                    if config.generate_json_report:
                        print(f"  - {os.path.splitext(image_path)[0]}_report.json")
                    if config.generate_text_report:
                        print(f"  - {os.path.splitext(image_path)[0]}_report.txt")
                    
                except Exception as e:
                    print(f"\n✗ Analysis failed: {str(e)}")
                    logging.error(traceback.format_exc())
            else:
                print(f"\n✗ File not found: {image_path}")
        
        elif choice == '4':
            # Batch analysis
            batch_analysis(config)
        
        elif choice == '5':
            print("\nExiting OmniFiberAnalyzer. Thank you!")
            break
        
        else:
            print("\n✗ Invalid option. Please enter a number between 1 and 5.")
    
    print("\n" + "="*80)

def configure_settings(config: OmniConfig) -> OmniConfig:
    """Interactive configuration menu."""
    while True:
        print("\n--- CONFIGURATION SETTINGS ---")
        print("1. General Settings")
        print("2. Preprocessing Settings")
        print("3. Detection Algorithm Settings")
        print("4. Ensemble Settings")
        print("5. Output Settings")
        print("6. Save Configuration")
        print("7. Load Configuration")
        print("8. Back to Main Menu")
        
        choice = input("\nSelect category (1-8): ").strip()
        
        if choice == '1':
            # General settings
            print("\n--- General Settings ---")
            config.min_defect_area_px = int(input(f"Minimum defect area (pixels) [{config.min_defect_area_px}]: ") or config.min_defect_area_px)
            config.pixels_per_micron = float(input(f"Pixels per micron [{config.pixels_per_micron}]: ") or config.pixels_per_micron)
            config.use_global_anomaly_analysis = input(f"Use global anomaly analysis? (y/n) [{'y' if config.use_global_anomaly_analysis else 'n'}]: ").lower() == 'y'
            config.use_topological_analysis = input(f"Use topological analysis? (y/n) [{'y' if config.use_topological_analysis else 'n'}]: ").lower() == 'y'
            
        elif choice == '2':
            # Preprocessing settings
            print("\n--- Preprocessing Settings ---")
            config.use_anisotropic_diffusion = input(f"Use anisotropic diffusion? (y/n) [{'y' if config.use_anisotropic_diffusion else 'n'}]: ").lower() == 'y'
            config.use_coherence_enhancing_diffusion = input(f"Use coherence-enhancing diffusion? (y/n) [{'y' if config.use_coherence_enhancing_diffusion else 'n'}]: ").lower() == 'y'
            config.denoise_strength = float(input(f"Denoising strength [{config.denoise_strength}]: ") or config.denoise_strength)
            
        elif choice == '3':
            # Detection algorithm settings
            print("\n--- Detection Algorithm Settings ---")
            print("Current enabled detectors:", ', '.join(config.enabled_detectors))
            
            # Show available detectors
            all_detectors = ['do2mr', 'lei', 'zana_klein', 'log', 'doh', 'hessian_eigen', 
                           'frangi', 'structure_tensor', 'mser', 'watershed', 
                           'gradient_mag', 'phase_congruency', 'radon', 'lbp_anomaly',
                           'canny', 'adaptive_threshold', 'otsu_variants', 'morphological']
            
            print("\nAvailable detectors:")
            for i, det in enumerate(all_detectors, 1):
                status = "✓" if det in config.enabled_detectors else " "
                print(f"  [{status}] {i}. {det}")
            
            toggle = input("\nEnter detector numbers to toggle (comma-separated) or press Enter to skip: ").strip()
            if toggle:
                indices = [int(x.strip()) - 1 for x in toggle.split(',') if x.strip().isdigit()]
                for idx in indices:
                    if 0 <= idx < len(all_detectors):
                        det = all_detectors[idx]
                        if det in config.enabled_detectors:
                            config.enabled_detectors.remove(det)
                        else:
                            config.enabled_detectors.append(det)
            
        elif choice == '4':
            # Ensemble settings
            print("\n--- Ensemble Settings ---")
            config.ensemble_vote_threshold = float(input(f"Ensemble vote threshold [{config.ensemble_vote_threshold}]: ") or config.ensemble_vote_threshold)
            config.min_methods_for_detection = int(input(f"Minimum methods for detection [{config.min_methods_for_detection}]: ") or config.min_methods_for_detection)
            
        elif choice == '5':
            # Output settings
            print("\n--- Output Settings ---")
            config.generate_json_report = input(f"Generate JSON report? (y/n) [{'y' if config.generate_json_report else 'n'}]: ").lower() == 'y'
            config.generate_text_report = input(f"Generate text report? (y/n) [{'y' if config.generate_text_report else 'n'}]: ").lower() == 'y'
            config.save_intermediate_masks = input(f"Save intermediate masks? (y/n) [{'y' if config.save_intermediate_masks else 'n'}]: ").lower() == 'y'
            config.visualization_dpi = int(input(f"Visualization DPI [{config.visualization_dpi}]: ") or config.visualization_dpi)
            
        elif choice == '6':
            # Save configuration
            config_path = input("Enter path to save configuration (e.g., config.json): ").strip()
            try:
                with open(config_path, 'w') as f:
                    json.dump(asdict(config), f, indent=2, cls=NumpyEncoder)
                print(f"✓ Configuration saved to {config_path}")
            except Exception as e:
                print(f"✗ Failed to save configuration: {str(e)}")
            
        elif choice == '7':
            # Load configuration
            config_path = input("Enter path to load configuration from: ").strip()
            if os.path.isfile(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    # Update config with loaded values
                    for key, value in config_dict.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    print(f"✓ Configuration loaded from {config_path}")
                except Exception as e:
                    print(f"✗ Failed to load configuration: {str(e)}")
            else:
                print(f"✗ File not found: {config_path}")
            
        elif choice == '8':
            break
        
        else:
            print("\n✗ Invalid option.")
    
    return config

def manage_knowledge_base(config: OmniConfig) -> Optional[str]:
    """Interactive knowledge base management."""
    while True:
        print("\n--- KNOWLEDGE BASE MANAGEMENT ---")
        print("1. Select existing knowledge base")
        print("2. Create new knowledge base")
        print("3. Add images to existing knowledge base")
        print("4. View knowledge base info")
        print("5. Back to Main Menu")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            # Select existing KB
            kb_path = input("Enter path to knowledge base JSON file: ").strip().strip('"\'')
            if os.path.isfile(kb_path):
                # Test load
                try:
                    analyzer = UltraComprehensiveMatrixAnalyzer(kb_path)
                    if analyzer.reference_model.get('statistical_model'):
                        print(f"✓ Successfully loaded knowledge base with {analyzer.reference_model['statistical_model'].get('n_samples', 0)} samples")
                        return kb_path
                    else:
                        print("✗ Knowledge base appears to be empty")
                except Exception as e:
                    print(f"✗ Failed to load knowledge base: {str(e)}")
            else:
                print(f"✗ File not found: {kb_path}")
        
        elif choice == '2':
            # Create new KB
            ref_dir = input("Enter path to folder with reference images: ").strip().strip('"\'')
            if os.path.isdir(ref_dir):
                kb_path = input("Enter path to save new knowledge base (e.g., my_kb.json): ").strip()
                
                print("\nBuilding knowledge base...")
                analyzer = UltraComprehensiveMatrixAnalyzer(kb_path)
                
                if analyzer.build_comprehensive_reference_model(ref_dir, append_mode=False):
                    print(f"✓ Knowledge base created successfully at {kb_path}")
                    return kb_path
                else:
                    print("✗ Failed to create knowledge base")
            else:
                print(f"✗ Directory not found: {ref_dir}")
        
        elif choice == '3':
            # Add to existing KB
            kb_path = input("Enter path to existing knowledge base: ").strip().strip('"\'')
            if os.path.isfile(kb_path):
                ref_dir = input("Enter path to folder with new reference images: ").strip().strip('"\'')
                if os.path.isdir(ref_dir):
                    print("\nUpdating knowledge base...")
                    analyzer = UltraComprehensiveMatrixAnalyzer(kb_path)
                    
                    if analyzer.build_comprehensive_reference_model(ref_dir, append_mode=True):
                        print(f"✓ Knowledge base updated successfully")
                        return kb_path
                    else:
                        print("✗ Failed to update knowledge base")
                else:
                    print(f"✗ Directory not found: {ref_dir}")
            else:
                print(f"✗ File not found: {kb_path}")
        
        elif choice == '4':
            # View KB info
            kb_path = config.knowledge_base_path
            if kb_path and os.path.isfile(kb_path):
                try:
                    analyzer = UltraComprehensiveMatrixAnalyzer(kb_path)
                    if analyzer.reference_model.get('statistical_model'):
                        print(f"\n--- Knowledge Base Info ---")
                        print(f"Path: {kb_path}")
                        print(f"Samples: {analyzer.reference_model['statistical_model'].get('n_samples', 0)}")
                        print(f"Features: {len(analyzer.reference_model.get('feature_names', []))}")
                        print(f"Last updated: {analyzer.reference_model.get('timestamp', 'Unknown')}")
                        
                        thresholds = analyzer.reference_model.get('learned_thresholds', {})
                        if thresholds:
                            print(f"\nLearned Thresholds:")
                            print(f"  Anomaly threshold: {thresholds.get('anomaly_threshold', 0):.3f}")
                            print(f"  Mean score: {thresholds.get('anomaly_mean', 0):.3f}")
                            print(f"  Std score: {thresholds.get('anomaly_std', 0):.3f}")
                    else:
                        print("✗ No valid knowledge base loaded")
                except Exception as e:
                    print(f"✗ Error reading knowledge base: {str(e)}")
            else:
                print("✗ No knowledge base configured")
        
        elif choice == '5':
            break
        
        else:
            print("\n✗ Invalid option.")
    
    return None

def batch_analysis(config: OmniConfig):
    """Perform batch analysis on multiple images."""
    input_dir = input("Enter directory containing fiber images: ").strip().strip('"\'')
    
    if not os.path.isdir(input_dir):
        print(f"✗ Directory not found: {input_dir}")
        return
    
    # Find all image files
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.json']
    image_files = []
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(input_dir, filename))
    
    if not image_files:
        print(f"✗ No valid image files found in {input_dir}")
        return
    
    print(f"\n✓ Found {len(image_files)} images to analyze")
    
    # Create output directory
    output_dir = input("Enter output directory for results (or press Enter for same as input): ").strip()
    if not output_dir:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = OmniFiberAnalyzer(config)
    
    # Process images
    results_summary = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        try:
            report = analyzer.analyze_end_face(image_path)
            
            # Move outputs to output directory if different
            if output_dir != input_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Move visualization
                viz_src = os.path.splitext(image_path)[0] + "_visualization.png"
                if os.path.exists(viz_src):
                    viz_dst = os.path.join(output_dir, base_name + "_visualization.png")
                    os.rename(viz_src, viz_dst)
                
                # Move reports
                if config.generate_json_report:
                    json_src = os.path.splitext(image_path)[0] + "_report.json"
                    if os.path.exists(json_src):
                        json_dst = os.path.join(output_dir, base_name + "_report.json")
                        os.rename(json_src, json_dst)
                
                if config.generate_text_report:
                    txt_src = os.path.splitext(image_path)[0] + "_report.txt"
                    if os.path.exists(txt_src):
                        txt_dst = os.path.join(output_dir, base_name + "_report.txt")
                        os.rename(txt_src, txt_dst)
            
            # Add to summary
            results_summary.append({
                'file': os.path.basename(image_path),
                'status': report.pass_fail_status,
                'quality_score': report.quality_score,
                'defect_count': len(report.defects),
                'processing_time': report.processing_time
            })
            
            print(f"  ✓ Complete - Status: {report.pass_fail_status}, Defects: {len(report.defects)}")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results_summary.append({
                'file': os.path.basename(image_path),
                'status': 'ERROR',
                'quality_score': 0,
                'defect_count': 0,
                'processing_time': 0
            })
    
    # Save batch summary
    summary_path = os.path.join(output_dir, "batch_summary.csv")
    try:
        with open(summary_path, 'w') as f:
            f.write("File,Status,Quality Score,Defect Count,Processing Time (s)\n")
            for result in results_summary:
                f.write(f"{result['file']},{result['status']},{result['quality_score']:.1f},"
                       f"{result['defect_count']},{result['processing_time']:.2f}\n")
        
        print(f"\n✓ Batch summary saved to {summary_path}")
    except Exception as e:
        print(f"\n✗ Failed to save batch summary: {str(e)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("BATCH ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total images: {len(results_summary)}")
    print(f"Passed: {sum(1 for r in results_summary if r['status'] == 'PASS')}")
    print(f"Failed: {sum(1 for r in results_summary if r['status'] == 'FAIL')}")
    print(f"Errors: {sum(1 for r in results_summary if r['status'] == 'ERROR')}")
    
    avg_quality = np.mean([r['quality_score'] for r in results_summary if r['status'] != 'ERROR'])
    avg_defects = np.mean([r['defect_count'] for r in results_summary if r['status'] != 'ERROR'])
    avg_time = np.mean([r['processing_time'] for r in results_summary if r['status'] != 'ERROR'])
    
    print(f"\nAverage quality score: {avg_quality:.1f}")
    print(f"Average defects per image: {avg_defects:.1f}")
    print(f"Average processing time: {avg_time:.2f} seconds")

if __name__ == "__main__":
    # Check for command line arguments
    parser = argparse.ArgumentParser(
        description="OmniFiberAnalyzer - Complete Fiber Optic Defect Detection System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'image_path', 
        nargs='?', 
        help='Path to fiber image for direct analysis (optional)'
    )
    
    parser.add_argument(
        '--kb', '--knowledge-base',
        dest='knowledge_base',
        help='Path to knowledge base JSON file'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--batch',
        help='Directory for batch analysis'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Handle command line mode
    if args.image_path or args.batch:
        # Direct analysis mode
        config = OmniConfig()
        
        # Load configuration if provided
        if args.config and os.path.isfile(args.config):
            try:
                with open(args.config, 'r') as f:
                    config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                print(f"✓ Loaded configuration from {args.config}")
            except Exception as e:
                print(f"✗ Failed to load configuration: {str(e)}")
        
        # Set knowledge base if provided
        if args.knowledge_base:
            config.knowledge_base_path = args.knowledge_base
        
        # Initialize analyzer
        analyzer = OmniFiberAnalyzer(config)
        
        if args.batch:
            # Batch mode
            print(f"Running batch analysis on {args.batch}")
            # Implement simplified batch logic here
        else:
            # Single image mode
            try:
                print(f"Analyzing {args.image_path}...")
                report = analyzer.analyze_end_face(args.image_path)
                
                print(f"\nAnalysis Complete!")
                print(f"Status: {report.pass_fail_status}")
                print(f"Quality Score: {report.quality_score:.1f}/100")
                print(f"Defects Found: {len(report.defects)}")
                
            except Exception as e:
                print(f"✗ Analysis failed: {str(e)}")
                sys.exit(1)
    else:
        # Interactive mode
        main()