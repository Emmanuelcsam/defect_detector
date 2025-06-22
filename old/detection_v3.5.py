#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omni_fiber_analyzer.py - Enhanced OmniFiberAnalyzer System (Optimized Production Version)

This script implements a master system for analyzing fiber optic end-face images.
It unifies robust engineering, exhaustive ensemble methods, statistical anomaly detection,
feature extraction, and advanced mathematical concepts including Topological Data Analysis.

Author: Unified Fiber Analysis Team
Version: 4.0 - Optimized Production Implementation
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
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from enum import Enum, auto
import warnings
from pathlib import Path
import gc

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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)

@dataclass
class OmniConfig:
    """Centralizes all tunable parameters for the OmniFiberAnalyzer."""
    # --- General Settings ---
    min_defect_area_px: int = 5
    max_defect_area_px: int = 10000
    max_defect_area_ratio: float = 0.1
    max_defect_eccentricity: float = 0.98
    pixels_per_micron: float = 1.0
    
    # --- Memory Management ---
    max_intermediate_results: int = 10
    cleanup_intermediate: bool = True
    
    # --- Output Settings ---
    output_dpi: int = 200
    save_intermediate_masks: bool = False
    generate_json_report: bool = True
    generate_text_report: bool = True
    visualization_dpi: int = 150
    
    # --- Global Anomaly Analysis Settings ---
    use_global_anomaly_analysis: bool = True
    knowledge_base_path: Optional[str] = None
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
    
    # --- Preprocessing Settings ---
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
    primary_masking_method: str = 'ensemble'
    cladding_core_ratio: float = 125.0 / 9.0
    ferrule_buffer_ratio: float = 1.2
    hough_dp_values: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    hough_param1_values: List[int] = field(default_factory=lambda: [50, 70, 100])
    hough_param2_values: List[int] = field(default_factory=lambda: [30, 40, 50])
    adaptive_threshold_block_size: int = 51
    adaptive_threshold_c: int = 10
    
    # --- Detection Algorithm Settings ---
    enabled_detectors: Set[str] = field(default_factory=lambda: {
        'do2mr', 'lei', 'zana_klein', 'log', 'doh', 'hessian_eigen', 
        'frangi', 'structure_tensor', 'mser', 'watershed', 
        'gradient_mag', 'phase_congruency', 'radon', 'lbp_anomaly',
        'canny', 'adaptive_threshold', 'otsu_variants', 'morphological'
    })
    
    # Algorithm-specific parameters
    do2mr_kernel_sizes: List[int] = field(default_factory=lambda: [5, 11, 15, 21])
    do2mr_gamma_values: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 3.5])
    lei_kernel_lengths: List[int] = field(default_factory=lambda: [9, 11, 15, 19, 21])
    lei_angle_steps: List[int] = field(default_factory=lambda: [5, 10, 15])
    lei_threshold_factors: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    zana_opening_length: int = 15
    zana_laplacian_threshold: float = 1.5
    log_min_sigma: int = 2
    log_max_sigma: int = 10
    log_num_sigma: int = 5
    log_threshold: float = 0.05
    hessian_scales: List[float] = field(default_factory=lambda: [1, 2, 3, 4])
    frangi_scales: List[float] = field(default_factory=lambda: [1, 1.5, 2, 2.5, 3])
    frangi_beta: float = 0.5
    frangi_gamma: float = 15
    
    # --- Ensemble Settings ---
    ensemble_confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'do2mr': 1.0, 'lei': 1.0, 'zana_klein': 0.95, 'log': 0.9,
        'doh': 0.85, 'hessian_eigen': 0.9, 'frangi': 0.9,
        'structure_tensor': 0.85, 'mser': 0.8, 'watershed': 0.75,
        'gradient_mag': 0.8, 'phase_congruency': 0.85, 'radon': 0.8,
        'lbp_anomaly': 0.7, 'canny': 0.7, 'adaptive_threshold': 0.75,
        'otsu_variants': 0.7, 'morphological': 0.75
    })
    ensemble_vote_threshold: float = 0.3
    min_methods_for_detection: int = 2
    
    # --- Segmentation Refinement Settings ---
    segmentation_refinement_method: str = 'morphological'
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

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.min_defect_area_px < 1:
            raise ValueError("min_defect_area_px must be at least 1")
        if self.ensemble_vote_threshold < 0 or self.ensemble_vote_threshold > 1:
            raise ValueError("ensemble_vote_threshold must be between 0 and 1")
        if self.knowledge_base_path and not os.path.exists(self.knowledge_base_path):
            logging.warning(f"Knowledge base path does not exist: {self.knowledge_base_path}")

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
# PART 2: KNOWLEDGE BASE ANALYZER (Optimized jake.py implementation)
# =============================================================================

class KnowledgeBaseAnalyzer:
    """Optimized statistical anomaly detection engine."""
    
    def __init__(self, kb_path: Optional[str] = None):
        self.knowledge_base_path = kb_path
        self.reference_model = None
        self._feature_cache = {}
        self._comparison_cache = {}
        
        if kb_path and os.path.exists(kb_path):
            self.load_knowledge_base()
    
    def load_knowledge_base(self) -> bool:
        """Load previously saved knowledge base from JSON."""
        if not self.knowledge_base_path or not os.path.exists(self.knowledge_base_path):
            return False
            
        try:
            with open(self.knowledge_base_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Convert lists back to numpy arrays
            if 'archetype_image' in loaded_data and loaded_data['archetype_image']:
                loaded_data['archetype_image'] = np.array(loaded_data['archetype_image'], dtype=np.uint8)
            
            if 'statistical_model' in loaded_data:
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    if key in loaded_data['statistical_model'] and loaded_data['statistical_model'][key]:
                        loaded_data['statistical_model'][key] = np.array(
                            loaded_data['statistical_model'][key], dtype=np.float64
                        )
            
            self.reference_model = loaded_data
            logging.info(f"Successfully loaded knowledge base from {self.knowledge_base_path}")
            return True
            
        except Exception as e:
            logging.error(f"Could not load knowledge base: {e}")
            return False
    
    def save_knowledge_base(self) -> bool:
        """Save current knowledge base to JSON."""
        if not self.knowledge_base_path or not self.reference_model:
            return False
            
        try:
            # Create directory if needed
            kb_dir = os.path.dirname(self.knowledge_base_path)
            if kb_dir and not os.path.exists(kb_dir):
                os.makedirs(kb_dir, exist_ok=True)
            
            # Prepare data for JSON serialization
            save_data = self.reference_model.copy()
            
            # Convert numpy arrays to lists
            if 'archetype_image' in save_data and isinstance(save_data['archetype_image'], np.ndarray):
                save_data['archetype_image'] = save_data['archetype_image'].tolist()
            
            if 'statistical_model' in save_data:
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    if key in save_data['statistical_model'] and isinstance(save_data['statistical_model'][key], np.ndarray):
                        save_data['statistical_model'][key] = save_data['statistical_model'][key].tolist()
            
            # Save to file
            save_data['timestamp'] = datetime.now().isoformat()
            
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(save_data, f, indent=2, cls=NumpyEncoder)
            
            logging.info(f"Knowledge base saved to {self.knowledge_base_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving knowledge base: {e}")
            return False
    
    def extract_features(self, image: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
        """Extract optimized feature set from image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Check cache
        img_hash = hash(gray.tobytes())
        if img_hash in self._feature_cache:
            return self._feature_cache[img_hash]
        
        features = {}
        
        # Statistical features
        flat = gray.flatten()
        features.update({
            'stat_mean': float(np.mean(gray)),
            'stat_std': float(np.std(gray)),
            'stat_median': float(np.median(gray)),
            'stat_mad': float(np.median(np.abs(gray - np.median(gray)))),
            'stat_entropy': float(stats.entropy(np.histogram(gray, bins=256)[0] + 1e-10))
        })
        
        # Texture features (simplified)
        features.update(self._extract_texture_features(gray))
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        features.update({
            'gradient_mean': float(np.mean(grad_mag)),
            'gradient_std': float(np.std(grad_mag)),
            'gradient_max': float(np.max(grad_mag))
        })
        
        # Sanitize features
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        feature_names = sorted(features.keys())
        result = (features, feature_names)
        
        # Cache result
        self._feature_cache[img_hash] = result
        
        # Limit cache size
        if len(self._feature_cache) > 100:
            self._feature_cache.clear()
        
        return result
    
    def _extract_texture_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract simplified texture features."""
        features = {}
        
        # Simple GLCM features
        try:
            # Quantize image for faster computation
            img_q = (gray // 32).astype(np.uint8)
            
            # Compute GLCM for one distance and angle
            glcm = graycomatrix(img_q, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
            
            features['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
            features['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
            features['glcm_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
            
        except Exception:
            features.update({'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0, 'glcm_energy': 0.0})
        
        return features
    
    def detect_anomalies(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform anomaly detection on test image."""
        if not self.reference_model or 'statistical_model' not in self.reference_model:
            return {
                "mahalanobis_distance": 999.0,
                "ssim": 0.0,
                "anomaly_verdict": "UNKNOWN_NO_KB",
                "deviant_features": []
            }
        
        # Extract features
        test_features, _ = self.extract_features(image)
        
        # Get reference statistics
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model.get('feature_names', list(test_features.keys()))
        
        # Create feature vector
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        ref_mean = np.array(stat_model.get('mean', np.zeros_like(test_vector)))
        ref_std = np.array(stat_model.get('std', np.ones_like(test_vector)))
        
        # Compute Mahalanobis distance (simplified)
        z_scores = np.abs((test_vector - ref_mean) / (ref_std + 1e-10))
        mahalanobis_dist = float(np.sqrt(np.sum(z_scores**2)))
        
        # Find most deviant features
        top_indices = np.argsort(z_scores)[::-1][:5]
        deviant_features = [
            (feature_names[i], float(z_scores[i]), float(test_vector[i]), float(ref_mean[i]))
            for i in top_indices
        ]
        
        # Compute SSIM if archetype available
        ssim_score = 0.0
        if 'archetype_image' in self.reference_model and self.reference_model['archetype_image'] is not None:
            archetype = self.reference_model['archetype_image']
            if len(image.shape) == 3:
                test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                test_gray = image
            
            # Resize if needed
            if test_gray.shape != archetype.shape:
                test_gray = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
            
            # Simple SSIM calculation
            c1, c2 = 0.01**2, 0.03**2
            mu1, mu2 = test_gray.mean(), archetype.mean()
            sigma1, sigma2 = test_gray.std(), archetype.std()
            sigma12 = ((test_gray - mu1) * (archetype - mu2)).mean()
            
            ssim_score = float(
                ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) /
                ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
            )
        
        # Determine verdict
        is_anomalous = mahalanobis_dist > 3.0 or ssim_score < 0.7
        
        return {
            "mahalanobis_distance": mahalanobis_dist,
            "ssim": ssim_score,
            "anomaly_verdict": "ANOMALOUS" if is_anomalous else "NORMAL",
            "deviant_features": deviant_features,
            "confidence": min(1.0, mahalanobis_dist / 5.0)
        }

# =============================================================================
# PART 3: TOPOLOGICAL DATA ANALYSIS ENGINE (Optimized)
# =============================================================================

class TDAAnalyzer:
    """Optimized Topological Data Analysis implementation."""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.enabled = TDA_AVAILABLE and config.use_topological_analysis
        
    def analyze_region(self, image_region: np.ndarray) -> Dict[str, Any]:
        """Analyze topological features of image region."""
        if not self.enabled:
            return {
                'mean_connectivity': 1.0,
                'min_connectivity': 1.0,
                'enabled': False
            }
        
        try:
            # Ensure uint8
            if image_region.dtype != np.uint8:
                image_region = cv2.normalize(image_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Simplified analysis - just compute connectivity
            connectivity_scores = []
            
            for threshold in range(50, 200, 50):
                _, binary = cv2.threshold(image_region, threshold, 255, cv2.THRESH_BINARY)
                
                # Count connected components
                num_labels = cv2.connectedComponents(binary, connectivity=8)[0]
                
                # Simple connectivity score (inverse of component count)
                score = 1.0 / max(1, num_labels - 1)
                connectivity_scores.append(score)
            
            return {
                'mean_connectivity': float(np.mean(connectivity_scores)),
                'min_connectivity': float(np.min(connectivity_scores)),
                'enabled': True
            }
            
        except Exception as e:
            logging.warning(f"TDA analysis failed: {str(e)}")
            return {
                'mean_connectivity': 1.0,
                'min_connectivity': 1.0,
                'enabled': False
            }

# =============================================================================
# PART 4: MAIN ANALYZER CLASS (Optimized)
# =============================================================================

class OmniFiberAnalyzer:
    """Optimized master class for fiber optic defect detection."""
    
    def __init__(self, config: Optional[OmniConfig] = None):
        self.config = config or OmniConfig()
        self.config.validate()
        
        self.logger = logging.getLogger(__name__)
        self.warnings = []
        self.intermediate_results = {}
        
        # Initialize sub-analyzers
        self.kb_analyzer = None
        if self.config.knowledge_base_path:
            self.kb_analyzer = KnowledgeBaseAnalyzer(self.config.knowledge_base_path)
        
        self.tda_analyzer = TDAAnalyzer(self.config) if self.config.use_topological_analysis else None
        
        # Detector registry
        self._detector_registry = {
            'do2mr': self._detect_do2mr,
            'lei': self._detect_lei,
            'zana_klein': self._detect_zana_klein,
            'log': self._detect_log,
            'doh': self._detect_doh,
            'hessian_eigen': self._detect_hessian_eigen,
            'frangi': self._detect_frangi,
            'structure_tensor': self._detect_structure_tensor,
            'mser': self._detect_mser,
            'watershed': self._detect_watershed,
            'gradient_mag': self._detect_gradient_magnitude,
            'phase_congruency': self._detect_phase_congruency,
            'radon': self._detect_radon,
            'lbp_anomaly': self._detect_lbp_anomaly,
            'canny': self._detect_canny,
            'adaptive_threshold': self._detect_adaptive_threshold,
            'otsu_variants': self._detect_otsu_variants,
            'morphological': self._detect_morphological
        }
        
        self.logger.info("OmniFiberAnalyzer initialized successfully")
    
    def analyze_end_face(self, image_path: str, output_dir: Optional[str] = None) -> AnalysisReport:
        """Main analysis pipeline for fiber optic end-face images."""
        self.logger.info(f"Starting analysis of: {image_path}")
        start_time = time.time()
        self.warnings.clear()
        self.intermediate_results.clear()
        
        try:
            # Stage 1: Load and validate image
            original_image, gray_image = self._load_and_prepare_image(image_path)
            if original_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Stage 2: Global anomaly analysis (if enabled)
            global_results = None
            if self.config.use_global_anomaly_analysis and self.kb_analyzer:
                try:
                    global_results = self.kb_analyzer.detect_anomalies(gray_image)
                except Exception as e:
                    self.logger.warning(f"Global analysis failed: {str(e)}")
            
            # Stage 3: Preprocessing
            preprocessed_maps = self._run_preprocessing(gray_image)
            
            # Stage 4: Region detection
            fiber_info, zone_masks = self._detect_fiber_regions(preprocessed_maps)
            
            # Stage 5: Defect detection
            raw_detections = self._run_detectors(preprocessed_maps, zone_masks)
            
            # Stage 6: Ensemble combination
            ensemble_masks = self._ensemble_detections(raw_detections, gray_image.shape)
            
            # Stage 7: Defect characterization
            analyzed_defects = self._characterize_defects(ensemble_masks, preprocessed_maps, gray_image)
            
            # Stage 8: Generate report
            duration = time.time() - start_time
            final_report = self._generate_report(
                image_path, original_image, global_results,
                analyzed_defects, fiber_info, zone_masks, duration
            )
            
            # Stage 9: Save outputs (if requested)
            if output_dir:
                self._save_outputs(final_report, output_dir, os.path.basename(image_path))
            
            # Clean up intermediate results to save memory
            if self.config.cleanup_intermediate:
                self.intermediate_results.clear()
                gc.collect()
            
            self.logger.info(f"Analysis completed in {duration:.2f} seconds")
            return final_report
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return error report
            return self._create_error_report(image_path, str(e), time.time() - start_time)
    
    def _load_and_prepare_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and validate image from file path."""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None, None
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not read image: {image_path}")
                return None, None
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Validate image size
            if gray.shape[0] < 100 or gray.shape[1] < 100:
                self.logger.warning(f"Image is very small: {gray.shape}")
            
            return image, gray
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            return None, None
    
    def _run_preprocessing(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply preprocessing techniques."""
        self.logger.info("Running preprocessing...")
        
        preprocessed = {
            'original': gray_image.copy()
        }
        
        try:
            # Denoising
            if self.config.denoise_strength > 0:
                preprocessed['denoised'] = cv2.fastNlMeansDenoising(
                    gray_image, None, self.config.denoise_strength, 7, 21
                )
            
            # CLAHE
            if self.config.clahe_params:
                for i, (clip, grid) in enumerate(self.config.clahe_params[:1]):  # Just use first
                    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
                    preprocessed[f'clahe_{i}'] = clahe.apply(gray_image)
            
            # Bilateral filter
            if self.config.bilateral_params:
                d, sc, ss = self.config.bilateral_params[0]  # Just use first
                preprocessed['bilateral_0'] = cv2.bilateralFilter(gray_image, d, sc, ss)
            
            # Gaussian blur
            if self.config.gaussian_blur_sizes:
                size = self.config.gaussian_blur_sizes[0]  # Just use first
                preprocessed['gaussian_5'] = cv2.GaussianBlur(gray_image, size, 0)
            
            # Illumination correction
            if self.config.use_illumination_correction:
                kernel_size = max(gray_image.shape) // 8
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                background = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
                corrected = cv2.subtract(gray_image, background)
                mean_bg = np.mean(background).astype(np.uint8)
                preprocessed['illumination_corrected'] = cv2.add(corrected, mean_bg)
            
        except Exception as e:
            self.logger.warning(f"Preprocessing error: {str(e)}")
        
        # Store intermediate results if requested
        if self.config.save_intermediate_masks and len(self.intermediate_results) < self.config.max_intermediate_results:
            self.intermediate_results['preprocessed'] = preprocessed
        
        return preprocessed
    
    def _detect_fiber_regions(self, preprocessed_maps: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Detect fiber regions and create zone masks."""
        self.logger.info("Detecting fiber regions...")
        
        # Use the best preprocessed image
        img = preprocessed_maps.get('clahe_0', preprocessed_maps.get('illumination_corrected', preprocessed_maps['original']))
        
        fiber_info = None
        
        # Try multiple detection methods
        detection_methods = [
            self._detect_fiber_hough,
            self._detect_fiber_adaptive,
            self._detect_fiber_contour
        ]
        
        for method in detection_methods:
            try:
                result = method(img)
                if result and result.get('confidence', 0) > 0.3:
                    fiber_info = result
                    break
            except Exception as e:
                self.logger.debug(f"Detection method failed: {str(e)}")
        
        # Fallback to center estimation
        if not fiber_info:
            self.logger.warning("Using estimated fiber location")
            h, w = img.shape
            fiber_info = {
                'center': (w // 2, h // 2),
                'cladding_radius': min(h, w) // 3,
                'core_radius': min(h, w) // 42,  # Approximate ratio
                'confidence': 0.1,
                'method': 'estimated'
            }
            self.warnings.append("Fiber location estimated - results may be inaccurate")
        
        # Create zone masks
        zone_masks = self._create_zone_masks(img.shape, fiber_info)
        
        # Calculate pixels per micron
        fiber_info['pixels_per_micron'] = (2 * fiber_info['cladding_radius']) / 125.0  # Assuming 125um cladding
        
        return fiber_info, zone_masks
    
    def _detect_fiber_hough(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect fiber using Hough circles."""
        blurred = cv2.GaussianBlur(img, (9, 9), 2)
        
        for dp in self.config.hough_dp_values:
            for p1 in self.config.hough_param1_values:
                for p2 in self.config.hough_param2_values:
                    circles = cv2.HoughCircles(
                        blurred, cv2.HOUGH_GRADIENT, dp=dp,
                        minDist=img.shape[0]//8,
                        param1=p1, param2=p2,
                        minRadius=img.shape[0]//10,
                        maxRadius=img.shape[0]//2
                    )
                    
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        cx, cy, cr = circles[0, 0]
                        
                        return {
                            'center': (int(cx), int(cy)),
                            'cladding_radius': int(cr),
                            'core_radius': max(1, int(cr / self.config.cladding_core_ratio)),
                            'confidence': 0.8,
                            'method': 'hough'
                        }
        
        return None
    
    def _detect_fiber_adaptive(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect fiber using adaptive thresholding."""
        block_size = self.config.adaptive_threshold_block_size
        if block_size % 2 == 0:
            block_size += 1
        
        adaptive = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, self.config.adaptive_threshold_c
        )
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        # Find largest circular contour
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img.shape[0] * img.shape[1] * 0.01:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > best_circularity:
                    best_circularity = circularity
                    best_contour = contour
        
        if best_contour is not None and best_circularity > 0.5:
            (cx, cy), cr = cv2.minEnclosingCircle(best_contour)
            
            return {
                'center': (int(cx), int(cy)),
                'cladding_radius': int(cr),
                'core_radius': max(1, int(cr / self.config.cladding_core_ratio)),
                'confidence': best_circularity,
                'method': 'adaptive'
            }
        
        return None
    
    def _detect_fiber_contour(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect fiber using contour analysis."""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find largest component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            component_mask = (labels == largest_idx).astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                (cx, cy), cr = cv2.minEnclosingCircle(contours[0])
                
                if cr > img.shape[0] * 0.1:
                    return {
                        'center': (int(cx), int(cy)),
                        'cladding_radius': int(cr),
                        'core_radius': max(1, int(cr / self.config.cladding_core_ratio)),
                        'confidence': 0.6,
                        'method': 'contour'
                    }
        
        return None
    
    def _create_zone_masks(self, shape: Tuple[int, int], fiber_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create masks for different fiber zones."""
        h, w = shape[:2]
        cx, cy = fiber_info['center']
        core_r = fiber_info['core_radius']
        cladding_r = fiber_info['cladding_radius']
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        # Create masks
        masks = {
            'Core': (dist_from_center <= core_r).astype(np.uint8) * 255,
            'Cladding': ((dist_from_center > core_r) & (dist_from_center <= cladding_r)).astype(np.uint8) * 255,
            'Ferrule': (dist_from_center > cladding_r).astype(np.uint8) * 255
        }
        
        # Combined fiber mask
        masks['Fiber'] = cv2.bitwise_or(masks['Core'], masks['Cladding'])
        
        return masks
    
    def _run_detectors(self, preprocessed_maps: Dict[str, np.ndarray], 
                       zone_masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run enabled detection algorithms."""
        self.logger.info("Running defect detection algorithms...")
        all_detections = {}
        
        # Select best preprocessed image for each detector type
        detector_inputs = {
            'scratch': preprocessed_maps.get('bilateral_0', preprocessed_maps['original']),
            'region': preprocessed_maps.get('clahe_0', preprocessed_maps['original']),
            'general': preprocessed_maps.get('denoised', preprocessed_maps['original'])
        }
        
        # Process each zone
        for zone_name, zone_mask in zone_masks.items():
            if zone_name == 'Fiber':  # Skip combined mask
                continue
            
            self.logger.info(f"  Processing {zone_name} zone...")
            zone_detections = {}
            
            # Run only enabled detectors
            for detector_name in self.config.enabled_detectors:
                if detector_name not in self._detector_registry:
                    continue
                
                try:
                    # Select appropriate input
                    if detector_name in ['lei', 'zana_klein', 'hessian_eigen', 'frangi']:
                        input_img = detector_inputs['scratch']
                    elif detector_name in ['do2mr', 'log', 'doh', 'mser']:
                        input_img = detector_inputs['region']
                    else:
                        input_img = detector_inputs['general']
                    
                    # Run detector
                    detection_mask = self._detector_registry[detector_name](input_img, zone_mask)
                    if detection_mask is not None:
                        zone_detections[detector_name] = detection_mask
                    
                except Exception as e:
                    self.logger.debug(f"    {detector_name} failed: {str(e)}")
            
            all_detections[zone_name] = zone_detections
        
        # Store intermediate results if requested
        if self.config.save_intermediate_masks and len(self.intermediate_results) < self.config.max_intermediate_results:
            self.intermediate_results['raw_detections'] = all_detections
        
        return all_detections
    
    # =============================================================================
    # DETECTION ALGORITHMS (Optimized implementations)
    # =============================================================================
    
    def _detect_do2mr(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Difference of Min-Max Ranking detection."""
        try:
            # Use smaller set of kernel sizes for efficiency
            kernel_sizes = [5, 11, 21]
            vote_map = np.zeros_like(image, dtype=np.float32)
            
            for k_size in kernel_sizes:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                img_max = cv2.dilate(image, kernel)
                img_min = cv2.erode(image, kernel)
                residual = cv2.subtract(img_max, img_min)
                
                # Apply zone mask
                masked_values = residual[zone_mask > 0]
                if len(masked_values) == 0:
                    continue
                
                mean_val = np.mean(masked_values)
                std_val = np.std(masked_values)
                
                # Use fixed gamma value
                threshold = mean_val + 2.5 * std_val
                
                binary = (residual > threshold).astype(np.uint8) * 255
                binary = cv2.bitwise_and(binary, binary, mask=zone_mask)
                
                vote_map += binary / 255.0
            
            # Threshold votes
            threshold = len(kernel_sizes) * 0.4
            result = (vote_map >= threshold).astype(np.uint8) * 255
            
            # Clean up small regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_lei(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Linear Enhancement Inspector for scratch detection."""
        try:
            scratch_strength = np.zeros_like(image, dtype=np.float32)
            
            # Use subset of angles for efficiency
            angles = range(0, 180, 15)
            kernel_lengths = [11, 21]
            
            for length in kernel_lengths:
                for angle in angles:
                    angle_rad = np.deg2rad(angle)
                    
                    # Create line kernel
                    kernel = np.zeros((length, length), dtype=np.float32)
                    center = length // 2
                    
                    # Draw line
                    for i in range(length):
                        x = int(center + (i - center) * np.cos(angle_rad))
                        y = int(center + (i - center) * np.sin(angle_rad))
                        if 0 <= x < length and 0 <= y < length:
                            kernel[y, x] = 1.0
                    
                    if kernel.sum() > 0:
                        kernel /= kernel.sum()
                    
                    # Convolve
                    response = cv2.filter2D(image.astype(np.float32), -1, kernel)
                    scratch_strength = np.maximum(scratch_strength, response)
            
            # Normalize and threshold
            scratch_strength = cv2.normalize(scratch_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply zone mask
            masked_values = scratch_strength[zone_mask > 0]
            if len(masked_values) > 0:
                threshold = np.mean(masked_values) + 2.5 * np.std(masked_values)
                _, result = cv2.threshold(scratch_strength, threshold, 255, cv2.THRESH_BINARY)
                result = cv2.bitwise_and(result, result, mask=zone_mask)
                
                # Morphological refinement
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
                
                return result
            
            return np.zeros_like(image, dtype=np.uint8)
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_zana_klein(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Zana-Klein algorithm for linear features."""
        try:
            l = self.config.zana_opening_length
            reconstructed = np.zeros_like(image)
            
            # Use subset of angles
            for angle in range(0, 180, 30):
                kernel_size = l
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
                center = kernel_size // 2
                
                # Create line structuring element
                angle_rad = np.deg2rad(angle)
                for i in range(l):
                    x = int(center + (i - l//2) * np.cos(angle_rad))
                    y = int(center + (i - l//2) * np.sin(angle_rad))
                    if 0 <= x < kernel_size and 0 <= y < kernel_size:
                        kernel[y, x] = 1
                
                # Morphological opening
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
            _, result = cv2.threshold(laplacian_norm, threshold, 255, cv2.THRESH_BINARY)
            
            # Apply zone mask
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_log(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Laplacian of Gaussian blob detection."""
        try:
            # Use skimage blob_log
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
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_doh(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Determinant of Hessian blob detection."""
        try:
            doh_response = np.zeros_like(image, dtype=np.float32)
            
            for scale in range(self.config.log_min_sigma, self.config.log_max_sigma, 2):
                # Smooth image
                smoothed = gaussian_filter(image.astype(np.float32), scale)
                
                # Compute Hessian
                Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
                Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
                Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
                
                # Determinant
                det = Hxx * Hyy - Hxy**2
                
                # Scale normalize
                det *= scale**4
                
                # Keep maximum
                doh_response = np.maximum(doh_response, np.abs(det))
            
            # Normalize and threshold
            doh_response = cv2.normalize(doh_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, result = cv2.threshold(doh_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply zone mask
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_hessian_eigen(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Hessian eigenvalue analysis."""
        try:
            ridge_response = np.zeros_like(image, dtype=np.float32)
            
            for scale in self.config.hessian_scales[:2]:  # Use fewer scales
                smoothed = gaussian_filter(image.astype(np.float32), scale)
                
                # Compute Hessian
                Hxx = gaussian_filter(smoothed, scale, order=(0, 2))
                Hyy = gaussian_filter(smoothed, scale, order=(2, 0))
                Hxy = gaussian_filter(smoothed, scale, order=(1, 1))
                
                # Eigenvalues
                trace = Hxx + Hyy
                det = Hxx * Hyy - Hxy * Hxy
                discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
                
                lambda1 = 0.5 * (trace + discriminant)
                lambda2 = 0.5 * (trace - discriminant)
                
                # Ridge measure
                with np.errstate(divide='ignore', invalid='ignore'):
                    Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
                    Rb[~np.isfinite(Rb)] = 0
                
                # Ridge strength
                S = np.sqrt(lambda1**2 + lambda2**2)
                
                # Response
                response = np.exp(-Rb**2 / 2) * (1 - np.exp(-S**2 / 100))
                response[lambda2 > 0] = 0
                
                ridge_response = np.maximum(ridge_response, scale**2 * response)
            
            # Threshold
            ridge_response = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, result = cv2.threshold(ridge_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_frangi(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Frangi vesselness filter."""
        try:
            vesselness = np.zeros_like(image, dtype=np.float32)
            
            for scale in self.config.frangi_scales[:2]:  # Use fewer scales
                smoothed = gaussian_filter(image.astype(np.float32), scale)
                
                # Hessian matrix
                Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
                Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
                Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
                
                # Eigenvalues
                tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
                lambda1 = 0.5 * (Hxx + Hyy + tmp)
                lambda2 = 0.5 * (Hxx + Hyy - tmp)
                
                # Frangi measures
                with np.errstate(divide='ignore', invalid='ignore'):
                    Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
                    Rb[~np.isfinite(Rb)] = 0
                
                S = np.sqrt(lambda1**2 + lambda2**2)
                
                # Vesselness
                v = np.exp(-Rb**2 / (2 * self.config.frangi_beta**2)) * (1 - np.exp(-S**2 / (2 * self.config.frangi_gamma**2)))
                v[lambda2 > 0] = 0
                
                vesselness = np.maximum(vesselness, v)
            
            # Normalize and threshold
            vesselness = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, result = cv2.threshold(vesselness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_structure_tensor(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Structure tensor coherence detection."""
        try:
            # Compute gradients
            Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Structure tensor
            Jxx = gaussian_filter(Ix * Ix, 2.0)
            Jxy = gaussian_filter(Ix * Iy, 2.0)
            Jyy = gaussian_filter(Iy * Iy, 2.0)
            
            # Eigenvalues
            trace = Jxx + Jyy
            det = Jxx * Jyy - Jxy * Jxy
            discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
            
            mu1 = 0.5 * (trace + discriminant)
            mu2 = 0.5 * (trace - discriminant)
            
            # Coherence
            with np.errstate(divide='ignore', invalid='ignore'):
                coherence = ((mu1 - mu2) / (mu1 + mu2 + 1e-10))**2
                coherence[~np.isfinite(coherence)] = 0
            
            # Threshold high coherence
            coherence_uint8 = (coherence * 255).astype(np.uint8)
            threshold = np.percentile(coherence_uint8[zone_mask > 0], 90)
            _, result = cv2.threshold(coherence_uint8, threshold, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_mser(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """MSER (Maximally Stable Extremal Regions) detection."""
        try:
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
            
            # Create mask
            mask = np.zeros_like(image)
            for region in regions:
                cv2.fillPoly(mask, [region], 255)
            
            # Apply zone mask
            mask = cv2.bitwise_and(mask, mask, mask=zone_mask)
            
            return mask
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_watershed(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Watershed segmentation."""
        try:
            # Apply zone mask
            masked_img = cv2.bitwise_and(image, image, mask=zone_mask)
            
            # Threshold
            _, binary = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Distance transform
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find sure foreground
            _, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            # Find markers
            _, markers = cv2.connectedComponents(sure_fg)
            
            # Apply watershed
            img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_color, markers)
            
            # Create defect mask
            result = np.zeros_like(image)
            result[markers == -1] = 255
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_gradient_magnitude(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Gradient magnitude detection."""
        try:
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, result = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_phase_congruency(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Simplified phase congruency using edges."""
        try:
            edges = cv2.Canny(image, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
            
            return edges
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_radon(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Radon transform line detection using Hough."""
        try:
            edges = cv2.Canny(image, 50, 150)
            edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
            
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180,
                threshold=50,
                minLineLength=20,
                maxLineGap=10
            )
            
            line_mask = np.zeros_like(image)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            line_mask = cv2.bitwise_and(line_mask, line_mask, mask=zone_mask)
            
            return line_mask
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_lbp_anomaly(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """LBP-based texture anomaly detection."""
        try:
            # Compute local variance
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            
            local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((image - local_mean)**2, -1, kernel)
            
            local_var = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, result = cv2.threshold(local_var, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_canny(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Canny edge detection."""
        try:
            edges = cv2.Canny(image, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            closed = cv2.bitwise_and(closed, closed, mask=zone_mask)
            
            return closed
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_adaptive_threshold(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Adaptive threshold detection."""
        try:
            adaptive = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21, 5
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.bitwise_and(cleaned, cleaned, mask=zone_mask)
            
            return cleaned
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_otsu_variants(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Otsu thresholding variants."""
        try:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            defects = cv2.absdiff(opened, closed)
            defects = cv2.bitwise_and(defects, defects, mask=zone_mask)
            
            return defects
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _detect_morphological(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Morphological defect detection."""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            
            combined = cv2.add(tophat, blackhat)
            _, result = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(result, result, mask=zone_mask)
            
            return result
            
        except Exception:
            return np.zeros_like(image, dtype=np.uint8)
    
    def _ensemble_detections(self, raw_detections: Dict[str, Dict[str, np.ndarray]], 
                            image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Combine detection results using weighted voting."""
        self.logger.info("Performing ensemble combination...")
        
        h, w = image_shape
        combined_masks = {}
        
        # Process each zone
        for zone_name, zone_detections in raw_detections.items():
            if not zone_detections:
                continue
            
            # Create vote map
            vote_map = np.zeros((h, w), dtype=np.float32)
            total_weight = 0
            
            # Accumulate weighted votes
            for method_name, mask in zone_detections.items():
                if mask is None:
                    continue
                
                weight = self.config.ensemble_confidence_weights.get(method_name, 0.5)
                vote_map += (mask > 0).astype(np.float32) * weight
                total_weight += weight
            
            # Normalize and threshold
            if total_weight > 0:
                vote_map /= total_weight
                ensemble_mask = (vote_map >= self.config.ensemble_vote_threshold).astype(np.uint8) * 255
                
                # Morphological cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                ensemble_mask = cv2.morphologyEx(ensemble_mask, cv2.MORPH_OPEN, kernel)
                
                combined_masks[zone_name] = ensemble_mask
        
        return combined_masks
    
    def _characterize_defects(self, ensemble_masks: Dict[str, np.ndarray],
                             preprocessed_maps: Dict[str, np.ndarray],
                             gray_image: np.ndarray) -> List[Defect]:
        """Extract features and characterize each defect."""
        self.logger.info("Characterizing defects...")
        
        all_defects = []
        defect_id = 0
        
        # Get gradient map
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_map = np.sqrt(grad_x**2 + grad_y**2)
        
        # Process each zone
        for zone_name, mask in ensemble_masks.items():
            if mask is None:
                continue
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):
                defect_id += 1
                
                # Get basic properties
                area_px = stats[i, cv2.CC_STAT_AREA]
                if area_px < self.config.min_defect_area_px:
                    continue
                
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Get defect mask
                component_mask = (labels == i).astype(np.uint8)
                
                # Extract features
                try:
                    defect = self._extract_defect_features(
                        defect_id, zone_name, component_mask,
                        (x, y, w, h), (cx, cy), area_px,
                        gray_image, gradient_map, grad_x, grad_y
                    )
                    
                    all_defects.append(defect)
                    
                except Exception as e:
                    self.logger.debug(f"Failed to characterize defect {defect_id}: {str(e)}")
        
        return all_defects
    
    def _extract_defect_features(self, defect_id: int, zone: str, mask: np.ndarray,
                                bbox: Tuple[int, int, int, int], center: Tuple[int, int],
                                area_px: int, gray_image: np.ndarray, gradient_map: np.ndarray,
                                grad_x: np.ndarray, grad_y: np.ndarray) -> Defect:
        """Extract comprehensive features for a single defect."""
        x, y, w, h = bbox
        cx, cy = center
        
        # Extract contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else np.array([])
        
        # Geometric properties
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        # Fit ellipse
        major_axis = minor_axis = orientation = eccentricity = 0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (MA, ma), angle = ellipse
                major_axis = max(MA, ma)
                minor_axis = min(MA, ma)
                orientation = angle
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
            except cv2.error:
                pass
        
        # Intensity features
        component_pixels = gray_image[mask > 0]
        mean_intensity = np.mean(component_pixels) if len(component_pixels) > 0 else 0
        std_intensity = np.std(component_pixels) if len(component_pixels) > 0 else 0
        min_intensity = np.min(component_pixels) if len(component_pixels) > 0 else 0
        max_intensity = np.max(component_pixels) if len(component_pixels) > 0 else 0
        
        # Skewness and kurtosis
        if len(component_pixels) > 3:
            intensity_skewness = float(stats.skew(component_pixels))
            intensity_kurtosis = float(stats.kurtosis(component_pixels))
        else:
            intensity_skewness = intensity_kurtosis = 0.0
        
        # Contrast
        dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        surrounding_mask = dilated - mask
        surrounding_pixels = gray_image[surrounding_mask > 0]
        if len(surrounding_pixels) > 0:
            contrast = abs(mean_intensity - np.mean(surrounding_pixels))
        else:
            contrast = std_intensity
        
        # Gradient features
        component_gradients = gradient_map[mask > 0]
        mean_gradient = np.mean(component_gradients) if len(component_gradients) > 0 else 0
        max_gradient = np.max(component_gradients) if len(component_gradients) > 0 else 0
        std_gradient = np.std(component_gradients) if len(component_gradients) > 0 else 0
        
        # Gradient orientation
        grad_x_comp = grad_x[mask > 0]
        grad_y_comp = grad_y[mask > 0]
        if len(grad_x_comp) > 0:
            gradient_orientation = float(np.mean(np.arctan2(grad_y_comp, grad_x_comp)))
        else:
            gradient_orientation = 0.0
        
        # Simple texture features
        roi = gray_image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        glcm_features = self._extract_simple_glcm(roi, roi_mask)
        lbp_features = self._extract_simple_lbp(roi, roi_mask)
        
        # Advanced features (simplified)
        hessian_ratio = coherence = frangi_response = 0.0
        
        # Determine defect type
        defect_type = self._classify_defect_type(
            eccentricity, circularity, solidity,
            major_axis / minor_axis if minor_axis > 0 else 1,
            area_px, mean_gradient, coherence
        )
        
        # Assess severity
        severity = self._assess_defect_severity(defect_type, area_px, zone, contrast)
        
        # Calculate confidence
        confidence = min(1.0, contrast / 50.0 * min(1.0, area_px / 100.0))
        
        # Convert to microns if possible
        area_um = area_px / (self.config.pixels_per_micron ** 2) if self.config.pixels_per_micron > 0 else None
        
        return Defect(
            defect_id=defect_id,
            defect_type=defect_type,
            zone=zone,
            severity=severity,
            confidence=confidence,
            area_px=area_px,
            area_um=area_um,
            perimeter=perimeter,
            eccentricity=eccentricity,
            solidity=solidity,
            circularity=circularity,
            location_xy=(cx, cy),
            bbox=bbox,
            major_axis_length=major_axis,
            minor_axis_length=minor_axis,
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
            std_dev_gradient=std_gradient,
            gradient_orientation=gradient_orientation,
            glcm_contrast=glcm_features['contrast'],
            glcm_homogeneity=glcm_features['homogeneity'],
            glcm_energy=glcm_features['energy'],
            glcm_correlation=glcm_features['correlation'],
            lbp_mean=lbp_features['mean'],
            lbp_std=lbp_features['std'],
            mean_hessian_eigen_ratio=hessian_ratio,
            mean_coherence=coherence,
            frangi_response=frangi_response,
            tda_local_connectivity_score=None,
            tda_betti_0_persistence=None,
            tda_betti_1_persistence=None,
            contributing_algorithms=['ensemble'],
            detection_strength=confidence
        )
    
    def _extract_simple_glcm(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract simplified GLCM features."""
        try:
            if roi.size < 25:  # Too small
                return {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0, 'correlation': 0.0}
            
            # Apply mask
            masked_roi = roi.copy()
            masked_roi[mask == 0] = 0
            
            # Compute GLCM
            glcm = graycomatrix(masked_roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            return {
                'contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
                'homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0]),
                'energy': float(graycoprops(glcm, 'energy')[0, 0]),
                'correlation': float(graycoprops(glcm, 'correlation')[0, 0])
            }
        except Exception:
            return {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0, 'correlation': 0.0}
    
    def _extract_simple_lbp(self, roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract simplified LBP features."""
        try:
            if roi.size < 100:  # Too small
                return {'mean': 0.0, 'std': 0.0}
            
            # Simple LBP using local variance
            kernel_size = min(5, min(roi.shape) // 2)
            if kernel_size < 3:
                return {'mean': 0.0, 'std': 0.0}
            
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            local_mean = cv2.filter2D(roi.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((roi - local_mean)**2, -1, kernel)
            
            lbp_values = local_var[mask > 0]
            
            if len(lbp_values) > 0:
                return {
                    'mean': float(np.mean(lbp_values)),
                    'std': float(np.std(lbp_values))
                }
            else:
                return {'mean': 0.0, 'std': 0.0}
                
        except Exception:
            return {'mean': 0.0, 'std': 0.0}
    
    def _classify_defect_type(self, eccentricity: float, circularity: float, solidity: float,
                             aspect_ratio: float, area_px: int, mean_gradient: float,
                             coherence: float) -> DefectType:
        """Classify defect type based on features."""
        # Scratch detection
        if eccentricity > 0.95 and aspect_ratio > 4:
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
    
    def _generate_report(self, image_path: str, original_image: np.ndarray,
                        global_results: Optional[Dict[str, Any]],
                        analyzed_defects: List[Defect],
                        fiber_info: Dict[str, Any],
                        zone_masks: Dict[str, np.ndarray],
                        processing_time: float) -> AnalysisReport:
        """Generate comprehensive analysis report."""
        self.logger.info("Generating report...")
        
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
        
        # Check global anomaly
        if global_results and global_results.get('anomaly_verdict') == 'ANOMALOUS':
            failure_reasons.append("Global statistical anomaly detected")
        
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
        
        # Global TDA analysis
        global_tda_analysis = None
        if self.tda_analyzer and self.config.use_topological_analysis:
            try:
                fiber_mask = zone_masks.get('Fiber')
                if fiber_mask is not None:
                    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if len(original_image.shape) == 3 else original_image
                    masked_fiber = cv2.bitwise_and(gray, gray, mask=fiber_mask)
                    
                    tda_results = self.tda_analyzer.analyze_region(masked_fiber)
                    global_tda_analysis = {
                        'mean_connectivity': tda_results.get('mean_connectivity', 1.0),
                        'min_connectivity': tda_results.get('min_connectivity', 1.0)
                    }
                    
                    if global_tda_analysis['mean_connectivity'] < self.config.min_global_connectivity:
                        failure_reasons.append(
                            f"Low topological connectivity ({global_tda_analysis['mean_connectivity']:.3f})"
                        )
            except Exception as e:
                self.logger.warning(f"Global TDA analysis failed: {str(e)}")
        
        return AnalysisReport(
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
    
    def _save_outputs(self, report: AnalysisReport, output_dir: str, base_filename: str) -> None:
        """Save analysis outputs to files."""
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(base_filename)[0]
        
        # Save JSON report
        if self.config.generate_json_report:
            json_path = os.path.join(output_dir, f"{base_name}_report.json")
            self._save_json_report(report, json_path)
        
        # Save text report
        if self.config.generate_text_report:
            text_path = os.path.join(output_dir, f"{base_name}_report.txt")
            self._save_text_report(report, text_path)
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        self._save_visualization(report, viz_path)
    
    def _save_json_report(self, report: AnalysisReport, filepath: str) -> None:
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
    
    def _save_text_report(self, report: AnalysisReport, filepath: str) -> None:
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
                
                if report.recommendations:
                    f.write("\nRECOMMENDATIONS\n")
                    f.write("-"*40 + "\n")
                    for rec in report.recommendations:
                        f.write(f"  - {rec}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
            
            self.logger.info(f"Text report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save text report: {str(e)}")
    
    def _save_visualization(self, report: AnalysisReport, filepath: str) -> None:
        """Save simple visualization of results."""
        try:
            # Load original image
            original = cv2.imread(report.image_path)
            if original is None:
                self.logger.error("Could not load original image for visualization")
                return
            
            # Convert to RGB
            if len(original.shape) == 2:
                display_img = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            else:
                display_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Draw defects
            severity_colors = {
                DefectSeverity.CRITICAL: (255, 0, 0),
                DefectSeverity.HIGH: (255, 128, 0),
                DefectSeverity.MEDIUM: (255, 255, 0),
                DefectSeverity.LOW: (0, 255, 0),
                DefectSeverity.NEGLIGIBLE: (0, 255, 255)
            }
            
            for defect in report.defects:
                color = severity_colors.get(defect.severity, (255, 255, 255))
                x, y, w, h = defect.bbox
                cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_img, str(defect.defect_id), (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original with defects
            ax1.imshow(display_img)
            ax1.set_title(f'Detected Defects ({len(report.defects)})')
            ax1.axis('off')
            
            # Summary text
            ax2.axis('off')
            summary = f"Analysis Summary\n{'='*30}\n"
            summary += f"Status: {report.pass_fail_status}\n"
            summary += f"Quality Score: {report.quality_score:.1f}/100\n"
            summary += f"Total Defects: {len(report.defects)}\n\n"
            
            summary += "Defects by Zone:\n"
            for zone, count in report.defects_by_zone.items():
                summary += f"  {zone}: {count}\n"
            
            if report.recommendations:
                summary += "\nRecommendations:\n"
                for rec in report.recommendations[:3]:
                    summary += f"• {rec}\n"
            
            ax2.text(0.1, 0.9, summary, transform=ax2.transAxes,
                    fontsize=12, va='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.config.visualization_dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {str(e)}")
    
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
# MAIN EXECUTION
# =============================================================================

def get_user_input(prompt: str, default: str = "") -> str:
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no response from user."""
    default_str = "Y" if default else "N"
    response = get_user_input(f"{prompt} (Y/N)", default_str)
    return response.upper().startswith('Y')

def main():
    """Main entry point with interactive configuration."""
    print("="*60)
    print("OMNIFIBER ANALYZER - Interactive Setup")
    print("="*60)
    print()
    
    # Mode selection
    print("What would you like to do?")
    print("1. Analyze a single image")
    print("2. Analyze multiple images in a directory")
    print("3. Exit")
    
    mode = get_user_input("Enter your choice (1-3)", "1")
    
    if mode == "3":
        print("Exiting...")
        return
    
    # Get paths based on mode
    if mode == "1":
        # Single image mode
        image_path = get_user_input("Enter the path to the fiber image").strip()
        if not os.path.exists(image_path):
            print(f"✗ Error: Image file not found: {image_path}")
            return
        image_paths = [image_path]
    else:
        # Directory mode
        image_dir = get_user_input("Enter the directory containing fiber images").strip()
        if not os.path.isdir(image_dir):
            print(f"✗ Error: Directory not found: {image_dir}")
            return
        
        # Find images in directory
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"✗ Error: No image files found in {image_dir}")
            return
        
        print(f"Found {len(image_paths)} image(s) to process")
    
    # Output directory
    default_output = "results"
    output_dir = get_user_input("Enter output directory for results", default_output).strip()
    
    # Configuration file
    use_config = get_yes_no("Do you have a configuration JSON file?", False)
    config_path = None
    if use_config:
        config_path = get_user_input("Enter path to configuration file").strip()
        if not os.path.isfile(config_path):
            print(f"✗ Warning: Configuration file not found: {config_path}")
            config_path = None
    
    # Knowledge base
    use_kb = get_yes_no("Do you have a knowledge base file for anomaly detection?", False)
    kb_path = None
    if use_kb:
        kb_path = get_user_input("Enter path to knowledge base file").strip()
        if not os.path.isfile(kb_path):
            print(f"✗ Warning: Knowledge base file not found: {kb_path}")
            kb_path = None
    
    # Advanced options
    show_advanced = get_yes_no("Would you like to configure advanced options?", False)
    
    # Initialize configuration
    config = OmniConfig()
    
    # Load config file if provided
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            print(f"✓ Loaded configuration from {config_path}")
        except Exception as e:
            print(f"✗ Failed to load configuration: {str(e)}")
    
    # Set knowledge base path
    if kb_path:
        config.knowledge_base_path = kb_path
    
    # Advanced configuration
    if show_advanced:
        print("\nAdvanced Configuration:")
        print("-" * 40)
        
        # Report generation options
        config.generate_json_report = get_yes_no("Generate JSON reports?", config.generate_json_report)
        config.generate_text_report = get_yes_no("Generate text reports?", config.generate_text_report)
        
        # Analysis options
        config.use_global_anomaly_analysis = get_yes_no(
            "Enable global anomaly analysis?", 
            config.use_global_anomaly_analysis and kb_path is not None
        )
        
        if TDA_AVAILABLE:
            config.use_topological_analysis = get_yes_no(
                "Enable topological data analysis?", 
                config.use_topological_analysis
            )
        
        # Detection thresholds
        adjust_thresholds = get_yes_no("Adjust detection thresholds?", False)
        if adjust_thresholds:
            min_area = get_user_input(
                f"Minimum defect area in pixels", 
                str(config.min_defect_area_px)
            )
            try:
                config.min_defect_area_px = int(min_area)
            except ValueError:
                print("Invalid value, using default")
            
            vote_threshold = get_user_input(
                f"Ensemble vote threshold (0-1)", 
                str(config.ensemble_vote_threshold)
            )
            try:
                config.ensemble_vote_threshold = float(vote_threshold)
            except ValueError:
                print("Invalid value, using default")
    
    # Create analyzer
    print("\nInitializing analyzer...")
    analyzer = OmniFiberAnalyzer(config)
    
    # Process images
    print(f"\nProcessing {len(image_paths)} image(s)...")
    print("-" * 40)
    
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Analyzing: {image_path}")
        
        try:
            # Create output subdirectory for each image if processing multiple
            if len(image_paths) > 1:
                image_output_dir = os.path.join(output_dir, Path(image_path).stem)
            else:
                image_output_dir = output_dir
            
            # Run analysis
            report = analyzer.analyze_end_face(str(image_path), image_output_dir)
            
            # Display results
            print(f"  Status: {report.pass_fail_status}")
            print(f"  Quality Score: {report.quality_score:.1f}/100")
            print(f"  Defects Found: {len(report.defects)}")
            
            if report.defects_by_zone:
                print(f"  Defects by Zone:")
                for zone, count in report.defects_by_zone.items():
                    print(f"    - {zone}: {count}")
            
            if report.recommendations:
                print(f"  Recommendations:")
                for rec in report.recommendations[:2]:
                    print(f"    - {rec}")
            
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Analysis failed: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_dir}")
    
    # Ask if user wants to view results
    if successful > 0 and len(image_paths) == 1:
        if get_yes_no("\nWould you like to open the results directory?", True):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_dir)
                elif os.name == 'posix':  # macOS and Linux
                    os.system(f'open {output_dir}' if sys.platform == 'darwin' else f'xdg-open {output_dir}')
            except:
                print(f"Please manually navigate to: {output_dir}")

if __name__ == "__main__":
    import sys
    main()
