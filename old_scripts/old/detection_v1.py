#!/usr/bin/env python3
"""
ENHANCED FIBER OPTIC DEFECT DETECTION SYSTEM
=============================================
Advanced Multi-Method Defect Analysis Engine (Version 5.2)

This system provides comprehensive defect detection for fiber optic end faces
using state-of-the-art computer vision and machine learning techniques.

Features:
- Robust automated mask generation with multiple fallback methods
- Multi-algorithmic defect detection (DO2MR, LEI, Zana-Klein, LoG, and more)
- Advanced anomaly detection using statistical and ML methods
- Comprehensive error handling and recovery
- Enhanced visualization and reporting
- Command-line interface for analyzing user-provided images.

Author: Advanced Fiber Optics Analysis Team
Version: 5.2 - Production Ready
"""

import cv2
import numpy as np
from scipy import ndimage, stats, signal
from scipy.signal import find_peaks
from skimage import morphology, feature, filters, measure, transform, segmentation, restoration
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle, Ellipse
import json
import os
from datetime import datetime
import traceback
import argparse # Added for command-line arguments

# --- Configuration & Setup ---

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# FIX: Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@dataclass
class DefectDetectionConfig:
    """Enhanced configuration for all detection methods."""
    # Mask Generation
    cladding_core_ratio: float = 125.0 / 9.0
    ferrule_buffer_ratio: float = 1.2
    mask_fallback_mode: bool = True
    adaptive_threshold_block_size: int = 51
    adaptive_threshold_c: int = 10
    
    # Preprocessing
    use_illumination_correction: bool = True
    gaussian_blur_sigma: float = 1.0
    use_adaptive_histogram_eq: bool = True
    denoise_strength: float = 5.0
    
    # Region-Based Defect Detection (DO2MR)
    do2mr_kernel_size: int = 5
    do2mr_threshold_gamma: float = 4.0 # FIX: Increased from 3.5
    
    # Scratch Detection (LEI)
    lei_kernel_length: int = 21
    lei_angle_step: int = 15
    lei_threshold_gamma: float = 3.5 # FIX: Increased from 3.0
    
    # Scratch Detection (Zana-Klein)
    zana_opening_length: int = 15
    zana_laplacian_threshold: float = 1.5 # FIX: Increased from 1.2
    
    # Laplacian of Gaussian (LoG) for Blobs/Digs
    log_min_sigma: int = 2
    log_max_sigma: int = 10
    log_num_sigma: int = 5
    log_threshold: float = 0.05
    
    # Anomaly Detection
    use_isolation_forest: bool = True
    isolation_contamination: float = 0.1
    use_statistical_outliers: bool = True
    outlier_zscore_threshold: float = 3.5 # FIX: Increased from 3.0
    
    # Texture Analysis
    use_texture_analysis: bool = True
    lbp_radius: int = 3
    lbp_n_points: int = 24
    
    # Validation
    min_defect_area_px: int = 15  # FIX: Increased from 10 to further reduce noise
    max_defect_area_ratio: float = 0.1
    max_defect_eccentricity: float = 0.98
    
    # Output
    visualization_dpi: int = 200
    save_intermediate_results: bool = True
    generate_report: bool = True
    report_format: str = "json"

class DefectType(Enum):
    """Extended enumeration of defect types."""
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
    """Enhanced defect representation with severity and features."""
    id: int
    type: DefectType
    severity: DefectSeverity
    confidence: float
    location: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    area_px: int
    perimeter: float
    eccentricity: float
    solidity: float
    mean_intensity: float
    std_intensity: float
    contrast: float
    detection_method: str
    mask: np.ndarray
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert defect to dictionary for serialization."""
        d = asdict(self)
        d['type'] = self.type.name
        d['severity'] = self.severity.name
        d['mask'] = None  # Don't serialize mask
        return d

@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""
    timestamp: str
    image_info: Dict[str, Any]
    fiber_metrics: Dict[str, Any]
    defects: List[Defect]
    quality_score: float
    pass_fail: bool
    analysis_time: float
    warnings: List[str]
    recommendations: List[str]

# --- Main Enhanced Detector Class ---

class EnhancedDefectDetector:
    """Enhanced fiber optic defect detection system with improved robustness."""
    
    def __init__(self, config: Optional[DefectDetectionConfig] = None):
        self.config = config or DefectDetectionConfig()
        self.logger = logging.getLogger(__name__)
        self.warnings = []
        self.intermediate_results = {}
        
    def analyze_fiber_image(self, image: np.ndarray, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs comprehensive analysis with enhanced error handling and features.
        """
        start_time = time.time()
        self.logger.info("Starting enhanced fiber image analysis.")
        self.warnings.clear()
        self.intermediate_results.clear()
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
            
            # Store image info
            image_info = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "path": image_path,
                "mean_intensity": np.mean(gray_image),
                "std_intensity": np.std(gray_image)
            }
            
            # 1. Enhanced Fiber Region Mask Generation
            self.logger.info("Step 1: Enhanced fiber region mask generation...")
            masks, localization = self._generate_fiber_masks_enhanced(gray_image)
            
            if not masks:
                self.logger.info("Primary mask generation failed. Attempting fallback methods...")
                masks, localization = self._fallback_mask_generation(gray_image)
                
            if not masks:
                error_msg = "Could not localize fiber structure with any method."
                self.logger.error(error_msg)
                return self._create_error_result(error_msg, image, start_time)
            
            # 2. Advanced Preprocessing
            self.logger.info("Step 2: Advanced preprocessing...")
            preprocessed_image = self._preprocess_image_advanced(gray_image)
            
            # 3. Multi-Algorithmic Defect Detection
            self.logger.info("Step 3: Running enhanced defect detection...")
            all_defects = self._run_defect_detection(preprocessed_image, masks)
            
            # 4. Anomaly Detection
            if self.config.use_isolation_forest or self.config.use_statistical_outliers:
                self.logger.info("Step 4: Running anomaly detection...")
                anomalies = self._detect_anomalies(preprocessed_image, masks)
                all_defects.extend(anomalies)
            
            # 5. Defect Analysis and Severity Assessment
            self.logger.info("Step 5: Analyzing defects and assessing severity...")
            analyzed_defects = self._analyze_and_classify_defects(all_defects, gray_image)
            
            # 6. Quality Assessment
            self.logger.info("Step 6: Performing quality assessment...")
            quality_metrics = self._assess_quality(analyzed_defects, masks, localization)
            
            # 7. Generate Report
            analysis_time = time.time() - start_time
            report = self._generate_report(
                image_info, localization, analyzed_defects, 
                quality_metrics, analysis_time
            )
            
            self.logger.info(f"Analysis complete in {analysis_time:.2f} seconds. "
                           f"Found {len(analyzed_defects)} defects.")
            
            return {
                "success": True,
                "defects": analyzed_defects,
                "masks": masks,
                "localization": localization,
                "quality_metrics": quality_metrics,
                "report": report,
                "analysis_time": analysis_time,
                "intermediate_results": self.intermediate_results if self.config.save_intermediate_results else {},
                "original_image": image,
                "preprocessed_image": preprocessed_image,
                "warnings": self.warnings
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed with error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._create_error_result(str(e), image, time.time() - start_time)
    
    def _generate_fiber_masks_enhanced(self, image: np.ndarray) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]]:
        """Enhanced mask generation with multiple methods and better error handling."""
        try:
            # Method 1: Adaptive thresholding for better contrast handling
            block_size = self.config.adaptive_threshold_block_size
            if block_size % 2 == 0:
                block_size += 1
                
            adaptive_thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
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
            
            if not contours:
                return None, None
            
            # Find the most circular contour (likely the fiber)
            best_contour = None
            best_circularity = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image.shape[0] * image.shape[1] * 0.01:
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > best_circularity:
                        best_circularity = circularity
                        best_contour = contour
            
            if best_contour is None or best_circularity < 0.3:
                self.warnings.append("No circular fiber structure found using adaptive threshold")
                return None, None
            
            # Get enclosing circle
            (cx, cy), cr = cv2.minEnclosingCircle(best_contour)
            cx, cy, cr = int(cx), int(cy), int(cr)
            
            # Validate the found circle
            if cr < image.shape[0] * 0.1 or cr > image.shape[0] * 0.45:
                self.warnings.append(f"Found circle radius {cr} is outside expected range")
                return None, None
            
            # Calculate core radius
            core_r = max(1, int(cr / self.config.cladding_core_ratio))
            
            # Create masks
            masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
            localization = {
                "center": (cx, cy),
                "cladding_radius_px": cr,
                "core_radius_px": core_r,
                "circularity": best_circularity,
                "method": "adaptive_threshold"
            }
            
            self.logger.info(f"Successfully localized fiber at ({cx},{cy}) with radius {cr}px")
            return masks, localization
            
        except Exception as e:
            self.logger.error(f"Enhanced mask generation failed: {str(e)}")
            return None, None
    
    def _fallback_mask_generation(self, image: np.ndarray) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]]:
        """Fallback mask generation using multiple alternative methods."""
        self.logger.info("Attempting fallback mask generation methods...")
        
        # Method 2: Hough Circle Detection
        try:
            blurred = cv2.GaussianBlur(image, (9, 9), 2)
            param_sets = [
                (50, 30),
                (100, 50),
                (30, 20),
            ]
            
            for param1, param2 in param_sets:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=1,
                    minDist=image.shape[0]//2,
                    param1=param1, param2=param2,
                    minRadius=int(image.shape[0] * 0.2),
                    maxRadius=int(image.shape[0] * 0.45)
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    cx, cy, cr = circles[0, 0]
                    core_r = max(1, int(cr / self.config.cladding_core_ratio))
                    
                    masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                    localization = {
                        "center": (int(cx), int(cy)),
                        "cladding_radius_px": int(cr),
                        "core_radius_px": core_r,
                        "method": "hough_circles"
                    }
                    
                    self.logger.info("Successfully used Hough circles for mask generation")
                    return masks, localization
                
        except Exception as e:
            self.logger.warning(f"Hough circle detection failed: {str(e)}")
        
        # Method 3: Simple thresholding with largest component
        try:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            if num_labels > 1:
                largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                component_mask = (labels == largest_idx).astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    (cx, cy), cr = cv2.minEnclosingCircle(contours[0])
                    cx, cy, cr = int(cx), int(cy), int(cr)
                    
                    if cr > image.shape[0] * 0.1:
                        core_r = max(1, int(cr / self.config.cladding_core_ratio))
                        masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                        localization = {
                            "center": (cx, cy),
                            "cladding_radius_px": cr,
                            "core_radius_px": core_r,
                            "method": "largest_component"
                        }
                        
                        self.logger.info("Successfully used largest component for mask generation")
                        return masks, localization
                        
        except Exception as e:
            self.logger.warning(f"Largest component detection failed: {str(e)}")
        
        # Method 4: Watershed segmentation
        try:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
            cx, cy = max_loc
            cr = int(max_val * 0.9)
            
            if cr > image.shape[0] * 0.1:
                core_r = max(1, int(cr / self.config.cladding_core_ratio))
                masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
                localization = {
                    "center": (cx, cy),
                    "cladding_radius_px": cr,
                    "core_radius_px": core_r,
                    "method": "watershed"
                }
                
                self.logger.info("Successfully used watershed for mask generation")
                return masks, localization
                
        except Exception as e:
            self.logger.warning(f"Watershed segmentation failed: {str(e)}")
        
        # Method 5: Use entire image with estimated center
        self.logger.warning("Using entire image with estimated center as last resort")
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        cr = min(image.shape) // 3
        core_r = max(1, int(cr / self.config.cladding_core_ratio))
        
        masks = self._create_fiber_masks(image.shape, cx, cy, core_r, cr)
        localization = {
            "center": (cx, cy),
            "cladding_radius_px": cr,
            "core_radius_px": core_r,
            "method": "estimated"
        }
        
        self.warnings.append("Using estimated fiber location - results may be inaccurate")
        return masks, localization
    
    def _create_fiber_masks(self, shape: Tuple[int, int], cx: int, cy: int, 
                           core_r: int, cladding_r: int) -> Dict[str, np.ndarray]:
        """Create fiber region masks."""
        h, w = shape[:2]
        
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core_mask, (cx, cy), core_r, 255, -1)
        
        cladding_mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cladding_mask_full, (cx, cy), cladding_r, 255, -1)
        cladding_mask = cv2.subtract(cladding_mask_full, core_mask)
        
        ferrule_mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(ferrule_mask, (cx, cy), cladding_r, 0, -1)
        
        fiber_mask = cv2.add(core_mask, cladding_mask)
        
        return {
            "Core": core_mask,
            "Cladding": cladding_mask,
            "Ferrule": ferrule_mask,
            "Fiber": fiber_mask
        }
    
    def _preprocess_image_advanced(self, image: np.ndarray) -> np.ndarray:
        """Advanced preprocessing with multiple enhancement techniques."""
        processed = image.copy()
        
        if self.config.denoise_strength > 0:
            processed = cv2.fastNlMeansDenoising(
                processed, None, 
                h=self.config.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        
        if self.config.use_illumination_correction:
            kernel_size = max(processed.shape) // 8
            if kernel_size % 2 == 0: kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            processed = cv2.subtract(processed, background)
            mean_background = np.full_like(processed, np.mean(background).astype(np.uint8))
            processed = cv2.add(processed, mean_background)
        
        if self.config.use_adaptive_histogram_eq:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        if self.config.gaussian_blur_sigma > 0:
            processed = cv2.GaussianBlur(
                processed, (0, 0), 
                self.config.gaussian_blur_sigma
            )
        
        if self.config.save_intermediate_results:
            self.intermediate_results['preprocessed'] = processed
        
        return processed
    
    def _run_defect_detection(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> List[Dict]:
        """Run all defect detection algorithms."""
        all_defects = []
        detection_maps = {}
        
        for region_name, region_mask in masks.items():
            if region_name == "Fiber":
                continue
                
            if np.sum(region_mask) < self.config.min_defect_area_px:
                continue
            
            self.logger.info(f"Analyzing '{region_name}' region...")
            
            algorithms = [
                ("DO2MR", self._do2mr_region_defect_detection),
                ("LEI", self._lei_scratch_detection),
                ("Zana-Klein", self._zana_klein_scratch_detection),
                ("LoG", self._log_blob_detection),
                ("Gradient", self._gradient_based_detection),
                ("Morphological", self._morphological_defect_detection)
            ]
            
            region_defects = []
            for algo_name, algo_func in algorithms:
                try:
                    defect_map = algo_func(image)
                    defects = self._find_defects_from_map(defect_map, region_mask, algo_name)
                    region_defects.extend(defects)
                    detection_maps[f"{region_name}_{algo_name}"] = defect_map
                except Exception as e:
                    self.logger.warning(f"{algo_name} failed for {region_name}: {str(e)}")
            
            unique_defects = self._remove_duplicate_defects(region_defects)
            all_defects.extend(unique_defects)
        
        if self.config.save_intermediate_results:
            self.intermediate_results['detection_maps'] = detection_maps
        
        return all_defects
    
    def _gradient_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect defects using gradient analysis."""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, defect_map = cv2.threshold(
            grad_mag, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return defect_map
    
    def _morphological_defect_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect defects using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        black_tophat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        combined = cv2.add(white_tophat, black_tophat)
        
        _, defect_map = cv2.threshold(
            combined, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return defect_map
    
    def _detect_anomalies(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> List[Dict]:
        """Detect anomalies using statistical and ML methods."""
        anomalies = []
        
        for region_name, mask in masks.items():
            if region_name == "Fiber" or region_name == "Ferrule":
                continue
                
            region_pixels = image[mask > 0]
            if len(region_pixels) < 100:
                continue
            
            if self.config.use_statistical_outliers:
                mean = np.mean(region_pixels)
                std = np.std(region_pixels)
                
                outlier_mask = np.zeros_like(image, dtype=np.uint8)
                z_scores = np.abs((image - mean) / (std + 1e-7))
                outlier_pixels = (z_scores > self.config.outlier_zscore_threshold) & (mask > 0)
                outlier_mask[outlier_pixels] = 255
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                outlier_mask = cv2.morphologyEx(outlier_mask, cv2.MORPH_OPEN, kernel)
                
                outlier_defects = self._find_defects_from_map(
                    outlier_mask, mask, f"Statistical_{region_name}"
                )
                anomalies.extend(outlier_defects)
            
            if self.config.use_isolation_forest and len(region_pixels) > 1000:
                try:
                    coords = np.column_stack(np.where(mask > 0))
                    intensities = image[mask > 0]
                    
                    features = []
                    for idx, (y, x) in enumerate(coords):
                        y_min, y_max = max(0, y-3), min(image.shape[0], y+4)
                        x_min, x_max = max(0, x-3), min(image.shape[1], x+4)
                        window = image[y_min:y_max, x_min:x_max]
                        features.append([
                            intensities[idx],
                            np.std(window) if window.size > 0 else 0,
                            (np.max(window) - np.min(window)) if window.size > 0 else 0
                        ])
                    
                    features = np.array(features)
                    
                    iso_forest = IsolationForest(
                        contamination=self.config.isolation_contamination,
                        random_state=42,
                        n_estimators=100
                    )
                    predictions = iso_forest.fit_predict(features)
                    
                    anomaly_mask = np.zeros_like(image, dtype=np.uint8)
                    anomaly_coords = coords[predictions == -1]
                    anomaly_mask[anomaly_coords[:, 0], anomaly_coords[:, 1]] = 255
                    
                    kernel = np.ones((3, 3), np.uint8)
                    anomaly_mask = cv2.dilate(anomaly_mask, kernel, iterations=1)
                    
                    iso_defects = self._find_defects_from_map(
                        anomaly_mask, mask, f"IsolationForest_{region_name}"
                    )
                    anomalies.extend(iso_defects)
                    
                except Exception as e:
                    self.logger.warning(f"Isolation Forest failed for {region_name}: {str(e)}")
        
        return anomalies
    
    def _analyze_and_classify_defects(self, raw_defects: List[Dict], 
                                     original_image: np.ndarray) -> List[Defect]:
        """Analyze defects and assign severity levels."""
        analyzed_defects = []
        
        unique_defects = self._remove_duplicate_defects(raw_defects)

        for i, defect_props in enumerate(unique_defects):
            try:
                defect = self._characterize_defect(i, defect_props, original_image)
                features = self._extract_defect_features(defect, original_image)
                defect.features = features
                defect.type = self._classify_defect_type(defect, features)
                defect.severity = self._assess_defect_severity(defect, features)
                defect.confidence = self._calculate_confidence(defect, features)
                analyzed_defects.append(defect)
            except Exception as e:
                self.logger.warning(f"Failed to analyze defect {i}: {str(e)}")
                continue
        
        return analyzed_defects
    
    def _extract_defect_features(self, defect: Defect, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive features for a defect."""
        features = {}
        
        x, y, w, h = defect.bbox
        roi = image[y:y+h, x:x+w]
        mask_roi = defect.mask[y:y+h, x:x+w]
        
        pixels = roi[mask_roi > 0]
        if len(pixels) > 0:
            features['mean_intensity'] = np.mean(pixels)
            features['std_intensity'] = np.std(pixels)
            features['min_intensity'] = np.min(pixels)
            features['max_intensity'] = np.max(pixels)
            features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        
        if self.config.use_texture_analysis and roi.size > 0 and roi.shape[0] > 5 and roi.shape[1] > 5:
            try:
                lbp = local_binary_pattern(
                    roi, self.config.lbp_n_points, 
                    self.config.lbp_radius, method='uniform'
                )
                masked_lbp = lbp[mask_roi > 0]
                features['lbp_mean'] = np.mean(masked_lbp) if len(masked_lbp) > 0 else 0
                features['lbp_std'] = np.std(masked_lbp) if len(masked_lbp) > 0 else 0
            except Exception as e:
                self.logger.debug(f"LBP feature extraction failed: {str(e)}")
                features['lbp_mean'], features['lbp_std'] = 0, 0
        
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            features['compactness'] = 4 * np.pi * defect.area_px / (defect.perimeter ** 2) if defect.perimeter > 0 else 0
            features['aspect_ratio'] = w / h if h > 0 else 1
            
            moments = cv2.HuMoments(cv2.moments(contour)).flatten()
            for i, moment in enumerate(moments):
                features[f'hu_moment_{i}'] = -np.sign(moment) * np.log10(abs(moment) + 1e-10)
        
        return features
    
    def _classify_defect_type(self, defect: Defect, features: Dict[str, float]) -> DefectType:
        """Classify defect type based on features."""
        if defect.eccentricity > 0.95 and features.get('aspect_ratio', 1) > 4:
            return DefectType.SCRATCH
        
        if "LoG" in defect.detection_method:
            return DefectType.PIT if defect.area_px < 50 else DefectType.DIG
        
        if features.get('compactness', 0) < 0.5:
            return DefectType.CONTAMINATION
        
        if features.get('intensity_range', 0) > 100:
            return DefectType.BURN
        
        if "Statistical" in defect.detection_method or "Isolation" in defect.detection_method:
            return DefectType.ANOMALY
        
        if defect.solidity < 0.8:
            return DefectType.CHIP
        
        return DefectType.UNKNOWN
    
    def _assess_defect_severity(self, defect: Defect, features: Dict[str, float]) -> DefectSeverity:
        """Assess defect severity based on type, size, and location."""
        type_severity = {
            DefectType.CRACK: DefectSeverity.CRITICAL,
            DefectType.BURN: DefectSeverity.CRITICAL,
            DefectType.CHIP: DefectSeverity.HIGH,
            DefectType.DELAMINATION: DefectSeverity.HIGH,
            DefectType.SCRATCH: DefectSeverity.MEDIUM,
            DefectType.DIG: DefectSeverity.MEDIUM,
            DefectType.CONTAMINATION: DefectSeverity.LOW,
            DefectType.PIT: DefectSeverity.LOW,
            DefectType.ANOMALY: DefectSeverity.MEDIUM,
            DefectType.UNKNOWN: DefectSeverity.LOW
        }
        
        base_severity = type_severity.get(defect.type, DefectSeverity.LOW)
        
        if defect.area_px > 1000:
            if base_severity == DefectSeverity.LOW: base_severity = DefectSeverity.MEDIUM
            elif base_severity == DefectSeverity.MEDIUM: base_severity = DefectSeverity.HIGH
        
        if features.get('intensity_range', 0) > 150 and base_severity == DefectSeverity.LOW:
            base_severity = DefectSeverity.MEDIUM
        
        return base_severity
    
    def _calculate_confidence(self, defect: Defect, features: Dict[str, float]) -> float:
        """Calculate detection confidence."""
        confidence = 0.5
        
        if features.get('intensity_range', 0) > 50: confidence += 0.2
        if defect.area_px > self.config.min_defect_area_px * 2: confidence += 0.1
        if defect.solidity > 0.9: confidence += 0.1
        if defect.type == DefectType.ANOMALY: confidence -= 0.2
        
        return min(max(confidence, 0.1), 1.0)
    
    def _assess_quality(self, defects: List[Defect], masks: Dict[str, np.ndarray], 
                       localization: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall fiber quality."""
        fiber_area = np.sum(masks.get("Fiber", 0)) / 255.0
        if fiber_area == 0:
            radius = localization.get("cladding_radius_px", 0)
            fiber_area = np.pi * (radius ** 2) if radius > 0 else 1

        metrics = {
            "total_defects": len(defects),
            "critical_defects": sum(1 for d in defects if d.severity == DefectSeverity.CRITICAL),
            "high_severity_defects": sum(1 for d in defects if d.severity == DefectSeverity.HIGH),
            "defect_density": len(defects) / fiber_area,
            "affected_area_ratio": sum(d.area_px for d in defects) / fiber_area
        }
        
        severity_penalties = {
            DefectSeverity.CRITICAL: 25, DefectSeverity.HIGH: 15,
            DefectSeverity.MEDIUM: 8, DefectSeverity.LOW: 3, DefectSeverity.NEGLIGIBLE: 1
        }
        
        quality_score = 100.0 - sum(severity_penalties.get(d.severity, 1) for d in defects)
        metrics["quality_score"] = max(0, quality_score)
        
        metrics["pass"] = (metrics["critical_defects"] == 0 and
                           metrics["high_severity_defects"] <= 2 and
                           quality_score >= 70)
        
        return metrics
    
    def _generate_report(self, image_info: Dict, localization: Dict, 
                        defects: List[Defect], quality_metrics: Dict, 
                        analysis_time: float) -> AnalysisReport:
        """Generate comprehensive analysis report."""
        recommendations = []
        if quality_metrics["critical_defects"] > 0:
            recommendations.append("Critical defects detected. Fiber should be re-terminated.")
        if quality_metrics["high_severity_defects"] > 2:
            recommendations.append("Multiple high-severity defects found. Consider re-polishing.")
        if quality_metrics["defect_density"] > 0.001:
            recommendations.append("High defect density detected. Thorough cleaning recommended.")
        if sum(1 for d in defects if d.type == DefectType.CONTAMINATION) > 3:
            recommendations.append("Multiple contamination spots found. Clean with appropriate solution.")
        
        return AnalysisReport(
            timestamp=datetime.now().isoformat(),
            image_info=image_info,
            fiber_metrics={
                "localization": localization,
                "core_diameter_um": localization.get("core_radius_px", 0) * 1.0,
                "cladding_diameter_um": localization.get("cladding_radius_px", 0) * 1.0
            },
            defects=defects,
            quality_score=quality_metrics["quality_score"],
            pass_fail=quality_metrics["pass"],
            analysis_time=analysis_time,
            warnings=self.warnings,
            recommendations=recommendations
        )
    
    def _create_error_result(self, error_msg: str, image: np.ndarray, 
                           elapsed_time: float) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            "success": False, "error": error_msg, "defects": [], "masks": {},
            "localization": {}, "quality_metrics": {}, "report": None,
            "analysis_time": elapsed_time, "original_image": image, "warnings": self.warnings
        }
    
    def _do2mr_region_defect_detection(self, image: np.ndarray) -> np.ndarray:
        """Implementation of the Difference of Min-Max Ranking (DO2MR) filter."""
        k = self.config.do2mr_kernel_size
        kernel = np.ones((k, k), np.uint8)
        img_min, img_max = cv2.erode(image, kernel), cv2.dilate(image, kernel)
        residual = cv2.subtract(img_max, img_min)
        mean_res, std_res = np.mean(residual), np.std(residual)
        threshold = mean_res + self.config.do2mr_threshold_gamma * std_res
        _, defect_map = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
        return defect_map
    
    def _lei_scratch_detection(self, image: np.ndarray) -> np.ndarray:
        """Implementation of the Linear Enhancement Inspector (LEI)."""
        l, step = self.config.lei_kernel_length, self.config.lei_angle_step
        max_response = np.zeros_like(image, dtype=np.float32)
        
        for angle_deg in range(0, 180, step):
            kernel = np.zeros((l, l), dtype=np.float32)
            center = l // 2
            cv2.line(kernel, (0, center), (l-1, center), 1.0, 1)
            if np.sum(kernel) > 0: kernel /= np.sum(kernel)
            
            M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
            rotated_kernel = cv2.warpAffine(kernel, M, (l, l))
            
            response = cv2.filter2D(image.astype(np.float32), -1, rotated_kernel)
            np.maximum(max_response, response, out=max_response)
        
        mean_res, std_res = np.mean(max_response), np.std(max_response)
        threshold = mean_res + self.config.lei_threshold_gamma * std_res
        _, defect_map = cv2.threshold(max_response, threshold, 255, cv2.THRESH_BINARY)
        return defect_map.astype(np.uint8)
    
    def _zana_klein_scratch_detection(self, image: np.ndarray) -> np.ndarray:
        """Implementation of the Zana & Klein algorithm."""
        l = self.config.zana_opening_length
        reconstructed = np.zeros_like(image)
        for angle in range(0, 180, 15):
            angle_rad = np.deg2rad(angle)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (l, 1))
            M = cv2.getRotationMatrix2D((l//2, l//2), -angle, 1)
            rotated_kernel = cv2.warpAffine(np.eye(l, dtype=np.uint8), M, (l,l))
            
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, rotated_kernel)
            np.maximum(reconstructed, opened, out=reconstructed)
        
        tophat = cv2.subtract(image, reconstructed)
        if tophat.max() == 0: return np.zeros_like(image, dtype=np.uint8)
        
        laplacian = cv2.Laplacian(tophat, cv2.CV_64F, ksize=5)
        laplacian[laplacian < 0] = 0
        laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        threshold = np.mean(laplacian_norm) + self.config.zana_laplacian_threshold * np.std(laplacian_norm)
        _, defect_map = cv2.threshold(laplacian_norm, threshold, 255, cv2.THRESH_BINARY)
        return defect_map
    
    def _log_blob_detection(self, image: np.ndarray) -> np.ndarray:
        """Laplacian of Gaussian blob detection."""
        blobs = feature.blob_log(
            image, min_sigma=self.config.log_min_sigma, max_sigma=self.config.log_max_sigma,
            num_sigma=self.config.log_num_sigma, threshold=self.config.log_threshold
        )
        defect_map = np.zeros_like(image, dtype=np.uint8)
        for y, x, r in blobs:
            cv2.circle(defect_map, (int(x), int(y)), int(r * np.sqrt(2)), 255, -1)
        return defect_map
    
    def _find_defects_from_map(self, defect_map: np.ndarray, region_mask: np.ndarray, 
                              method_name: str) -> List[Dict]:
        """Extract defect properties from binary map."""
        defect_map_binary = (defect_map > 0).astype(np.uint8)
        masked_defects = cv2.bitwise_and(defect_map_binary, defect_map_binary, mask=region_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked_defects, 8)
        
        defects = []
        total_region_area = np.sum(region_mask) / 255.0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.config.min_defect_area_px: continue
            
            if total_region_area > 0 and (area / total_region_area) > self.config.max_defect_area_ratio:
                self.warnings.append(f"Defect too large ({area}px) - possible false positive")
                continue
            
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            defects.append({
                "bbox": (x, y, w, h), "centroid": tuple(centroids[i]), "area": area,
                "mask": (labels == i).astype(np.uint8), "detection_method": method_name
            })
        return defects
    
    def _remove_duplicate_defects(self, defects: List[Dict]) -> List[Dict]:
        """Remove duplicate detections using Intersection over Union (IoU)."""
        if not defects: return []
        
        defects = sorted(defects, key=lambda x: x['area'], reverse=True)
        
        keep = []
        while defects:
            current_defect = defects.pop(0)
            keep.append(current_defect)
            
            x1, y1, w1, h1 = current_defect['bbox']
            
            remaining_defects = []
            for other_defect in defects:
                x2, y2, w2, h2 = other_defect['bbox']
                
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                
                intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union = (w1 * h1) + (w2 * h2) - intersection
                iou = intersection / (union + 1e-7)
                
                if iou < 0.3:
                    remaining_defects.append(other_defect)
            
            defects = remaining_defects
        
        return keep
    
    def _characterize_defect(self, defect_id: int, props: Dict, 
                           original_image: np.ndarray) -> Defect:
        """Create defect object with basic characterization."""
        mask, bbox = props['mask'], props['bbox']
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else np.array([])
        
        area, perimeter = props['area'], cv2.arcLength(contour, True)
        
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                (x, y), (ma, MA), angle = cv2.fitEllipse(contour)
                if MA > 0 and ma > 0: eccentricity = np.sqrt(1 - (ma/MA)**2)
            except cv2.error: pass
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 1.0
        
        roi_mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        roi_image = original_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        
        mean_intensity, std_intensity, contrast = 0, 0, 0
        if roi_mask.sum() > 0:
            pixels = roi_image[roi_mask > 0]
            mean_intensity, std_intensity = np.mean(pixels), np.std(pixels)
            
            dilated_mask = cv2.dilate(roi_mask, np.ones((5,5), np.uint8))
            surrounding_mask = dilated_mask - roi_mask
            if surrounding_mask.sum() > 0:
                surrounding_pixels = roi_image[surrounding_mask > 0]
                contrast = abs(mean_intensity - np.mean(surrounding_pixels))
            else:
                contrast = std_intensity
        
        return Defect(
            id=defect_id, type=DefectType.UNKNOWN, severity=DefectSeverity.LOW, confidence=0.5,
            location=(int(props['centroid'][0]), int(props['centroid'][1])),
            bbox=bbox, area_px=area, perimeter=perimeter, eccentricity=eccentricity,
            solidity=solidity, mean_intensity=mean_intensity, std_intensity=std_intensity,
            contrast=contrast, detection_method=props['detection_method'], mask=mask
        )
    
    @staticmethod
    def visualize_results(analysis_results: Dict[str, Any], save_path: Optional[str] = None):
        """Enhanced visualization with multiple views."""
        if not analysis_results or not analysis_results.get('success', False):
            print("No valid results to visualize.")
            return
        
        image = analysis_results['original_image']
        defects = analysis_results['defects']
        masks = analysis_results.get('masks', {})
        quality_metrics = analysis_results.get('quality_metrics', {})
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Original image with defect overlay
        ax1 = fig.add_subplot(gs[0, 0])
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        severity_colors = {
            DefectSeverity.CRITICAL: (255, 0, 0), DefectSeverity.HIGH: (255, 128, 0),
            DefectSeverity.MEDIUM: (255, 255, 0), DefectSeverity.LOW: (0, 255, 0),
            DefectSeverity.NEGLIGIBLE: (0, 255, 255)
        }
        
        for defect in defects:
            color = severity_colors.get(defect.severity, (255, 255, 255))
            x, y, w, h = defect.bbox
            cv2.rectangle(display_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_image, f"{defect.id}:{defect.type.name[:3]}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        ax1.imshow(display_image)
        ax1.set_title('Defect Detection Results')
        ax1.axis('off')
        
        # 2. Mask visualization
        ax2 = fig.add_subplot(gs[0, 1])
        if masks:
            mask_display = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
            colors = {"Core": (255, 0, 0), "Cladding": (0, 255, 0), "Ferrule": (0, 0, 255)}
            for name, mask in masks.items():
                if name in colors: mask_display[mask > 0] = colors[name]
            ax2.imshow(mask_display)
            ax2.set_title('Region Masks (R:Core, G:Cladding, B:Ferrule)')
        else:
            ax2.text(0.5, 0.5, 'No masks available', ha='center', va='center')
        ax2.axis('off')
        
        # 3. Defect type distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if defects:
            types, counts = np.unique([d.type.name for d in defects], return_counts=True)
            ax3.bar(types, counts)
            ax3.set_title('Defect Type Distribution')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'No defects found', ha='center', va='center')
        
        # 4. Severity distribution
        ax4 = fig.add_subplot(gs[1, 0])
        if defects:
            severities, counts = np.unique([d.severity.name for d in defects], return_counts=True)
            colors_list = [tuple(c/255 for c in severity_colors.get(DefectSeverity[s], (128,128,128))) for s in severities]
            ax4.bar(severities, counts, color=colors_list)
            ax4.set_title('Defect Severity Distribution')
        else:
            ax4.text(0.5, 0.5, 'No defects found', ha='center', va='center')
        
        # 5. Quality metrics
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        qm = quality_metrics
        quality_text = (f"Quality Metrics\n"
                        f"{'='*30}\n"
                        f"Quality Score: {qm.get('quality_score', 0):.1f}/100\n"
                        f"Status: {'PASS' if qm.get('pass', False) else 'FAIL'}\n"
                        f"Total Defects: {qm.get('total_defects', 0)}\n"
                        f"Critical Defects: {qm.get('critical_defects', 0)}\n"
                        f"Defect Density: {qm.get('defect_density', 0):.4f}")
        ax5.text(0.0, 0.95, quality_text, fontsize=12, va='top', fontfamily='monospace')
        
        # 6. Defect details table
        # FIX: Corrected subplot creation. This was the source of the TypeError.
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('tight'), ax6.axis('off')
        ax6.set_title('Defect Details (Top 10)')
        if defects:
            headers = ['ID', 'Type', 'Sev', 'Area', 'Conf']
            table_data = [[d.id, d.type.name[:8], d.severity.name[:4], d.area_px, f"{d.confidence:.2f}"]
                          for d in defects[:10]]
            table = ax6.table(cellText=table_data, colLabels=headers, loc='center')
            table.auto_set_font_size(False), table.set_fontsize(9), table.scale(1, 1.5)
        
        # 7. Warnings and recommendations
        # FIX: Corrected subplot creation to span the entire bottom row.
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        report = analysis_results.get('report')
        if report:
            info_text = f"Analysis Summary\n{'='*50}\n"
            info_text += f"Analysis Time: {report.analysis_time:.2f} seconds\n\n"
            if report.warnings:
                info_text += "Warnings:\n" + "".join(f"  • {w}\n" for w in report.warnings) + "\n"
            if report.recommendations:
                info_text += "Recommendations:\n" + "".join(f"  • {r}\n" for r in report.recommendations)
            ax7.text(0.01, 0.95, info_text, va='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", alpha=0.8))

        fig.suptitle('Enhanced Fiber Optic Defect Analysis Report', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logging.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def save_report(self, report: AnalysisReport, filepath: str):
        """Save analysis report to file."""
        if self.config.report_format == "json":
            report_dict = asdict(report)
            report_dict["defects"] = [d.to_dict() for d in report.defects]
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, cls=NumpyEncoder)
            self.logger.info(f"Report saved to {filepath}")


# --- Enhanced Test Image Generator ---

def create_realistic_fiber_image(size: int = 800, defect_complexity: str = "medium") -> np.ndarray:
    """Create a more realistic fiber image with various defects."""
    center = (size // 2, size // 2)
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    image = 200 - 0.1 * dist_from_center
    
    cladding_radius = int(size * 0.35)
    cladding_mask = dist_from_center <= cladding_radius
    image[cladding_mask] = 160 + np.random.normal(0, 2, image.shape)[cladding_mask]
    
    core_radius = max(10, int(cladding_radius / 14))
    core_mask = dist_from_center <= core_radius
    image[core_mask] = 100 + np.random.normal(0, 1, image.shape)[core_mask]
    
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    if defect_complexity != "none":
        if defect_complexity in ["medium", "high"] or (defect_complexity == "low" and np.random.rand() > 0.5):
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(core_radius, cladding_radius * 0.9)
            x, y = int(center[0] + r*np.cos(angle)), int(center[1] + r*np.sin(angle))
            axes = (np.random.randint(15, 25), np.random.randint(10, 20))
            cv2.ellipse(image, (x, y), axes, np.random.uniform(0, 180), 0, 360, 140, -1)

    if defect_complexity in ["medium", "high"]:
        angle = np.random.uniform(0, 2*np.pi)
        l = cladding_radius * 0.6
        x1, y1 = int(center[0] + core_radius*2*np.cos(angle)), int(center[1] + core_radius*2*np.sin(angle))
        x2, y2 = int(x1 + l*np.cos(angle)), int(y1 + l*np.sin(angle))
        cv2.line(image, (x1, y1), (x2, y2), 130, 2)

    if defect_complexity == "high":
        angle = np.random.uniform(0, 2*np.pi)
        x, y = int(center[0] + cladding_radius*0.9*np.cos(angle)), int(center[1] + cladding_radius*0.9*np.sin(angle))
        pts = np.array([[x, y], [x+20, y+10], [x+15, y-15]], np.int32)
        cv2.fillPoly(image, [pts], 180)
    
    image = cv2.GaussianBlur(image.astype(np.float32), (3, 3), 0.5)
    noise = np.random.normal(0, 3, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image


def run_analysis_on_image(image_path: str):
    """Loads an image, runs the full analysis, and saves the results."""
    print(f"\n{'='*70}\nANALYZING IMAGE: {image_path}\n{'='*70}\n")
    if not os.path.exists(image_path):
        logging.error(f"Image path does not exist: {image_path}"); return
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error(f"Failed to load image: {image_path}"); return

    detector = EnhancedDefectDetector()
    results = detector.analyze_fiber_image(image, image_path)

    if results['success']:
        print("✓ Analysis Status: SUCCESS")
        print(f"  - Analysis Time: {results['analysis_time']:.2f} seconds")
        print(f"  - Mask Method: {results['localization'].get('method', 'unknown')}")
        print(f"  - Total Defects Found: {len(results['defects'])}")
        if results['defects']:
            types = {d.type.name: 0 for d in results['defects']}
            for d in results['defects']: types[d.type.name] += 1
            print(f"  - Defect Types: {', '.join([f'{k}:{v}' for k,v in types.items()])}")
        
        quality = results['quality_metrics']
        print(f"  - Quality Score: {quality['quality_score']:.1f}/100 ({'PASS' if quality['pass'] else 'FAIL'})")
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        viz_filename = f"analysis_results_{base_filename}.png"
        EnhancedDefectDetector.visualize_results(results, viz_filename)
        
        if results['report']:
            report_filename = f"analysis_report_{base_filename}.json"
            detector.save_report(results['report'], report_filename)
    else:
        print(f"✗ Analysis Status: FAILED\n  Error: {results.get('error', 'Unknown error')}")
    print(f"\n{'='*70}\nAnalysis complete! Check generated files.\n{'='*70}")


def run_test_suite():
    """Runs the original test suite with generated images."""
    print(f"\n{'='*70}\nENHANCED FIBER OPTIC DEFECT DETECTION SYSTEM - v5.2\n{'='*70}\n")
    
    # Simple test (no defects)
    print("Running simple functionality test (no defects)...")
    print("-"*70)
    simple_image = create_realistic_fiber_image(size=400, defect_complexity="none")
    detector = EnhancedDefectDetector()
    result = detector.analyze_fiber_image(simple_image)
    if result['success']: print(f"✓ Simple test PASSED! Found {len(result['defects'])} defects.")
    else: print(f"✗ Simple test FAILED! Error: {result.get('error', 'Unknown')}")
    print(f"\n{'='*70}\n")
    
    # Test with different complexity levels
    for complexity in ["low", "medium", "high"]:
        print(f"Testing with {complexity.upper()} complexity fiber image...")
        print("-"*70)
        
        test_image = create_realistic_fiber_image(size=800, defect_complexity=complexity)
        filename = f"test_fiber_{complexity}_complexity.png"
        cv2.imwrite(filename, test_image)
        print(f"✓ Created test image: {filename}")
        
        detector = EnhancedDefectDetector()
        results = detector.analyze_fiber_image(test_image, filename)
        
        if results['success']:
            print(f"✓ Analysis Status: SUCCESS")
            print(f"  - Total Defects Found: {len(results['defects'])}")
            quality = results['quality_metrics']
            print(f"  - Quality Score: {quality['quality_score']:.1f}/100 ({'PASS' if quality['pass'] else 'FAIL'})")
            
            viz_filename = f"enhanced_analysis_{complexity}_complexity.png"
            EnhancedDefectDetector.visualize_results(results, viz_filename)
            
            if results['report']:
                report_filename = f"analysis_report_{complexity}_complexity.json"
                detector.save_report(results['report'], report_filename)
        else:
            print(f"✗ Analysis Status: FAILED\n  Error: {results.get('error', 'Unknown error')}")
        print()
    
    print(f"{'='*70}\nAll tests completed! Check generated files.\n{'='*70}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Fiber Optic Defect Detection System.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'image_path', nargs='?', default=None,
        help="Path to the fiber optic image to analyze.\nIf not provided, runs the built-in test suite.")
    args = parser.parse_args()

    if args.image_path:
        run_analysis_on_image(args.image_path)
    else:
        run_test_suite()