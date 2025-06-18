#!/usr/bin/env python3
"""
ENHANCED FIBER OPTIC DEFECT DETECTION SYSTEM
=============================================
Advanced Multi-Method Defect Analysis Engine (Version 5.0)

This system provides comprehensive defect detection for fiber optic end faces
using state-of-the-art computer vision and machine learning techniques.

Features:
- Robust automated mask generation with multiple fallback methods
- Multi-algorithmic defect detection (DO2MR, LEI, Zana-Klein, LoG, and more)
- Advanced anomaly detection using statistical and ML methods
- Comprehensive error handling and recovery
- Enhanced visualization and reporting

Author: Advanced Fiber Optics Analysis Team
Version: 5.0 - Production Ready
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
from matplotlib.patches import Rectangle, Circle, Ellipse
import json
import os
from datetime import datetime
import traceback

# --- Configuration & Setup ---

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DefectDetectionConfig:
    """Enhanced configuration for all detection methods."""
    # Mask Generation
    cladding_core_ratio: float = 125.0 / 9.0
    ferrule_buffer_ratio: float = 1.2
    mask_fallback_mode: bool = True
    adaptive_threshold_block_size: int = 51  # Larger block size for better results
    adaptive_threshold_c: int = 10  # Higher C value for better contrast
    
    # Preprocessing
    use_illumination_correction: bool = True
    gaussian_blur_sigma: float = 1.0
    use_adaptive_histogram_eq: bool = True
    denoise_strength: float = 5.0  # Reduced from 10.0 for faster processing
    
    # Region-Based Defect Detection (DO2MR)
    do2mr_kernel_size: int = 5
    do2mr_threshold_gamma: float = 3.5
    
    # Scratch Detection (LEI)
    lei_kernel_length: int = 21
    lei_angle_step: int = 15
    lei_threshold_gamma: float = 3.0
    
    # Scratch Detection (Zana-Klein)
    zana_opening_length: int = 15
    zana_laplacian_threshold: float = 1.2
    
    # Laplacian of Gaussian (LoG) for Blobs/Digs
    log_min_sigma: int = 2
    log_max_sigma: int = 10
    log_num_sigma: int = 5
    log_threshold: float = 0.05
    
    # Anomaly Detection
    use_isolation_forest: bool = True
    isolation_contamination: float = 0.1
    use_statistical_outliers: bool = True
    outlier_zscore_threshold: float = 3.0
    
    # Texture Analysis
    use_texture_analysis: bool = True
    lbp_radius: int = 3
    lbp_n_points: int = 24
    
    # Validation
    min_defect_area_px: int = 5
    max_defect_area_ratio: float = 0.1  # Max defect area as ratio of fiber area
    max_defect_eccentricity: float = 0.98
    
    # Output
    visualization_dpi: int = 200
    save_intermediate_results: bool = True
    generate_report: bool = True
    report_format: str = "json"  # json, html, pdf

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
                self.logger.error("Failed to generate masks. Attempting fallback methods...")
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
            # Use a larger block size for better results
            block_size = self.config.adaptive_threshold_block_size
            if block_size % 2 == 0:
                block_size += 1  # Ensure odd number
                
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
                if area < image.shape[0] * image.shape[1] * 0.01:  # Skip small contours
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > best_circularity:
                        best_circularity = circularity
                        best_contour = contour
            
            if best_contour is None or best_circularity < 0.3:  # Lower threshold for circularity
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
                "circularity": best_circularity
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
            # Try multiple parameter sets
            param_sets = [
                (50, 30),  # Original
                (100, 50), # More strict
                (30, 20),  # More lenient
            ]
            
            for param1, param2 in param_sets:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=1,
                    minDist=image.shape[0]//2,
                    param1=param1, param2=param2,
                    minRadius=int(image.shape[0] * 0.2),
                    maxRadius=int(image.shape[0] * 0.4)
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
            # Try simple thresholding
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find largest component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            if num_labels > 1:
                # Find largest component (excluding background)
                largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                # Get the mask of the largest component
                component_mask = (labels == largest_idx).astype(np.uint8) * 255
                
                # Find minimum enclosing circle
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
            # Compute distance transform
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            
            # Find the peak (center of fiber)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
            cx, cy = max_loc
            cr = int(max_val * 0.9)  # Approximate radius
            
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
        
        # Core mask
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core_mask, (cx, cy), core_r, 255, -1)
        
        # Cladding mask (excluding core)
        cladding_mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cladding_mask_full, (cx, cy), cladding_r, 255, -1)
        cladding_mask = cv2.subtract(cladding_mask_full, core_mask)
        
        # Ferrule mask (outside cladding)
        ferrule_mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(ferrule_mask, (cx, cy), cladding_r, 0, -1)
        
        # Full fiber mask (core + cladding)
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
        
        # 1. Denoise
        if self.config.denoise_strength > 0:
            processed = cv2.fastNlMeansDenoising(
                processed, None, 
                h=self.config.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        
        # 2. Illumination correction
        if self.config.use_illumination_correction:
            # Rolling ball background subtraction
            kernel_size = max(processed.shape) // 8
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            processed = cv2.subtract(processed, background)
            # Create an array filled with the mean background value
            mean_background = np.full_like(processed, np.mean(background).astype(np.uint8))
            processed = cv2.add(processed, mean_background)
        
        # 3. Adaptive histogram equalization
        if self.config.use_adaptive_histogram_eq:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        # 4. Gaussian blur
        if self.config.gaussian_blur_sigma > 0:
            processed = cv2.GaussianBlur(
                processed, (0, 0), 
                self.config.gaussian_blur_sigma
            )
        
        # Store intermediate result
        if self.config.save_intermediate_results:
            self.intermediate_results['preprocessed'] = processed
        
        return processed
    
    def _run_defect_detection(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> List[Dict]:
        """Run all defect detection algorithms."""
        all_defects = []
        detection_maps = {}
        
        for region_name, region_mask in masks.items():
            if region_name == "Fiber":  # Skip combined mask
                continue
                
            if np.sum(region_mask) < self.config.min_defect_area_px:
                continue
            
            self.logger.info(f"Analyzing '{region_name}' region...")
            
            # Run each detection algorithm
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
            
            # Remove duplicates
            unique_defects = self._remove_duplicate_defects(region_defects)
            all_defects.extend(unique_defects)
        
        # Store detection maps
        if self.config.save_intermediate_results:
            self.intermediate_results['detection_maps'] = detection_maps
        
        return all_defects
    
    def _gradient_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect defects using gradient analysis."""
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and threshold
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Adaptive thresholding on gradient magnitude
        _, defect_map = cv2.threshold(
            grad_mag, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return defect_map
    
    def _morphological_defect_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect defects using morphological operations."""
        # White top-hat for bright defects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Black top-hat for dark defects
        black_tophat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine both
        combined = cv2.add(white_tophat, black_tophat)
        
        # Threshold
        _, defect_map = cv2.threshold(
            combined, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return defect_map
    
    def _detect_anomalies(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> List[Dict]:
        """Detect anomalies using statistical and ML methods."""
        anomalies = []
        
        for region_name, mask in masks.items():
            if region_name == "Fiber":
                continue
                
            # Extract region pixels
            region_pixels = image[mask > 0]
            if len(region_pixels) < 100:
                continue
            
            # Statistical outlier detection
            if self.config.use_statistical_outliers:
                mean = np.mean(region_pixels)
                std = np.std(region_pixels)
                
                # Find outlier pixels
                outlier_mask = np.zeros_like(image, dtype=np.uint8)
                z_scores = np.abs((image - mean) / (std + 1e-7))
                outlier_pixels = (z_scores > self.config.outlier_zscore_threshold) & (mask > 0)
                outlier_mask[outlier_pixels] = 255
                
                # Find connected components
                outlier_defects = self._find_defects_from_map(
                    outlier_mask, mask, f"Statistical_{region_name}"
                )
                anomalies.extend(outlier_defects)
            
            # Isolation Forest for anomaly detection
            if self.config.use_isolation_forest and len(region_pixels) > 1000:
                try:
                    # Prepare features
                    coords = np.column_stack(np.where(mask > 0))
                    intensities = image[mask > 0]
                    
                    # Local statistics as features
                    features = []
                    for idx, (y, x) in enumerate(coords):
                        y_min = max(0, y-3)
                        y_max = min(image.shape[0], y+4)
                        x_min = max(0, x-3)
                        x_max = min(image.shape[1], x+4)
                        window = image[y_min:y_max, x_min:x_max]
                        features.append([
                            intensities[idx],
                            np.std(window) if window.size > 0 else 0,
                            (np.max(window) - np.min(window)) if window.size > 0 else 0
                        ])
                    
                    features = np.array(features)
                    
                    # Train Isolation Forest
                    iso_forest = IsolationForest(
                        contamination=self.config.isolation_contamination,
                        random_state=42,
                        n_estimators=100
                    )
                    predictions = iso_forest.fit_predict(features)
                    
                    # Create anomaly mask
                    anomaly_mask = np.zeros_like(image, dtype=np.uint8)
                    anomaly_coords = coords[predictions == -1]
                    for y, x in anomaly_coords:
                        anomaly_mask[y, x] = 255
                    
                    # Dilate to connect nearby anomalies
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
        
        for i, defect_props in enumerate(raw_defects):
            try:
                # Create basic defect object
                defect = self._characterize_defect(i, defect_props, original_image)
                
                # Extract additional features
                features = self._extract_defect_features(defect, original_image)
                defect.features = features
                
                # Classify type more accurately
                defect.type = self._classify_defect_type(defect, features)
                
                # Assess severity
                defect.severity = self._assess_defect_severity(defect, features)
                
                # Calculate confidence
                defect.confidence = self._calculate_confidence(defect, features)
                
                analyzed_defects.append(defect)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze defect {i}: {str(e)}")
                continue
        
        return analyzed_defects
    
    def _extract_defect_features(self, defect: Defect, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive features for a defect."""
        features = {}
        
        # Get defect region
        x, y, w, h = defect.bbox
        roi = image[y:y+h, x:x+w]
        mask_roi = defect.mask[y:y+h, x:x+w]
        
        # Intensity features
        pixels = roi[mask_roi > 0]
        if len(pixels) > 0:
            features['mean_intensity'] = np.mean(pixels)
            features['std_intensity'] = np.std(pixels)
            features['min_intensity'] = np.min(pixels)
            features['max_intensity'] = np.max(pixels)
            features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        
        # Texture features
        if self.config.use_texture_analysis and roi.size > 0 and roi.shape[0] > 5 and roi.shape[1] > 5:
            try:
                # Local Binary Pattern
                lbp = local_binary_pattern(
                    roi, self.config.lbp_n_points, 
                    self.config.lbp_radius, method='uniform'
                )
                masked_lbp = lbp[mask_roi > 0]
                if len(masked_lbp) > 0:
                    features['lbp_mean'] = np.mean(masked_lbp)
                    features['lbp_std'] = np.std(masked_lbp)
                else:
                    features['lbp_mean'] = 0
                    features['lbp_std'] = 0
            except Exception as e:
                self.logger.debug(f"LBP feature extraction failed: {str(e)}")
                features['lbp_mean'] = 0
                features['lbp_std'] = 0
        
        # Shape features
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            features['compactness'] = 4 * np.pi * defect.area_px / (defect.perimeter ** 2) if defect.perimeter > 0 else 0
            features['aspect_ratio'] = w / h if h > 0 else 1
            
            # Hu moments
            moments = cv2.HuMoments(cv2.moments(contour)).flatten()
            for i, moment in enumerate(moments):
                features[f'hu_moment_{i}'] = -np.sign(moment) * np.log10(abs(moment) + 1e-10)
        
        return features
    
    def _classify_defect_type(self, defect: Defect, features: Dict[str, float]) -> DefectType:
        """Classify defect type based on features."""
        # Rule-based classification with feature thresholds
        
        if defect.eccentricity > 0.9 and features.get('aspect_ratio', 1) > 3:
            return DefectType.SCRATCH
        
        if defect.detection_method == "LoG":
            if defect.area_px < 50:
                return DefectType.PIT
            else:
                return DefectType.DIG
        
        if features.get('compactness', 0) < 0.5:
            return DefectType.CONTAMINATION
        
        if features.get('intensity_range', 0) > 100:
            return DefectType.BURN
        
        if "Statistical" in defect.detection_method:
            return DefectType.ANOMALY
        
        if defect.solidity < 0.8:
            return DefectType.CHIP
        
        return DefectType.UNKNOWN
    
    def _assess_defect_severity(self, defect: Defect, features: Dict[str, float]) -> DefectSeverity:
        """Assess defect severity based on type, size, and location."""
        # Base severity on defect type
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
        
        # Adjust based on size
        if defect.area_px > 1000:
            if base_severity == DefectSeverity.LOW:
                base_severity = DefectSeverity.MEDIUM
            elif base_severity == DefectSeverity.MEDIUM:
                base_severity = DefectSeverity.HIGH
        
        # Adjust based on contrast
        if features.get('intensity_range', 0) > 150:
            if base_severity == DefectSeverity.LOW:
                base_severity = DefectSeverity.MEDIUM
        
        return base_severity
    
    def _calculate_confidence(self, defect: Defect, features: Dict[str, float]) -> float:
        """Calculate detection confidence."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for clear defects
        if features.get('intensity_range', 0) > 50:
            confidence += 0.2
        
        if defect.area_px > self.config.min_defect_area_px * 2:
            confidence += 0.1
        
        if defect.solidity > 0.9:
            confidence += 0.1
        
        # Decrease confidence for anomalies
        if defect.type == DefectType.ANOMALY:
            confidence -= 0.2
        
        return min(max(confidence, 0.1), 1.0)
    
    def _assess_quality(self, defects: List[Defect], masks: Dict[str, np.ndarray], 
                       localization: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall fiber quality."""
        # Calculate fiber area
        fiber_area = 0
        if "Fiber" in masks:
            fiber_area = np.sum(masks["Fiber"]) / 255
        elif "Cladding" in masks and "Core" in masks:
            fiber_area = (np.sum(masks["Cladding"]) + np.sum(masks["Core"])) / 255
        else:
            # Estimate from localization
            if "cladding_radius_px" in localization:
                fiber_area = np.pi * (localization["cladding_radius_px"] ** 2)
            else:
                fiber_area = 1  # Avoid division by zero
        
        metrics = {
            "total_defects": len(defects),
            "critical_defects": sum(1 for d in defects if d.severity == DefectSeverity.CRITICAL),
            "high_severity_defects": sum(1 for d in defects if d.severity == DefectSeverity.HIGH),
            "defect_density": len(defects) / max(fiber_area, 1),
            "affected_area_ratio": sum(d.area_px for d in defects) / max(fiber_area, 1)
        }
        
        # Calculate quality score (0-100)
        quality_score = 100.0
        
        # Deduct points based on defects
        severity_penalties = {
            DefectSeverity.CRITICAL: 25,
            DefectSeverity.HIGH: 15,
            DefectSeverity.MEDIUM: 8,
            DefectSeverity.LOW: 3,
            DefectSeverity.NEGLIGIBLE: 1
        }
        
        for defect in defects:
            quality_score -= severity_penalties.get(defect.severity, 1)
        
        quality_score = max(0, quality_score)
        metrics["quality_score"] = quality_score
        
        # Pass/fail determination
        metrics["pass"] = (
            metrics["critical_defects"] == 0 and
            metrics["high_severity_defects"] <= 2 and
            quality_score >= 70
        )
        
        return metrics
    
    def _generate_report(self, image_info: Dict, localization: Dict, 
                        defects: List[Defect], quality_metrics: Dict, 
                        analysis_time: float) -> AnalysisReport:
        """Generate comprehensive analysis report."""
        recommendations = []
        
        if quality_metrics["critical_defects"] > 0:
            recommendations.append("Critical defects detected. Fiber should be re-terminated.")
        
        if quality_metrics["high_severity_defects"] > 2:
            recommendations.append("Multiple high-severity defects found. Consider cleaning or re-polishing.")
        
        if quality_metrics["defect_density"] > 0.01:
            recommendations.append("High defect density detected. Thorough cleaning recommended.")
        
        contamination_count = sum(1 for d in defects if d.type == DefectType.CONTAMINATION)
        if contamination_count > 3:
            recommendations.append("Multiple contamination spots found. Clean with appropriate fiber cleaning solution.")
        
        report = AnalysisReport(
            timestamp=datetime.now().isoformat(),
            image_info=image_info,
            fiber_metrics={
                "localization": localization,
                "core_diameter_um": localization.get("core_radius_px", 0) * 2 * 0.5,  # Assuming 0.5um/pixel
                "cladding_diameter_um": localization.get("cladding_radius_px", 0) * 2 * 0.5
            },
            defects=defects,
            quality_score=quality_metrics["quality_score"],
            pass_fail=quality_metrics["pass"],
            analysis_time=analysis_time,
            warnings=self.warnings,
            recommendations=recommendations
        )
        
        return report
    
    def _create_error_result(self, error_msg: str, image: np.ndarray, 
                           elapsed_time: float) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            "success": False,
            "error": error_msg,
            "defects": [],
            "masks": {},
            "localization": {},
            "quality_metrics": {},
            "report": None,
            "analysis_time": elapsed_time,
            "original_image": image,
            "warnings": self.warnings
        }
    
    # Keep all the original detection methods (DO2MR, LEI, Zana-Klein, LoG)
    def _do2mr_region_defect_detection(self, image: np.ndarray) -> np.ndarray:
        """Implementation of the Difference of Min-Max Ranking (DO2MR) filter."""
        k = self.config.do2mr_kernel_size
        kernel = np.ones((k, k), np.uint8)
        
        img_min = cv2.erode(image, kernel)
        img_max = cv2.dilate(image, kernel)
        
        residual = cv2.subtract(img_max, img_min)
        
        mean_res = np.mean(residual)
        std_res = np.std(residual)
        threshold = mean_res + self.config.do2mr_threshold_gamma * std_res
        
        _, defect_map = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
        return defect_map
    
    def _lei_scratch_detection(self, image: np.ndarray) -> np.ndarray:
        """Implementation of the Linear Enhancement Inspector (LEI)."""
        l = self.config.lei_kernel_length
        step = self.config.lei_angle_step
        max_response = np.zeros_like(image, dtype=np.float32)
        
        for angle_deg in range(0, 180, step):
            # Create a linear kernel
            kernel = np.zeros((l, l), dtype=np.float32)
            
            # Draw a line in the center
            center = l // 2
            cv2.line(kernel, (0, center), (l-1, center), 1.0, 1)
            
            # Normalize
            kernel = kernel / np.sum(kernel)
            
            # Rotate the kernel
            M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
            rotated_kernel = cv2.warpAffine(kernel, M, (l, l))
            
            # Apply filter
            response = cv2.filter2D(image.astype(np.float32), -1, rotated_kernel)
            np.maximum(max_response, response, out=max_response)
        
        # Threshold
        mean_res = np.mean(max_response)
        std_res = np.std(max_response)
        threshold = mean_res + self.config.lei_threshold_gamma * std_res
        
        _, defect_map = cv2.threshold(max_response, threshold, 255, cv2.THRESH_BINARY)
        return defect_map.astype(np.uint8)
    
    def _zana_klein_scratch_detection(self, image: np.ndarray) -> np.ndarray:
        """Implementation of the Zana & Klein algorithm."""
        l = self.config.zana_opening_length
        
        # Linear openings at different angles
        reconstructed = np.zeros_like(image)
        for angle in range(0, 180, 15):
            # Create linear structuring element
            if angle == 0 or angle == 180:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (l, 1))
            elif angle == 90:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, l))
            else:
                # Create rotated line kernel
                kernel = np.zeros((l, l), dtype=np.uint8)
                angle_rad = np.deg2rad(angle)
                x1 = int(l/2 - l/2 * np.cos(angle_rad))
                y1 = int(l/2 - l/2 * np.sin(angle_rad))
                x2 = int(l/2 + l/2 * np.cos(angle_rad))
                y2 = int(l/2 + l/2 * np.sin(angle_rad))
                cv2.line(kernel, (x1, y1), (x2, y2), 1, 1)
            
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            np.maximum(reconstructed, opened, out=reconstructed)
        
        # Top-hat transform
        tophat = cv2.subtract(image, reconstructed)
        
        # Ensure non-zero for Laplacian
        if tophat.max() == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Laplacian for curvature
        laplacian = cv2.Laplacian(tophat, cv2.CV_64F, ksize=5)
        laplacian[laplacian < 0] = 0
        
        # Normalize and threshold
        laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Adaptive threshold
        threshold = np.mean(laplacian_norm) + self.config.zana_laplacian_threshold * np.std(laplacian_norm)
        _, defect_map = cv2.threshold(laplacian_norm, threshold, 255, cv2.THRESH_BINARY)
        
        return defect_map
    
    def _log_blob_detection(self, image: np.ndarray) -> np.ndarray:
        """Laplacian of Gaussian blob detection."""
        # Use scikit-image blob detection
        blobs = feature.blob_log(
            image,
            min_sigma=self.config.log_min_sigma,
            max_sigma=self.config.log_max_sigma,
            num_sigma=self.config.log_num_sigma,
            threshold=self.config.log_threshold
        )
        
        defect_map = np.zeros_like(image, dtype=np.uint8)
        for blob in blobs:
            y, x, r = blob
            cv2.circle(defect_map, (int(x), int(y)), int(r * np.sqrt(2)), 255, -1)
        
        return defect_map
    
    def _find_defects_from_map(self, defect_map: np.ndarray, region_mask: np.ndarray, 
                              method_name: str) -> List[Dict]:
        """Extract defect properties from binary map."""
        # Ensure binary
        defect_map_binary = (defect_map > 0).astype(np.uint8)
        
        # Apply region mask
        masked_defects = cv2.bitwise_and(defect_map_binary, defect_map_binary, mask=region_mask)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked_defects, 8)
        
        defects = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area
            if area < self.config.min_defect_area_px:
                continue
            
            # Filter by maximum area ratio
            total_region_area = np.sum(region_mask) / 255
            if area / total_region_area > self.config.max_defect_area_ratio:
                self.warnings.append(f"Defect too large ({area}px) - possible false positive")
                continue
            
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            
            defects.append({
                "bbox": (x, y, w, h),
                "centroid": tuple(centroids[i]),
                "area": area,
                "mask": (labels == i).astype(np.uint8),
                "detection_method": method_name
            })
        
        return defects
    
    def _remove_duplicate_defects(self, defects: List[Dict]) -> List[Dict]:
        """Remove duplicate detections using NMS."""
        if not defects:
            return []
        
        # Sort by area (largest first)
        defects = sorted(defects, key=lambda x: x['area'], reverse=True)
        
        keep = []
        for i, defect in enumerate(defects):
            duplicate = False
            
            for kept_defect in keep:
                # Calculate IoU
                x1, y1, w1, h1 = defect['bbox']
                x2, y2, w2, h2 = kept_defect['bbox']
                
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                
                if xi2 > xi1 and yi2 > yi1:
                    intersection = (xi2 - xi1) * (yi2 - yi1)
                    union = w1 * h1 + w2 * h2 - intersection
                    iou = intersection / union
                    
                    if iou > 0.3:
                        duplicate = True
                        break
            
            if not duplicate:
                keep.append(defect)
        
        return keep
    
    def _characterize_defect(self, defect_id: int, props: Dict, 
                           original_image: np.ndarray) -> Defect:
        """Create defect object with basic characterization."""
        mask = props['mask']
        bbox = props['bbox']
        
        # Calculate shape properties
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            contour = np.array([[bbox[0], bbox[1]], 
                              [bbox[0]+bbox[2], bbox[1]+bbox[3]]])
        else:
            contour = contours[0]
        
        area = props['area']
        perimeter = cv2.arcLength(contour, True)
        
        # Eccentricity
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                (x, y), (ma, MA), angle = cv2.fitEllipse(contour)
                if MA > 0 and ma > 0:
                    eccentricity = np.sqrt(1 - (ma/MA)**2)
            except:
                pass
        
        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 1.0
        
        # Intensity statistics
        roi_mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        roi_image = original_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        
        if roi_mask.sum() > 0:
            pixels = roi_image[roi_mask > 0]
            mean_intensity = np.mean(pixels)
            std_intensity = np.std(pixels)
            
            # Contrast relative to surrounding
            dilated_mask = cv2.dilate(roi_mask, np.ones((5,5), np.uint8))
            surrounding_mask = dilated_mask - roi_mask
            if surrounding_mask.sum() > 0:
                surrounding_pixels = roi_image[surrounding_mask > 0]
                contrast = abs(mean_intensity - np.mean(surrounding_pixels))
            else:
                contrast = std_intensity
        else:
            mean_intensity = 0
            std_intensity = 0
            contrast = 0
        
        return Defect(
            id=defect_id,
            type=DefectType.UNKNOWN,  # Will be classified later
            severity=DefectSeverity.LOW,  # Will be assessed later
            confidence=0.5,  # Will be calculated later
            location=(int(props['centroid'][0]), int(props['centroid'][1])),
            bbox=bbox,
            area_px=area,
            perimeter=perimeter,
            eccentricity=eccentricity,
            solidity=solidity,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            contrast=contrast,
            detection_method=props['detection_method'],
            mask=mask
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
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original image with defect overlay
        ax1 = fig.add_subplot(gs[0, 0])
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = image.copy()
        
        # Color code by severity
        severity_colors = {
            DefectSeverity.CRITICAL: (255, 0, 0),      # Red
            DefectSeverity.HIGH: (255, 128, 0),        # Orange
            DefectSeverity.MEDIUM: (255, 255, 0),      # Yellow
            DefectSeverity.LOW: (0, 255, 0),           # Green
            DefectSeverity.NEGLIGIBLE: (0, 255, 255)   # Cyan
        }
        
        for defect in defects:
            color = severity_colors.get(defect.severity, (255, 255, 255))
            # Draw bounding box
            x, y, w, h = defect.bbox
            cv2.rectangle(display_image, (x, y), (x+w, y+h), color, 2)
            # Add label
            label = f"{defect.id}:{defect.type.name[:3]}"
            cv2.putText(display_image, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        ax1.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Defect Detection Results')
        ax1.axis('off')
        
        # 2. Mask visualization
        ax2 = fig.add_subplot(gs[0, 1])
        if masks:
            mask_display = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for i, (name, mask) in enumerate(masks.items()):
                if name != "Fiber":
                    mask_display[mask > 0] = colors[i % len(colors)]
            ax2.imshow(mask_display)
            ax2.set_title('Region Masks (R:Core, G:Cladding, B:Ferrule)')
        else:
            ax2.text(0.5, 0.5, 'No masks available', ha='center', va='center')
        ax2.axis('off')
        
        # 3. Defect type distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if defects:
            defect_types = [d.type.name for d in defects]
            unique_types, counts = np.unique(defect_types, return_counts=True)
            ax3.bar(unique_types, counts)
            ax3.set_xlabel('Defect Type')
            ax3.set_ylabel('Count')
            ax3.set_title('Defect Type Distribution')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'No defects found', ha='center', va='center')
        
        # 4. Severity distribution
        ax4 = fig.add_subplot(gs[1, 0])
        if defects:
            severities = [d.severity.name for d in defects]
            unique_severities, counts = np.unique(severities, return_counts=True)
            colors_list = [severity_colors.get(DefectSeverity[s], (128,128,128)) for s in unique_severities]
            colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors_list]
            ax4.bar(unique_severities, counts, color=colors_normalized)
            ax4.set_xlabel('Severity')
            ax4.set_ylabel('Count')
            ax4.set_title('Defect Severity Distribution')
        else:
            ax4.text(0.5, 0.5, 'No defects found', ha='center', va='center')
        
        # 5. Quality metrics
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        quality_text = "Quality Metrics\n" + "="*30 + "\n"
        quality_text += f"Quality Score: {quality_metrics.get('quality_score', 0):.1f}/100\n"
        quality_text += f"Status: {'PASS' if quality_metrics.get('pass', False) else 'FAIL'}\n"
        quality_text += f"Total Defects: {quality_metrics.get('total_defects', 0)}\n"
        quality_text += f"Critical Defects: {quality_metrics.get('critical_defects', 0)}\n"
        quality_text += f"Defect Density: {quality_metrics.get('defect_density', 0):.4f}\n"
        ax5.text(0.1, 0.9, quality_text, transform=ax5.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # 6. Defect details table
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('tight')
        ax6.axis('off')
        
        if defects:
            # Create table data
            headers = ['ID', 'Type', 'Severity', 'Area(px)', 'Location', 'Confidence']
            table_data = []
            for d in defects[:10]:  # Show first 10
                table_data.append([
                    d.id,
                    d.type.name[:8],
                    d.severity.name[:4],
                    d.area_px,
                    f"({d.location[0]},{d.location[1]})",
                    f"{d.confidence:.2f}"
                ])
            
            table = ax6.table(cellText=table_data, colLabels=headers, 
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
        
        ax6.set_title('Defect Details (First 10)')
        
        # 7. Warnings and recommendations
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        report = analysis_results.get('report')
        if report:
            info_text = "Analysis Summary\n" + "="*50 + "\n"
            info_text += f"Analysis Time: {report.analysis_time:.2f} seconds\n\n"
            
            if report.warnings:
                info_text += "Warnings:\n"
                for warning in report.warnings:
                    info_text += f"  • {warning}\n"
                info_text += "\n"
            
            if report.recommendations:
                info_text += "Recommendations:\n"
                for rec in report.recommendations:
                    info_text += f"  • {rec}\n"
        else:
            info_text = "No report available"
        
        ax7.text(0.05, 0.95, info_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.suptitle('Enhanced Fiber Optic Defect Analysis Report', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logging.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_report(self, report: AnalysisReport, filepath: str):
        """Save analysis report to file."""
        if self.config.report_format == "json":
            report_dict = {
                "timestamp": report.timestamp,
                "image_info": report.image_info,
                "fiber_metrics": report.fiber_metrics,
                "defects": [d.to_dict() for d in report.defects],
                "quality_score": report.quality_score,
                "pass_fail": report.pass_fail,
                "analysis_time": report.analysis_time,
                "warnings": report.warnings,
                "recommendations": report.recommendations
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            self.logger.info(f"Report saved to {filepath}")


# --- Enhanced Test Image Generator ---

def create_realistic_fiber_image(size: int = 800, defect_complexity: str = "medium") -> np.ndarray:
    """Create a more realistic fiber image with various defects."""
    # Create base image with gradient background
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add radial gradient for more realistic appearance
    center = (size // 2, size // 2)
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Background with vignetting
    image = 200 - 0.1 * dist_from_center
    
    # Add cladding with texture
    cladding_radius = int(size * 0.35)
    cladding_mask = dist_from_center <= cladding_radius
    image[cladding_mask] = 160 + np.random.normal(0, 2, np.sum(cladding_mask))
    
    # Add core
    core_radius = max(10, int(cladding_radius / 14))  # More realistic ratio
    core_mask = dist_from_center <= core_radius
    image[core_mask] = 100 + np.random.normal(0, 1, np.sum(core_mask))
    
    # Convert to uint8 for drawing operations
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Add realistic defects based on complexity
    if defect_complexity in ["medium", "high"]:
        # Scratch in cladding
        scratch_angle = np.random.uniform(0, 2*np.pi)
        scratch_length = cladding_radius * 0.6
        x1 = int(center[0] + core_radius * 2 * np.cos(scratch_angle))
        y1 = int(center[1] + core_radius * 2 * np.sin(scratch_angle))
        x2 = int(x1 + scratch_length * np.cos(scratch_angle + np.pi/6))
        y2 = int(y1 + scratch_length * np.sin(scratch_angle + np.pi/6))
        cv2.line(image, (x1, y1), (x2, y2), 130, 2)
        
        # Pits in cladding
        for _ in range(3):
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(core_radius + 10, cladding_radius - 10)
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            cv2.circle(image, (x, y), np.random.randint(3, 7), 
                      int(np.random.uniform(70, 90)), -1)
    
    if defect_complexity == "high":
        # Contamination blob
        blob_x = int(center[0] + cladding_radius * 0.7)
        blob_y = int(center[1] - cladding_radius * 0.3)
        axes = (np.random.randint(15, 30), np.random.randint(10, 20))
        angle = np.random.uniform(0, 180)
        cv2.ellipse(image, (blob_x, blob_y), axes, angle, 0, 360, 140, -1)
        
        # Chip on edge
        chip_angle = np.random.uniform(0, 2*np.pi)
        chip_x = int(center[0] + cladding_radius * 0.9 * np.cos(chip_angle))
        chip_y = int(center[1] + cladding_radius * 0.9 * np.sin(chip_angle))
        chip_pts = np.array([[chip_x, chip_y],
                           [chip_x + 20, chip_y + 10],
                           [chip_x + 15, chip_y - 15]], np.int32)
        cv2.fillPoly(image, [chip_pts], 180)
    
    # Add realistic noise
    noise = np.random.normal(0, 3, image.shape)
    image = image.astype(np.float32) + noise
    
    # Add slight blur for realism
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # Convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


# --- Main Execution ---

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED FIBER OPTIC DEFECT DETECTION SYSTEM - v5.0")
    print("="*70 + "\n")
    
    # Simple test first
    print("Running simple functionality test...")
    print("-"*70)
    
    # Create a simple test image
    simple_image = np.ones((400, 400), dtype=np.uint8) * 200
    center = (200, 200)
    cv2.circle(simple_image, center, 140, 160, -1)  # Cladding
    cv2.circle(simple_image, center, 10, 100, -1)   # Core
    cv2.circle(simple_image, (250, 250), 5, 80, -1) # A defect
    
    # Test with simple config
    simple_config = DefectDetectionConfig(
        use_isolation_forest=False,
        use_statistical_outliers=False,
        use_texture_analysis=False,
        save_intermediate_results=False
    )
    
    detector = EnhancedDefectDetector(simple_config)
    result = detector.analyze_fiber_image(simple_image)
    
    if result['success']:
        print("✓ Simple test PASSED!")
        print(f"  Found {len(result['defects'])} defects")
        print(f"  Mask generation method: {result['localization'].get('method', 'unknown')}")
    else:
        print("✗ Simple test FAILED!")
        print(f"  Error: {result.get('error', 'Unknown')}")
    
    print("\n" + "="*70 + "\n")
    
    # Test with different complexity levels
    complexity_levels = ["low", "medium", "high"]
    
    for complexity in complexity_levels:
        print(f"Testing with {complexity.upper()} complexity fiber image...")
        print("-"*70)
        
        # Create test image
        test_image = create_realistic_fiber_image(size=800, defect_complexity=complexity)
        filename = f"test_fiber_{complexity}_complexity.png"
        cv2.imwrite(filename, test_image)
        print(f"✓ Created test image: {filename}")
        
        # Initialize detector with enhanced configuration
        config = DefectDetectionConfig(
            use_isolation_forest=True if complexity == "high" else False,
            use_statistical_outliers=True,
            use_texture_analysis=True,
            save_intermediate_results=False,  # Disable for cleaner output
            generate_report=True
        )
        
        detector = EnhancedDefectDetector(config)
        
        # Run analysis
        results = detector.analyze_fiber_image(test_image, filename)
        
        # Print summary
        if results['success']:
            print(f"✓ Analysis Status: SUCCESS")
            print(f"  - Analysis Time: {results['analysis_time']:.2f} seconds")
            print(f"  - Mask Method: {results['localization'].get('method', 'unknown')}")
            print(f"  - Total Defects Found: {len(results['defects'])}")
            
            # Defect type summary
            if results['defects']:
                defect_types = {}
                for d in results['defects']:
                    defect_types[d.type.name] = defect_types.get(d.type.name, 0) + 1
                print(f"  - Defect Types: {', '.join([f'{k}:{v}' for k,v in defect_types.items()])}")
            
            # Quality assessment
            quality = results['quality_metrics']
            print(f"  - Quality Score: {quality['quality_score']:.1f}/100 ({'PASS' if quality['pass'] else 'FAIL'})")
            print(f"  - Critical/High Severity: {quality['critical_defects']}/{quality['high_severity_defects']}")
            
            # Save outputs
            viz_filename = f"enhanced_analysis_{complexity}_complexity.png"
            EnhancedDefectDetector.visualize_results(results, viz_filename)
            print(f"✓ Visualization saved to: {viz_filename}")
            
            if results['report']:
                report_filename = f"analysis_report_{complexity}_complexity.json"
                detector.save_report(results['report'], report_filename)
                print(f"✓ Report saved to: {report_filename}")
        else:
            print(f"✗ Analysis Status: FAILED")
            print(f"  Error: {results.get('error', 'Unknown error')}")
        
        print()  # Empty line between tests
    
    print("="*70)
    print("All tests completed! Check the generated files for detailed results.")
    print("="*70)