#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_core_cladding_detector.py - Comprehensive Core/Cladding Detection System

This script implements a unified approach for detecting fiber optic core and cladding
regions, similar to how detection.py handles defect detection. It combines multiple
detection algorithms internally and uses ensemble methods to produce robust results.

Author: Unified Detection System
Version: 1.0
"""

import cv2
import numpy as np
import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, asdict
from scipy import ndimage, signal
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize, least_squares
from skimage import morphology, feature, measure, filters
from skimage.feature import canny
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DetectorConfig:
    """Configuration for the unified detector"""
    # General settings
    min_core_radius: int = 5
    max_core_radius_ratio: float = 0.3  # Max 30% of image size
    min_cladding_thickness: int = 10
    max_cladding_radius_ratio: float = 0.45  # Max 45% of image size
    
    # Enhancement settings
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Detection methods weights
    method_weights: Dict[str, float] = None
    
    # Ensemble settings
    min_methods_agreement: int = 2
    confidence_threshold: float = 0.3
    
    # Algorithm-specific parameters
    hough_params: Dict[str, Any] = None
    radial_params: Dict[str, Any] = None
    gradient_params: Dict[str, Any] = None
    intensity_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.method_weights is None:
            self.method_weights = {
                'hough_circles': 1.0,
                'radial_profile': 1.2,
                'gradient_based': 1.1,
                'intensity_based': 0.9,
                'contour_based': 0.8,
                'edge_based': 1.0,
                'morphological': 0.7
            }
        
        if self.hough_params is None:
            self.hough_params = {
                'dp': 1.2,
                'min_dist': 50,
                'param1': 50,
                'param2': 30,
                'iterations': 3  # Try multiple parameter sets
            }
        
        if self.radial_params is None:
            self.radial_params = {
                'smoothing_sigma': 2,
                'prominence_factor': 0.1,
                'min_peak_distance': 10
            }
        
        if self.gradient_params is None:
            self.gradient_params = {
                'sobel_ksize': 5,
                'blur_ksize': (5, 5)
            }
        
        if self.intensity_params is None:
            self.intensity_params = {
                'brightness_percentile': 95,
                'core_percentile': 85,
                'cladding_percentile': 50
            }

# =============================================================================
# DETECTION RESULT CLASS
# =============================================================================

@dataclass
class DetectionResult:
    """Result from a single detection method"""
    method_name: str
    center: Optional[Tuple[float, float]] = None
    core_radius: Optional[float] = None
    cladding_radius: Optional[float] = None
    confidence: float = 0.0
    details: Dict[str, Any] = None
    
    def is_valid(self) -> bool:
        """Check if the result is valid"""
        return (self.center is not None and 
                self.core_radius is not None and 
                self.core_radius > 0 and
                self.cladding_radius is not None and
                self.cladding_radius > self.core_radius)

# =============================================================================
# MAIN UNIFIED DETECTOR CLASS
# =============================================================================

class UnifiedCoreCladingDetector:
    """Main detector class that combines multiple methods"""
    
    def __init__(self, config: DetectorConfig = None):
        self.config = config or DetectorConfig()
        self.logger = logging.getLogger(__name__)
        self.image = None
        self.gray = None
        self.enhanced = None
        self.shape = None
        self.results = []
        
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Main detection method"""
        self.logger.info(f"Starting unified detection for: {image_path}")
        
        # Initialize result
        result = {
            'success': False,
            'method': 'unified_core_cladding_detector',
            'image_path': image_path,
            'center': None,
            'core_radius': None,
            'cladding_radius': None,
            'confidence': 0.0,
            'error': None,
            'details': {}
        }
        
        try:
            # Load and prepare image
            if not self._load_and_prepare_image(image_path):
                result['error'] = "Failed to load or prepare image"
                return result
            
            # Run all detection methods
            self._run_all_detectors()
            
            # Combine results using ensemble
            ensemble_result = self._ensemble_combination()
            
            if ensemble_result and ensemble_result.is_valid():
                result['success'] = True
                result['center'] = ensemble_result.center
                result['core_radius'] = ensemble_result.core_radius
                result['cladding_radius'] = ensemble_result.cladding_radius
                result['confidence'] = ensemble_result.confidence
                result['details'] = {
                    'num_methods_agreed': len([r for r in self.results if r.is_valid()]),
                    'ensemble_details': ensemble_result.details,
                    'individual_results': [asdict(r) for r in self.results]
                }
            else:
                result['error'] = "No valid consensus could be reached"
                
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            result['error'] = str(e)
            
        return result
    
    def _load_and_prepare_image(self, image_path: str) -> bool:
        """Load and prepare the image"""
        if not os.path.exists(image_path):
            return False
            
        self.image = cv2.imread(image_path)
        if self.image is None:
            return False
            
        # Convert to grayscale
        if len(self.image.shape) == 3:
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.image.copy()
            
        self.shape = self.gray.shape
        
        # Enhance contrast if configured
        if self.config.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size
            )
            self.enhanced = clahe.apply(self.gray)
        else:
            self.enhanced = self.gray.copy()
            
        return True
    
    def _run_all_detectors(self):
        """Run all detection methods"""
        self.results = []
        
        # Method 1: Hough Circles
        result = self._detect_hough_circles()
        if result:
            self.results.append(result)
            
        # Method 2: Radial Profile Analysis
        result = self._detect_radial_profile()
        if result:
            self.results.append(result)
            
        # Method 3: Gradient-based Detection
        result = self._detect_gradient_based()
        if result:
            self.results.append(result)
            
        # Method 4: Intensity-based Detection
        result = self._detect_intensity_based()
        if result:
            self.results.append(result)
            
        # Method 5: Contour-based Detection
        result = self._detect_contour_based()
        if result:
            self.results.append(result)
            
        # Method 6: Edge-based Detection
        result = self._detect_edge_based()
        if result:
            self.results.append(result)
            
        # Method 7: Morphological Detection
        result = self._detect_morphological()
        if result:
            self.results.append(result)
    
    # ==========================================================================
    # DETECTION METHODS
    # ==========================================================================
    
    def _detect_hough_circles(self) -> Optional[DetectionResult]:
        """Detect using Hough Circle Transform"""
        try:
            h, w = self.shape
            blurred = cv2.GaussianBlur(self.enhanced, (9, 9), 2)
            
            best_result = None
            best_score = 0
            
            # Try multiple parameter combinations
            for iteration in range(self.config.hough_params['iterations']):
                param1 = self.config.hough_params['param1'] + iteration * 10
                param2 = self.config.hough_params['param2'] - iteration * 5
                
                # Detect circles
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=self.config.hough_params['dp'],
                    minDist=self.config.hough_params['min_dist'],
                    param1=param1,
                    param2=max(10, param2),
                    minRadius=self.config.min_core_radius,
                    maxRadius=int(min(h, w) * self.config.max_cladding_radius_ratio)
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    
                    # Find best circle pair (core and cladding)
                    for i in range(circles.shape[1]):
                        cx1, cy1, r1 = circles[0, i]
                        
                        # Look for concentric circle
                        for j in range(circles.shape[1]):
                            if i == j:
                                continue
                                
                            cx2, cy2, r2 = circles[0, j]
                            
                            # Check if circles are roughly concentric
                            center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                            if center_dist < min(r1, r2) * 0.2:  # Centers are close
                                # Determine which is core and which is cladding
                                if r1 < r2:
                                    core_r, clad_r = r1, r2
                                    cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                                else:
                                    core_r, clad_r = r2, r1
                                    cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                                
                                # Score based on expected ratios
                                score = 1.0 / (1.0 + center_dist)
                                if clad_r > core_r + self.config.min_cladding_thickness:
                                    score *= 1.5
                                    
                                if score > best_score:
                                    best_score = score
                                    best_result = DetectionResult(
                                        method_name='hough_circles',
                                        center=(cx, cy),
                                        core_radius=float(core_r),
                                        cladding_radius=float(clad_r),
                                        confidence=min(1.0, score),
                                        details={'iterations': iteration + 1}
                                    )
            
            # If no pair found, try single circle detection
            if best_result is None and circles is not None and circles.shape[1] > 0:
                # Use the most prominent circle as cladding
                cx, cy, r = circles[0, 0]
                best_result = DetectionResult(
                    method_name='hough_circles',
                    center=(float(cx), float(cy)),
                    core_radius=float(r * 0.3),  # Estimate core
                    cladding_radius=float(r),
                    confidence=0.5,
                    details={'single_circle': True}
                )
                
            return best_result
            
        except Exception as e:
            self.logger.warning(f"Hough detection failed: {str(e)}")
            return None
    
    def _detect_radial_profile(self) -> Optional[DetectionResult]:
        """Detect using radial intensity profile analysis"""
        try:
            # Find initial center estimate
            h, w = self.shape
            
            # Use brightness centroid as initial guess
            bright_thresh = np.percentile(self.enhanced, self.config.intensity_params['brightness_percentile'])
            bright_mask = self.enhanced > bright_thresh
            
            M = cv2.moments(bright_mask.astype(np.uint8))
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = w / 2, h / 2
            
            # Create radial profile
            max_radius = int(min(cx, cy, w - cx, h - cy))
            if max_radius < 20:
                return None
                
            # Calculate distances from center
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            
            # Create radial profile
            radial_profile = []
            radial_counts = []
            
            for r in range(max_radius):
                mask = (distances >= r) & (distances < r + 1)
                if np.any(mask):
                    radial_profile.append(np.mean(self.enhanced[mask]))
                    radial_counts.append(np.sum(mask))
                else:
                    radial_profile.append(0)
                    radial_counts.append(0)
                    
            radial_profile = np.array(radial_profile)
            
            # Smooth the profile
            if len(radial_profile) > 10:
                radial_profile = savgol_filter(
                    radial_profile, 
                    window_length=min(11, len(radial_profile) // 2 * 2 - 1),
                    polyorder=3
                )
            
            # Find peaks in gradient (boundaries)
            gradient = np.gradient(radial_profile)
            gradient2 = np.gradient(gradient)
            
            # Find peaks in negative second derivative (inflection points)
            peaks, properties = find_peaks(
                -gradient2,
                prominence=np.std(gradient2) * self.config.radial_params['prominence_factor'],
                distance=self.config.radial_params['min_peak_distance']
            )
            
            if len(peaks) >= 2:
                # Sort by prominence and take top 2
                prominences = properties['prominences']
                sorted_indices = np.argsort(prominences)[::-1]
                top_peaks = sorted(peaks[sorted_indices[:2]])
                
                return DetectionResult(
                    method_name='radial_profile',
                    center=(cx, cy),
                    core_radius=float(top_peaks[0]),
                    cladding_radius=float(top_peaks[1]),
                    confidence=0.8,
                    details={'num_peaks': len(peaks)}
                )
            elif len(peaks) == 1:
                # Single boundary detected
                return DetectionResult(
                    method_name='radial_profile',
                    center=(cx, cy),
                    core_radius=float(peaks[0] * 0.3),
                    cladding_radius=float(peaks[0]),
                    confidence=0.5,
                    details={'num_peaks': 1}
                )
                
        except Exception as e:
            self.logger.warning(f"Radial profile detection failed: {str(e)}")
            
        return None
    
    def _detect_gradient_based(self) -> Optional[DetectionResult]:
        """Detect using gradient magnitude analysis"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(self.enhanced, cv2.CV_64F, 1, 0, ksize=self.config.gradient_params['sobel_ksize'])
            grad_y = cv2.Sobel(self.enhanced, cv2.CV_64F, 0, 1, ksize=self.config.gradient_params['sobel_ksize'])
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Blur gradient magnitude
            grad_mag_blurred = cv2.GaussianBlur(grad_mag, self.config.gradient_params['blur_ksize'], 0)
            
            # Threshold to get edge regions
            thresh = np.percentile(grad_mag_blurred, 90)
            edge_mask = grad_mag_blurred > thresh
            
            # Find contours in edge mask
            contours, _ = cv2.findContours(
                edge_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
                
            # Find circular contours
            circular_boundaries = []
            
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                    
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Check circularity
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                    if circularity > 0.7:
                        circular_boundaries.append((x, y, radius, circularity))
            
            if len(circular_boundaries) >= 2:
                # Sort by radius
                circular_boundaries.sort(key=lambda x: x[2])
                
                # Take innermost and outermost
                inner = circular_boundaries[0]
                outer = circular_boundaries[-1]
                
                # Average centers
                cx = (inner[0] + outer[0]) / 2
                cy = (inner[1] + outer[1]) / 2
                
                return DetectionResult(
                    method_name='gradient_based',
                    center=(cx, cy),
                    core_radius=float(inner[2]),
                    cladding_radius=float(outer[2]),
                    confidence=float((inner[3] + outer[3]) / 2),
                    details={'num_boundaries': len(circular_boundaries)}
                )
                
        except Exception as e:
            self.logger.warning(f"Gradient-based detection failed: {str(e)}")
            
        return None
    
    def _detect_intensity_based(self) -> Optional[DetectionResult]:
        """Detect using intensity thresholding"""
        try:
            h, w = self.shape
            
            # Multi-level thresholding
            core_thresh = np.percentile(self.enhanced, self.config.intensity_params['core_percentile'])
            clad_thresh = np.percentile(self.enhanced, self.config.intensity_params['cladding_percentile'])
            
            # Create masks
            core_mask = self.enhanced > core_thresh
            clad_mask = self.enhanced > clad_thresh
            
            # Clean masks
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            core_mask = cv2.morphologyEx(core_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            clad_mask = cv2.morphologyEx(clad_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Find largest contours
            core_contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clad_contours, _ = cv2.findContours(clad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if core_contours and clad_contours:
                # Get largest contours
                core_contour = max(core_contours, key=cv2.contourArea)
                clad_contour = max(clad_contours, key=cv2.contourArea)
                
                # Fit circles
                (cx1, cy1), r1 = cv2.minEnclosingCircle(core_contour)
                (cx2, cy2), r2 = cv2.minEnclosingCircle(clad_contour)
                
                # Average centers
                cx = (cx1 + cx2) / 2
                cy = (cy1 + cy2) / 2
                
                # Ensure proper ordering
                core_r = min(r1, r2)
                clad_r = max(r1, r2)
                
                if clad_r > core_r + self.config.min_cladding_thickness:
                    return DetectionResult(
                        method_name='intensity_based',
                        center=(cx, cy),
                        core_radius=float(core_r),
                        cladding_radius=float(clad_r),
                        confidence=0.7,
                        details={'thresholds': {'core': core_thresh, 'cladding': clad_thresh}}
                    )
                    
        except Exception as e:
            self.logger.warning(f"Intensity-based detection failed: {str(e)}")
            
        return None
    
    def _detect_contour_based(self) -> Optional[DetectionResult]:
        """Detect using contour analysis"""
        try:
            # Apply Otsu's thresholding
            _, binary = cv2.threshold(self.enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours or hierarchy is None:
                return None
                
            # Find nested circular contours
            circular_contours = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                    
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                    
                    if circularity > 0.6:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        parent_idx = hierarchy[0][i][3]
                        circular_contours.append({
                            'center': (x, y),
                            'radius': radius,
                            'circularity': circularity,
                            'parent': parent_idx,
                            'index': i
                        })
            
            # Find nested pairs
            best_pair = None
            best_score = 0
            
            for outer in circular_contours:
                for inner in circular_contours:
                    if inner['index'] == outer['index']:
                        continue
                        
                    # Check if inner is inside outer
                    center_dist = np.sqrt(
                        (inner['center'][0] - outer['center'][0])**2 + 
                        (inner['center'][1] - outer['center'][1])**2
                    )
                    
                    if (inner['radius'] < outer['radius'] and 
                        center_dist < outer['radius'] * 0.3):
                        
                        score = (inner['circularity'] + outer['circularity']) / 2
                        if score > best_score:
                            best_score = score
                            best_pair = (inner, outer)
            
            if best_pair:
                inner, outer = best_pair
                cx = (inner['center'][0] + outer['center'][0]) / 2
                cy = (inner['center'][1] + outer['center'][1]) / 2
                
                return DetectionResult(
                    method_name='contour_based',
                    center=(cx, cy),
                    core_radius=float(inner['radius']),
                    cladding_radius=float(outer['radius']),
                    confidence=best_score,
                    details={'num_circular_contours': len(circular_contours)}
                )
                
        except Exception as e:
            self.logger.warning(f"Contour-based detection failed: {str(e)}")
            
        return None
    
    def _detect_edge_based(self) -> Optional[DetectionResult]:
        """Detect using edge detection and circle fitting"""
        try:
            # Detect edges using Canny
            edges = cv2.Canny(self.enhanced, 50, 150)
            
            # Get edge points
            edge_points = np.column_stack(np.where(edges.T))
            
            if len(edge_points) < 100:
                return None
                
            # RANSAC-like circle fitting
            best_circles = []
            
            for _ in range(50):  # Random trials
                # Sample 3 points
                if len(edge_points) < 3:
                    continue
                    
                indices = np.random.choice(len(edge_points), 3, replace=False)
                p1, p2, p3 = edge_points[indices]
                
                # Calculate circle from 3 points
                circle = self._circle_from_three_points(p1, p2, p3)
                if circle is None:
                    continue
                    
                cx, cy, r = circle
                
                # Count inliers
                distances = np.sqrt((edge_points[:, 0] - cx)**2 + (edge_points[:, 1] - cy)**2)
                inliers = np.abs(distances - r) < 3  # 3 pixel tolerance
                inlier_count = np.sum(inliers)
                
                if inlier_count > len(edge_points) * 0.1:  # At least 10% inliers
                    best_circles.append((cx, cy, r, inlier_count))
            
            if len(best_circles) >= 2:
                # Sort by inlier count
                best_circles.sort(key=lambda x: x[3], reverse=True)
                
                # Take top 2 and sort by radius
                top_circles = sorted(best_circles[:2], key=lambda x: x[2])
                
                inner = top_circles[0]
                outer = top_circles[1]
                
                cx = (inner[0] + outer[0]) / 2
                cy = (inner[1] + outer[1]) / 2
                
                return DetectionResult(
                    method_name='edge_based',
                    center=(cx, cy),
                    core_radius=float(inner[2]),
                    cladding_radius=float(outer[2]),
                    confidence=0.7,
                    details={'num_circles_found': len(best_circles)}
                )
                
        except Exception as e:
            self.logger.warning(f"Edge-based detection failed: {str(e)}")
            
        return None
    
    def _detect_morphological(self) -> Optional[DetectionResult]:
        """Detect using morphological operations"""
        try:
            # Apply morphological gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            gradient = cv2.morphologyEx(self.enhanced, cv2.MORPH_GRADIENT, kernel)
            
            # Threshold gradient
            _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply closing to connect edges
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Process contours to find circular regions
            circular_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                    
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Check if contour is roughly circular
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    if solidity > 0.8:  # High solidity indicates circular shape
                        circular_regions.append((x, y, radius, solidity))
            
            if len(circular_regions) >= 2:
                # Sort by radius
                circular_regions.sort(key=lambda x: x[2])
                
                # Take smallest and largest
                inner = circular_regions[0]
                outer = circular_regions[-1]
                
                cx = (inner[0] + outer[0]) / 2
                cy = (inner[1] + outer[1]) / 2
                
                confidence = (inner[3] + outer[3]) / 2 * 0.8  # Scale down morphological confidence
                
                return DetectionResult(
                    method_name='morphological',
                    center=(cx, cy),
                    core_radius=float(inner[2]),
                    cladding_radius=float(outer[2]),
                    confidence=confidence,
                    details={'num_regions': len(circular_regions)}
                )
                
        except Exception as e:
            self.logger.warning(f"Morphological detection failed: {str(e)}")
            
        return None
    
    # ==========================================================================
    # ENSEMBLE COMBINATION
    # ==========================================================================
    
    def _ensemble_combination(self) -> Optional[DetectionResult]:
        """Combine results from multiple methods using weighted voting"""
        valid_results = [r for r in self.results if r.is_valid()]
        
        if len(valid_results) < self.config.min_methods_agreement:
            self.logger.warning(f"Only {len(valid_results)} valid results, need at least {self.config.min_methods_agreement}")
            return None
            
        # Weighted averaging of parameters
        total_weight = 0
        weighted_cx = 0
        weighted_cy = 0
        weighted_core_r = 0
        weighted_clad_r = 0
        
        for result in valid_results:
            weight = self.config.method_weights.get(result.method_name, 1.0) * result.confidence
            
            weighted_cx += result.center[0] * weight
            weighted_cy += result.center[1] * weight
            weighted_core_r += result.core_radius * weight
            weighted_clad_r += result.cladding_radius * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
            
        # Calculate weighted averages
        final_cx = weighted_cx / total_weight
        final_cy = weighted_cy / total_weight
        final_core_r = weighted_core_r / total_weight
        final_clad_r = weighted_clad_r / total_weight
        
        # Calculate confidence based on agreement
        # Higher confidence when results are close to each other
        center_variance = np.var([r.center for r in valid_results], axis=0)
        core_variance = np.var([r.core_radius for r in valid_results])
        clad_variance = np.var([r.cladding_radius for r in valid_results])
        
        # Normalize variances and calculate confidence
        h, w = self.shape
        center_agreement = 1.0 / (1.0 + np.mean(center_variance) / (min(h, w) * 0.1))
        radius_agreement = 1.0 / (1.0 + (core_variance + clad_variance) / (final_clad_r * 0.1))
        
        final_confidence = (center_agreement + radius_agreement) / 2
        final_confidence *= len(valid_results) / len(self.config.method_weights)  # Scale by participation
        
        # Create ensemble result
        ensemble_result = DetectionResult(
            method_name='ensemble',
            center=(final_cx, final_cy),
            core_radius=final_core_r,
            cladding_radius=final_clad_r,
            confidence=min(1.0, final_confidence),
            details={
                'num_methods': len(valid_results),
                'contributing_methods': [r.method_name for r in valid_results],
                'center_variance': float(np.mean(center_variance)),
                'radius_variances': {'core': float(core_variance), 'cladding': float(clad_variance)}
            }
        )
        
        return ensemble_result
    
    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================
    
    def _circle_from_three_points(self, p1, p2, p3):
        """Calculate circle parameters from three points"""
        try:
            # Convert to homogeneous coordinates
            temp = p2[0] * p2[0] + p2[1] * p2[1]
            bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
            cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
            
            if abs(det) < 1e-6:
                return None
                
            # Center
            cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
            cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
            
            # Radius
            radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
            
            return (cx, cy, radius)
            
        except:
            return None

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def detect_core_cladding(image_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Main entry point for unified core/cladding detection
    
    Args:
        image_path: Path to the fiber optic image
        output_dir: Optional output directory for saving results
        
    Returns:
        Dictionary with detection results
    """
    # Create detector with default config
    config = DetectorConfig()
    detector = UnifiedCoreCladingDetector(config)
    
    # Run detection
    result = detector.detect(image_path)
    
    # Save result if output directory specified
    if output_dir and result['success']:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save JSON result
        with open(os.path.join(output_dir, f'{base_name}_unified_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
            
        # Create and save visualization
        try:
            img = cv2.imread(image_path)
            if img is not None:
                # Draw detection results
                cx, cy = int(result['center'][0]), int(result['center'][1])
                core_r = int(result['core_radius'])
                clad_r = int(result['cladding_radius'])
                
                # Draw circles
                cv2.circle(img, (cx, cy), core_r, (0, 255, 0), 2)  # Green for core
                cv2.circle(img, (cx, cy), clad_r, (0, 0, 255), 2)  # Red for cladding
                cv2.circle(img, (cx, cy), 3, (255, 255, 0), -1)    # Yellow center
                
                # Add text
                text = f"Confidence: {result['confidence']:.2f}"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Save annotated image
                cv2.imwrite(os.path.join(output_dir, f'{base_name}_unified_annotated.png'), img)
                
        except Exception as e:
            logging.warning(f"Failed to create visualization: {str(e)}")
    
    return result


def main():
    """Main function for standalone testing"""
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"').strip("'")
    
    # Run detection
    result = detect_core_cladding(image_path, output_dir='output_unified')
    
    # Print results
    if result['success']:
        print(f"\n✓ Detection successful!")
        print(f"  Center: {result['center']}")
        print(f"  Core radius: {result['core_radius']:.2f}")
        print(f"  Cladding radius: {result['cladding_radius']:.2f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        
        if 'details' in result and 'ensemble_details' in result['details']:
            details = result['details']['ensemble_details']
            print(f"  Methods agreed: {details['num_methods']}")
            print(f"  Contributing: {', '.join(details['contributing_methods'])}")
    else:
        print(f"\n✗ Detection failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
