#!/usr/bin/env python3
"""
Unified Fiber Optic End Face Defect Detection System
====================================================
This script combines all defect detection methods from multiple sources
into a single, highly accurate system.

Author: Unified System
Version: 2.0
"""

import cv2
import numpy as np
from scipy import ndimage, signal, stats
from scipy.ndimage import gaussian_filter, median_filter
from scipy.sparse.linalg import svds
from scipy.optimize import minimize
from skimage import morphology, measure, feature
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import warnings
import time
from datetime import datetime
import concurrent.futures
from functools import partial

warnings.filterwarnings('ignore')

@dataclass
class DefectDetectionConfig:
    """Unified configuration for all detection methods"""
    # General parameters
    min_defect_area_px: int = 5
    max_defect_area_px: int = 10000
    
    # Preprocessing parameters
    gaussian_blur_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(3,3), (5,5), (7,7)])
    bilateral_params: List[Tuple[int, int, int]] = field(default_factory=lambda: [(9,75,75), (7,50,50)])
    clahe_params: List[Tuple[float, Tuple[int, int]]] = field(default_factory=lambda: [(2.0,(8,8)), (3.0,(8,8))])
    
    # Fiber detection parameters
    hough_dp_values: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    hough_param1_values: List[int] = field(default_factory=lambda: [50, 70, 100])
    hough_param2_values: List[int] = field(default_factory=lambda: [30, 40, 50])
    
    # DO2MR parameters
    do2mr_kernel_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(5,5), (11,11), (15,15), (21,21)])
    do2mr_gamma_values: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 3.5])
    
    # LEI parameters
    lei_kernel_lengths: List[int] = field(default_factory=lambda: [9, 11, 15, 19, 21])
    lei_angle_steps: List[int] = field(default_factory=lambda: [5, 10, 15])
    lei_threshold_factors: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    
    # Advanced detection parameters
    hessian_scales: List[float] = field(default_factory=lambda: [1, 2, 3, 4])
    frangi_scales: List[float] = field(default_factory=lambda: [1, 1.5, 2, 2.5, 3])
    log_scales: List[float] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8])
    phase_congruency_scales: int = 4
    phase_congruency_orientations: int = 6
    
    # Ensemble parameters
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'do2mr': 1.0,
        'lei': 1.0,
        'hessian': 0.9,
        'frangi': 0.9,
        'phase_congruency': 0.85,
        'radon': 0.8,
        'gradient': 0.8,
        'log': 0.9,
        'doh': 0.85,
        'mser': 0.8,
        'watershed': 0.85,
        'canny': 0.7,
        'adaptive': 0.75,
        'lbp': 0.7,
        'otsu': 0.7,
        'morphological': 0.75
    })
    min_methods_for_detection: int = 2
    ensemble_vote_threshold: float = 0.3

@dataclass
class DefectInfo:
    """Information about a detected defect"""
    defect_id: int
    zone_name: str
    defect_type: str  # 'scratch', 'dig', 'contamination', etc.
    centroid_px: Tuple[int, int]
    area_px: float
    area_um: Optional[float] = None
    major_dimension_px: float = 0
    major_dimension_um: Optional[float] = None
    minor_dimension_px: float = 0
    minor_dimension_um: Optional[float] = None
    confidence_score: float = 0.0
    detection_methods: List[str] = field(default_factory=list)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    eccentricity: float = 0.0
    solidity: float = 0.0
    orientation: float = 0.0

class UnifiedFiberDefectDetector:
    """Main class for unified defect detection"""
    
    def __init__(self, config: Optional[DefectDetectionConfig] = None):
        self.config = config or DefectDetectionConfig()
        self.debug = True
        self.intermediate_results = {}
        self.pixels_per_micron = None
        
    def detect_defects(self, image_path: Union[str, Path], 
                      cladding_diameter_um: Optional[float] = None,
                      core_diameter_um: Optional[float] = None) -> Dict[str, Any]:
        """
        Main method to detect all defects in an image
        
        Args:
            image_path: Path to the fiber optic image
            cladding_diameter_um: Known cladding diameter in microns (optional)
            core_diameter_um: Known core diameter in microns (optional)
            
        Returns:
            Dictionary containing detection results
        """
        print(f"Starting defect detection for: {image_path}")
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Advanced preprocessing
        print("Step 1: Advanced preprocessing...")
        preprocessed_images = self._advanced_preprocessing(gray)
        
        # Step 2: Find fiber center and create zones
        print("Step 2: Finding fiber center and creating zones...")
        fiber_info = self._find_fiber_center_enhanced(preprocessed_images)
        if not fiber_info:
            print("Warning: Could not detect fiber center accurately")
            # Use image center as fallback
            h, w = gray.shape
            fiber_info = {
                'center': (w//2, h//2),
                'radius': min(h, w) // 4,
                'confidence': 0.0
            }
        
        # Calculate pixel to micron conversion if possible
        if cladding_diameter_um and fiber_info['radius'] > 0:
            self.pixels_per_micron = (2 * fiber_info['radius']) / cladding_diameter_um
            print(f"Calculated scale: {self.pixels_per_micron:.4f} pixels/micron")
        
        # Create zone masks
        zone_masks = self._create_zone_masks(gray.shape, fiber_info['center'], 
                                           fiber_info['radius'], core_diameter_um, 
                                           cladding_diameter_um)
        
        # Step 3: Apply all detection methods
        print("Step 3: Applying all detection methods...")
        all_detections = self._apply_all_detection_methods(preprocessed_images, zone_masks)
        
        # Step 4: Ensemble combination
        print("Step 4: Ensemble combination of results...")
        combined_masks = self._ensemble_combination(all_detections, gray.shape)
        
        # Step 5: False positive reduction
        print("Step 5: Reducing false positives...")
        refined_masks = self._reduce_false_positives(combined_masks, preprocessed_images)
        
        # Step 6: Analyze and classify defects
        print("Step 6: Analyzing and classifying defects...")
        defects = self._analyze_defects(refined_masks, zone_masks)
        
        # Step 7: Apply pass/fail criteria
        print("Step 7: Applying pass/fail criteria...")
        pass_fail_result = self._apply_pass_fail_criteria(defects, zone_masks)
        
        duration = time.time() - start_time
        print(f"Detection completed in {duration:.2f} seconds")
        
        return {
            'defects': defects,
            'pass_fail': pass_fail_result,
            'fiber_info': fiber_info,
            'zone_masks': zone_masks,
            'detection_masks': refined_masks,
            'intermediate_results': self.intermediate_results if self.debug else None,
            'processing_time': duration,
            'image_path': str(image_path),
            'pixels_per_micron': self.pixels_per_micron
        }
    
    def _advanced_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply all preprocessing methods"""
        preprocessed = {
            'original': image.copy()
        }
        
        # 1. Anisotropic diffusion (Perona-Malik)
        preprocessed['anisotropic'] = self._anisotropic_diffusion(image.astype(np.float64))
        
        # 2. Total Variation denoising
        preprocessed['tv_denoised'] = denoise_tv_chambolle(image.astype(np.float64) / 255.0, weight=0.1) * 255
        
        # 3. Coherence-enhancing diffusion
        preprocessed['coherence'] = self._coherence_enhancing_diffusion(preprocessed['tv_denoised'])
        
        # 4. Multiple Gaussian blurs
        for size in self.config.gaussian_blur_sizes:
            preprocessed[f'gaussian_{size[0]}'] = cv2.GaussianBlur(image, size, 0)
        
        # 5. Multiple bilateral filters
        for i, (d, sc, ss) in enumerate(self.config.bilateral_params):
            preprocessed[f'bilateral_{i}'] = cv2.bilateralFilter(image, d, sc, ss)
        
        # 6. Multiple CLAHE variants
        for i, (clip, grid) in enumerate(self.config.clahe_params):
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            preprocessed[f'clahe_{i}'] = clahe.apply(image)
        
        # 7. Standard preprocessing
        preprocessed['median'] = cv2.medianBlur(image, 5)
        preprocessed['nlmeans'] = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        preprocessed['histeq'] = cv2.equalizeHist(image)
        
        # 8. Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        preprocessed['morph_gradient'] = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        preprocessed['tophat'] = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        preprocessed['blackhat'] = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        return preprocessed
    
    def _anisotropic_diffusion(self, image: np.ndarray, iterations: int = 10, 
                               kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """Perona-Malik anisotropic diffusion"""
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
        """Coherence-enhancing diffusion for linear structures"""
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
    
    def _find_fiber_center_enhanced(self, preprocessed_images: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Find fiber center using multiple methods and vote"""
        candidates = []
        
        # Method 1: Hough circles on different preprocessed images
        for img_name in ['gaussian_5', 'bilateral_0', 'clahe_0', 'median']:
            if img_name not in preprocessed_images:
                continue
                
            img = preprocessed_images[img_name]
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
        for img_name in ['bilateral_0', 'nlmeans']:
            if img_name not in preprocessed_images:
                continue
                
            img = preprocessed_images[img_name]
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                candidates.append({
                    'center': (int(x), int(y)),
                    'radius': float(radius),
                    'method': f'contour_{img_name}',
                    'confidence': 0.8
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
        
        return {
            'center': best_center,
            'radius': best_radius,
            'confidence': avg_confidence,
            'num_candidates': len(candidates)
        }
    
    def _create_zone_masks(self, image_shape: Tuple[int, int], center: Tuple[int, int],
                          cladding_radius: float, core_diameter_um: Optional[float] = None,
                          cladding_diameter_um: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Create masks for different fiber zones"""
        h, w = image_shape
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
        
        masks = {}
        
        # Define zone radii
        if core_diameter_um and cladding_diameter_um:
            core_radius = cladding_radius * (core_diameter_um / cladding_diameter_um) / 2
        else:
            core_radius = cladding_radius * 0.072  # Default 9/125 for SMF
        
        ferrule_radius = cladding_radius * 2.0
        adhesive_radius = ferrule_radius * 1.1
        
        # Create masks
        masks['core'] = (dist_from_center <= core_radius).astype(np.uint8) * 255
        masks['cladding'] = ((dist_from_center > core_radius) & 
                            (dist_from_center <= cladding_radius)).astype(np.uint8) * 255
        masks['ferrule'] = ((dist_from_center > cladding_radius) & 
                           (dist_from_center <= ferrule_radius)).astype(np.uint8) * 255
        masks['adhesive'] = ((dist_from_center > ferrule_radius) & 
                            (dist_from_center <= adhesive_radius)).astype(np.uint8) * 255
        
        return masks
    
    def _apply_all_detection_methods(self, preprocessed_images: Dict[str, np.ndarray],
                                    zone_masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Apply all detection methods to all zones"""
        all_detections = {}
        
        # Get best preprocessed images for different methods
        img_for_region = preprocessed_images.get('clahe_0', preprocessed_images['original'])
        img_for_scratch = preprocessed_images.get('coherence', preprocessed_images['original'])
        img_for_general = preprocessed_images.get('bilateral_0', preprocessed_images['original'])
        
        for zone_name, zone_mask in zone_masks.items():
            print(f"  Processing zone: {zone_name}")
            zone_detections = {}
            
            # 1. DO2MR for region defects
            zone_detections['do2mr'] = self._detect_do2mr_enhanced(img_for_region, zone_mask)
            
            # 2. LEI for scratches
            zone_detections['lei'] = self._detect_lei_enhanced(img_for_scratch, zone_mask)
            
            # 3. Hessian ridge detection
            zone_detections['hessian'] = self._hessian_ridge_detection(img_for_scratch, zone_mask)
            
            # 4. Frangi vesselness
            zone_detections['frangi'] = self._frangi_vesselness(img_for_scratch, zone_mask)
            
            # 5. Phase congruency
            zone_detections['phase_congruency'] = self._phase_congruency_detection(img_for_general, zone_mask)
            
            # 6. Radon transform
            zone_detections['radon'] = self._radon_line_detection(img_for_scratch, zone_mask)
            
            # 7. Gradient-based
            zone_detections['gradient'] = self._gradient_based_detection(img_for_general, zone_mask)
            
            # 8. Scale-normalized LoG
            zone_detections['log'] = self._scale_normalized_log(img_for_region, zone_mask)
            
            # 9. Determinant of Hessian
            zone_detections['doh'] = self._determinant_of_hessian(img_for_region, zone_mask)
            
            # 10. MSER
            zone_detections['mser'] = self._mser_detection(img_for_region, zone_mask)
            
            # 11. Watershed
            zone_detections['watershed'] = self._watershed_detection(img_for_general, zone_mask)
            
            # 12. Canny
            zone_detections['canny'] = self._canny_detection(img_for_general, zone_mask)
            
            # 13. Adaptive threshold
            zone_detections['adaptive'] = self._adaptive_threshold_detection(img_for_general, zone_mask)
            
            # 14. LBP
            zone_detections['lbp'] = self._lbp_detection(img_for_general, zone_mask)
            
            # 15. Otsu variants
            zone_detections['otsu'] = self._otsu_based_detection(img_for_general, zone_mask)
            
            # 16. Morphological
            zone_detections['morphological'] = self._morphological_detection(img_for_general, zone_mask)
            
            all_detections[zone_name] = zone_detections
        
        return all_detections
    
    def _detect_do2mr_enhanced(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Enhanced DO2MR detection"""
        combined_mask = np.zeros_like(image)
        vote_map = np.zeros_like(image, dtype=np.float32)
        
        for kernel_size in self.config.do2mr_kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
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
    
    def _detect_lei_enhanced(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Enhanced LEI scratch detection"""
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
        """Apply linear detector for scratch detection"""
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
    
    def _hessian_ridge_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Multi-scale Hessian ridge detection"""
        ridge_response = np.zeros_like(image, dtype=np.float32)
        
        for scale in self.config.hessian_scales:
            smoothed = gaussian_filter(image.astype(np.float32), scale)
            
            Hxx = gaussian_filter(smoothed, scale, order=(0, 2))
            Hyy = gaussian_filter(smoothed, scale, order=(2, 0))
            Hxy = gaussian_filter(smoothed, scale, order=(1, 1))
            
            trace = Hxx + Hyy
            det = Hxx * Hyy - Hxy * Hxy
            discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
            
            lambda1 = 0.5 * (trace + discriminant)
            lambda2 = 0.5 * (trace - discriminant)
            
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            beta = 0.5
            c = 0.5 * np.max(S)
            
            response = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*c**2)))
            response[lambda2 > 0] = 0
            
            ridge_response = np.maximum(ridge_response, scale**2 * response)
        
        _, ridge_mask = cv2.threshold(ridge_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ridge_mask = cv2.bitwise_and(ridge_mask, ridge_mask, mask=zone_mask)
        
        return ridge_mask
    
    def _frangi_vesselness(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Frangi vesselness filter"""
        vesselness = np.zeros_like(image, dtype=np.float32)
        
        for scale in self.config.frangi_scales:
            smoothed = gaussian_filter(image.astype(np.float32), scale)
            
            Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
            Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
            Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
            
            tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
            lambda1 = 0.5 * (Hxx + Hyy + tmp)
            lambda2 = 0.5 * (Hxx + Hyy - tmp)
            
            idx = np.abs(lambda1) < np.abs(lambda2)
            lambda1[idx], lambda2[idx] = lambda2[idx], lambda1[idx]
            
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            beta = 0.5
            gamma = 15
            
            v = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*gamma**2)))
            v[lambda2 > 0] = 0
            
            vesselness = np.maximum(vesselness, v)
        
        cv2.normalize(vesselness, vesselness, 0, 255, cv2.NORM_MINMAX)
        _, vessel_mask = cv2.threshold(vesselness.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=zone_mask)
        
        return vessel_mask
    
    def _phase_congruency_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Simplified phase congruency detection"""
        # Using edge detection as a simplified version
        edges = cv2.Canny(image, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
        
        # Apply morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def _radon_line_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Radon transform for line detection"""
        edges = cv2.Canny(image, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
        
        # Simplified implementation using Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
        
        line_mask = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        
        line_mask = cv2.bitwise_and(line_mask, line_mask, mask=zone_mask)
        return line_mask
    
    def _gradient_based_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Gradient-based defect detection"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
        
        _, grad_mask = cv2.threshold(magnitude.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        grad_mask = cv2.bitwise_and(grad_mask, grad_mask, mask=zone_mask)
        
        return grad_mask
    
    def _scale_normalized_log(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Scale-normalized Laplacian of Gaussian"""
        blob_response = np.zeros_like(image, dtype=np.float32)
        
        for scale in self.config.log_scales:
            log = ndimage.gaussian_laplace(image.astype(np.float32), scale)
            log *= scale**2
            blob_response = np.maximum(blob_response, np.abs(log))
        
        cv2.normalize(blob_response, blob_response, 0, 255, cv2.NORM_MINMAX)
        _, blob_mask = cv2.threshold(blob_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blob_mask = cv2.bitwise_and(blob_mask, blob_mask, mask=zone_mask)
        
        return blob_mask
    
    def _determinant_of_hessian(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Determinant of Hessian blob detection"""
        doh_response = np.zeros_like(image, dtype=np.float32)
        
        for scale in self.config.log_scales[::2]:  # Use subset of scales
            smoothed = gaussian_filter(image.astype(np.float32), scale)
            
            Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
            Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
            Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
            
            det = Hxx * Hyy - Hxy**2
            det *= scale**4
            
            doh_response = np.maximum(doh_response, np.abs(det))
        
        cv2.normalize(doh_response, doh_response, 0, 255, cv2.NORM_MINMAX)
        _, doh_mask = cv2.threshold(doh_response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        doh_mask = cv2.bitwise_and(doh_mask, doh_mask, mask=zone_mask)
        
        return doh_mask
    
    def _mser_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """MSER blob detection"""
        mser = cv2.MSER_create(delta=5, min_area=10, max_area=1000)
        regions, _ = mser.detectRegions(image)
        
        mask = np.zeros_like(image)
        for region in regions:
            cv2.fillPoly(mask, [region], 255)
        
        mask = cv2.bitwise_and(mask, mask, mask=zone_mask)
        return mask
    
    def _watershed_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Watershed segmentation"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(binary, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        watershed_mask = np.zeros_like(image)
        watershed_mask[markers == -1] = 255
        watershed_mask = cv2.bitwise_and(watershed_mask, watershed_mask, mask=zone_mask)
        
        return watershed_mask
    
    def _canny_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Canny edge detection"""
        edges = cv2.Canny(image, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        closed = cv2.bitwise_and(closed, closed, mask=zone_mask)
        return closed
    
    def _adaptive_threshold_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Adaptive threshold detection"""
        adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)
        adaptive = cv2.bitwise_and(adaptive, adaptive, mask=zone_mask)
        return adaptive
    
    def _lbp_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Simplified LBP-based anomaly detection"""
        # Using texture variance as anomaly measure
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((image - local_mean)**2, -1, kernel)
        
        cv2.normalize(local_var, local_var, 0, 255, cv2.NORM_MINMAX)
        _, anomaly_mask = cv2.threshold(local_var.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        anomaly_mask = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=zone_mask)
        
        return anomaly_mask
    
    def _otsu_based_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Otsu-based defect detection"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        defects = cv2.absdiff(opened, closed)
        defects = cv2.bitwise_and(defects, defects, mask=zone_mask)
        
        return defects
    
    def _morphological_detection(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """Morphological-based defect detection"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Top-hat for bright defects
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat for dark defects
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine both
        combined = cv2.add(tophat, blackhat)
        
        _, morph_mask = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
        morph_mask = cv2.bitwise_and(morph_mask, morph_mask, mask=zone_mask)
        
        return morph_mask
    
    def _ensemble_combination(self, all_detections: Dict[str, Dict[str, np.ndarray]], 
                            image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Combine all detection results using weighted voting"""
        h, w = image_shape
        combined_masks = {}
        
        # Separate masks for scratches and regions
        scratch_methods = ['lei', 'hessian', 'frangi', 'radon', 'phase_congruency']
        region_methods = ['do2mr', 'log', 'doh', 'mser', 'lbp', 'otsu']
        general_methods = ['gradient', 'watershed', 'canny', 'adaptive', 'morphological']
        
        for zone_name, zone_detections in all_detections.items():
            # Initialize vote maps
            scratch_votes = np.zeros((h, w), dtype=np.float32)
            region_votes = np.zeros((h, w), dtype=np.float32)
            
            # Accumulate weighted votes
            for method_name, mask in zone_detections.items():
                if mask is None:
                    continue
                    
                weight = self.config.confidence_weights.get(method_name, 0.5)
                
                if method_name in scratch_methods:
                    scratch_votes += (mask > 0).astype(np.float32) * weight
                elif method_name in region_methods:
                    region_votes += (mask > 0).astype(np.float32) * weight
                else:  # general methods contribute to both
                    scratch_votes += (mask > 0).astype(np.float32) * weight * 0.5
                    region_votes += (mask > 0).astype(np.float32) * weight * 0.5
            
            # Normalize vote maps
            max_scratch_vote = sum(self.config.confidence_weights.get(m, 0.5) 
                                  for m in scratch_methods + general_methods)
            max_region_vote = sum(self.config.confidence_weights.get(m, 0.5) 
                                 for m in region_methods + general_methods)
            
            scratch_votes /= max_scratch_vote
            region_votes /= max_region_vote
            
            # Apply threshold
            scratch_mask = (scratch_votes >= self.config.ensemble_vote_threshold).astype(np.uint8) * 255
            region_mask = (region_votes >= self.config.ensemble_vote_threshold).astype(np.uint8) * 255
            
            # Morphological refinement
            scratch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, scratch_kernel)
            
            region_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, region_kernel)
            
            combined_masks[f'{zone_name}_scratches'] = scratch_mask
            combined_masks[f'{zone_name}_regions'] = region_mask
            combined_masks[f'{zone_name}_all'] = cv2.bitwise_or(scratch_mask, region_mask)
        
        return combined_masks
    
    def _reduce_false_positives(self, combined_masks: Dict[str, np.ndarray],
                               preprocessed_images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply advanced false positive reduction"""
        refined_masks = {}
        
        # Use best quality image for validation
        validation_image = preprocessed_images.get('bilateral_0', preprocessed_images['original'])
        
        for mask_name, mask in combined_masks.items():
            if 'scratches' in mask_name:
                refined = self._reduce_false_positives_scratches(mask, validation_image)
            elif 'regions' in mask_name:
                refined = self._reduce_false_positives_regions(mask, validation_image)
            else:
                refined = mask.copy()
            
            refined_masks[mask_name] = refined
        
        return refined_masks
    
    def _reduce_false_positives_scratches(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Reduce false positives for scratches"""
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
                    if np.mean(residuals) > 3:  # threshold in pixels
                        refined[labels == i] = 0
                        continue
                
                # Check aspect ratio
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                if aspect_ratio < 3:  # Not elongated enough for a scratch
                    refined[labels == i] = 0
                    continue
                
                # Check contrast
                component_pixels = image[component_mask > 0]
                surrounding_mask = cv2.dilate(component_mask, np.ones((5, 5), np.uint8)) - component_mask
                if np.sum(surrounding_mask) > 0:
                    surrounding_pixels = image[surrounding_mask > 0]
                    contrast = abs(np.mean(component_pixels) - np.mean(surrounding_pixels))
                    if contrast < 10:  # Low contrast threshold
                        refined[labels == i] = 0
        
        return refined
    
    def _reduce_false_positives_regions(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Reduce false positives for region defects"""
        refined = mask.copy()
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)
        
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
            
            # Check circularity
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity < 0.3:  # Too irregular
                        refined[labels == i] = 0
                        continue
                
                # Check solidity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.5:  # Too concave
                        refined[labels == i] = 0
                        continue
        
        return refined
    
    def _analyze_defects(self, refined_masks: Dict[str, np.ndarray],
                        zone_masks: Dict[str, np.ndarray]) -> List[DefectInfo]:
        """Analyze and classify all detected defects"""
        all_defects = []
        defect_id = 0
        
        for mask_name, mask in refined_masks.items():
            if '_all' not in mask_name:  # Skip combined masks to avoid duplicates
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
                
                # Get detailed properties
                component_mask = (labels == i).astype(np.uint8)
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                    
                contour = contours[0]
                
                # Calculate shape properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area_px / hull_area if hull_area > 0 else 0
                
                # Fit ellipse for orientation and dimensions
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center_x, center_y), (MA, ma), angle = ellipse
                    major_dimension_px = max(MA, ma)
                    minor_dimension_px = min(MA, ma)
                    orientation = angle
                else:
                    major_dimension_px = max(w, h)
                    minor_dimension_px = min(w, h)
                    orientation = 0
                
                # Calculate eccentricity
                if major_dimension_px > 0:
                    eccentricity = np.sqrt(1 - (minor_dimension_px / major_dimension_px) ** 2)
                else:
                    eccentricity = 0
                
                # Classify defect type
                aspect_ratio = major_dimension_px / minor_dimension_px if minor_dimension_px > 0 else 1
                
                if aspect_ratio > 3 and eccentricity > 0.8:
                    defect_type = 'scratch'
                elif circularity > 0.7 and solidity > 0.8:
                    defect_type = 'dig'
                elif area_px > 500:
                    defect_type = 'contamination'
                else:
                    defect_type = 'defect'
                
                # Convert to microns if possible
                area_um = area_px / (self.pixels_per_micron ** 2) if self.pixels_per_micron else None
                major_dimension_um = major_dimension_px / self.pixels_per_micron if self.pixels_per_micron else None
                minor_dimension_um = minor_dimension_px / self.pixels_per_micron if self.pixels_per_micron else None
                
                # Determine which methods detected this defect
                detection_methods = []
                scratch_mask = refined_masks.get(f'{zone_name}_scratches', np.zeros_like(mask))
                region_mask = refined_masks.get(f'{zone_name}_regions', np.zeros_like(mask))
                
                if scratch_mask[cy, cx] > 0:
                    detection_methods.extend(['lei', 'hessian', 'frangi'])
                if region_mask[cy, cx] > 0:
                    detection_methods.extend(['do2mr', 'log', 'mser'])
                
                # Create DefectInfo
                defect = DefectInfo(
                    defect_id=defect_id,
                    zone_name=zone_name,
                    defect_type=defect_type,
                    centroid_px=(cx, cy),
                    area_px=area_px,
                    area_um=area_um,
                    major_dimension_px=major_dimension_px,
                    major_dimension_um=major_dimension_um,
                    minor_dimension_px=minor_dimension_px,
                    minor_dimension_um=minor_dimension_um,
                    confidence_score=0.8,  # Could be calculated based on detection agreement
                    detection_methods=detection_methods,
                    bounding_box=(x, y, w, h),
                    eccentricity=eccentricity,
                    solidity=solidity,
                    orientation=orientation
                )
                
                all_defects.append(defect)
        
        return all_defects
    
    def _apply_pass_fail_criteria(self, defects: List[DefectInfo], 
                                 zone_masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Apply IEC 61300-3-35 based pass/fail criteria"""
        # Zone-specific criteria (in microns)
        zone_criteria = {
            'core': {'max_defect_um': 3, 'max_count': 5},
            'cladding': {'max_defect_um': 10, 'max_count': 10},
            'ferrule': {'max_defect_um': 25, 'max_count': 20},
            'adhesive': {'max_defect_um': 50, 'max_count': None}
        }
        
        status = 'PASS'
        failures = []
        
        # Group defects by zone
        defects_by_zone = {}
        for defect in defects:
            if defect.zone_name not in defects_by_zone:
                defects_by_zone[defect.zone_name] = []
            defects_by_zone[defect.zone_name].append(defect)
        
        # Check each zone
        for zone_name, criteria in zone_criteria.items():
            zone_defects = defects_by_zone.get(zone_name, [])
            
            # Check defect count
            if criteria['max_count'] and len(zone_defects) > criteria['max_count']:
                status = 'FAIL'
                failures.append(f"{zone_name}: Too many defects ({len(zone_defects)} > {criteria['max_count']})")
            
            # Check defect sizes
            if self.pixels_per_micron:  # Only check sizes if we have micron conversion
                for defect in zone_defects:
                    size_um = defect.major_dimension_um if defect.defect_type == 'scratch' else defect.area_um ** 0.5
                    if size_um and size_um > criteria['max_defect_um']:
                        status = 'FAIL'
                        failures.append(
                            f"{zone_name}: {defect.defect_type} exceeds size limit "
                            f"({size_um:.1f}µm > {criteria['max_defect_um']}µm)"
                        )
        
        return {
            'status': status,
            'failures': failures,
            'defects_by_zone': {k: len(v) for k, v in defects_by_zone.items()},
            'total_defects': len(defects)
        }
    
    def visualize_results(self, image_path: str, results: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize detection results"""
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image for visualization: {image_path}")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Fiber Optic Defect Detection Results - {results['pass_fail']['status']}", fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Zone masks overlay
        zone_overlay = np.zeros_like(image)
        zone_colors = {
            'core': [255, 0, 0],      # Red
            'cladding': [0, 255, 0],   # Green
            'ferrule': [0, 0, 255],    # Blue
            'adhesive': [255, 255, 0]  # Yellow
        }
        
        for zone_name, mask in results['zone_masks'].items():
            if zone_name in zone_colors:
                zone_overlay[mask > 0] = zone_colors[zone_name]
        
        axes[0, 1].imshow(zone_overlay)
        axes[0, 1].set_title('Fiber Zones')
        axes[0, 1].axis('off')
        
        # All defects overlay
        defect_overlay = image.copy()
        for defect in results['defects']:
            x, y, w, h = defect.bounding_box
            
            # Color based on defect type
            if defect.defect_type == 'scratch':
                color = (255, 0, 255)  # Magenta
            elif defect.defect_type == 'dig':
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 128, 0)  # Orange
            
            cv2.rectangle(defect_overlay, (x, y), (x+w, y+h), color, 2)
            cv2.putText(defect_overlay, f"{defect.defect_id}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        axes[0, 2].imshow(cv2.cvtColor(defect_overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'Detected Defects ({len(results["defects"])})')
        axes[0, 2].axis('off')
        
        # Detection masks
        if 'detection_masks' in results:
            # Scratches
            scratch_combined = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask_name, mask in results['detection_masks'].items():
                if 'scratches' in mask_name:
                    scratch_combined = cv2.bitwise_or(scratch_combined, mask)
            
            axes[1, 0].imshow(scratch_combined, cmap='hot')
            axes[1, 0].set_title('Scratch Detections')
            axes[1, 0].axis('off')
            
            # Regions
            region_combined = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask_name, mask in results['detection_masks'].items():
                if 'regions' in mask_name:
                    region_combined = cv2.bitwise_or(region_combined, mask)
            
            axes[1, 1].imshow(region_combined, cmap='hot')
            axes[1, 1].set_title('Region Defect Detections')
            axes[1, 1].axis('off')
        
        # Results summary
        summary_text = f"Status: {results['pass_fail']['status']}\n"
        summary_text += f"Total Defects: {results['pass_fail']['total_defects']}\n\n"
        summary_text += "Defects by Zone:\n"
        for zone, count in results['pass_fail']['defects_by_zone'].items():
            summary_text += f"  {zone}: {count}\n"
        
        if results['pass_fail']['failures']:
            summary_text += "\nFailure Reasons:\n"
            for failure in results['pass_fail']['failures'][:5]:  # Show first 5
                summary_text += f"  - {failure}\n"
        
        summary_text += f"\nProcessing Time: {results['processing_time']:.2f}s"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=10, family='monospace')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to run the unified defect detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Fiber Optic Defect Detection System')
    parser.add_argument('image_path', type=str, help='Path to fiber optic image')
    parser.add_argument('--cladding-diameter', type=float, default=None,
                       help='Cladding diameter in microns (e.g., 125)')
    parser.add_argument('--core-diameter', type=float, default=None,
                       help='Core diameter in microns (e.g., 9 for SMF, 50/62.5 for MMF)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    parser.add_argument('--save-masks', action='store_true',
                       help='Save individual detection masks')
    
    args = parser.parse_args()
    
    # Create detector
    print("Initializing Unified Fiber Defect Detector...")
    detector = UnifiedFiberDefectDetector()
    
    # Run detection
    try:
        results = detector.detect_defects(
            args.image_path,
            cladding_diameter_um=args.cladding_diameter,
            core_diameter_um=args.core_diameter
        )
        
        # Print summary
        print("\n" + "="*50)
        print("DETECTION RESULTS")
        print("="*50)
        print(f"Status: {results['pass_fail']['status']}")
        print(f"Total Defects: {len(results['defects'])}")
        print(f"\nDefects by Zone:")
        for zone, count in results['pass_fail']['defects_by_zone'].items():
            print(f"  {zone}: {count}")
        
        if results['pass_fail']['failures']:
            print(f"\nFailure Reasons:")
            for failure in results['pass_fail']['failures']:
                print(f"  - {failure}")
        
        print(f"\nDefect Types:")
        defect_types = {}
        for defect in results['defects']:
            defect_types[defect.defect_type] = defect_types.get(defect.defect_type, 0) + 1
        for dtype, count in defect_types.items():
            print(f"  {dtype}: {count}")
        
        # Save visualization
        output_path = args.output or f"{Path(args.image_path).stem}_results.png"
        detector.visualize_results(args.image_path, results, output_path)
        
        # Save masks if requested
        if args.save_masks and 'detection_masks' in results:
            mask_dir = Path(args.image_path).parent / f"{Path(args.image_path).stem}_masks"
            mask_dir.mkdir(exist_ok=True)
            
            for mask_name, mask in results['detection_masks'].items():
                cv2.imwrite(str(mask_dir / f"{mask_name}.png"), mask)
            
            print(f"\nMasks saved to: {mask_dir}")
        
        # Save detailed results as JSON
        import json
        
        # Convert results to JSON-serializable format
        json_results = {
            'image_path': results['image_path'],
            'status': results['pass_fail']['status'],
            'total_defects': len(results['defects']),
            'defects_by_zone': results['pass_fail']['defects_by_zone'],
            'failures': results['pass_fail']['failures'],
            'processing_time': results['processing_time'],
            'fiber_center': results['fiber_info']['center'],
            'fiber_radius': results['fiber_info']['radius'],
            'pixels_per_micron': results['pixels_per_micron'],
            'defects': [
                {
                    'id': d.defect_id,
                    'type': d.defect_type,
                    'zone': d.zone_name,
                    'centroid': d.centroid_px,
                    'area_px': d.area_px,
                    'area_um': d.area_um,
                    'major_dim_px': d.major_dimension_px,
                    'major_dim_um': d.major_dimension_um,
                    'confidence': d.confidence_score
                }
                for d in results['defects']
            ]
        }
        
        json_path = f"{Path(args.image_path).stem}_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {json_path}")
        
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
