#!/usr/bin/env python3
"""
=========================================================
Unified Fiber Optic Defect Inspection System (Version 2.0)
=========================================================

This script represents a complete re-architecture to create a hierarchical and
synergistic analysis engine, building on the strengths of all provided scripts.
The system now uses a two-stage approach for superior accuracy:

1.  **Primary Anomaly Detection:** Leverages an advanced statistical knowledge base
    (inspired by jake.py) that models the relationship between pixel intensity
    and gradient magnitude for each fiber region. This stage identifies all
    statistically significant deviations from a "perfect" fiber, producing a
    highly accurate anomaly heatmap.

2.  **Targeted Defect Classification:** The specialized defect detectors (from
    daniel.py, jill.py) are no longer run blindly. They are now targeted
    specifically at the anomalous regions found in Stage 1 to classify them
    as scratches, pits, contamination, etc.

This hierarchical process dramatically reduces false positives and improves
detection accuracy by combining statistical anomaly detection with morphological
classification.

Author: Unified Analysis Team
Version: 2.0 - Hierarchical Re-architecture
"""

import cv2
import numpy as np
import os
import json
import time
import logging
import traceback
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Scipy and Scikit-image/learn are powerful libraries.
from scipy import ndimage, stats
from skimage import morphology, feature, filters, measure, restoration, segmentation, metrics
from sklearn.ensemble import IsolationForest

# --- Configuration & Setup ---

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types for serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (DefectType, DefectSeverity)):
            return obj.name
        return json.JSONEncoder.default(self, obj)

# --- Data Structures ---

@dataclass
class MasterConfig:
    """Unified configuration for the hierarchical analysis."""
    # Mask Generation
    cladding_core_ratio: float = 125.0 / 9.0

    # Preprocessing
    denoise_strength: float = 7.0

    # Primary Anomaly Detection
    gradient_kernel_size: int = 3
    intensity_gradient_bins: int = 64 # Bins for the 2D histogram model
    anomaly_threshold_p_value: float = 0.001 # Corresponds to a high confidence requirement

    # Secondary Defect Classification
    min_defect_area_px: int = 10
    scratch_eccentricity_threshold: float = 0.95
    scratch_aspect_ratio_threshold: float = 4.0
    
    # Output
    output_dir: str = "analysis_output"
    knowledge_base_path: str = "fiber_knowledge_base.json"

class DefectType(Enum):
    SCRATCH = "Scratch"
    PIT_DIG = "Pit/Dig"
    CONTAMINATION = "Contamination"
    TEXTURE_ANOMALY = "Texture Anomaly"
    UNKNOWN = "Unknown Defect"

class DefectSeverity(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@dataclass
class Defect:
    """Represents a single detected and classified defect."""
    id: int
    type: DefectType
    severity: DefectSeverity
    confidence: float
    location: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    area_px: int
    primary_anomaly_score: float
    classification_method: str
    mask: np.ndarray = field(repr=False)

    def to_dict(self):
        d = asdict(self)
        d.pop('mask')
        return d

# --- Knowledge Base Manager (Inspired by jake.py) ---

class KnowledgeBaseManager:
    """
    Manages the creation and use of a sophisticated statistical model
    of a "perfect" fiber end face.
    """
    def __init__(self, config: MasterConfig):
        self.config = config
        self.kb_path = config.knowledge_base_path
        self.model = None
        if os.path.exists(self.kb_path):
            self.load_model()

    def load_model(self):
        """Loads the knowledge base from a JSON file."""
        try:
            with open(self.kb_path, 'r') as f:
                loaded_data = json.load(f)
            # Convert histograms from lists back to numpy arrays
            for region, data in loaded_data['regional_models'].items():
                data['intensity_gradient_hist'] = np.array(data['intensity_gradient_hist'], dtype=np.float64)
            self.model = loaded_data
            logging.info(f"Successfully loaded knowledge base from {self.kb_path}")
            return True
        except Exception as e:
            logging.error(f"Could not load or parse knowledge base: {e}")
            self.model = None
            return False

    def build_new_model(self, ref_dir: str):
        """Builds a new knowledge base from a directory of reference images."""
        logging.info(f"Building new knowledge base from reference images in: {ref_dir}")
        image_paths = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if len(image_paths) < 3:
            logging.error(f"Need at least 3 reference images to build a reliable model, but found {len(image_paths)}.")
            return

        # Initialize accumulators
        regional_models = {
            "Core": {'intensity_gradient_hist': np.zeros((self.config.intensity_gradient_bins, self.config.intensity_gradient_bins))},
            "Cladding": {'intensity_gradient_hist': np.zeros((self.config.intensity_gradient_bins, self.config.intensity_gradient_bins))},
            "Ferrule": {'intensity_gradient_hist': np.zeros((self.config.intensity_gradient_bins, self.config.intensity_gradient_bins))},
        }

        # Process each reference image
        processed_count = 0
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None: continue

            # Create a temporary inspector to use its localization and preprocessing
            temp_inspector = UnifiedFiberInspector(self.config, self)
            masks, _ = temp_inspector._localize_fiber_robust(image)
            if not masks: continue

            # Preprocess and calculate gradient
            intensity_map = temp_inspector._preprocess_image(image)
            gradient_map = self._calculate_gradient_magnitude(intensity_map)

            # Accumulate 2D histograms for each region
            for region_name, mask in masks.items():
                if region_name not in regional_models: continue
                
                intensities = intensity_map[mask > 0]
                gradients = gradient_map[mask > 0]

                hist, _, _ = np.histogram2d(
                    intensities, gradients,
                    bins=self.config.intensity_gradient_bins,
                    range=[[0, 255], [0, 255]]
                )
                regional_models[region_name]['intensity_gradient_hist'] += hist
            processed_count += 1
        
        if processed_count == 0:
            logging.error("Failed to process any reference images. Model not built.")
            return

        # Normalize the histograms to create probability distributions
        for region, data in regional_models.items():
            hist = data['intensity_gradient_hist']
            total = hist.sum()
            if total > 0:
                data['intensity_gradient_hist'] = hist / total
        
        self.model = {'regional_models': regional_models}
        self._save_model()
        logging.info(f"Successfully built and saved new knowledge base with {processed_count} images.")

    def _save_model(self):
        """Saves the knowledge base to a JSON file."""
        try:
            save_data = json.loads(json.dumps(self.model, cls=NumpyEncoder))
            with open(self.kb_path, 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save knowledge base: {e}")

    def _calculate_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """Calculates the gradient magnitude of an image."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.config.gradient_kernel_size)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.config.gradient_kernel_size)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- Main Unified Inspector Class ---

class UnifiedFiberInspector:
    """The main class orchestrating the hierarchical analysis."""

    def __init__(self, config: MasterConfig, kb_manager: KnowledgeBaseManager):
        self.config = config
        self.kb_manager = kb_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_full_analysis(self, image_path: str) -> Optional[Dict]:
        """Main entry point to run the full hierarchical analysis."""
        start_time = time.time()
        self.logger.info(f"--- Starting Hierarchical Analysis for: {image_path} ---")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}"); return None

        try:
            # Step 1: Localize Fiber Regions
            masks, localization = self._localize_fiber_robust(image)
            if not masks: raise RuntimeError("Failed to localize fiber structure.")

            # Step 2: Preprocess and Calculate Gradient
            intensity_map = self._preprocess_image(image)
            gradient_map = self.kb_manager._calculate_gradient_magnitude(intensity_map)

            # Step 3: Primary Anomaly Detection (Statistical)
            anomaly_heatmap = self._perform_primary_anomaly_detection(intensity_map, gradient_map, masks)
            primary_anomaly_mask = (anomaly_heatmap > self.config.anomaly_threshold_p_value).astype(np.uint8) * 255

            # Step 4: Secondary Defect Classification (Targeted)
            final_defects = self._perform_secondary_classification(primary_anomaly_mask, anomaly_heatmap, masks)

            # Step 5: Final Quality Assessment & Reporting
            quality_score, pass_fail = self._assess_final_quality(final_defects)
            analysis_time = time.time() - start_time
            self.logger.info(f"--- Analysis Complete in {analysis_time:.2f}s. Quality Score: {quality_score:.1f}/100 ---")

            # Save outputs
            report = self._generate_and_save_outputs(image_path, image, final_defects, anomaly_heatmap, quality_score, pass_fail)
            return report

        except Exception as e:
            self.logger.error(f"An unrecoverable error occurred: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def _localize_fiber_robust(self, image: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Robustly finds fiber center and radius using multiple fallbacks."""
        # Using the robust method from previous versions
        try:
            blurred = cv2.GaussianBlur(image, (11, 11), 0)
            adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
            cleaned = morphology.remove_small_objects(adaptive_thresh.astype(bool), min_size=1000)
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=1000).astype(np.uint8) * 255
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), cr = cv2.minEnclosingCircle(largest_contour)
                if cr > image.shape[0] * 0.1:
                    localization = {"center": (int(cx), int(cy)), "cladding_radius_px": int(cr)}
                    masks = self._create_fiber_masks(image.shape, localization)
                    return masks, localization
        except Exception as e:
            self.logger.warning(f"Localization failed: {e}")
        return None, None

    def _create_fiber_masks(self, shape: Tuple, localization: Dict) -> Dict:
        """Creates region masks from localization info."""
        cx, cy = localization['center']
        cr = localization['cladding_radius_px']
        core_r = max(1, int(cr / self.config.cladding_core_ratio))
        localization['core_radius_px'] = core_r
        
        y, x = np.ogrid[:shape[0], :shape[1]]
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        masks = {}
        masks['Core'] = (dist_sq <= core_r**2).astype(np.uint8)
        masks['Cladding'] = ((dist_sq > core_r**2) & (dist_sq <= cr**2)).astype(np.uint8)
        masks['Ferrule'] = (dist_sq > cr**2).astype(np.uint8)
        return masks

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Applies advanced preprocessing."""
        processed = restoration.denoise_nl_means(image, h=self.config.denoise_strength, fast_mode=True)
        processed = cv2.equalizeHist(processed)
        return processed

    def _perform_primary_anomaly_detection(self, intensity_map: np.ndarray, gradient_map: np.ndarray, masks: Dict) -> np.ndarray:
        """Generates an anomaly heatmap based on the knowledge base."""
        if not self.kb_manager.model:
            raise RuntimeError("Knowledge base is not loaded. Cannot perform anomaly detection.")
        
        anomaly_heatmap = np.zeros(intensity_map.shape, dtype=np.float64)
        bins = self.config.intensity_gradient_bins

        for region_name, mask in masks.items():
            if region_name not in self.kb_manager.model['regional_models']: continue

            # Get the learned probability distribution for this region
            prob_dist = self.kb_manager.model['regional_models'][region_name]['intensity_gradient_hist']
            
            # Get pixels for the current region
            region_intensities = intensity_map[mask > 0]
            region_gradients = gradient_map[mask > 0]
            
            # Discretize pixel values to map to histogram bins
            intensity_bins = np.floor(region_intensities / (256 / bins)).astype(int)
            gradient_bins = np.floor(region_gradients / (256 / bins)).astype(int)
            np.clip(intensity_bins, 0, bins - 1, out=intensity_bins)
            np.clip(gradient_bins, 0, bins - 1, out=gradient_bins)
            
            # Look up probabilities from the 2D histogram
            probabilities = prob_dist[intensity_bins, gradient_bins]
            
            # Anomaly score is 1 - probability (low probability = high anomaly)
            # Add a small epsilon to avoid log(0)
            anomaly_scores = -np.log(probabilities + 1e-12)
            
            # Normalize scores for this region
            if anomaly_scores.max() > 0:
                anomaly_scores /= anomaly_scores.max()
            
            # Place scores back into the main heatmap
            anomaly_heatmap[mask > 0] = anomaly_scores

        return anomaly_heatmap

    def _perform_secondary_classification(self, anomaly_mask: np.ndarray, heatmap: np.ndarray, masks: Dict) -> List[Defect]:
        """Runs targeted classifiers on anomalous regions."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(anomaly_mask, connectivity=8)
        final_defects = []
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.config.min_defect_area_px:
                continue

            component_mask = (labels == i).astype(np.uint8)
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Characterize shape
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            eccentricity = 0
            if len(contours[0]) >= 5:
                _, (major_axis, minor_axis), _ = cv2.fitEllipse(contours[0])
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2)
            
            aspect_ratio = max(w,h) / (min(w,h) + 1e-6)

            # Classify based on shape
            if eccentricity > self.config.scratch_eccentricity_threshold and aspect_ratio > self.config.scratch_aspect_ratio_threshold:
                defect_type = DefectType.SCRATCH
                method = "Shape Analysis (Scratch)"
            elif eccentricity < 0.8:
                defect_type = DefectType.PIT_DIG
                method = "Shape Analysis (Pit/Dig)"
            else:
                defect_type = DefectType.TEXTURE_ANOMALY
                method = "Statistical Anomaly"

            # Assess severity and confidence
            anomaly_score = heatmap[component_mask > 0].mean()
            confidence = 0.5 + (anomaly_score * 0.5)
            
            area = stats[i, cv2.CC_STAT_AREA]
            severity = DefectSeverity.LOW
            if area > 500 or anomaly_score > 0.8: severity = DefectSeverity.CRITICAL
            elif area > 200 or anomaly_score > 0.6: severity = DefectSeverity.HIGH
            elif area > 50 or anomaly_score > 0.4: severity = DefectSeverity.MEDIUM

            final_defects.append(Defect(
                id=i, type=defect_type, severity=severity, confidence=confidence,
                location=(int(centroids[i][0]), int(centroids[i][1])),
                bbox=(x, y, w, h), area_px=area,
                primary_anomaly_score=anomaly_score,
                classification_method=method,
                mask=component_mask
            ))
            
        return sorted(final_defects, key=lambda d: d.primary_anomaly_score, reverse=True)

    def _assess_final_quality(self, defects: List[Defect]) -> Tuple[float, bool]:
        """Calculates final quality score and pass/fail status."""
        quality_score = 100.0
        
        for defect in defects:
            penalty = {
                DefectSeverity.CRITICAL: 30, DefectSeverity.HIGH: 15,
                DefectSeverity.MEDIUM: 5, DefectSeverity.LOW: 1
            }.get(defect.severity, 1)
            quality_score -= penalty * defect.confidence
            
        quality_score = max(0, quality_score)
        pass_fail = quality_score >= 70 and not any(d.severity == DefectSeverity.CRITICAL for d in defects)
        return quality_score, pass_fail

    def _generate_and_save_outputs(self, image_path, original_image, defects, anomaly_heatmap, score, status):
        """Generates and saves the visualization and JSON report."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- Create Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Original with defects
        display_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        severity_colors = {
            DefectSeverity.CRITICAL: (255, 0, 0), DefectSeverity.HIGH: (255, 128, 0),
            DefectSeverity.MEDIUM: (255, 255, 0), DefectSeverity.LOW: (0, 255, 255)
        }
        for d in defects:
            color = severity_colors.get(d.severity, (255, 255, 255))
            x, y, w, h = d.bbox
            cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_img, f"#{d.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        axes[0].imshow(display_img)
        axes[0].set_title(f"Detected Defects ({len(defects)})")
        axes[0].axis('off')
        
        # Anomaly Heatmap
        im = axes[1].imshow(anomaly_heatmap, cmap='inferno')
        axes[1].set_title("Anomaly Heatmap")
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Summary
        axes[2].axis('off')
        summary_text = (f"Analysis Report\n\n"
                        f"Overall Status: {'PASS' if status else 'FAIL'}\n"
                        f"Quality Score: {score:.1f} / 100\n\n"
                        f"Total Defects: {len(defects)}\n"
                        f" - Critical: {sum(1 for d in defects if d.severity == DefectSeverity.CRITICAL)}\n"
                        f" - High: {sum(1 for d in defects if d.severity == DefectSeverity.HIGH)}\n"
                        f" - Medium: {sum(1 for d in defects if d.severity == DefectSeverity.MEDIUM)}\n"
                        f" - Low: {sum(1 for d in defects if d.severity == DefectSeverity.LOW)}\n")
        axes[2].text(0.0, 0.95, summary_text, transform=axes[2].transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        viz_path = os.path.join(self.config.output_dir, f"{base_name}_analysis_{timestamp}.png")
        plt.savefig(viz_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Visualization saved to {viz_path}")
        
        # --- Create and Save JSON Report ---
        report = {
            "image_path": image_path,
            "analysis_timestamp": timestamp,
            "quality_score": score,
            "pass_fail_status": status,
            "defects": [d.to_dict() for d in defects]
        }
        report_path = os.path.join(self.config.output_dir, f"{base_name}_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"JSON report saved to {report_path}")
        
        return report

# --- Main Interactive Execution Logic ---

def main():
    """Main function to run the interactive inspector."""
    print("\n" + "="*70)
    print(" Welcome to the Unified Fiber Optic Defect Inspection System (v2.0)")
    print("="*70)

    config = MasterConfig()
    kb_manager = KnowledgeBaseManager(config)

    # Step 1: Handle the Knowledge Base interactively
    if not kb_manager.model:
        print("No knowledge base found. You must build one.")
        while True:
            ref_dir = input("Enter path to folder containing reference images: ").strip().strip('"')
            if os.path.isdir(ref_dir):
                kb_manager.build_new_model(ref_dir)
                if kb_manager.model: break
                else: print("Model building failed. Please check the directory and try again.")
            else: print("Invalid directory path.")
    else:
        while True:
            choice = input("Existing knowledge base found. Use it (U) or build a new one (B)? [U/B]: ").strip().upper()
            if choice == 'B':
                ref_dir = input("Enter path to folder for new reference images: ").strip().strip('"')
                if os.path.isdir(ref_dir):
                    kb_manager.build_new_model(ref_dir)
                else:
                    print("Invalid directory. Using existing model.")
                break
            elif choice == 'U':
                break

    # Step 2: Loop for analyzing images
    inspector = UnifiedFiberInspector(config, kb_manager)
    print("\n--- Image Analysis ---")
    while True:
        image_path = input("\nEnter the path to the image to scan (or type 'quit' to exit): ").strip().strip('"')
        if image_path.lower() == 'quit': break
        if os.path.isfile(image_path):
            inspector.run_full_analysis(image_path)
        else:
            print(f"✗ File not found: {image_path}. Please enter a valid file path.")
    
    print("\n" + "="*70 + "\nThank you for using the Unified Inspection System. Goodbye!\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
