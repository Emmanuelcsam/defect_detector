#!/usr/bin/env python3
"""
Correlation method
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, binary_fill_holes
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore')

class IntelligentSegmenter:
    """
    An advanced segmentation system that uses correlation data, spatial trends,
    and image classification to make intelligent segmentation decisions for
    fiber optic images.
    """

    def __init__(self, data_path: str = ".", knowledge_base_name: str = "segmentation_knowledge.json"):
        """
        Initializes the IntelligentSegmenter.

        Args:
            data_path (str): Path to the directory containing correlation and feature files.
            knowledge_base_name (str): Filename of the JSON knowledge base.
        """
        self.data_path = Path(data_path)
        self.knowledge_base_path = self.data_path / knowledge_base_name

        # --- Configuration ---
        # File names for data loading
        self.feature_file = "extracted_features.csv"
        self.correlation_files = {
            'pearson': 'correlation_pearson.csv',
            'spearman': 'correlation_spearman.csv',
            'mutual_info': 'mutual_information.csv'
        }

        # Image classification thresholds
        self.contrast_threshold_low = 35
        self.contrast_threshold_high = 40

        # Features used for finding similar images
        self.similarity_features = [
            'mean', 'std_dev', 'entropy', 'fractal_dimension',
            'contrast_d1', 'homogeneity_d1', 'energy_d1'
        ]

        # --- Data Storage ---
        self.features_df: Optional[pd.DataFrame] = None
        self.correlations: Dict[str, pd.DataFrame] = {}
        self.knowledge_base: Dict[str, Any] = {}

        # --- Load Initial Data ---
        self._load_data()
        self._load_knowledge_base()

    def _load_data(self):
        """
        Loads correlation matrices and extracted features from disk.
        Combines loading logic from both original scripts.
        """
        print("[1/5] Loading correlation and feature data...")
        # Load feature matrix
        try:
            features_path = self.data_path / self.feature_file
            if features_path.exists():
                self.features_df = pd.read_csv(features_path, index_col=0)
                print(f"  ✓ Loaded features for {len(self.features_df)} images.")
        except Exception as e:
            print(f"  ! Warning: Could not load feature file: {e}")
            self.features_df = None

        # Load correlation matrices
        for corr_type, filename in self.correlation_files.items():
            try:
                corr_path = self.data_path / filename
                if corr_path.exists():
                    self.correlations[corr_type] = pd.read_csv(corr_path, index_col=0)
                    print(f"  ✓ Loaded {corr_type} correlation matrix.")
            except Exception as e:
                print(f"  ! Warning: Could not load {corr_type} correlation file: {e}")

    def _load_knowledge_base(self):
        """Loads the segmentation knowledge base with successful parameters."""
        print("[2/5] Loading knowledge base...")
        try:
            if self.knowledge_base_path.exists():
                with open(self.knowledge_base_path, 'r') as f:
                    self.knowledge_base = json.load(f)
                print("  ✓ Knowledge base loaded successfully.")
            else:
                print("  ! Knowledge base not found. Proceeding with default parameters.")
        except Exception as e:
            print(f"! Error loading knowledge base: {e}")
            self.knowledge_base = {}

    def classify_image(self, image: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        Classifies the image based on contrast and spatial trends, and generates a profile.

        Args:
            image (np.ndarray): The input image (can be color or grayscale).

        Returns:
            A tuple containing the classification string and the image profile dictionary.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Calculate basic statistics
        mean_val = float(np.mean(gray_image))
        std_val = float(np.std(gray_image))
        entropy = self._calculate_entropy(gray_image)

        # Analyze spatial trends
        spatial_trends = self._analyze_spatial_trends(gray_image)

        # Primary classification based on contrast (std deviation)
        if std_val < self.contrast_threshold_low:
            classification = "Textural/Unified"
        elif std_val > self.contrast_threshold_high:
            classification = "Structured/Outlier"
        else:
            # Secondary classification: use vertical trend for tie-breaking
            vertical_b = spatial_trends['vertical']['b']
            classification = "Textural/Unified" if vertical_b < 0 else "Structured/Outlier"

        profile = {
            'classification': classification,
            'mean_grayscale': mean_val,
            'std_deviation': std_val,
            'entropy': entropy,
            'spatial_trends': spatial_trends
        }

        print("\n--- Image Classification ---")
        print(f"  - Classification: {classification}")
        print(f"  - Contrast Score (Std Dev): {std_val:.2f}")
        print(f"  - Mean Grayscale: {mean_val:.2f}")
        print(f"  - Entropy: {entropy:.4f}")
        
        return classification, profile

    def find_similar_images(self, image_profile: Dict[str, Any], n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        Finds the most similar images from the database using feature correlation.

        Args:
            image_profile (Dict): The profile of the current image.
            n_similar (int): The number of similar images to return.

        Returns:
            A list of tuples, each containing an image name and its distance score.
        """
        if self.features_df is None or self.features_df.empty:
            print("  ! No feature database available to find similar images.")
            return []

        current_features = self._extract_feature_vector(image_profile)
        
        # Ensure the feature vector is a DataFrame-like structure for consistent access
        current_features_s = pd.Series(current_features)

        # Calculate normalized Euclidean distance to all images in the database
        similarities = []
        for img_name, db_row in self.features_df.iterrows():
            # Use a consistent subset of features for comparison
            db_features = db_row[self.similarity_features].values
            curr_subset = current_features_s[self.similarity_features].values
            
            distance = euclidean(
                StandardScaler().fit_transform([curr_subset])[0],
                StandardScaler().fit_transform([db_features])[0]
            )
            similarities.append((img_name, distance))

        similarities.sort(key=lambda x: x[1])
        return similarities[:n_similar]

    def get_segmentation_parameters(self, classification: str, similar_images: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Determines optimal segmentation parameters based on classification and similar images.

        Args:
            classification (str): The image classification.
            similar_images (List): A list of similar images found in the database.

        Returns:
            A dictionary of optimized parameters for segmentation.
        """
        # Start with default parameters based on classification
        if classification == "Textural/Unified":
            params = {
                'segmentation_method': self._segment_textural_unified,
                'preprocessing': {'blur_kernel': 5},
            }
        else:  # Structured/Outlier
            params = {
                'segmentation_method': self._segment_structured_outlier,
                'preprocessing': {'blur_kernel': 3},
            }
        
        # Learn from similar images if available in the knowledge base
        if similar_images and self.knowledge_base:
            # Placeholder for potential adjustments. The original logic was to adjust
            # method_weights, but the new structure directly selects a method.
            # This could be extended to adjust preprocessing parameters.
            print("  - Analyzing knowledge base for similar images...")
            # Example: self._adjust_parameters_from_similar(params, similar_images)
        
        return params
        
    def apply_spatial_trend_correction(self, image: np.ndarray, spatial_trends: Dict) -> np.ndarray:
        """
        Applies spatial trend correction to normalize brightness variations.

        Args:
            image (np.ndarray): The grayscale image to correct.
            spatial_trends (Dict): The dictionary of trend coefficients.

        Returns:
            The corrected grayscale image.
        """
        h, w = image.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Calculate inverse correction map from horizontal and vertical trends
        h_trend = spatial_trends['horizontal']
        v_trend = spatial_trends['vertical']
        h_correction = -(h_trend['a'] * x_coords**2 + h_trend['b'] * x_coords)
        v_correction = -(v_trend['a'] * y_coords**2 + v_trend['b'] * y_coords)

        # Combine corrections and apply to the image
        total_correction = 0.5 * h_correction + 0.5 * v_correction
        corrected_image = np.clip(image.astype(float) + total_correction, 0, 255).astype(np.uint8)

        return corrected_image

    def segment_image(self, image: np.ndarray, image_name: str) -> Dict[str, Any]:
        """
        Main segmentation pipeline that uses correlation-based guidance.

        Args:
            image (np.ndarray): The input image.
            image_name (str): The name or path of the image being processed.

        Returns:
            A dictionary containing the full segmentation result.
        """
        print("\n" + "="*60)
        print(f"STARTING INTELLIGENT SEGMENTATION FOR: {os.path.basename(image_name)}")
        print("="*60)

        # Step 1: Classify the image and create its profile
        classification, profile = self.classify_image(image)

        # Step 2: Find similar images in the database
        print("\n[3/5] Finding similar images...")
        similar_images = self.find_similar_images(profile)
        if similar_images:
            print(f"  ✓ Found {len(similar_images)} similar images. Top 3:")
            for img_name, dist in similar_images[:3]:
                print(f"    - {img_name} (Distance: {dist:.3f})")
        
        # Step 3: Get optimal segmentation parameters
        print("\n[4/5] Determining optimal parameters...")
        params = self.get_segmentation_parameters(classification, similar_images)
        print(f"  ✓ Selected segmentation method: {params['segmentation_method'].__name__}")

        # Step 4: Preprocess the image (apply correction and blur)
        print("\n[5/5] Performing segmentation...")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corrected_image = self.apply_spatial_trend_correction(gray_image, profile['spatial_trends'])
        blurred_image = cv2.GaussianBlur(
            corrected_image,
            (params['preprocessing']['blur_kernel'],) * 2, 0
        )
        print("  - Applied spatial trend correction and Gaussian blur.")

        # Step 5: Perform segmentation using the selected method
        result = params['segmentation_method'](blurred_image, params)
        print("  - Executed core segmentation algorithm.")
        
        # Step 6: Validate and refine results using geometric constraints
        result = self._validate_and_refine(result, image.shape[:2], classification)
        print("  - Validated and refined segmentation geometry.")
        
        return {
            'success': True,
            'classification': classification,
            'masks': result['masks'],
            'center': result['center'],
            'core_radius': result['core_radius'],
            'cladding_radius': result['cladding_radius'],
            'confidence': result['confidence'],
            'spatial_trends': profile['spatial_trends'],
            'similar_images': similar_images[:3]
        }

    # --- Segmentation Strategies ---

    def _segment_textural_unified(self, image: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Segmentation strategy for textural/unified images using radial profiling."""
        h, w = image.shape
        center = self._find_fiber_center_weighted(image)
        
        # Analyze radial intensity profile to find transitions
        radial_profile = self._compute_radial_profile(image, center)
        gradients = np.gradient(gaussian_filter(radial_profile, sigma=2))
        
        # Find core-cladding boundary (first significant negative gradient)
        core_radius = self._find_transition_radius(gradients, mode='negative')
        
        # Find cladding-ferrule boundary (next significant negative gradient)
        cladding_radius = self._find_transition_radius(
            gradients[int(core_radius):], mode='negative', offset=core_radius
        )
        
        masks = self._create_circular_masks((h, w), center, core_radius, cladding_radius)
        return {'masks': masks, 'center': center, 'core_radius': core_radius, 'cladding_radius': cladding_radius, 'confidence': 0.85}

    def _segment_structured_outlier(self, image: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Segmentation strategy for structured/outlier images using Hough transform."""
        h, w = image.shape
        
        # Use Hough Circle Transform to find distinct circular features
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=int(min(h, w) * 0.45)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort circles by radius (smallest is likely core, largest is cladding)
            circles = sorted(circles, key=lambda x: x[2])
            
            center = (int(np.mean([c[0] for c in circles])), int(np.mean([c[1] for c in circles])))
            core_radius = circles[0][2]
            cladding_radius = circles[-1][2]
        else:
            # Fallback if no circles are found
            center = self._find_fiber_center_weighted(image)
            core_radius = min(h, w) * 0.1  # Estimate
            cladding_radius = min(h, w) * 0.35 # Estimate
        
        masks = self._create_circular_masks((h, w), center, core_radius, cladding_radius)
        return {'masks': masks, 'center': center, 'core_radius': core_radius, 'cladding_radius': cladding_radius, 'confidence': 0.75}

    def _validate_and_refine(self, result: Dict, shape: Tuple[int, int], classification: str) -> Dict[str, Any]:
        """Validates and refines segmentation results based on geometric constraints."""
        h, w = shape
        max_dim = min(h, w)
        
        # Define base geometric constraints
        constraints = {
            'core_ratio': (0.05, 0.25),
            'cladding_ratio': (0.15, 0.45),
            'defaults': {'core': max_dim * 0.15, 'cladding': max_dim * 0.35}
        }
        
        # Adjust constraints for structured images which may have larger features
        if classification == "Structured/Outlier":
            constraints['core_ratio'] = (0.05, 0.35)
            constraints['cladding_ratio'] = (0.15, 0.55)

        # Validate core radius
        core_ratio = result['core_radius'] / max_dim
        if not (constraints['core_ratio'][0] < core_ratio < constraints['core_ratio'][1]):
            result['core_radius'] = constraints['defaults']['core']
            result['confidence'] *= 0.8
            
        # Validate cladding radius
        cladding_ratio = result['cladding_radius'] / max_dim
        if not (constraints['cladding_ratio'][0] < cladding_ratio < constraints['cladding_ratio'][1]):
            result['cladding_radius'] = constraints['defaults']['cladding']
            result['confidence'] *= 0.8
        
        # Ensure core radius is smaller than cladding radius
        if result['core_radius'] >= result['cladding_radius']:
            result['cladding_radius'] = result['core_radius'] * 2.5
        
        # Recreate masks with validated parameters
        result['masks'] = self._create_circular_masks(shape, result['center'], result['core_radius'], result['cladding_radius'])
        return result

    # --- Helper Methods ---

    @staticmethod
    def _analyze_spatial_trends(image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Fits quadratic models to horizontal and vertical brightness profiles."""
        h, w = image.shape
        h_profile, v_profile = np.mean(image, axis=0), np.mean(image, axis=1)
        
        h_coeffs = np.polyfit(np.arange(w), h_profile, 2)
        v_coeffs = np.polyfit(np.arange(h), v_profile, 2)
        
        return {
            'horizontal': {'a': h_coeffs[0], 'b': h_coeffs[1], 'c': h_coeffs[2]},
            'vertical': {'a': v_coeffs[0], 'b': v_coeffs[1], 'c': v_coeffs[2]}
        }

    @staticmethod
    def _calculate_entropy(image: np.ndarray) -> float:
        """Calculates the Shannon entropy of a grayscale image."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist /= hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _extract_feature_vector(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Creates a feature vector from an image profile."""
        return {
            'mean': profile['mean_grayscale'],
            'std_dev': profile['std_deviation'],
            'entropy': profile['entropy'],
            'fractal_dimension': 1.5,  # Placeholder: requires specific implementation
            'contrast_d1': profile['std_deviation'] * 0.8,
            'homogeneity_d1': 1.0 / (1.0 + profile['std_deviation'] * 0.1),
            'energy_d1': 1.0 / (1.0 + profile['entropy'])
        }

    @staticmethod
    def _find_fiber_center_weighted(image: np.ndarray) -> Tuple[int, int]:
        """Finds the fiber center using an intensity-weighted centroid."""
        h, w = image.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        total_weight = np.sum(image)
        if total_weight == 0: return (w // 2, h // 2) # Avoid division by zero
        
        cy = int(np.sum(y_coords * image) / total_weight)
        cx = int(np.sum(x_coords * image) / total_weight)
        return (cx, cy)

    @staticmethod
    def _compute_radial_profile(image: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Computes the radial intensity profile from the center out."""
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        max_radius = int(dist_from_center.max())
        profile = np.zeros(max_radius)
        for r in range(max_radius):
            mask = (dist_from_center >= r) & (dist_from_center < r + 1)
            if np.any(mask):
                profile[r] = np.mean(image[mask])
        
        return gaussian_filter(profile, sigma=2)

    @staticmethod
    def _find_transition_radius(gradients: np.ndarray, mode: str, offset: int = 0) -> float:
        """Finds a transition radius from a gradient profile."""
        if gradients.size == 0: return offset # No gradient to analyze
        
        if mode == 'negative':
            threshold = np.mean(gradients) - 0.5 * np.std(gradients)
            candidates = np.where(gradients < threshold)[0]
        else: # 'positive'
            threshold = np.mean(gradients) + 0.5 * np.std(gradients)
            candidates = np.where(gradients > threshold)[0]
        
        # Return the first candidate found, or a fallback
        return (candidates[0] + offset) if len(candidates) > 0 else (len(gradients) // 2 + offset)

    @staticmethod
    def _create_circular_masks(shape: Tuple[int, int], center: Tuple[int, int], 
                              core_r: float, clad_r: float) -> Dict[str, np.ndarray]:
        """Creates circular binary masks for core, cladding, and ferrule."""
        h, w = shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        core_mask = (dist <= core_r).astype(np.uint8)
        cladding_mask = ((dist > core_r) & (dist <= clad_r)).astype(np.uint8)
        ferrule_mask = (dist > clad_r).astype(np.uint8)
        
        return {
            'core': binary_fill_holes(core_mask).astype(np.uint8),
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        }




def main():
    """
    Main execution function.
    Parses command-line arguments, runs the segmentation, and saves the results.
    """
    if len(sys.argv) < 3:
        print("Usage: python intelligent_segmenter.py <image_path> <output_dir>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        sys.exit(1)
        
    # Initialize and run the segmenter
    # Assumes data files (correlations, etc.) are in the current directory
    segmenter = IntelligentSegmenter(data_path=".")
    result = segmenter.segment_image(image, image_path)
    
    # --- Save Results ---
    # Prepare result for JSON serialization (convert numpy types to native Python types)
    result_json = {
        'success': result['success'],
        'classification': result['classification'],
        'center': (int(result['center'][0]), int(result['center'][1])),
        'core_radius': float(result['core_radius']),
        'cladding_radius': float(result['cladding_radius']),
        'confidence': float(result['confidence']),
        'spatial_trends': result['spatial_trends'],
        'similar_images': result['similar_images']
    }
    
    json_path = output_dir / "segmentation_result.json"
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=4)
    
    # Save masks as images
    if result['success']:
        cv2.imwrite(str(output_dir / "mask_core.png"), result['masks']['core'] * 255)
        cv2.imwrite(str(output_dir / "mask_cladding.png"), result['masks']['cladding'] * 255)
        cv2.imwrite(str(output_dir / "mask_ferrule.png"), result['masks']['ferrule'] * 255)
    
    print("\n" + "="*60)
    print(f"✓ Segmentation complete. Results saved to '{output_dir}'.")
    print("="*60)


def run_intelligent_segmentation(image_path: str, output_dir: str) -> Dict[str, Any]:
    """
    A wrapper function to make the IntelligentSegmenter compatible with the
    UnifiedSegmentationSystem in separation.py.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {'success': False, 'error': f'Could not load image: {image_path}'}

    # Initialize the segmenter. Assumes data files are in the current working directory.
    segmenter = IntelligentSegmenter(data_path=".")
    result = segmenter.segment_image(image, image_path)

    # The result from segment_image already contains all necessary keys.
    # Convert numpy types for clean JSON output, as the main script would.
    result_json = {
        'success': result.get('success', False),
        'center': result.get('center'),
        'core_radius': result.get('core_radius'),
        'cladding_radius': result.get('cladding_radius'),
        'confidence': result.get('confidence', 0.5),
        'error': result.get('error')
    }
    return result_json

if __name__ == "__main__":
    main()
