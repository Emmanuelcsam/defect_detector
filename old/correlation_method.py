import os
import sys
import json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats, optimize
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter, binary_fill_holes
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ImageProfile:
    """Stores comprehensive image statistics and classification"""
    filename: str
    classification: str  # 'Textural/Unified' or 'Structured/Outlier'
    mean_grayscale: float
    std_deviation: float
    entropy: float
    spatial_trends: Dict[str, Dict[str, float]]  # horizontal/vertical -> a, b, c coefficients
    feature_vector: Optional[np.ndarray] = None
    
class CorrelationBasedSeparator:
    """
    Advanced segmentation system that uses correlation data, spatial trends,
    and image classification to make intelligent segmentation decisions.
    """
    
    def __init__(self, correlation_data_path: str = ".", knowledge_base_path: str = "segmentation_knowledge.json"):
        self.correlation_data_path = Path(correlation_data_path)
        self.knowledge_base_path = Path(knowledge_base_path)
        
        # Classification thresholds from the analysis
        self.contrast_threshold_low = 35
        self.contrast_threshold_high = 40
        
        # Load correlation matrices and feature data
        self.load_correlation_data()
        self.load_knowledge_base()
        
        # Initialize spatial trend analyzer
        self.spatial_analyzer = SpatialTrendAnalyzer()
        
    def load_correlation_data(self):
        """Load correlation matrices and extracted features"""
        try:
            # Load feature matrix
            features_path = self.correlation_data_path / "extracted_features.csv"
            if features_path.exists():
                self.features_df = pd.read_csv(features_path, index_col=0)
                print(f"✓ Loaded features for {len(self.features_df)} images")
            
            # Load correlation matrices
            self.correlations = {}
            for corr_type in ['pearson', 'spearman']:
                corr_path = self.correlation_data_path / f"correlation_{corr_type}.csv"
                if corr_path.exists():
                    self.correlations[corr_type] = pd.read_csv(corr_path, index_col=0)
                    print(f"✓ Loaded {corr_type} correlation matrix")
                    
        except Exception as e:
            print(f"! Error loading correlation data: {e}")
            self.features_df = None
            self.correlations = {}
    
    def load_knowledge_base(self):
        """Load segmentation knowledge base with successful parameters"""
        self.knowledge_base = {}
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    self.knowledge_base = json.load(f)
                print(f"✓ Loaded knowledge base")
            except Exception as e:
                print(f"! Could not load knowledge base: {e}")
    
    def classify_image(self, image: np.ndarray) -> Tuple[str, ImageProfile]:
        """Classify image based on contrast score and spatial trends"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate basic statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        entropy = self._calculate_entropy(gray)
        
        # Calculate spatial trends
        spatial_trends = self.spatial_analyzer.analyze_trends(gray)
        
        # Primary classification based on contrast score
        if std_val < self.contrast_threshold_low:
            classification = "Textural/Unified"
        elif std_val > self.contrast_threshold_high:
            classification = "Structured/Outlier"
        else:
            # Use secondary classifier: vertical trend coefficient
            vertical_b = spatial_trends['vertical']['b']
            classification = "Textural/Unified" if vertical_b < 0 else "Structured/Outlier"
        
        profile = ImageProfile(
            filename="current_image",
            classification=classification,
            mean_grayscale=mean_val,
            std_deviation=std_val,
            entropy=entropy,
            spatial_trends=spatial_trends
        )
        
        print(f"\nImage Classification: {classification}")
        print(f"  - Contrast Score: {std_val:.2f}")
        print(f"  - Mean Grayscale: {mean_val:.2f}")
        print(f"  - Entropy: {entropy:.4f}")
        
        return classification, profile
    
    def find_similar_images(self, image_profile: ImageProfile, n_similar: int = 5) -> List[Tuple[str, float]]:
        """Find most similar images based on feature correlation"""
        if self.features_df is None or len(self.features_df) == 0:
            return []
        
        # Create feature vector for current image
        current_features = self._extract_features(image_profile)
        
        # Calculate distances to all images in database
        similarities = []
        for idx, row in self.features_df.iterrows():
            # Use subset of features for comparison
            feature_subset = ['mean', 'std_dev', 'entropy', 'fractal_dimension', 
                            'contrast_d1', 'homogeneity_d1', 'energy_d1']
            
            ref_features = row[feature_subset].values
            curr_subset = current_features[feature_subset] if isinstance(current_features, dict) else current_features[:len(feature_subset)]
            
            # Calculate normalized Euclidean distance
            distance = euclidean(
                StandardScaler().fit_transform([curr_subset])[0],
                StandardScaler().fit_transform([ref_features])[0]
            )
            similarities.append((idx, distance))
        
        # Sort by distance and return top N
        similarities.sort(key=lambda x: x[1])
        return similarities[:n_similar]
    
    def get_segmentation_parameters(self, image: np.ndarray, classification: str, 
                                   similar_images: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Determine optimal segmentation parameters based on classification and similar images"""
        
        params = {
            'method_weights': {},
            'preprocessing': {},
            'geometric_constraints': {},
            'confidence_threshold': 0.7
        }
        
        # Set method weights based on classification
        if classification == "Textural/Unified":
            # These images have uniform texture, favor methods that work well with gradual transitions
            params['method_weights'] = {
                'bright_core_extractor': 1.2,
                'unified_core_cladding_detector': 1.1,
                'computational_separation': 1.0,
                'geometric_approach': 0.9,
                'hough_separation': 1.1,
                'adaptive_intensity': 0.8,
                'gradient_approach': 0.7,
                'threshold_separation': 0.6
            }
            params['preprocessing']['blur_kernel'] = 5
            params['preprocessing']['morphology_kernel'] = 3
            
        else:  # Structured/Outlier
            # These images have distinct structures, favor edge-based methods
            params['method_weights'] = {
                'gradient_approach': 1.2,
                'geometric_approach': 1.1,
                'hough_separation': 1.0,
                'computational_separation': 0.9,
                'bright_core_extractor': 0.8,
                'unified_core_cladding_detector': 0.8,
                'adaptive_intensity': 0.7,
                'threshold_separation': 0.6
            }
            params['preprocessing']['blur_kernel'] = 3
            params['preprocessing']['morphology_kernel'] = 5
        
        # Learn from similar images if available
        if similar_images and self.knowledge_base:
            self._adjust_parameters_from_similar(params, similar_images)
        
        return params
    
    def apply_spatial_trend_correction(self, image: np.ndarray, spatial_trends: Dict) -> np.ndarray:
        """Apply spatial trend correction to normalize brightness variations"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Create correction masks based on spatial trends
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Apply inverse of spatial trends to flatten the image
        h_trend = spatial_trends['horizontal']
        v_trend = spatial_trends['vertical']
        
        h_correction = -(h_trend['a'] * x_coords**2 + h_trend['b'] * x_coords)
        v_correction = -(v_trend['a'] * y_coords**2 + v_trend['b'] * y_coords)
        
        # Combine corrections with weights
        total_correction = 0.5 * h_correction + 0.5 * v_correction
        
        # Apply correction
        corrected = np.clip(gray.astype(float) + total_correction, 0, 255).astype(np.uint8)
        
        return corrected
    
    def segment_with_correlation_guidance(self, image: np.ndarray, image_path: str) -> Dict[str, Any]:
        """Main segmentation method using correlation-based guidance"""
        print("\n" + "="*60)
        print("CORRELATION-BASED SEGMENTATION ANALYSIS")
        print("="*60)
        
        # Step 1: Classify the image
        classification, profile = self.classify_image(image)
        
        # Step 2: Find similar images
        similar_images = self.find_similar_images(profile)
        if similar_images:
            print(f"\nFound {len(similar_images)} similar images:")
            for img_name, distance in similar_images[:3]:
                print(f"  - {img_name}: distance = {distance:.3f}")
        
        # Step 3: Get optimal parameters
        params = self.get_segmentation_parameters(image, classification, similar_images)
        
        # Step 4: Apply spatial trend correction
        corrected_image = self.apply_spatial_trend_correction(image, profile.spatial_trends)
        
        # Step 5: Perform advanced segmentation
        result = self._perform_advanced_segmentation(corrected_image, params, profile)
        
        # Step 6: Validate and refine using geometric constraints
        result = self._validate_and_refine(result, image.shape[:2], classification)
        
        return {
            'success': True,
            'classification': classification,
            'masks': result['masks'],
            'center': result['center'],
            'core_radius': result['core_radius'],
            'cladding_radius': result['cladding_radius'],
            'confidence': result['confidence'],
            'spatial_trends': profile.spatial_trends,
            'similar_images': similar_images[:3] if similar_images else []
        }
    
    def _perform_advanced_segmentation(self, image: np.ndarray, params: Dict, 
                                     profile: ImageProfile) -> Dict[str, Any]:
        """Perform segmentation using advanced techniques based on image profile"""
        h, w = image.shape[:2]
        
        if profile.classification == "Textural/Unified":
            # Use gradient-based region growing for smooth textures
            result = self._segment_textural_unified(image, params)
        else:
            # Use edge detection and watershed for structured images
            result = self._segment_structured_outlier(image, params)
        
        return result
    
    def _segment_textural_unified(self, image: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Segmentation strategy for textural/unified images"""
        h, w = image.shape[:2]
        
        # Apply preprocessing
        blurred = cv2.GaussianBlur(image, (params['preprocessing']['blur_kernel'],) * 2, 0)
        
        # Find fiber center using weighted centroid
        center = self._find_fiber_center_weighted(blurred)
        
        # Use radial intensity profiling
        radial_profile = self._compute_radial_profile(blurred, center)
        
        # Find transitions using gradient analysis
        gradients = np.gradient(gaussian_filter(radial_profile, sigma=2))
        
        # Find core-cladding boundary (first significant negative gradient)
        core_radius = self._find_transition_radius(gradients, mode='negative')
        
        # Find cladding-ferrule boundary
        cladding_radius = self._find_transition_radius(
            gradients[int(core_radius):], mode='negative', offset=core_radius
        )
        
        # Create masks
        masks = self._create_circular_masks(center, core_radius, cladding_radius, (h, w))
        
        return {
            'masks': masks,
            'center': center,
            'core_radius': core_radius,
            'cladding_radius': cladding_radius,
            'confidence': 0.85
        }
    
    def _segment_structured_outlier(self, image: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Segmentation strategy for structured/outlier images"""
        h, w = image.shape[:2]
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find circles using Hough transform
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=int(min(h, w) * 0.4)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort by radius to find core and cladding
            circles = sorted(circles, key=lambda x: x[2])
            
            if len(circles) >= 2:
                center = (int(np.mean([c[0] for c in circles])), 
                         int(np.mean([c[1] for c in circles])))
                core_radius = circles[0][2]
                cladding_radius = circles[-1][2]
            else:
                # Fallback to single circle detection
                center = (circles[0][0], circles[0][1])
                core_radius = circles[0][2] * 0.3
                cladding_radius = circles[0][2]
        else:
            # Fallback to centroid method
            center = self._find_fiber_center_weighted(image)
            core_radius = min(h, w) * 0.1
            cladding_radius = min(h, w) * 0.3
        
        masks = self._create_circular_masks(center, core_radius, cladding_radius, (h, w))
        
        return {
            'masks': masks,
            'center': center,
            'core_radius': core_radius,
            'cladding_radius': cladding_radius,
            'confidence': 0.75
        }
    
    def _validate_and_refine(self, result: Dict, image_shape: Tuple[int, int], 
                           classification: str) -> Dict[str, Any]:
        """Validate and refine segmentation results"""
        h, w = image_shape
        
        # Geometric constraints
        min_core_ratio = 0.05
        max_core_ratio = 0.25
        min_cladding_ratio = 0.15
        max_cladding_ratio = 0.45
        
        # Adjust constraints based on classification
        if classification == "Structured/Outlier":
            max_core_ratio = 0.35
            max_cladding_ratio = 0.55
        
        # Validate radii
        max_dim = min(h, w)
        core_ratio = result['core_radius'] / max_dim
        cladding_ratio = result['cladding_radius'] / max_dim
        
        if core_ratio < min_core_ratio or core_ratio > max_core_ratio:
            result['core_radius'] = max_dim * 0.15
            result['confidence'] *= 0.8
            
        if cladding_ratio < min_cladding_ratio or cladding_ratio > max_cladding_ratio:
            result['cladding_radius'] = max_dim * 0.35
            result['confidence'] *= 0.8
        
        # Ensure core < cladding
        if result['core_radius'] >= result['cladding_radius']:
            result['cladding_radius'] = result['core_radius'] * 2.5
        
        # Recreate masks with validated parameters
        result['masks'] = self._create_circular_masks(
            result['center'], result['core_radius'], result['cladding_radius'], (h, w)
        )
        
        return result
    
    # Helper methods
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate Shannon entropy of image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _extract_features(self, profile: ImageProfile) -> Dict[str, float]:
        """Extract feature vector from image profile"""
        features = {
            'mean': profile.mean_grayscale,
            'std_dev': profile.std_deviation,
            'entropy': profile.entropy,
            'fractal_dimension': 1.5,  # Placeholder
            'contrast_d1': profile.std_deviation * 0.8,
            'homogeneity_d1': 1.0 / (1.0 + profile.std_deviation * 0.1),
            'energy_d1': 1.0 / (1.0 + profile.entropy)
        }
        return features
    
    def _adjust_parameters_from_similar(self, params: Dict, similar_images: List):
        """Adjust parameters based on successful segmentations of similar images"""
        if not self.knowledge_base:
            return
            
        # Average successful parameters from similar images
        for img_name, _ in similar_images[:3]:
            if img_name in self.knowledge_base:
                kb_entry = self.knowledge_base[img_name]
                if 'method_scores' in kb_entry:
                    for method, score in kb_entry['method_scores'].items():
                        if method in params['method_weights']:
                            # Weighted average
                            params['method_weights'][method] = (
                                params['method_weights'][method] * 0.7 + score * 0.3
                            )
    
    def _find_fiber_center_weighted(self, image: np.ndarray) -> Tuple[int, int]:
        """Find fiber center using intensity-weighted centroid"""
        h, w = image.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Use image as weights
        total_weight = np.sum(image)
        cx = int(np.sum(x_coords * image) / total_weight)
        cy = int(np.sum(y_coords * image) / total_weight)
        
        return (cx, cy)
    
    def _compute_radial_profile(self, image: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Compute radial intensity profile from center"""
        h, w = image.shape[:2]
        cx, cy = center
        
        # Maximum radius
        max_radius = int(np.sqrt((h/2)**2 + (w/2)**2))
        
        # Compute distances
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        # Bin the distances and compute mean intensity
        profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)
        
        for r in range(max_radius):
            mask = (distances >= r) & (distances < r + 1)
            if np.any(mask):
                profile[r] = np.mean(image[mask])
                counts[r] = np.sum(mask)
        
        # Smooth the profile
        valid = counts > 0
        profile[valid] = gaussian_filter(profile[valid], sigma=2)
        
        return profile
    
    def _find_transition_radius(self, gradients: np.ndarray, mode: str = 'negative', 
                              offset: int = 0) -> float:
        """Find transition radius based on gradient analysis"""
        if mode == 'negative':
            # Find first significant negative gradient
            threshold = np.std(gradients) * -0.5
            candidates = np.where(gradients < threshold)[0]
        else:
            # Find first significant positive gradient
            threshold = np.std(gradients) * 0.5
            candidates = np.where(gradients > threshold)[0]
        
        if len(candidates) > 0:
            return candidates[0] + offset
        else:
            return len(gradients) // 2 + offset
    
    def _create_circular_masks(self, center: Tuple[int, int], core_radius: float,
                             cladding_radius: float, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Create circular masks for core, cladding, and ferrule"""
        h, w = shape
        cx, cy = center
        
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        core_mask = (dist_from_center <= core_radius).astype(np.uint8)
        cladding_mask = ((dist_from_center > core_radius) & 
                        (dist_from_center <= cladding_radius)).astype(np.uint8)
        ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8)
        
        # Fill holes
        core_mask = binary_fill_holes(core_mask).astype(np.uint8)
        
        return {
            'core': core_mask,
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        }


class SpatialTrendAnalyzer:
    """Analyzes spatial brightness trends in images"""
    
    def analyze_trends(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Fit quadratic models to horizontal and vertical brightness profiles"""
        h, w = image.shape[:2]
        
        # Compute average profiles
        h_profile = np.mean(image, axis=0)
        v_profile = np.mean(image, axis=1)
        
        # Fit quadratic models
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        
        h_coeffs = np.polyfit(x_coords, h_profile, 2)
        v_coeffs = np.polyfit(y_coords, v_profile, 2)
        
        return {
            'horizontal': {'a': h_coeffs[0], 'b': h_coeffs[1], 'c': h_coeffs[2]},
            'vertical': {'a': v_coeffs[0], 'b': v_coeffs[1], 'c': v_coeffs[2]}
        }


def main():
    """Integration point with the main separation.py system"""
    if len(sys.argv) < 3:
        print("Usage: python correlation_based_separation.py <image_path> <output_dir>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the separator
    separator = CorrelationBasedSeparator()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    # Perform segmentation
    result = separator.segment_with_correlation_guidance(image, image_path)
    
    # Save results
    output_path = os.path.join(output_dir, "correlation_based_result.json")
    
    # Convert numpy arrays to lists for JSON serialization
    result_json = {
        'success': result['success'],
        'classification': result['classification'],
        'center': result['center'],
        'core_radius': float(result['core_radius']),
        'cladding_radius': float(result['cladding_radius']),
        'confidence': float(result['confidence']),
        'spatial_trends': result['spatial_trends'],
        'similar_images': result['similar_images']
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_json, f, indent=4)
    
    # Save masks
    if result['success']:
        cv2.imwrite(os.path.join(output_dir, "correlation_core_mask.png"), 
                   result['masks']['core'] * 255)
        cv2.imwrite(os.path.join(output_dir, "correlation_cladding_mask.png"), 
                   result['masks']['cladding'] * 255)
        cv2.imwrite(os.path.join(output_dir, "correlation_ferrule_mask.png"), 
                   result['masks']['ferrule'] * 255)
    
    print(f"\n✓ Results saved to {output_dir}")
    return result


if __name__ == "__main__":
    main()
