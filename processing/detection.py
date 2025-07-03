#!/usr/bin/env python3

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import time

# Configure logging system to display timestamps, log levels, and messages
logging.basicConfig(
    level=logging.INFO,  # Set minimum log level to INFO (won't show DEBUG messages)
    format='%(asctime)s - [%(levelname)s] - %(message)s'  # Define log message format with timestamp
)


@dataclass  # Decorator to automatically generate __init__ and other methods
class OmniConfig:
    """Configuration for OmniFiberAnalyzer - matches expected structure from app.py"""
    # Path to saved knowledge base file containing reference model data (None means use default)
    knowledge_base_path: Optional[str] = None
    # Minimum pixel area for a region to be considered a defect (filters noise)
    min_defect_size: int = 10
    # Maximum pixel area for a defect (larger areas might be image artifacts)
    max_defect_size: int = 5000
    # Dictionary mapping severity levels to confidence thresholds for classification
    severity_thresholds: Optional[Dict[str, float]] = None
    # Minimum confidence score (0-1) to report a detected anomaly
    confidence_threshold: float = 0.3
    # Multiplier for standard deviation to set anomaly detection threshold
    anomaly_threshold_multiplier: float = 2.5
    # Whether to generate and save visualization images
    enable_visualization: bool = True
    
    def __post_init__(self):
        # Initialize default severity thresholds if none provided
        if self.severity_thresholds is None:
            # Create mapping from severity levels to minimum confidence scores
            self.severity_thresholds = {
                'CRITICAL': 0.9,  # 90%+ confidence = critical defect
                'HIGH': 0.7,      # 70-89% = high severity
                'MEDIUM': 0.5,    # 50-69% = medium severity
                'LOW': 0.3,       # 30-49% = low severity
                'NEGLIGIBLE': 0.1 # 10-29% = negligible
            }


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        # Convert numpy integer types to Python int for JSON compatibility
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert numpy float types to Python float for JSON compatibility
        if isinstance(obj, np.floating):
            return float(obj)
        # Convert numpy arrays to Python lists for JSON compatibility
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Fall back to default JSON encoder for other types
        return super(NumpyEncoder, self).default(obj)


class OmniFiberAnalyzer:
    """The ultimate fiber optic anomaly detection system - pipeline compatible version."""
    
    def __init__(self, config: OmniConfig):
        # Store configuration object containing all analysis parameters
        self.config = config
        # Set knowledge base path, defaulting to "fiber_anomaly_kb.json" if not specified
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        # Initialize empty reference model structure for storing learned patterns
        self.reference_model = {
            'features': [],              # List of feature dictionaries from reference images
            'statistical_model': None,   # Statistical parameters (mean, std, covariance)
            'archetype_image': None,     # Median image representing typical fiber
            'feature_names': [],         # List of feature names in consistent order
            'comparison_results': {},    # Cached comparison results
            'learned_thresholds': {},    # Learned anomaly detection thresholds
            'timestamp': None           # When model was created/updated
        }
        # Initialize metadata storage for current image being processed
        self.current_metadata = None
        # Create logger instance for this class
        self.logger = logging.getLogger(__name__)
        # Attempt to load existing knowledge base from disk
        self.load_knowledge_base()
        
    def analyze_end_face(self, image_path: str, output_dir: str):
        """Main analysis method - compatible with pipeline expectations"""
        # Log start of analysis for debugging
        self.logger.info(f"Analyzing fiber end face: {image_path}")
        
        # Create Path object for easier directory manipulation
        output_path = Path(output_dir)
        # Create output directory and any missing parent directories
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if reference model exists (needed for comparison)
        if not self.reference_model.get('statistical_model'):
            # Warn that no reference exists and create minimal one
            self.logger.warning("No reference model available. Building from single image...")
            # Build minimal reference using current image as sole reference
            self._build_minimal_reference(image_path)
        
        # Run comprehensive anomaly detection analysis
        results = self.detect_anomalies_comprehensive(image_path)
        
        # Check if analysis succeeded
        if results:
            # Convert internal results format to pipeline-expected format
            pipeline_report = self._convert_to_pipeline_format(results, image_path)
            
            # Construct path for JSON report file
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            # Open file for writing
            with open(report_path, 'w') as f:
                # Write JSON with indentation and custom numpy encoder
                json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
            # Log successful save
            self.logger.info(f"Saved detection report to {report_path}")
            
            # Generate visualizations if enabled in config
            if self.config.enable_visualization:
                # Construct path for visualization image
                viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
                # Generate and save comprehensive visualization
                self.visualize_comprehensive_results(results, str(viz_path))
                
                # Construct path for defect mask file
                mask_path = output_path / f"{Path(image_path).stem}_defect_mask.npy"
                # Create binary mask showing defect locations
                defect_mask = self._create_defect_mask(results)
                # Save mask as numpy array for later processing
                np.save(mask_path, defect_mask)
            
            # Construct path for detailed text report
            text_report_path = output_path / f"{Path(image_path).stem}_detailed.txt"
            # Generate human-readable text report
            self.generate_detailed_report(results, str(text_report_path))
            
            # Return the pipeline report
            return pipeline_report
            
        else:
            # Log analysis failure
            self.logger.error(f"Analysis failed for {image_path}")
            # Create minimal error report structure
            empty_report = {
                'image_path': image_path,
                'timestamp': self._get_timestamp(),
                'success': False,
                'error': 'Analysis failed',
                'defects': []
            }
            # Save error report
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            with open(report_path, 'w') as f:
                json.dump(empty_report, f, indent=2)
            
            # Return the empty report
            return empty_report
    
    def _convert_to_pipeline_format(self, results: Dict, image_path: str) -> Dict:
        """Convert internal results format to pipeline-expected format"""
        # Initialize empty list to store formatted defects
        defects = []
        # Initialize defect ID counter
        defect_id = 1
        
        # Process each detected anomaly region
        for region in results['local_analysis']['anomaly_regions']:
            # Extract bounding box coordinates (x, y, width, height)
            x, y, w, h = region['bbox']
            # Extract centroid coordinates
            cx, cy = region['centroid']
            
            # Get confidence score for this region
            confidence = region['confidence']
            # Convert confidence to severity level using thresholds
            severity = self._confidence_to_severity(confidence)
            
            # Create defect dictionary in pipeline format
            defect = {
                'defect_id': f"ANOM_{defect_id:04d}",  # Format: ANOM_0001, ANOM_0002, etc.
                'defect_type': 'ANOMALY',               # Generic anomaly type
                'location_xy': [cx, cy],                # Center point coordinates
                'bbox': [x, y, w, h],                   # Bounding box
                'area_px': region['area'],              # Area in pixels
                'confidence': float(confidence),        # Ensure float type for JSON
                'severity': severity,                   # Calculated severity level
                'orientation': None,                    # No orientation for generic anomalies
                'contributing_algorithms': ['ultra_comprehensive_matrix_analyzer'],  # Algorithm name
                'detection_metadata': {                 # Additional detection details
                    'max_intensity': region.get('max_intensity', 0),  # Peak anomaly value
                    'anomaly_score': float(confidence)   # Redundant but expected by pipeline
                }
            }
            # Add to defects list
            defects.append(defect)
            # Increment ID counter
            defect_id += 1
        
        # Extract specific defect types from results
        specific_defects = results['specific_defects']
        
        # Process detected scratches (linear defects)
        for scratch in specific_defects['scratches']:
            # Extract line endpoints
            x1, y1, x2, y2 = scratch['line']
            # Calculate center point of line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Create scratch defect entry
            defect = {
                'defect_id': f"SCR_{defect_id:04d}",   # Format: SCR_0001, etc.
                'defect_type': 'SCRATCH',               # Specific type
                'location_xy': [cx, cy],                # Center of scratch
                'bbox': [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)],  # Bounding box
                'area_px': int(scratch['length'] * 2),  # Approximate area (length * assumed width)
                'confidence': 0.7,                      # Fixed confidence for scratches
                'severity': 'MEDIUM' if scratch['length'] > 50 else 'LOW',  # Length-based severity
                'orientation': float(scratch['angle']), # Angle in degrees
                'contributing_algorithms': ['hough_line_detection'],  # Detection method
                'detection_metadata': {
                    'length': float(scratch['length']),     # Scratch length in pixels
                    'angle_degrees': float(scratch['angle']) # Redundant angle info
                }
            }
            defects.append(defect)
            defect_id += 1
        
        # Process detected digs/pits (small dark spots)
        for dig in specific_defects['digs']:
            # Extract center coordinates
            cx, cy = dig['center']
            # Calculate approximate radius from area (A = πr²)
            radius = int(np.sqrt(dig['area'] / np.pi))
            
            # Create dig defect entry
            defect = {
                'defect_id': f"DIG_{defect_id:04d}",   # Format: DIG_0001, etc.
                'defect_type': 'DIG',                   # Specific type
                'location_xy': [cx, cy],                # Center of dig
                'bbox': [cx-radius, cy-radius, radius*2, radius*2],  # Square bounding box
                'area_px': int(dig['area']),           # Actual area in pixels
                'confidence': 0.8,                      # Fixed confidence for digs
                'severity': 'HIGH' if dig['area'] > 100 else 'MEDIUM',  # Area-based severity
                'orientation': None,                    # Digs have no orientation
                'contributing_algorithms': ['morphological_blackhat'],  # Detection method
                'detection_metadata': {
                    'contour_area': float(dig['area'])  # Precise contour area
                }
            }
            defects.append(defect)
            defect_id += 1
        
        # Process detected blobs/contamination
        for blob in specific_defects['blobs']:
            # Extract bounding box
            x, y, w, h = blob['bbox']
            # Calculate center point
            cx, cy = x + w//2, y + h//2
            
            # Create contamination defect entry
            defect = {
                'defect_id': f"CONT_{defect_id:04d}",  # Format: CONT_0001, etc.
                'defect_type': 'CONTAMINATION',         # Specific type
                'location_xy': [cx, cy],                # Center of blob
                'bbox': [x, y, w, h],                   # Bounding box
                'area_px': int(blob['area']),           # Area in pixels
                'confidence': 0.6,                      # Lower confidence for blobs
                'severity': 'MEDIUM' if blob['area'] > 500 else 'LOW',  # Area-based severity
                'orientation': None,                    # Blobs have no orientation
                'contributing_algorithms': ['blob_detection'],  # Detection method
                'detection_metadata': {
                    'circularity': float(blob['circularity']),    # Shape metric (0-1)
                    'aspect_ratio': float(blob['aspect_ratio'])   # Width/height ratio
                }
            }
            defects.append(defect)
            defect_id += 1
        
        # Extract summary data from results
        verdict = results['verdict']
        global_stats = results['global_analysis']
        
        # Calculate overall quality score (0-100 scale)
        quality_score = float(100 * (1 - verdict['confidence']))
        if len(defects) > 0:
            # Reduce quality based on number and severity of defects
            quality_score = max(0, quality_score - len(defects) * 2)
        
        # Construct final pipeline-format report
        report = {
            'source_image': image_path,                            # Source image path (pipeline expects this)
            'image_path': image_path,                              # Also keep for compatibility
            'timestamp': self._get_timestamp(),                     # Analysis timestamp
            'analysis_complete': True,                              # Pipeline expects this field
            'success': True,                                        # Analysis succeeded
            'overall_quality_score': quality_score,                 # Pipeline expects this field
            'defects': defects,                                     # List of all defects
            'zones': {                                              # Pipeline expects zone info
                'core': {'detected': True},
                'cladding': {'detected': True},
                'ferrule': {'detected': True}
            },
            'summary': {
                'total_defects': len(defects),                     # Defect count
                'is_anomalous': verdict['is_anomalous'],           # Boolean verdict
                'anomaly_confidence': float(verdict['confidence']), # Overall confidence
                'quality_score': quality_score,                     # Quality score
                'mahalanobis_distance': float(global_stats['mahalanobis_distance']),  # Statistical distance
                'ssim_score': float(results['structural_analysis']['ssim'])  # Structural similarity
            },
            'analysis_metadata': {
                'analyzer': 'ultra_comprehensive_matrix_analyzer',  # Algorithm identifier
                'version': '1.5',                                   # Version number
                'knowledge_base': self.knowledge_base_path,         # Reference model path
                'reference_samples': len(self.reference_model.get('features', [])),  # Reference count
                'features_extracted': len(results.get('test_features', {}))  # Feature count
            }
        }
        
        return report
    
    def _confidence_to_severity(self, confidence: float) -> str:
        """Convert confidence score to severity level"""
        # Iterate through severity levels from highest to lowest
        for severity, threshold in sorted(self.config.severity_thresholds.items(), 
                                        key=lambda x: x[1], reverse=True):
            # Return first severity level where confidence exceeds threshold
            if confidence >= threshold:
                return severity
        # Default to negligible if below all thresholds
        return 'NEGLIGIBLE'
    
    def _create_defect_mask(self, results: Dict) -> np.ndarray:
        """Create a binary mask of all detected defects"""
        # Extract grayscale test image from results
        test_gray = results['test_gray']
        # Initialize blank mask with same dimensions as image
        mask = np.zeros(test_gray.shape, dtype=np.uint8)
        
        # Fill in anomaly regions on mask
        for region in results['local_analysis']['anomaly_regions']:
            # Extract bounding box
            x, y, w, h = region['bbox']
            # Set pixels in region to white (255)
            mask[y:y+h, x:x+w] = 255
        
        # Extract specific defects
        defects = results['specific_defects']
        
        # Draw scratches as lines on mask
        for scratch in defects['scratches']:
            # Extract line endpoints
            x1, y1, x2, y2 = scratch['line']
            # Draw white line with thickness 3
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
        
        # Draw digs as filled circles on mask
        for dig in defects['digs']:
            # Extract center point
            cx, cy = dig['center']
            # Calculate radius from area, minimum 3 pixels
            radius = max(3, int(np.sqrt(dig['area'] / np.pi)))
            # Draw filled white circle
            cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Draw blob contours on mask
        # Extract list of contours and draw all as filled white regions
        cv2.drawContours(mask, [b['contour'] for b in defects['blobs']], -1, 255, -1)
        
        return mask
    
    def _build_minimal_reference(self, image_path: str):
        """Build a minimal reference model from a single image"""
        # Log the operation
        self.logger.info("Building minimal reference model from current image...")
        
        # Load the image from path
        image = self.load_image(image_path)
        # Check if load succeeded
        if image is None:
            return
        
        # Extract comprehensive features from image
        features, feature_names = self.extract_ultra_comprehensive_features(image)
        
        # Convert feature dictionary to numpy array in consistent order
        feature_vector = np.array([features[fname] for fname in feature_names])
        
        # Create minimal statistical model with assumed variance
        self.reference_model = {
            'features': [features],                    # Single feature set
            'feature_names': feature_names,            # Feature name list
            'statistical_model': {
                'mean': feature_vector,                # Use single sample as mean
                'std': np.ones_like(feature_vector) * 0.1,  # Assume 10% standard deviation
                'median': feature_vector,              # Single sample is also median
                'robust_mean': feature_vector,         # Use as robust mean too
                'robust_cov': np.eye(len(feature_vector)),     # Identity covariance matrix
                'robust_inv_cov': np.eye(len(feature_vector)),  # Identity inverse covariance
                'n_samples': 1,                        # Only one sample
            },
            # Convert to grayscale if needed and store as archetype
            'archetype_image': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image,
            'learned_thresholds': {                    # Default thresholds
                'anomaly_mean': 1.0,                   # Mean anomaly score
                'anomaly_std': 0.5,                    # Std of anomaly scores
                'anomaly_p90': 1.5,                    # 90th percentile
                'anomaly_p95': 2.0,                    # 95th percentile
                'anomaly_p99': 3.0,                    # 99th percentile
                'anomaly_threshold': self.config.anomaly_threshold_multiplier,  # Final threshold
            },
            'timestamp': self._get_timestamp(),        # Creation time
        }
    
    def load_knowledge_base(self):
        """Load previously saved knowledge base from JSON."""
        # Check if knowledge base file exists
        if os.path.exists(self.knowledge_base_path):
            try:
                # Open and read JSON file
                with open(self.knowledge_base_path, 'r') as f:
                    loaded_data = json.load(f)
                
                # Convert archetype image list back to numpy array
                if loaded_data.get('archetype_image'):
                    loaded_data['archetype_image'] = np.array(loaded_data['archetype_image'], dtype=np.uint8)
                
                # Convert statistical model lists back to numpy arrays
                if loaded_data.get('statistical_model'):
                    # Iterate through array fields
                    for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                        # Check if field exists and is not None
                        if key in loaded_data['statistical_model'] and loaded_data['statistical_model'][key] is not None:
                            # Convert list to numpy array
                            loaded_data['statistical_model'][key] = np.array(loaded_data['statistical_model'][key], dtype=np.float64)
                
                # Store loaded data as reference model
                self.reference_model = loaded_data
                # Log successful load
                self.logger.info(f"Loaded knowledge base from {self.knowledge_base_path}")
            except Exception as e:
                # Log warning if load fails
                self.logger.warning(f"Could not load knowledge base: {e}")
    
    def save_knowledge_base(self):
        """Save current knowledge base to JSON."""
        try:
            # Create copy to avoid modifying original
            save_data = self.reference_model.copy()
            
            # Convert archetype image numpy array to list for JSON
            if isinstance(save_data.get('archetype_image'), np.ndarray):
                save_data['archetype_image'] = save_data['archetype_image'].tolist()
            
            # Convert statistical model numpy arrays to lists
            if save_data.get('statistical_model'):
                # Iterate through array fields
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    # Check if field is numpy array
                    if key in save_data['statistical_model'] and isinstance(save_data['statistical_model'][key], np.ndarray):
                        # Convert to list
                        save_data['statistical_model'][key] = save_data['statistical_model'][key].tolist()
            
            # Remove large comparison scores if present to reduce file size
            if 'comparison_scores' in save_data:
                del save_data['comparison_scores']
            
            # Update timestamp to current time
            save_data['timestamp'] = self._get_timestamp()
            
            # Write to JSON file
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            # Log successful save
            self.logger.info(f"Knowledge base saved to {self.knowledge_base_path}")
        except Exception as e:
            # Log error if save fails
            self.logger.error(f"Error saving knowledge base: {e}")
    
    def _get_timestamp(self):
        """Get current timestamp as string."""
        # Import time module (redundant but kept for clarity)
        import time
        # Format current time as YYYY-MM-DD_HH:MM:SS
        return time.strftime("%Y-%m-%d_%H:%M:%S")
    
    # ==================== DATA LOADING ====================
    
    def load_image(self, path):
        """Load image from JSON or standard image file."""
        # Reset metadata for new image
        self.current_metadata = None
        
        # Check if file is JSON format
        if path.lower().endswith('.json'):
            # Use special JSON loader
            return self._load_from_json(path)
        else:
            # Use OpenCV to load standard image formats
            img = cv2.imread(path)
            # Check if load succeeded
            if img is None:
                # Log error
                self.logger.error(f"Could not read image: {path}")
                return None
            # Store basic metadata
            self.current_metadata = {'filename': os.path.basename(path)}
            return img
    
    def _load_from_json(self, json_path):
        """Load matrix from JSON file with bounds checking."""
        try:
            # Open and parse JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract image dimensions from JSON
            width = data['image_dimensions']['width']
            height = data['image_dimensions']['height']
            # Default to 3 channels (BGR) if not specified
            channels = data['image_dimensions'].get('channels', 3)
            
            # Initialize empty image array
            matrix = np.zeros((height, width, channels), dtype=np.uint8)
            
            # Counter for out-of-bounds pixels
            oob_count = 0
            
            # Iterate through pixel data
            for pixel in data['pixels']:
                # Extract pixel coordinates
                x = pixel['coordinates']['x']
                y = pixel['coordinates']['y']
                
                # Check if coordinates are within image bounds
                if 0 <= x < width and 0 <= y < height:
                    # Extract BGR values, handle both single value and list formats
                    bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                    # Convert single value to BGR triplet
                    if isinstance(bgr, (int, float)):
                        bgr = [bgr] * 3
                    # Set pixel values (only first 3 channels)
                    matrix[y, x] = bgr[:3]
                else:
                    # Increment out-of-bounds counter
                    oob_count += 1
            
            # Warn if any pixels were out of bounds
            if oob_count > 0:
                self.logger.warning(f"Skipped {oob_count} out-of-bounds pixels")
            
            # Store comprehensive metadata
            self.current_metadata = {
                'filename': data.get('filename', os.path.basename(json_path)),
                'width': width,
                'height': height,
                'channels': channels,
                'json_path': json_path
            }
            
            return matrix
            
        except Exception as e:
            # Log error and return None
            self.logger.error(f"Error loading JSON {json_path}: {e}")
            return None
    
    # ==================== STATISTICAL FUNCTIONS ====================
    
    def _compute_skewness(self, data):
        """Compute skewness of data."""
        # Calculate mean of data
        mean = np.mean(data)
        # Calculate standard deviation
        std = np.std(data)
        # Handle zero standard deviation case
        if std == 0:
            return 0.0
        # Compute third standardized moment (skewness)
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis of data."""
        # Calculate mean of data
        mean = np.mean(data)
        # Calculate standard deviation
        std = np.std(data)
        # Handle zero standard deviation case
        if std == 0:
            return 0.0
        # Compute fourth standardized moment minus 3 (excess kurtosis)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_entropy(self, data, bins=256):
        """Compute Shannon entropy."""
        # Create histogram with specified bins in range 0-255
        hist, _ = np.histogram(data, bins=bins, range=(0, 256))
        # Normalize histogram to get probability distribution
        hist = hist / (hist.sum() + 1e-10)
        # Remove zero bins to avoid log(0)
        hist = hist[hist > 0]
        # Compute Shannon entropy: -Σ(p * log2(p))
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def _compute_correlation(self, x, y):
        """Compute Pearson correlation coefficient."""
        # Need at least 2 points for correlation
        if len(x) < 2:
            return 0.0
        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        # Calculate covariance
        cov = np.mean((x - x_mean) * (y - y_mean))
        # Calculate standard deviations
        std_x = np.std(x)
        std_y = np.std(y)
        # Handle zero standard deviation
        if std_x == 0 or std_y == 0:
            return 0.0
        # Return correlation coefficient
        return cov / (std_x * std_y)
    
    def _compute_spearman_correlation(self, x, y):
        """Compute Spearman rank correlation."""
        # Need at least 2 points
        if len(x) < 2:
            return 0.0
        # Convert values to ranks using double argsort trick
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))
        # Compute Pearson correlation on ranks
        return self._compute_correlation(rank_x, rank_y)
    
    # ==================== FEATURE EXTRACTION ====================
    
    def _sanitize_feature_value(self, value):
        """Ensure feature value is finite and valid."""
        # Handle array-like values by taking first element
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[0]) if len(value) > 0 else 0.0
        
        # Convert to float
        val = float(value)
        # Replace NaN or infinity with 0
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    
    def extract_ultra_comprehensive_features(self, image):
        """Extract 100+ features using all available methods."""
        # Initialize empty feature dictionary
        features = {}
        
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Log feature extraction start
        self.logger.info("  Extracting features...")
        
        # Define list of feature extraction methods with names
        feature_extractors = [
            ("Stats", self._extract_statistical_features),         # Basic statistics
            ("Norms", self._extract_matrix_norms),                # Matrix norms
            ("LBP", self._extract_lbp_features),                  # Local Binary Patterns
            ("GLCM", self._extract_glcm_features),                # Gray-Level Co-occurrence
            ("FFT", self._extract_fourier_features),              # Fourier Transform
            ("MultiScale", self._extract_multiscale_features),    # Multi-scale analysis
            ("Morph", self._extract_morphological_features),      # Morphological features
            ("Shape", self._extract_shape_features),              # Shape descriptors
            ("SVD", self._extract_svd_features),                  # Singular Value Decomposition
            ("Entropy", self._extract_entropy_features),          # Entropy measures
            ("Gradient", self._extract_gradient_features),        # Gradient features
            ("Topology", self._extract_topological_proxy_features), # Topological features
        ]
        
        # Execute each feature extractor
        for name, extractor in feature_extractors:
            try:
                # Update features dictionary with new features
                features.update(extractor(gray))
            except Exception as e:
                # Log warning if extraction fails
                self.logger.warning(f"Feature extraction failed for {name}: {e}")
        
        # Create new dictionary with sanitized values
        sanitized_features = {}
        # Sanitize each feature value to ensure finite numbers
        for key, value in features.items():
            sanitized_features[key] = self._sanitize_feature_value(value)
        
        # Get sorted list of feature names for consistent ordering
        feature_names = sorted(sanitized_features.keys())
        return sanitized_features, feature_names
    
    def _extract_statistical_features(self, gray):
        """Extract comprehensive statistical features."""
        # Flatten 2D image to 1D array for statistics
        flat = gray.flatten()
        # Calculate percentiles at 10, 25, 50, 75, 90
        percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
        
        return {
            'stat_mean': float(np.mean(gray)),                    # Average pixel value
            'stat_std': float(np.std(gray)),                      # Standard deviation
            'stat_variance': float(np.var(gray)),                 # Variance
            'stat_skew': float(self._compute_skewness(flat)),     # Distribution skewness
            'stat_kurtosis': float(self._compute_kurtosis(flat)), # Distribution kurtosis
            'stat_min': float(np.min(gray)),                      # Minimum value
            'stat_max': float(np.max(gray)),                      # Maximum value
            'stat_range': float(np.max(gray) - np.min(gray)),     # Value range
            'stat_median': float(np.median(gray)),                # Median value
            'stat_mad': float(np.median(np.abs(gray - np.median(gray)))), # Median absolute deviation
            'stat_iqr': float(percentiles[3] - percentiles[1]),   # Interquartile range
            'stat_entropy': float(self._compute_entropy(gray)),   # Information entropy
            'stat_energy': float(np.sum(gray**2)),                # Energy (sum of squares)
            'stat_p10': float(percentiles[0]),                    # 10th percentile
            'stat_p25': float(percentiles[1]),                    # 25th percentile
            'stat_p50': float(percentiles[2]),                    # 50th percentile (median)
            'stat_p75': float(percentiles[3]),                    # 75th percentile
            'stat_p90': float(percentiles[4]),                    # 90th percentile
        }
    
    def _extract_matrix_norms(self, gray):
        """Extract various matrix norms."""
        return {
            'norm_frobenius': float(np.linalg.norm(gray, 'fro')), # Frobenius norm (sqrt of sum of squares)
            'norm_l1': float(np.sum(np.abs(gray))),               # L1 norm (sum of absolute values)
            'norm_l2': float(np.sqrt(np.sum(gray**2))),           # L2 norm (Euclidean norm)
            'norm_linf': float(np.max(np.abs(gray))),             # L-infinity norm (max absolute value)
            'norm_nuclear': float(np.sum(np.linalg.svd(gray, compute_uv=False))), # Nuclear norm (sum of singular values)
            'norm_trace': float(np.trace(gray)),                  # Trace (sum of diagonal elements)
        }
    
    def _extract_lbp_features(self, gray):
        """Extract Local Binary Pattern features using custom implementation."""
        # Initialize feature dictionary
        features = {}
        
        # Compute LBP at multiple radii
        for radius in [1, 2, 3, 5]:
            # Initialize LBP result array
            lbp = np.zeros_like(gray, dtype=np.float32)
            
            # Iterate through neighborhood offsets
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    # Skip center pixel
                    if dx == 0 and dy == 0:
                        continue
                    
                    # Create shifted version of image
                    shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                    
                    # Compare shifted with original (binary pattern)
                    lbp += (shifted >= gray).astype(np.float32)
            
            # Compute statistics of LBP
            features[f'lbp_r{radius}_mean'] = float(np.mean(lbp))        # Average LBP value
            features[f'lbp_r{radius}_std'] = float(np.std(lbp))          # LBP standard deviation
            features[f'lbp_r{radius}_entropy'] = float(self._compute_entropy(lbp))  # LBP entropy
            features[f'lbp_r{radius}_energy'] = float(np.sum(lbp**2) / lbp.size)    # Normalized energy
        
        return features
    
    def _extract_glcm_features(self, gray):
        """Extract Gray-Level Co-occurrence Matrix features using custom implementation."""
        # Quantize image to 8 levels for faster computation
        img_q = (gray // 32).astype(np.uint8)
        levels = 8
        
        # Initialize feature dictionary
        features = {}
        # Define distances and angles for GLCM
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]  # degrees
        
        # Compute GLCM for each distance and angle
        for dist in distances:
            for angle in angles:
                # Initialize GLCM matrix
                glcm = np.zeros((levels, levels), dtype=np.float32)
                
                # Determine pixel offset based on angle
                if angle == 0:
                    dy, dx = 0, dist       # Horizontal
                elif angle == 45:
                    dy, dx = -dist, dist   # Diagonal up-right
                elif angle == 90:
                    dy, dx = -dist, 0      # Vertical
                else:  # 135
                    dy, dx = -dist, -dist  # Diagonal up-left
                
                # Build GLCM by counting co-occurrences
                rows, cols = img_q.shape
                for i in range(rows):
                    for j in range(cols):
                        # Check if neighbor is within bounds
                        if 0 <= i + dy < rows and 0 <= j + dx < cols:
                            # Increment co-occurrence count
                            glcm[img_q[i, j], img_q[i + dy, j + dx]] += 1
                
                # Normalize GLCM to probabilities
                glcm = glcm / (glcm.sum() + 1e-10)
                
                # Compute GLCM properties
                # Contrast: measure of local variations
                contrast = 0
                for i in range(levels):
                    for j in range(levels):
                        contrast += ((i - j) ** 2) * glcm[i, j]
                
                # Energy: measure of uniformity
                energy = np.sum(glcm ** 2)
                
                # Homogeneity: measure of closeness to diagonal
                homogeneity = 0
                for i in range(levels):
                    for j in range(levels):
                        homogeneity += glcm[i, j] / (1 + abs(i - j))
                
                # Store features with descriptive names
                features[f'glcm_d{dist}_a{angle}_contrast'] = float(contrast)
                features[f'glcm_d{dist}_a{angle}_energy'] = float(energy)
                features[f'glcm_d{dist}_a{angle}_homogeneity'] = float(homogeneity)
        
        return features
    
    def _extract_fourier_features(self, gray):
        """Extract 2D Fourier Transform features."""
        # Compute 2D FFT
        f = np.fft.fft2(gray)
        # Shift zero frequency to center
        fshift = np.fft.fftshift(f)
        # Compute magnitude spectrum
        magnitude = np.abs(fshift)
        # Compute power spectrum
        power = magnitude**2
        # Compute phase spectrum
        phase = np.angle(fshift)
        
        # Calculate center coordinates
        center = np.array(power.shape) // 2
        # Create coordinate grids
        y, x = np.ogrid[:power.shape[0], :power.shape[1]]
        # Compute distance from center for each pixel
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Compute radial profile (average power at each radius)
        radial_prof = []
        for radius in range(1, min(center)):
            # Create ring mask
            mask = (r >= radius - 1) & (r < radius)
            # Average power in ring
            if mask.any():
                radial_prof.append(np.mean(power[mask]))
        
        # Convert to array
        radial_prof = np.array(radial_prof)
        
        # Compute spectral centroid and spread if profile exists
        if len(radial_prof) > 0:
            # Weighted average of frequencies
            spectral_centroid = float(np.sum(np.arange(len(radial_prof)) * radial_prof) / (np.sum(radial_prof) + 1e-10))
            # Weighted standard deviation of frequencies
            spectral_spread = float(np.sqrt(np.sum((np.arange(len(radial_prof)) - spectral_centroid)**2 * radial_prof) / (np.sum(radial_prof) + 1e-10)))
        else:
            spectral_centroid = 0.0
            spectral_spread = 0.0
        
        return {
            'fft_mean_magnitude': float(np.mean(magnitude)),       # Average magnitude
            'fft_std_magnitude': float(np.std(magnitude)),         # Magnitude variation
            'fft_max_magnitude': float(np.max(magnitude)),         # Peak magnitude
            'fft_total_power': float(np.sum(power)),               # Total power
            'fft_dc_component': float(magnitude[center[0], center[1]]), # DC (zero frequency) component
            'fft_mean_phase': float(np.mean(phase)),               # Average phase
            'fft_std_phase': float(np.std(phase)),                 # Phase variation
            'fft_spectral_centroid': spectral_centroid,            # Center of spectral mass
            'fft_spectral_spread': spectral_spread,                # Spread of spectral mass
        }
    
    def _extract_multiscale_features(self, gray):
        """Extract multi-scale features using Gaussian pyramids (replaces wavelets)."""
        # Initialize feature dictionary
        features = {}
        
        # Create Gaussian pyramid (progressively downsampled versions)
        pyramid = [gray]
        for i in range(3):
            # Downsample by factor of 2
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        
        # Compute features at each scale
        for level, img in enumerate(pyramid):
            # Basic statistics at this scale
            features[f'pyramid_L{level}_mean'] = float(np.mean(img))      # Mean at scale
            features[f'pyramid_L{level}_std'] = float(np.std(img))        # Std at scale
            features[f'pyramid_L{level}_energy'] = float(np.sum(img**2))  # Energy at scale
            
            # Compute difference between scales (detail information)
            if level > 0:
                # Upsample current level to match previous level size
                upsampled = cv2.pyrUp(img)
                # Get dimensions of previous level
                h, w = pyramid[level-1].shape
                # Resize to exact dimensions
                upsampled = cv2.resize(upsampled, (w, h))
                
                # Compute difference (approximates wavelet detail coefficients)
                detail = pyramid[level-1].astype(np.float32) - upsampled.astype(np.float32)
                
                # Statistics of detail coefficients
                features[f'pyramid_detail_L{level}_energy'] = float(np.sum(detail**2))  # Detail energy
                features[f'pyramid_detail_L{level}_mean'] = float(np.mean(np.abs(detail)))  # Mean absolute detail
                features[f'pyramid_detail_L{level}_std'] = float(np.std(detail))       # Detail variation
        
        # Laplacian pyramid features (edge information at multiple scales)
        for level in range(2):
            # Compute Laplacian (second derivative)
            laplacian = cv2.Laplacian(pyramid[level], cv2.CV_64F)
            features[f'laplacian_L{level}_energy'] = float(np.sum(laplacian**2))  # Edge energy
            features[f'laplacian_L{level}_mean'] = float(np.mean(np.abs(laplacian)))  # Mean edge strength
        
        return features
    
    def _extract_morphological_features(self, gray):
        """Extract morphological features."""
        # Initialize feature dictionary
        features = {}
        
        # Multi-scale morphological operations
        for size in [3, 5, 7, 11]:
            # Create circular structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            
            # White tophat: bright features smaller than kernel
            wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            # Black tophat: dark features smaller than kernel
            bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Statistics of tophat transforms
            features[f'morph_wth_{size}_mean'] = float(np.mean(wth))    # Average bright feature intensity
            features[f'morph_wth_{size}_max'] = float(np.max(wth))      # Maximum bright feature
            features[f'morph_wth_{size}_sum'] = float(np.sum(wth))      # Total bright feature energy
            features[f'morph_bth_{size}_mean'] = float(np.mean(bth))    # Average dark feature intensity
            features[f'morph_bth_{size}_max'] = float(np.max(bth))      # Maximum dark feature
            features[f'morph_bth_{size}_sum'] = float(np.sum(bth))      # Total dark feature energy
        
        # Binary morphology analysis
        # Otsu's threshold for automatic binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Define 5x5 square kernel
        kernel = np.ones((5, 5), np.uint8)
        # Erosion: shrinks white regions
        erosion = cv2.erode(binary, kernel, iterations=1)
        # Dilation: expands white regions
        dilation = cv2.dilate(binary, kernel, iterations=1)
        # Morphological gradient: difference between dilation and erosion
        gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        
        # Compute morphological statistics
        features['morph_binary_area_ratio'] = float(np.sum(binary) / binary.size)  # Fraction of white pixels
        features['morph_gradient_sum'] = float(np.sum(gradient))                   # Total edge pixels
        features['morph_erosion_ratio'] = float(np.sum(erosion) / (np.sum(binary) + 1e-10))  # Erosion effect
        features['morph_dilation_ratio'] = float(np.sum(dilation) / (np.sum(binary) + 1e-10))  # Dilation effect
        
        return features
    
    def _extract_shape_features(self, gray):
        """Extract shape features using Hu moments."""
        # Calculate image moments (statistical measures)
        moments = cv2.moments(gray)
        # Compute 7 Hu moments (rotation invariant)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Initialize feature dictionary
        features = {}
        # Store log-transformed Hu moments for scale invariance
        for i, hu in enumerate(hu_moments):
            # Log transform with sign preservation
            features[f'shape_hu_{i}'] = float(-np.sign(hu) * np.log10(abs(hu) + 1e-10))
        
        # Additional moment features
        if moments['m00'] > 0:  # Check for non-zero area
            # Calculate centroid coordinates
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            # Normalize centroid to [0,1] range
            features['shape_centroid_x'] = float(cx / gray.shape[1])
            features['shape_centroid_y'] = float(cy / gray.shape[0])
        
        return features
    
    def _extract_svd_features(self, gray):
        """Extract Singular Value Decomposition features."""
        try:
            # Compute SVD (decompose matrix into U*S*V^T)
            _, s, _ = np.linalg.svd(gray, full_matrices=False)
            # Normalize singular values
            s_norm = s / (np.sum(s) + 1e-10)
            
            # Compute cumulative energy
            cumsum = np.cumsum(s_norm)
            # Find components needed for 90% energy
            n_components_90 = np.argmax(cumsum >= 0.9) + 1
            # Find components needed for 95% energy
            n_components_95 = np.argmax(cumsum >= 0.95) + 1
            
            return {
                'svd_largest': float(s[0]) if len(s) > 0 else 0.0,                    # Largest singular value
                'svd_top5_ratio': float(np.sum(s_norm[:5])) if len(s) >= 5 else float(np.sum(s_norm)),  # Energy in top 5
                'svd_top10_ratio': float(np.sum(s_norm[:10])) if len(s) >= 10 else float(np.sum(s_norm)),  # Energy in top 10
                'svd_entropy': float(self._compute_entropy(s_norm * 1000)),           # Entropy of singular values
                'svd_energy_ratio': float(s[0] / (s[1] + 1e-10)) if len(s) > 1 else 0.0,  # Ratio of first two values
                'svd_n_components_90': float(n_components_90),                        # Components for 90% energy
                'svd_n_components_95': float(n_components_95),                        # Components for 95% energy
                'svd_effective_rank': float(np.exp(self._compute_entropy(s_norm * 1000))),  # Exponential of entropy
            }
        except:
            # Return zeros if SVD fails
            return {f'svd_{k}': 0.0 for k in ['largest', 'top5_ratio', 'top10_ratio', 'entropy', 
                                               'energy_ratio', 'n_components_90', 'n_components_95', 'effective_rank']}
    
    def _extract_entropy_features(self, gray):
        """Extract various entropy measures."""
        # Compute global histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        # Normalize to probability distribution
        hist_norm = hist / (hist.sum() + 1e-10)
        
        # Shannon entropy (already computed)
        shannon = self._compute_entropy(gray)
        
        # Renyi entropy with parameter alpha = 2
        renyi = -np.log2(np.sum(hist_norm**2) + 1e-10)
        
        # Tsallis entropy with parameter q = 2
        tsallis = (1 - np.sum(hist_norm**2)) / 1
        
        # Local entropy computation
        # Define local window size
        kernel_size = 9
        # Create averaging kernel
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
        
        # Compute local statistics for entropy approximation
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)      # Local mean
        local_sq_mean = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel)  # Local mean of squares
        local_var = local_sq_mean - local_mean**2                           # Local variance
        local_ent = np.log2(local_var + 1)                                  # Log variance as entropy proxy
        
        return {
            'entropy_shannon': float(shannon),              # Shannon entropy
            'entropy_renyi': float(renyi),                  # Renyi entropy
            'entropy_tsallis': float(tsallis),              # Tsallis entropy
            'entropy_local_mean': float(np.mean(local_ent)),  # Average local entropy
            'entropy_local_std': float(np.std(local_ent)),    # Variation in local entropy
            'entropy_local_max': float(np.max(local_ent)),    # Maximum local entropy
            'entropy_local_min': float(np.min(local_ent)),    # Minimum local entropy
        }
    
    def _extract_gradient_features(self, gray):
        """Extract gradient-based features."""
        # Compute Sobel gradients (first derivatives)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient
        
        # Compute gradient magnitude (edge strength)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        # Compute gradient orientation (edge direction)
        grad_orient = np.arctan2(grad_y, grad_x)
        
        # Compute Laplacian (second derivative)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Detect edges using Canny algorithm
        edges = cv2.Canny(gray, 50, 150)  # Lower and upper thresholds
        # Calculate edge density
        edge_density = np.sum(edges) / edges.size
        
        return {
            'gradient_magnitude_mean': float(np.mean(grad_mag)),    # Average edge strength
            'gradient_magnitude_std': float(np.std(grad_mag)),      # Edge strength variation
            'gradient_magnitude_max': float(np.max(grad_mag)),      # Maximum edge strength
            'gradient_magnitude_sum': float(np.sum(grad_mag)),      # Total edge energy
            'gradient_orientation_mean': float(np.mean(grad_orient)),  # Average edge direction
            'gradient_orientation_std': float(np.std(grad_orient)),    # Edge direction variation
            'laplacian_mean': float(np.mean(np.abs(laplacian))),    # Average second derivative
            'laplacian_std': float(np.std(laplacian)),              # Second derivative variation
            'laplacian_sum': float(np.sum(np.abs(laplacian))),      # Total second derivative
            'edge_density': float(edge_density),                    # Fraction of edge pixels
            'edge_count': float(np.sum(edges > 0)),                 # Total edge pixels
        }
    
    def _extract_topological_proxy_features(self, gray):
        """Extract topological proxy features using connected components analysis."""
        # Initialize feature dictionary
        features = {}
        
        # Create threshold values from 5th to 95th percentile
        thresholds = np.percentile(gray, np.linspace(5, 95, 20))
        
        # Track connected components at each threshold (proxy for Betti 0)
        n_components = []
        for t in thresholds:
            # Create binary image at threshold
            binary = (gray >= t).astype(np.uint8)
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            # Store count (subtract 1 for background)
            n_components.append(num_labels - 1)
        
        # Track holes at each threshold (proxy for Betti 1)
        n_holes = []
        for t in thresholds:
            # Create inverted binary image
            binary = (gray < t).astype(np.uint8)
            # Find connected components in inverted image
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            # Store count (subtract 1 for background)
            n_holes.append(num_labels - 1)
        
        # Compute statistics for connected components
        if len(n_components) > 1:
            # Compute persistence (changes between thresholds)
            persistence_b0 = np.diff(n_components)
            features['topo_b0_max_components'] = float(np.max(n_components))         # Maximum components
            features['topo_b0_mean_components'] = float(np.mean(n_components))       # Average components
            features['topo_b0_persistence_sum'] = float(np.sum(np.abs(persistence_b0)))  # Total persistence
            features['topo_b0_persistence_max'] = float(np.max(np.abs(persistence_b0)))  # Max persistence
        else:
            # Handle single threshold case
            features['topo_b0_max_components'] = float(n_components[0]) if n_components else 0.0
            features['topo_b0_mean_components'] = float(n_components[0]) if n_components else 0.0
            features['topo_b0_persistence_sum'] = 0.0
            features['topo_b0_persistence_max'] = 0.0
        
        # Compute statistics for holes
        if len(n_holes) > 1:
            # Compute persistence for holes
            persistence_b1 = np.diff(n_holes)
            features['topo_b1_max_holes'] = float(np.max(n_holes))              # Maximum holes
            features['topo_b1_mean_holes'] = float(np.mean(n_holes))            # Average holes
            features['topo_b1_persistence_sum'] = float(np.sum(np.abs(persistence_b1)))  # Total persistence
            features['topo_b1_persistence_max'] = float(np.max(np.abs(persistence_b1)))  # Max persistence
        else:
            # Handle single threshold case
            features['topo_b1_max_holes'] = float(n_holes[0]) if n_holes else 0.0
            features['topo_b1_mean_holes'] = float(n_holes[0]) if n_holes else 0.0
            features['topo_b1_persistence_sum'] = 0.0
            features['topo_b1_persistence_max'] = 0.0
        
        return features
    
    # ==================== COMPARISON METHODS ====================
    
    def compute_exhaustive_comparison(self, features1, features2):
        """Compute all possible comparison metrics between two feature sets."""
        # Get common feature keys between both sets
        keys = sorted(set(features1.keys()) & set(features2.keys()))
        # Handle case with no common features
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
        
        # Convert feature dictionaries to vectors
        vec1 = np.array([features1[k] for k in keys])
        vec2 = np.array([features2[k] for k in keys])
        
        # Handle empty vectors
        if len(vec1) == 0 or len(vec2) == 0:
            return self.compute_exhaustive_comparison({}, {})
        
        # Normalize vectors to unit length
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        vec1_norm = vec1 / (norm1 + 1e-10)
        vec2_norm = vec2 / (norm2 + 1e-10)
        
        # Initialize comparison dictionary
        comparison = {}
        
        # Distance metrics
        comparison['euclidean_distance'] = float(np.linalg.norm(vec1 - vec2))     # L2 distance
        comparison['manhattan_distance'] = float(np.sum(np.abs(vec1 - vec2)))     # L1 distance
        comparison['chebyshev_distance'] = float(np.max(np.abs(vec1 - vec2)))     # L∞ distance
        comparison['cosine_distance'] = float(1 - np.dot(vec1_norm, vec2_norm))   # 1 - cosine similarity
        
        # Correlation measures
        comparison['pearson_correlation'] = float(self._compute_correlation(vec1, vec2))      # Linear correlation
        comparison['spearman_correlation'] = float(self._compute_spearman_correlation(vec1, vec2))  # Rank correlation
        
        # Statistical tests
        comparison['ks_statistic'] = float(self._compute_ks_statistic(vec1, vec2))  # Kolmogorov-Smirnov
        
        # Information theoretic measures
        bins = min(30, len(vec1) // 2)  # Adaptive bin count
        if bins > 2:
            # Create normalized histograms for both vectors
            min_val = min(vec1.min(), vec2.min())
            max_val = max(vec1.max(), vec2.max())
            
            # Compute histograms with same bins
            hist1, bin_edges = np.histogram(vec1, bins=bins, range=(min_val, max_val))
            hist2, _ = np.histogram(vec2, bins=bin_edges)
            
            # Normalize to probabilities
            hist1 = hist1 / (hist1.sum() + 1e-10)
            hist2 = hist2 / (hist2.sum() + 1e-10)
            
            # KL divergence: D_KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
            kl_div = 0
            for i in range(len(hist1)):
                if hist1[i] > 0:
                    kl_div += hist1[i] * np.log((hist1[i] + 1e-10) / (hist2[i] + 1e-10))
            comparison['kl_divergence'] = float(kl_div)
            
            # JS divergence: symmetric version of KL
            m = 0.5 * (hist1 + hist2)  # Average distribution
            js_div = 0.5 * sum(hist1[i] * np.log((hist1[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist1)) if hist1[i] > 0)
            js_div += 0.5 * sum(hist2[i] * np.log((hist2[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist2)) if hist2[i] > 0)
            comparison['js_divergence'] = float(js_div)
            
            # Chi-square distance: χ² = 0.5 * Σ (P(i) - Q(i))² / (P(i) + Q(i))
            chi_sq = 0.5 * np.sum(np.where(hist1 + hist2 > 0, (hist1 - hist2)**2 / (hist1 + hist2 + 1e-10), 0))
            comparison['chi_square'] = float(chi_sq)
        else:
            # Default values if not enough bins
            comparison['kl_divergence'] = float('inf')
            comparison['js_divergence'] = 1.0
            comparison['chi_square'] = float('inf')
        
        # Wasserstein distance (1D approximation)
        comparison['wasserstein_distance'] = float(self._compute_wasserstein_distance(vec1, vec2))
        
        # Feature SSIM (simplified structural similarity)
        mean1, mean2 = np.mean(vec1), np.mean(vec2)
        comparison['feature_ssim'] = float((2 * mean1 * mean2 + 1e-10) / (mean1**2 + mean2**2 + 1e-10))
        
        return comparison
    
    def _compute_ks_statistic(self, x, y):
        """Compute Kolmogorov-Smirnov statistic."""
        # Sort both arrays
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        # Combine and sort all values
        combined = np.concatenate([x_sorted, y_sorted])
        combined_sorted = np.sort(combined)
        
        # Compute empirical CDFs and find maximum difference
        max_diff = 0
        for val in combined_sorted:
            # Compute CDF at this value for both distributions
            cdf_x = np.sum(x_sorted <= val) / len(x_sorted)
            cdf_y = np.sum(y_sorted <= val) / len(y_sorted)
            # Update maximum difference
            max_diff = max(max_diff, abs(cdf_x - cdf_y))
        
        return max_diff
    
    def _compute_wasserstein_distance(self, x, y):
        """Compute 1D Wasserstein distance."""
        # Sort both arrays
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        # Interpolate to same size for comparison
        n = max(len(x_sorted), len(y_sorted))
        # Create interpolation points
        x_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x_sorted)), x_sorted)
        y_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y_sorted)), y_sorted)
        
        # Compute average absolute difference
        return np.mean(np.abs(x_interp - y_interp))
    
    def compute_image_structural_comparison(self, img1, img2):
        """Compute structural similarity between images."""
        # Ensure images have same dimensions
        if img1.shape != img2.shape:
            # Use maximum dimensions
            h, w = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
            # Resize both images
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # SSIM implementation constants
        C1 = (0.01 * 255)**2  # Constant to stabilize luminance
        C2 = (0.03 * 255)**2  # Constant to stabilize contrast
        
        # Create Gaussian window for local statistics
        kernel = cv2.getGaussianKernel(11, 1.5)  # 11x11 kernel, sigma=1.5
        window = np.outer(kernel, kernel.transpose())  # 2D kernel
        
        # Compute local means
        mu1 = cv2.filter2D(img1.astype(float), -1, window)
        mu2 = cv2.filter2D(img2.astype(float), -1, window)
        
        # Compute local statistics
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = cv2.filter2D(img1.astype(float)**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2.astype(float)**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1.astype(float) * img2.astype(float), -1, window) - mu1_mu2
        
        # SSIM components
        # Luminance comparison
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        # Contrast comparison
        contrast = (2 * np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2) / (sigma1_sq + sigma2_sq + C2)
        # Structure comparison
        structure = (sigma12 + C2/2) / (np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2/2)
        
        # Combine components
        ssim_map = luminance * contrast * structure
        # Average SSIM
        ssim_index = np.mean(ssim_map)
        
        # Multi-scale SSIM at different resolutions
        ms_ssim_values = [ssim_index]
        for scale in [2, 4]:
            # Downsample images
            img1_scaled = cv2.resize(img1, (img1.shape[1]//scale, img1.shape[0]//scale))
            img2_scaled = cv2.resize(img2, (img2.shape[1]//scale, img2.shape[0]//scale))
            
            # Simplified SSIM for other scales
            diff = np.abs(img1_scaled.astype(float) - img2_scaled.astype(float))
            ms_ssim = 1 - np.mean(diff) / 255
            ms_ssim_values.append(ms_ssim)
        
        return {
            'ssim': float(ssim_index),                        # Overall SSIM
            'ssim_map': ssim_map,                             # Pixel-wise SSIM
            'ms_ssim': ms_ssim_values,                        # Multi-scale SSIM
            'luminance_map': luminance,                       # Luminance comparison map
            'contrast_map': contrast,                         # Contrast comparison map
            'structure_map': structure,                       # Structure comparison map
            'mean_luminance': float(np.mean(luminance)),     # Average luminance similarity
            'mean_contrast': float(np.mean(contrast)),       # Average contrast similarity
            'mean_structure': float(np.mean(structure)),     # Average structure similarity
        }
    
    # ==================== REFERENCE MODEL BUILDING ====================
    
    def build_comprehensive_reference_model(self, ref_dir):
        """Build an exhaustive reference model from a directory of JSON/image files."""
        # Log start of model building
        self.logger.info(f"Building Comprehensive Reference Model from: {ref_dir}")
        
        # Define supported file extensions
        valid_extensions = ['.json', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        all_files = []
        
        # List all files in directory
        try:
            for filename in os.listdir(ref_dir):
                # Get file extension
                ext = os.path.splitext(filename)[1].lower()
                # Check if valid format
                if ext in valid_extensions:
                    # Add full path to list
                    all_files.append(os.path.join(ref_dir, filename))
        except Exception as e:
            # Log error if directory read fails
            self.logger.error(f"Error reading directory: {e}")
            return False
        
        # Sort files for consistent processing order
        all_files.sort()
        
        # Check if any files found
        if not all_files:
            self.logger.error(f"No valid files found in {ref_dir}")
            return False
        
        # Log file count
        self.logger.info(f"Found {len(all_files)} files to process")
        
        # Initialize storage lists
        all_features = []     # Feature dictionaries
        all_images = []       # Grayscale images
        feature_names = []    # Feature name list
        
        # Process each file
        self.logger.info("Processing files:")
        for i, file_path in enumerate(all_files, 1):
            # Log progress
            self.logger.info(f"[{i}/{len(all_files)}] {os.path.basename(file_path)}")
            
            # Load image
            image = self.load_image(file_path)
            if image is None:
                # Log failure and skip
                self.logger.warning(f"  Failed to load")
                continue
            
            # Convert to grayscale for consistent storage
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Extract features
            features, f_names = self.extract_ultra_comprehensive_features(image)
            
            # Store feature names from first image
            if not feature_names:
                feature_names = f_names
            
            # Add to collections
            all_features.append(features)
            all_images.append(gray)
            
            # Log success
            self.logger.info(f"  Processed: {len(features)} features extracted")
        
        # Check if any features extracted
        if not all_features:
            self.logger.error("No features could be extracted from any file")
            return False
        
        # Check minimum sample requirement
        if len(all_features) < 2:
            self.logger.error(f"At least 2 reference files are required, but only {len(all_features)} were successfully processed.")
            return False
        
        self.logger.info("Building Statistical Model...")
        
        # Convert features to matrix (samples x features)
        feature_matrix = np.zeros((len(all_features), len(feature_names)))
        for i, features in enumerate(all_features):
            for j, fname in enumerate(feature_names):
                # Get feature value, default to 0 if missing
                feature_matrix[i, j] = features.get(fname, 0)
        
        # Compute basic statistics
        mean_vector = np.mean(feature_matrix, axis=0)    # Feature means
        std_vector = np.std(feature_matrix, axis=0)      # Feature standard deviations
        median_vector = np.median(feature_matrix, axis=0) # Feature medians
        
        # Compute robust statistics
        self.logger.info("Computing robust statistics...")
        robust_mean, robust_cov, robust_inv_cov = self._compute_robust_statistics(feature_matrix)
        
        # Create archetype image (median of all images)
        self.logger.info("Creating archetype image...")
        # Get target dimensions from first image
        target_shape = all_images[0].shape
        aligned_images = []
        # Resize all images to same size
        for img in all_images:
            if img.shape != target_shape:
                img = cv2.resize(img, (target_shape[1], target_shape[0]))
            aligned_images.append(img)
        
        # Compute pixel-wise median
        archetype_image = np.median(aligned_images, axis=0).astype(np.uint8)
        
        # Learn anomaly thresholds from pairwise comparisons
        self.logger.info("Computing pairwise comparisons for threshold learning...")
        # Calculate total number of pairwise comparisons
        n_comparisons = len(all_features) * (len(all_features) - 1) // 2
        self.logger.info(f"Total comparisons to compute: {n_comparisons}")
        
        # Initialize comparison tracking
        comparison_scores = []
        comparison_count = 0
        
        # Compare all pairs of reference samples
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                # Compute comprehensive comparison
                comp = self.compute_exhaustive_comparison(all_features[i], all_features[j])
                
                # Compute weighted anomaly score
                score = (comp['euclidean_distance'] * 0.2 +      # Euclidean weight
                        comp['manhattan_distance'] * 0.1 +        # Manhattan weight
                        comp['cosine_distance'] * 0.2 +           # Cosine weight
                        (1 - abs(comp['pearson_correlation'])) * 0.1 +  # Correlation weight
                        min(comp['kl_divergence'], 10.0) * 0.1 +  # KL divergence weight (capped)
                        comp['js_divergence'] * 0.1 +             # JS divergence weight
                        min(comp['chi_square'], 10.0) * 0.1 +     # Chi-square weight (capped)
                        min(comp['wasserstein_distance'], 10.0) * 0.1)  # Wasserstein weight (capped)
                
                # Store score
                comparison_scores.append(score)
                comparison_count += 1
                
                # Log progress every 100 comparisons
                if comparison_count % 100 == 0:
                    self.logger.info(f"  Progress: {comparison_count}/{n_comparisons} ({comparison_count/n_comparisons*100:.1f}%)")
        
        # Learn thresholds from comparison scores
        scores_array = np.array(comparison_scores)
        
        # Check if valid scores exist
        if len(scores_array) > 0 and not np.all(np.isnan(scores_array)):
            # Filter out invalid values
            valid_scores = scores_array[~np.isnan(scores_array)]
            valid_scores = valid_scores[np.isfinite(valid_scores)]
            
            if len(valid_scores) > 0:
                # Clip extreme outliers at 99.9th percentile
                valid_scores = np.clip(valid_scores, 0, np.percentile(valid_scores, 99.9))
                
                # Calculate statistics
                mean_score = float(np.mean(valid_scores))
                std_score = float(np.std(valid_scores))
                
                # Create threshold dictionary
                thresholds = {
                    'anomaly_mean': mean_score,                    # Mean comparison score
                    'anomaly_std': std_score,                      # Std of comparison scores
                    'anomaly_p90': float(np.percentile(valid_scores, 90)),   # 90th percentile
                    'anomaly_p95': float(np.percentile(valid_scores, 95)),   # 95th percentile
                    'anomaly_p99': float(np.percentile(valid_scores, 99)),   # 99th percentile
                    'anomaly_threshold': float(min(mean_score + self.config.anomaly_threshold_multiplier * std_score,  # Statistical threshold
                                                   np.percentile(valid_scores, 99.5),     # 99.5th percentile
                                                   10.0)),                                # Hard cap at 10.0
                }
            else:
                # Use defaults if no valid scores
                thresholds = self._get_default_thresholds()
        else:
            # Use defaults if no scores computed
            thresholds = self._get_default_thresholds()
        
        # Store complete reference model
        self.reference_model = {
            'features': all_features,                      # All feature dictionaries
            'feature_names': feature_names,                # Consistent feature ordering
            'statistical_model': {
                'mean': mean_vector,                       # Feature means
                'std': std_vector,                         # Feature standard deviations
                'median': median_vector,                   # Feature medians
                'robust_mean': robust_mean,                # Robust mean estimate
                'robust_cov': robust_cov,                  # Robust covariance matrix
                'robust_inv_cov': robust_inv_cov,          # Inverse covariance for Mahalanobis
                'n_samples': len(all_features),            # Number of reference samples
            },
            'archetype_image': archetype_image,            # Median reference image
            'learned_thresholds': thresholds,              # Learned anomaly thresholds
            'timestamp': self._get_timestamp(),            # Creation timestamp
        }
        
        # Save model to disk
        self.save_knowledge_base()
        
        # Log success summary
        self.logger.info("Reference Model Built Successfully!")
        self.logger.info(f"  - Samples: {len(all_features)}")
        self.logger.info(f"  - Features: {len(feature_names)}")
        self.logger.info(f"  - Anomaly threshold: {thresholds['anomaly_threshold']:.4f}")
        
        return True
    
    def _compute_robust_statistics(self, data):
        """Compute robust mean and covariance using custom implementation."""
        # Get data dimensions
        n_samples, n_features = data.shape
        
        # Use median as initial robust mean estimate
        robust_mean = np.median(data, axis=0)
        
        # Compute deviations from median
        deviations = data - robust_mean
        # Compute Median Absolute Deviation for each feature
        mad = np.median(np.abs(deviations), axis=0)
        
        # Scale MAD to approximate standard deviation (1.4826 is consistency factor)
        mad_scaled = mad * 1.4826
        
        # Replace near-zero values to avoid division by zero
        mad_scaled[mad_scaled < 1e-6] = 1.0
        
        # Compute robust covariance using weighted approach
        # Normalize deviations by scaled MAD
        normalized_deviations = deviations / mad_scaled
        # Compute distance from center for each sample
        distances = np.sqrt(np.sum(normalized_deviations**2, axis=1))
        
        # Clip distances to avoid numerical issues
        distances = np.clip(distances, 0, 10)
        
        # Compute weights using Gaussian kernel
        weights = np.exp(-0.5 * distances)
        # Normalize weights
        weight_sum = weights.sum()
        
        # Check if weights are valid
        if weight_sum < 1e-10 or n_samples < 2:
            # Fall back to standard covariance if weights fail
            robust_cov = np.cov(data, rowvar=False)
            # Handle single feature case
            if robust_cov.ndim == 0:
                robust_cov = np.array([[robust_cov]])
        else:
            # Normalize weights
            weights = weights / weight_sum
            
            # Compute weighted covariance
            # Weight data by square root of weights
            weighted_data = data * np.sqrt(weights[:, np.newaxis])
            # Compute covariance of weighted data
            robust_cov = np.dot(weighted_data.T, weighted_data)
            
            # Apply bias correction using effective sample size
            effective_n = 1.0 / np.sum(weights**2)
            if effective_n > 1:
                robust_cov = robust_cov * effective_n / (effective_n - 1)
        
        # Ensure covariance matrix is well-conditioned
        # Add small regularization to diagonal
        reg_value = np.trace(robust_cov) / n_features * 1e-4
        if reg_value < 1e-6:
            reg_value = 1e-6
        robust_cov = robust_cov + np.eye(n_features) * reg_value
        
        # Ensure positive semi-definite through eigenvalue decomposition
        try:
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
            # Clip negative eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            # Reconstruct covariance matrix
            robust_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except np.linalg.LinAlgError:
            # Fall back to diagonal matrix if decomposition fails
            var_scale = np.var(data)
            robust_cov = np.eye(n_features) * var_scale
        
        # Compute pseudo-inverse for Mahalanobis distance
        try:
            # Add extra regularization for stable inversion
            robust_inv_cov = np.linalg.pinv(robust_cov + np.eye(n_features) * 1e-4)
        except np.linalg.LinAlgError:
            # Fall back to diagonal approximation
            diag_values = np.diag(robust_cov)
            diag_values[diag_values < 1e-6] = 1e-6
            robust_inv_cov = np.diag(1.0 / diag_values)
        
        return robust_mean, robust_cov, robust_inv_cov
    
    def _get_default_thresholds(self):
        """Return default thresholds when learning fails."""
        return {
            'anomaly_mean': 1.0,                           # Default mean score
            'anomaly_std': 0.5,                            # Default standard deviation
            'anomaly_p90': 1.5,                            # Default 90th percentile
            'anomaly_p95': 2.0,                            # Default 95th percentile
            'anomaly_p99': 3.0,                            # Default 99th percentile
            'anomaly_threshold': self.config.anomaly_threshold_multiplier,  # Config multiplier
        }
    
    # ==================== ANOMALY DETECTION ====================
    
    def detect_anomalies_comprehensive(self, test_path):
        """Perform exhaustive anomaly detection on a test image."""
        # Log start of analysis
        self.logger.info(f"Analyzing: {test_path}")
        
        # Check if reference model exists
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Build one first.")
            return None
        
        # Load test image
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        # Log loaded image metadata
        self.logger.info(f"Loaded image: {self.current_metadata}")
        
        # Convert to grayscale for analysis
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Extract features from test image
        self.logger.info("Extracting features from test image...")
        test_features, _ = self.extract_ultra_comprehensive_features(test_image)
        
        # --- Global Analysis ---
        self.logger.info("Performing global anomaly analysis...")
        
        # Get reference statistics
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        
        # Ensure numpy arrays (in case loaded from JSON)
        for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
            if key in stat_model and isinstance(stat_model[key], list):
                stat_model[key] = np.array(stat_model[key], dtype=np.float64)
        
        # Create feature vector in consistent order
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        
        # Compute Mahalanobis distance
        diff = test_vector - stat_model['robust_mean']  # Difference from reference mean
        try:
            # Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
            mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
        except:
            # Fall back to normalized Euclidean if Mahalanobis fails
            std_vector = stat_model['std']
            std_vector[std_vector < 1e-6] = 1.0  # Avoid division by zero
            normalized_diff = diff / std_vector
            mahalanobis_dist = np.linalg.norm(normalized_diff)
        
        # Compute Z-scores for each feature
        z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
        
        # Find most deviant features
        top_indices = np.argsort(z_scores)[::-1][:10]  # Top 10 by Z-score
        deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                           for i in top_indices]
        
        # --- Individual Comparisons ---
        self.logger.info(f"Comparing against {len(self.reference_model['features'])} reference samples...")
        
        # Compare test against each reference sample
        individual_scores = []
        for i, ref_features in enumerate(self.reference_model['features']):
            # Compute comprehensive comparison
            comp = self.compute_exhaustive_comparison(test_features, ref_features)
            
            # Compute weighted anomaly score with bounds
            euclidean_term = min(comp['euclidean_distance'], 1000.0) * 0.2      # Cap at 1000
            manhattan_term = min(comp['manhattan_distance'], 10000.0) * 0.1     # Cap at 10000
            cosine_term = comp['cosine_distance'] * 0.2                         # Already bounded [0,2]
            correlation_term = (1 - abs(comp['pearson_correlation'])) * 0.1     # Bounded [0,1]
            kl_term = min(comp['kl_divergence'], 10.0) * 0.1                   # Cap at 10
            js_term = comp['js_divergence'] * 0.1                               # Already bounded
            chi_term = min(comp['chi_square'], 10.0) * 0.1                     # Cap at 10
            wasserstein_term = min(comp['wasserstein_distance'], 10.0) * 0.1   # Cap at 10
            
            # Sum weighted terms
            score = (euclidean_term + manhattan_term + cosine_term + 
                    correlation_term + kl_term + js_term + 
                    chi_term + wasserstein_term)
            
            # Cap the final score
            score = min(score, 100.0)
            
            individual_scores.append(score)
        
        # Compute statistics of individual comparisons
        scores_array = np.array(individual_scores)
        comparison_stats = {
            'mean': float(np.mean(scores_array)),      # Average comparison score
            'std': float(np.std(scores_array)),        # Variation in scores
            'min': float(np.min(scores_array)),        # Best match score
            'max': float(np.max(scores_array)),        # Worst match score
            'median': float(np.median(scores_array)),  # Median score
        }
        
        # --- Structural Analysis ---
        self.logger.info("Performing structural analysis...")
        
        # Get reference archetype image
        archetype = self.reference_model['archetype_image']
        # Convert from list if loaded from JSON
        if isinstance(archetype, list):
            archetype = np.array(archetype, dtype=np.uint8)
        # Resize test image to match archetype if needed
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
        else:
            test_gray_resized = test_gray
        
        # Compute structural similarity
        structural_comp = self.compute_image_structural_comparison(test_gray_resized, archetype)
        
        # --- Local Anomaly Detection ---
        self.logger.info("Detecting local anomalies...")
        
        # Compute local anomaly map using sliding window
        anomaly_map = self._compute_local_anomaly_map(test_gray_resized, archetype)
        
        # Find distinct anomaly regions
        anomaly_regions = self._find_anomaly_regions(anomaly_map, test_gray.shape)
        
        # --- Specific Defect Detection ---
        self.logger.info("Detecting specific defects...")
        specific_defects = self._detect_specific_defects(test_gray)
        
        # --- Determine Overall Status ---
        thresholds = self.reference_model['learned_thresholds']
        
        # Multiple criteria for anomaly detection
        is_anomalous = (
            mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6) or    # Statistical distance exceeds threshold
            comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6) or   # Worst match exceeds 95th percentile
            structural_comp['ssim'] < 0.7 or                                    # Low structural similarity
            len(anomaly_regions) > 3 or                                         # Many local anomalies
            any(region['confidence'] > 0.8 for region in anomaly_regions)       # High confidence anomaly
        )
        
        # Overall confidence score (maximum of normalized criteria)
        confidence = min(1.0, max(
            mahalanobis_dist / max(thresholds['anomaly_threshold'], 1e-6),      # Normalized Mahalanobis
            comparison_stats['max'] / max(thresholds['anomaly_p95'], 1e-6),     # Normalized comparison
            1 - structural_comp['ssim'],                                         # Inverted SSIM
            len(anomaly_regions) / 10                                           # Normalized region count
        ))
        
        self.logger.info("Analysis Complete!")
        
        # Return comprehensive results dictionary
        return {
            'test_image': test_image,                    # Original test image
            'test_gray': test_gray,                      # Grayscale version
            'test_features': test_features,              # Extracted features
            'metadata': self.current_metadata,           # Image metadata
            
            'global_analysis': {
                'mahalanobis_distance': float(mahalanobis_dist),  # Statistical distance
                'deviant_features': deviant_features,              # Most abnormal features
                'comparison_stats': comparison_stats,              # Individual comparison statistics
            },
            
            'structural_analysis': structural_comp,      # SSIM and related metrics
            
            'local_analysis': {
                'anomaly_map': anomaly_map,             # Pixel-wise anomaly scores
                'anomaly_regions': anomaly_regions,     # Detected anomaly regions
            },
            
            'specific_defects': specific_defects,        # Type-specific defects
            
            'verdict': {
                'is_anomalous': is_anomalous,           # Binary decision
                'confidence': float(confidence),         # Confidence in decision
                'criteria_triggered': {                  # Which criteria caused anomaly
                    'mahalanobis': mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6),
                    'comparison': comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6),
                    'structural': structural_comp['ssim'] < 0.7,
                    'local': len(anomaly_regions) > 3,
                }
            }
        }
    
    def _compute_local_anomaly_map(self, test_img, reference_img):
        """Compute local anomaly map using sliding window."""
        # Get image dimensions
        h, w = test_img.shape
        # Initialize anomaly map
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        
        # Multi-scale window sizes
        window_sizes = [16, 32, 64]
        
        # Process each window size
        for win_size in window_sizes:
            # Stride is half the window size for overlap
            stride = win_size // 2
            
            # Slide window over image
            for y in range(0, h - win_size + 1, stride):
                for x in range(0, w - win_size + 1, stride):
                    # Extract windows from both images
                    test_win = test_img[y:y+win_size, x:x+win_size]
                    ref_win = reference_img[y:y+win_size, x:x+win_size]
                    
                    # Compute local difference
                    diff = np.abs(test_win.astype(float) - ref_win.astype(float))
                    local_score = np.mean(diff) / 255.0  # Normalize to [0,1]
                    
                    # Simple SSIM approximation for window
                    mean_test = np.mean(test_win)      # Local mean of test
                    mean_ref = np.mean(ref_win)        # Local mean of reference
                    var_test = np.var(test_win)        # Local variance of test
                    var_ref = np.var(ref_win)          # Local variance of reference
                    cov = np.mean((test_win - mean_test) * (ref_win - mean_ref))  # Local covariance
                    
                    # SSIM constants
                    c1 = 0.01**2 * 255**2
                    c2 = 0.03**2 * 255**2
                    
                    # Compute SSIM approximation
                    ssim_approx = ((2 * mean_test * mean_ref + c1) * (2 * cov + c2)) / \
                                  ((mean_test**2 + mean_ref**2 + c1) * (var_test + var_ref + c2))
                    
                    # Use maximum of difference and inverted SSIM
                    local_score = max(local_score, 1 - ssim_approx)
                    
                    # Update anomaly map with maximum value
                    anomaly_map[y:y+win_size, x:x+win_size] = np.maximum(
                        anomaly_map[y:y+win_size, x:x+win_size],
                        local_score
                    )
        
        # Smooth anomaly map to reduce noise
        anomaly_map = cv2.GaussianBlur(anomaly_map, (15, 15), 0)
        
        return anomaly_map
    
    def _find_anomaly_regions(self, anomaly_map, original_shape):
        """Find distinct anomaly regions from the anomaly map."""
        # Extract positive anomaly values
        positive_values = anomaly_map[anomaly_map > 0]
        # Check if any anomalies exist
        if positive_values.size == 0:
            return []
        
        # Set threshold at 80th percentile of positive values
        threshold = np.percentile(positive_values, 80)
        # Create binary mask
        binary_map = (anomaly_map > threshold).astype(np.uint8)
        
        # Find connected components in binary map
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        
        # Initialize region list
        regions = []
        # Calculate scaling factors to original image size
        h_scale = original_shape[0] / anomaly_map.shape[0]
        w_scale = original_shape[1] / anomaly_map.shape[1]
        
        # Process each connected component (skip background at index 0)
        for i in range(1, num_labels):
            # Extract component statistics
            x, y, w, h, area = stats[i]
            
            # Scale coordinates to original image size
            x_orig = int(x * w_scale)
            y_orig = int(y * h_scale)
            w_orig = int(w * w_scale)
            h_orig = int(h * h_scale)
            
            # Create mask for this component
            region_mask = (labels == i)
            # Extract anomaly values in this region
            region_values = anomaly_map[region_mask]
            # Compute average anomaly score as confidence
            confidence = float(np.mean(region_values))
            
            # Filter out tiny regions (likely noise)
            if area > 20:
                regions.append({
                    'bbox': (x_orig, y_orig, w_orig, h_orig),  # Bounding box in original coordinates
                    'area': int(area * h_scale * w_scale),      # Area in original pixels
                    'confidence': confidence,                    # Average anomaly score
                    'centroid': (int(centroids[i][0] * w_scale), int(centroids[i][1] * h_scale)),  # Center point
                    'max_intensity': float(np.max(region_values)),  # Peak anomaly value
                })
        
        # Sort regions by confidence (highest first)
        regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return regions
    
    def _detect_specific_defects(self, gray):
        """Detect specific types of defects."""
        # Initialize defect dictionary
        defects = {
            'scratches': [],
            'digs': [],
            'blobs': [],
            'edges': [],
        }
        
        # Scratch detection using Hough line transform
        edges = cv2.Canny(gray, 30, 100)  # Edge detection
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                               minLineLength=20, maxLineGap=5)
        
        # Process detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]  # Extract endpoints
                # Calculate line length
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                # Filter short lines
                if length > 25:
                    defects['scratches'].append({
                        'line': (x1, y1, x2, y2),                          # Line endpoints
                        'length': float(length),                           # Line length
                        'angle': float(np.arctan2(y2-y1, x2-x1) * 180 / np.pi),  # Angle in degrees
                    })
        
        # Dig detection using morphological black-hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Circular kernel
        bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)       # Extract dark spots
        # Threshold at 95th percentile
        _, dig_mask = cv2.threshold(bth, np.percentile(bth, 95), 255, cv2.THRESH_BINARY)
        
        # Find contours of dark spots
        dig_contours, _ = cv2.findContours(dig_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in dig_contours:
            area = cv2.contourArea(contour)
            # Filter by size
            if self.config.min_defect_size < area < self.config.max_defect_size:
                # Calculate moments for centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])  # X centroid
                    cy = int(M["m01"] / M["m00"])  # Y centroid
                    defects['digs'].append({
                        'center': (cx, cy),         # Center point
                        'area': float(area),        # Area in pixels
                        'contour': contour,         # Contour points
                    })
        
        # Blob detection using adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5)  # Adaptive threshold
        
        # Morphological operations to clean up blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close gaps
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Find blob contours
        blob_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each blob
        for contour in blob_contours:
            area = cv2.contourArea(contour)
            # Filter large blobs
            if area > 100:
                perimeter = cv2.arcLength(contour, True)
                # Compute circularity (perfect circle = 1.0)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-10)  # Width/height ratio
                
                defects['blobs'].append({
                    'contour': contour,                     # Contour points
                    'bbox': (x, y, w, h),                   # Bounding box
                    'area': float(area),                    # Area
                    'circularity': float(circularity),      # Shape metric
                    'aspect_ratio': float(aspect_ratio),    # Shape ratio
                })
        
        # Edge irregularity detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # X gradient
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)  # Y gradient
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)   # Gradient magnitude
        
        # Threshold at 95th percentile
        edge_thresh = np.percentile(grad_mag, 95)
        edge_mask = (grad_mag > edge_thresh).astype(np.uint8)
        
        # Find edge contours
        edge_contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process edge irregularities
        for contour in edge_contours:
            area = cv2.contourArea(contour)
            # Filter by size
            if 50 < area < 2000:
                defects['edges'].append({
                    'contour': contour,      # Contour points
                    'area': float(area),     # Area
                })
        
        return defects
    
    # ==================== VISUALIZATION ====================
    
    def visualize_comprehensive_results(self, results, output_path):
        """Create comprehensive visualization of all anomaly detection results."""
        # Create large figure with 3x4 grid layout
        fig = plt.figure(figsize=(24, 16))
        
        # Create grid specification
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Get test image
        test_img = results['test_image']
        # Convert BGR to RGB for matplotlib
        if len(test_img.shape) == 3:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        else:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        
        # Get archetype image
        archetype = self.reference_model['archetype_image']
        archetype_rgb = cv2.cvtColor(archetype, cv2.COLOR_GRAY2RGB)
        
        # Panel 1: Original Test Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(test_img_rgb)
        ax1.set_title('Test Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Panel 2: Reference Archetype
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(archetype_rgb)
        ax2.set_title('Reference Archetype', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Panel 3: SSIM Map
        ax3 = fig.add_subplot(gs[0, 2])
        ssim_map = results['structural_analysis']['ssim_map']
        im3 = ax3.imshow(ssim_map, cmap='RdYlBu', vmin=0, vmax=1)  # Red-Yellow-Blue colormap
        ax3.set_title(f'SSIM Map (Index: {results["structural_analysis"]["ssim"]:.3f})', 
                     fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)  # Add colorbar
        
        # Panel 4: Local Anomaly Heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        anomaly_map = results['local_analysis']['anomaly_map']
        
        # Resize anomaly map to match test image if needed
        if anomaly_map.shape != test_img_rgb.shape[:2]:
            anomaly_map_resized = cv2.resize(anomaly_map, 
                                            (test_img_rgb.shape[1], test_img_rgb.shape[0]))
        else:
            anomaly_map_resized = anomaly_map
        
        ax4.imshow(test_img_rgb, alpha=0.7)  # Show test image as background
        im4 = ax4.imshow(anomaly_map_resized, cmap='hot', alpha=0.5, vmin=0)  # Overlay heatmap
        ax4.set_title('Local Anomaly Heatmap', fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # Panel 5: Detected Anomalies (Blue Highlights)
        ax5 = fig.add_subplot(gs[1, :2])  # Span two columns
        overlay = test_img_rgb.copy()
        
        # Draw anomaly regions in blue
        for region in results['local_analysis']['anomaly_regions']:
            x, y, w, h = region['bbox']
            # Draw blue rectangle outline
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Fill with semi-transparent blue
            roi = overlay[y:y+h, x:x+w]
            blue_overlay = np.zeros_like(roi)
            blue_overlay[:, :] = [0, 0, 255]  # Blue color
            cv2.addWeighted(roi, 0.7, blue_overlay, 0.3, 0, roi)  # Blend
            
            # Add confidence text
            cv2.putText(overlay, f'{region["confidence"]:.2f}', 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        ax5.imshow(overlay)
        ax5.set_title(f'Detected Anomalies ({len(results["local_analysis"]["anomaly_regions"])} regions)', 
                     fontsize=16, fontweight='bold', color='blue')
        ax5.axis('off')
        
        # Panel 6: Specific Defects
        ax6 = fig.add_subplot(gs[1, 2:])  # Span two columns
        defect_overlay = test_img_rgb.copy()
        
        # Draw specific defects with different colors
        defects = results['specific_defects']
        
        # Scratches - cyan lines
        for scratch in defects['scratches']:
            x1, y1, x2, y2 = scratch['line']
            cv2.line(defect_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Digs - magenta circles
        for dig in defects['digs']:
            cx, cy = dig['center']
            radius = int(np.sqrt(dig['area'] / np.pi))
            cv2.circle(defect_overlay, (cx, cy), max(3, radius), (255, 0, 255), -1)
        
        # Blobs - yellow contours
        cv2.drawContours(defect_overlay, [b['contour'] for b in defects['blobs']], 
                        -1, (255, 255, 0), 2)
        
        # Edges - green contours
        cv2.drawContours(defect_overlay, [e['contour'] for e in defects['edges']], 
                        -1, (0, 255, 0), 1)
        
        ax6.imshow(defect_overlay)
        # Create defect count string
        defect_counts = (f"Scratches: {len(defects['scratches'])}, " 
                        f"Digs: {len(defects['digs'])}, "
                        f"Blobs: {len(defects['blobs'])}, "
                        f"Edges: {len(defects['edges'])}")
        ax6.set_title(f'Specific Defects\n{defect_counts}', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        # Panel 7: Feature Deviation Chart
        ax7 = fig.add_subplot(gs[2, :2])  # Span two columns
        
        # Get top deviating features
        deviations = results['global_analysis']['deviant_features'][:8]  # Top 8
        names = [d[0].replace('_', '\n') for d in deviations]  # Format names
        z_scores = [d[1] for d in deviations]
        
        # Color code by severity
        colors = ['red' if z > 3 else 'orange' if z > 2 else 'yellow' for z in z_scores]
        
        # Create horizontal bar chart
        bars = ax7.barh(names, z_scores, color=colors)
        ax7.set_xlabel('Z-Score (Standard Deviations from Reference)', fontsize=12)
        ax7.set_title('Most Deviant Features', fontsize=14, fontweight='bold')
        ax7.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='2σ threshold')
        ax7.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3σ threshold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, z in zip(bars, z_scores):
            width = bar.get_width()
            ax7.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{z:.1f}', va='center', fontsize=10)
        
        # Panel 8: Analysis Summary
        ax8 = fig.add_subplot(gs[2, 2:])  # Span two columns
        ax8.axis('off')
        
        # Prepare summary text
        verdict = results['verdict']
        global_stats = results['global_analysis']
        structural = results['structural_analysis']
        
        # Create formatted summary
        summary_text = f"""COMPREHENSIVE ANALYSIS SUMMARY
        
Overall Verdict: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}
Confidence: {verdict['confidence']:.1%}

Global Analysis:
• Mahalanobis Distance: {global_stats['mahalanobis_distance']:.2f}
• Max Comparison Score: {global_stats['comparison_stats']['max']:.3f}
• Mean Comparison Score: {global_stats['comparison_stats']['mean']:.3f}

Structural Analysis:
• SSIM Index: {structural['ssim']:.3f}
• Mean Luminance Similarity: {structural['mean_luminance']:.3f}
• Mean Contrast Similarity: {structural['mean_contrast']:.3f}
• Mean Structure Similarity: {structural['mean_structure']:.3f}

Local Analysis:
• Anomaly Regions Found: {len(results['local_analysis']['anomaly_regions'])}
• Max Region Confidence: {max([r['confidence'] for r in results['local_analysis']['anomaly_regions']], default=0):.3f}

Criteria Triggered:
• Mahalanobis: {'✓' if verdict['criteria_triggered']['mahalanobis'] else '✗'}
• Comparison: {'✓' if verdict['criteria_triggered']['comparison'] else '✗'}
• Structural: {'✓' if verdict['criteria_triggered']['structural'] else '✗'}
• Local: {'✓' if verdict['criteria_triggered']['local'] else '✗'}"""
        
        # Add text with box
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Main title
        source_name = results['metadata'].get('filename', 'Unknown')
        fig.suptitle(f'Ultra-Comprehensive Anomaly Analysis\nTest: {source_name}', 
                    fontsize=20, fontweight='bold')
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Log save location
        self.logger.info(f"Visualization saved to: {output_path}")
        
        # Also save simplified version
        self._save_simple_anomaly_image(results, output_path.replace('.png', '_simple.png'))
    
    def _save_simple_anomaly_image(self, results, output_path):
        """Save a simple image with just anomalies highlighted in blue."""
        # Copy test image
        test_img = results['test_image'].copy()
        
        # Draw anomaly regions
        for region in results['local_analysis']['anomaly_regions']:
            x, y, w, h = region['bbox']
            
            # Draw blue rectangle
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # Fill with semi-transparent blue
            overlay = test_img.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, test_img, 0.7, 0, test_img)
        
        # Draw specific defects in blue
        defects = results['specific_defects']
        
        # All defects in blue
        for scratch in defects['scratches']:
            x1, y1, x2, y2 = scratch['line']
            cv2.line(test_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        for dig in defects['digs']:
            cx, cy = dig['center']
            radius = max(3, int(np.sqrt(dig['area'] / np.pi)))
            cv2.circle(test_img, (cx, cy), radius, (255, 0, 0), -1)
        
        cv2.drawContours(test_img, [b['contour'] for b in defects['blobs']], 
                        -1, (255, 0, 0), 2)
        
        # Add verdict text
        verdict = "ANOMALOUS" if results['verdict']['is_anomalous'] else "NORMAL"
        confidence = results['verdict']['confidence']
        
        cv2.putText(test_img, f"{verdict} ({confidence:.1%})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Save image
        cv2.imwrite(output_path, test_img)
        self.logger.info(f"Simple anomaly image saved to: {output_path}")
    
    # ==================== REPORT GENERATION ====================
    
    def generate_detailed_report(self, results, output_path):
        """Generate a detailed text report of the analysis."""
        # Open file for writing
        with open(output_path, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("ULTRA-COMPREHENSIVE ANOMALY DETECTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # File information section
            f.write("FILE INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Test File: {results['metadata'].get('filename', 'Unknown')}\n")
            f.write(f"Analysis Date: {self._get_timestamp()}\n")
            f.write(f"Image Dimensions: {results['test_gray'].shape}\n")
            f.write("\n")
            
            # Overall verdict section
            f.write("OVERALL VERDICT\n")
            f.write("-"*40 + "\n")
            verdict = results['verdict']
            f.write(f"Status: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}\n")
            f.write(f"Confidence: {verdict['confidence']:.1%}\n")
            f.write("\n")
            
            # Global analysis section
            f.write("GLOBAL STATISTICAL ANALYSIS\n")
            f.write("-"*40 + "\n")
            global_stats = results['global_analysis']
            f.write(f"Mahalanobis Distance: {global_stats['mahalanobis_distance']:.4f}\n")
            f.write(f"Comparison Scores:\n")
            f.write(f"  - Mean: {global_stats['comparison_stats']['mean']:.4f}\n")
            f.write(f"  - Std: {global_stats['comparison_stats']['std']:.4f}\n")
            f.write(f"  - Min: {global_stats['comparison_stats']['min']:.4f}\n")
            f.write(f"  - Max: {global_stats['comparison_stats']['max']:.4f}\n")
            f.write("\n")
            
            # Top deviant features section
            f.write("TOP DEVIANT FEATURES (Z-Score > 2)\n")
            f.write("-"*40 + "\n")
            # Iterate through top 10 deviant features
            for fname, z_score, test_val, ref_val in global_stats['deviant_features'][:10]:
                # Only show features with significant deviation
                if z_score > 2:
                    f.write(f"{fname:30} Z={z_score:6.2f}  Test={test_val:10.4f}  Ref={ref_val:10.4f}\n")
            f.write("\n")
            
            # Structural analysis section
            f.write("STRUCTURAL ANALYSIS\n")
            f.write("-"*40 + "\n")
            structural = results['structural_analysis']
            f.write(f"SSIM Index: {structural['ssim']:.4f}\n")
            f.write(f"Luminance Similarity: {structural['mean_luminance']:.4f}\n")
            f.write(f"Contrast Similarity: {structural['mean_contrast']:.4f}\n")
            f.write(f"Structure Similarity: {structural['mean_structure']:.4f}\n")
            f.write("\n")
            
            # Local anomalies section
            f.write("LOCAL ANOMALY REGIONS\n")
            f.write("-"*40 + "\n")
            regions = results['local_analysis']['anomaly_regions']
            f.write(f"Total Regions Found: {len(regions)}\n")
            # Detail first 5 regions
            for i, region in enumerate(regions[:5], 1):
                f.write(f"\nRegion {i}:\n")
                f.write(f"  - Location: {region['bbox']}\n")
                f.write(f"  - Area: {region['area']} pixels\n")
                f.write(f"  - Confidence: {region['confidence']:.3f}\n")
                f.write(f"  - Centroid: {region['centroid']}\n")
            # Note if more regions exist
            if len(regions) > 5:
                f.write(f"\n... and {len(regions) - 5} more regions\n")
            f.write("\n")
            
            # Specific defects section
            f.write("SPECIFIC DEFECTS DETECTED\n")
            f.write("-"*40 + "\n")
            defects = results['specific_defects']
            f.write(f"Scratches: {len(defects['scratches'])}\n")
            f.write(f"Digs: {len(defects['digs'])}\n")
            f.write(f"Blobs: {len(defects['blobs'])}\n")
            f.write(f"Edge Irregularities: {len(defects['edges'])}\n")
            f.write("\n")
            
            # Criteria summary section
            f.write("ANOMALY CRITERIA SUMMARY\n")
            f.write("-"*40 + "\n")
            criteria = verdict['criteria_triggered']
            f.write(f"Mahalanobis Threshold Exceeded: {'Yes' if criteria['mahalanobis'] else 'No'}\n")
            f.write(f"Comparison Threshold Exceeded: {'Yes' if criteria['comparison'] else 'No'}\n")
            f.write(f"Low Structural Similarity: {'Yes' if criteria['structural'] else 'No'}\n")
            f.write(f"Multiple Local Anomalies: {'Yes' if criteria['local'] else 'No'}\n")
            
            # Footer
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        # Log report location
        self.logger.info(f"Detailed report saved to: {output_path}")


def main():
    """Main execution function for standalone testing."""
    # Print banner
    print("\n" + "="*80)
    print("OMNIRIBER ANALYZER - DETECTION MODULE (v1.5)".center(80))
    print("="*80)
    print("\nThis module is designed to be called from app.py in the pipeline.")
    print("For standalone testing, you can analyze individual images.\n")
    
    # Create default configuration
    config = OmniConfig()
    
    # Initialize analyzer with configuration
    analyzer = OmniFiberAnalyzer(config)
    
    # Interactive testing loop
    while True:
        # Prompt user for image path
        test_path = input("\nEnter path to test image (or 'quit' to exit): ").strip()
        # Remove quotes if present
        test_path = test_path.strip('"\'')
        
        # Check for exit command
        if test_path.lower() == 'quit':
            break
            
        # Validate file exists
        if not os.path.isfile(test_path):
            print(f"✗ File not found: {test_path}")
            continue
            
        # Create unique output directory with timestamp
        output_dir = f"detection_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Run analysis
        print(f"\nAnalyzing {test_path}...")
        analyzer.analyze_end_face(test_path, output_dir)
        
        # Report output files
        print(f"\nResults saved to: {output_dir}/")
        print("  - JSON report: *_report.json")
        print("  - Visualization: *_analysis.png")
        print("  - Detailed text: *_detailed.txt")
    
    # Exit message
    print("\nThank you for using the OmniFiber Analyzer!")


# Entry point for script execution
if __name__ == "__main__":
    main()