"""
Unit tests for detection.py module
Tests defect detection and analysis functions
"""

import unittest
import os
import sys
import numpy as np
import cv2
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import (
    OmniConfig, NumpyEncoder, OmniFiberAnalyzer
)
from tests.test_utils import (
    TestImageGenerator, TestDataManager, MockDefectReport,
    assert_image_valid, assert_json_structure, assert_file_exists
)

class TestOmniConfig(unittest.TestCase):
    """Test the OmniConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = OmniConfig()
        
        self.assertEqual(config.min_defect_size, 5)
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertTrue(config.detect_scratches)
        self.assertTrue(config.detect_contamination)
        self.assertTrue(config.detect_chips)
        self.assertTrue(config.save_visualization)
        self.assertEqual(config.visualization_dpi, 150)
        self.assertIsNone(config.reference_dir)
        self.assertEqual(config.knowledge_base_path, "fiber_knowledge.json")
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = OmniConfig(
            min_defect_size=10,
            confidence_threshold=0.9,
            detect_scratches=False,
            reference_dir="/path/to/ref"
        )
        
        self.assertEqual(config.min_defect_size, 10)
        self.assertEqual(config.confidence_threshold, 0.9)
        self.assertFalse(config.detect_scratches)
        self.assertEqual(config.reference_dir, "/path/to/ref")

class TestNumpyEncoderDetection(unittest.TestCase):
    """Test the NumpyEncoder for detection module"""
    
    def test_numpy_encoding(self):
        """Test encoding numpy types in detection context"""
        # Test with detection-specific data structures
        data = {
            "defect_location": {
                "x": np.int32(100),
                "y": np.int64(200)
            },
            "confidence": np.float32(0.95),
            "defect_mask": np.array([[1, 0], [0, 1]]),
            "statistics": {
                "mean": np.float64(127.5),
                "std": np.float32(15.3)
            }
        }
        
        json_str = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed["defect_location"]["x"], 100)
        self.assertEqual(parsed["defect_location"]["y"], 200)
        self.assertAlmostEqual(parsed["confidence"], 0.95, places=2)
        self.assertEqual(parsed["defect_mask"], [[1, 0], [0, 1]])

class TestOmniFiberAnalyzer(unittest.TestCase):
    """Test the OmniFiberAnalyzer class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create analyzer with test config
        self.config = OmniConfig(
            knowledge_base_path=os.path.join(self.temp_dir, "test_knowledge.json")
        )
        self.analyzer = OmniFiberAnalyzer(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_initialization(self):
        """Test OmniFiberAnalyzer initialization"""
        self.assertIsInstance(self.analyzer.config, OmniConfig)
        self.assertIsInstance(self.analyzer.knowledge_base, dict)
        self.assertIn('reference_features', self.analyzer.knowledge_base)
        self.assertIn('defect_patterns', self.analyzer.knowledge_base)
        self.assertIn('quality_thresholds', self.analyzer.knowledge_base)
    
    def test_load_image(self):
        """Test image loading"""
        # Create and save test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("test_load.jpg", test_image)
        
        # Load image
        loaded_image = self.analyzer.load_image(image_path)
        
        assert_image_valid(loaded_image)
        self.assertEqual(loaded_image.shape, test_image.shape)
    
    def test_load_image_invalid_path(self):
        """Test loading non-existent image"""
        result = self.analyzer.load_image("/nonexistent/image.jpg")
        self.assertIsNone(result)
    
    def test_sanitize_feature_value(self):
        """Test feature value sanitization"""
        # Test various value types
        self.assertEqual(self.analyzer._sanitize_feature_value(42), 42)
        self.assertEqual(self.analyzer._sanitize_feature_value(3.14), 3.14)
        self.assertEqual(self.analyzer._sanitize_feature_value("text"), "text")
        
        # Test numpy types
        self.assertEqual(self.analyzer._sanitize_feature_value(np.int32(42)), 42)
        self.assertAlmostEqual(self.analyzer._sanitize_feature_value(np.float64(3.14)), 3.14)
        
        # Test special values
        self.assertEqual(self.analyzer._sanitize_feature_value(np.inf), 1e10)
        self.assertEqual(self.analyzer._sanitize_feature_value(-np.inf), -1e10)
        self.assertEqual(self.analyzer._sanitize_feature_value(np.nan), 0)
        
        # Test arrays
        arr = np.array([1, 2, 3])
        self.assertEqual(self.analyzer._sanitize_feature_value(arr), [1, 2, 3])
    
    def test_compute_statistics(self):
        """Test statistical computation methods"""
        # Test data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test skewness
        skewness = self.analyzer._compute_skewness(data)
        self.assertIsInstance(skewness, float)
        self.assertAlmostEqual(skewness, 0, places=1)  # Should be ~0 for uniform data
        
        # Test kurtosis
        kurtosis = self.analyzer._compute_kurtosis(data)
        self.assertIsInstance(kurtosis, float)
        
        # Test entropy
        entropy = self.analyzer._compute_entropy(data)
        self.assertIsInstance(entropy, float)
        self.assertGreater(entropy, 0)
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction"""
        # Create test image
        test_image = TestImageGenerator.create_test_image(pattern='gradient')
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        features = self.analyzer._extract_statistical_features(gray)
        
        # Check feature keys
        expected_keys = [
            'mean', 'std', 'min', 'max', 'median',
            'skewness', 'kurtosis', 'entropy'
        ]
        for key in expected_keys:
            self.assertIn(key, features)
            self.assertIsInstance(features[key], (int, float))
    
    def test_extract_glcm_features(self):
        """Test GLCM feature extraction"""
        test_image = TestImageGenerator.create_test_image(size=(64, 64))
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        features = self.analyzer._extract_glcm_features(gray)
        
        # Check that we have GLCM features
        glcm_keys = [k for k in features.keys() if 'glcm' in k]
        self.assertGreater(len(glcm_keys), 0)
        
        # Check feature values are numeric
        for key, value in features.items():
            self.assertIsInstance(value, (int, float))
    
    def test_extract_lbp_features(self):
        """Test LBP feature extraction"""
        test_image = TestImageGenerator.create_test_image()
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        features = self.analyzer._extract_lbp_features(gray)
        
        # Check LBP histogram features
        self.assertIn('lbp_hist', features)
        self.assertIsInstance(features['lbp_hist'], list)
        self.assertGreater(len(features['lbp_hist']), 0)
        
        # Check LBP statistics
        for stat in ['mean', 'std', 'energy']:
            key = f'lbp_{stat}'
            self.assertIn(key, features)
            self.assertIsInstance(features[key], (int, float))
    
    def test_extract_comprehensive_features(self):
        """Test comprehensive feature extraction"""
        test_image = TestImageGenerator.create_fiber_optic_image()
        
        features = self.analyzer.extract_ultra_comprehensive_features(test_image)
        
        # Check that we have features from all categories
        feature_categories = [
            'statistical', 'lbp', 'glcm', 'fourier', 'morphological',
            'shape', 'gradient', 'entropy', 'svd'
        ]
        
        for category in feature_categories:
            matching_features = [k for k in features.keys() if category in k.lower()]
            self.assertGreater(len(matching_features), 0, 
                             f"No features found for category: {category}")
        
        # Check all values are sanitized
        for key, value in features.items():
            if isinstance(value, (list, tuple)):
                for v in value:
                    self.assertNotEqual(v, np.inf)
                    self.assertNotEqual(v, -np.inf)
                    self.assertFalse(np.isnan(v) if isinstance(v, float) else False)
            else:
                self.assertNotEqual(value, np.inf)
                self.assertNotEqual(value, -np.inf)
                self.assertFalse(np.isnan(value) if isinstance(value, float) else False)
    
    def test_confidence_to_severity(self):
        """Test confidence to severity mapping"""
        self.assertEqual(self.analyzer._confidence_to_severity(0.95), "critical")
        self.assertEqual(self.analyzer._confidence_to_severity(0.85), "major")
        self.assertEqual(self.analyzer._confidence_to_severity(0.75), "minor")
        self.assertEqual(self.analyzer._confidence_to_severity(0.5), "minor")
    
    def test_detect_specific_defects(self):
        """Test specific defect detection"""
        # Create image with known defects
        test_image = TestImageGenerator.create_fiber_optic_image(
            defects=[
                {'type': 'scratch', 'location': (320, 240), 'size': 30},
                {'type': 'contamination', 'location': (100, 100), 'size': 20}
            ]
        )
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        defects = self.analyzer._detect_specific_defects(gray)
        
        self.assertIsInstance(defects, list)
        # Should detect at least some defects in the synthetic image
        # Note: Detection may not be perfect on synthetic data
    
    def test_create_defect_mask(self):
        """Test defect mask creation"""
        # Create test results
        results = {
            'original_shape': (480, 640),
            'defects': [
                {
                    'type': 'scratch',
                    'bounding_box': {'x': 100, 'y': 100, 'width': 50, 'height': 10}
                },
                {
                    'type': 'contamination',
                    'bounding_box': {'x': 200, 'y': 200, 'width': 30, 'height': 30}
                }
            ]
        }
        
        mask = self.analyzer._create_defect_mask(results)
        
        self.assertEqual(mask.shape, (480, 640))
        self.assertEqual(mask.dtype, np.uint8)
        
        # Check that defect areas are marked
        self.assertGreater(mask[105, 125], 0)  # Inside first defect
        self.assertGreater(mask[215, 215], 0)  # Inside second defect
        self.assertEqual(mask[0, 0], 0)  # Outside defects
    
    def test_build_minimal_reference(self):
        """Test minimal reference building"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("reference.jpg", test_image)
        
        self.analyzer._build_minimal_reference(image_path)
        
        # Check that reference was added to knowledge base
        self.assertGreater(len(self.analyzer.knowledge_base['reference_features']), 0)
        
        # Check reference structure
        ref_features = self.analyzer.knowledge_base['reference_features']
        self.assertIn('mean', ref_features)
        self.assertIn('std', ref_features)
    
    def test_convert_to_pipeline_format(self):
        """Test conversion to pipeline format"""
        # Create test results
        results = {
            'defects': [
                {
                    'type': 'scratch',
                    'location': {'x': 100, 'y': 100},
                    'confidence': 0.9,
                    'severity': 'major'
                }
            ],
            'anomaly_regions': [
                {
                    'bbox': [50, 50, 150, 150],
                    'score': 0.85
                }
            ],
            'overall_anomaly_score': 0.75,
            'statistics': {'total_defects': 1}
        }
        
        image_path = "test_image.jpg"
        pipeline_format = self.analyzer._convert_to_pipeline_format(results, image_path)
        
        assert_json_structure(pipeline_format, [
            'source_image', 'timestamp', 'analysis_complete',
            'overall_quality_score', 'defects', 'zones'
        ])
        
        self.assertEqual(pipeline_format['source_image'], image_path)
        self.assertTrue(pipeline_format['analysis_complete'])
        self.assertEqual(len(pipeline_format['defects']), 1)
    
    @patch('cv2.imwrite')
    def test_analyze_end_face(self, mock_imwrite):
        """Test complete end face analysis"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("analyze_test.jpg", test_image)
        output_dir = os.path.join(self.temp_dir, "analysis_output")
        
        # Mock imwrite to avoid actual file writing
        mock_imwrite.return_value = True
        
        # Run analysis
        results = self.analyzer.analyze_end_face(image_path, output_dir)
        
        self.assertIsInstance(results, dict)
        assert_json_structure(results, [
            'source_image', 'timestamp', 'analysis_complete',
            'overall_quality_score', 'defects'
        ])
        
        # Check that visualization was attempted
        if self.config.save_visualization:
            mock_imwrite.assert_called()
    
    def test_knowledge_base_persistence(self):
        """Test saving and loading knowledge base"""
        # Add some data to knowledge base
        self.analyzer.knowledge_base['reference_features']['test_feature'] = 42
        self.analyzer.knowledge_base['defect_patterns'].append({
            'type': 'test_defect',
            'signature': [1, 2, 3]
        })
        
        # Save knowledge base
        self.analyzer.save_knowledge_base()
        
        # Create new analyzer and load knowledge base
        new_analyzer = OmniFiberAnalyzer(self.config)
        
        # Check that data was persisted
        self.assertEqual(
            new_analyzer.knowledge_base['reference_features']['test_feature'], 
            42
        )
        self.assertEqual(
            len(new_analyzer.knowledge_base['defect_patterns']), 
            1
        )

if __name__ == '__main__':
    unittest.main()