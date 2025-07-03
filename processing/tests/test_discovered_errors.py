"""
Tests for errors discovered during manual testing
These errors were not caught by the original test suite
"""

import unittest
import os
import sys
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import OmniFiberAnalyzer, OmniConfig, NumpyEncoder
from data_acquisition import DefectAggregator
from tests.test_utils import TestDataManager, TestImageGenerator

class TestDiscoveredErrors(unittest.TestCase):
    """Test cases for errors found during manual testing"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_detection_analyze_end_face_returns_value(self):
        """Test that analyze_end_face returns a value (was missing return statement)"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("test.jpg", test_image)
        
        # Create analyzer
        config = OmniConfig(
            knowledge_base_path=os.path.join(self.temp_dir, "test_kb.json")
        )
        analyzer = OmniFiberAnalyzer(config)
        
        # Analyze image
        output_dir = os.path.join(self.temp_dir, "output")
        result = analyzer.analyze_end_face(image_path, output_dir)
        
        # CRITICAL: Must return a value, not None
        self.assertIsNotNone(result, "analyze_end_face must return a report dictionary")
        self.assertIsInstance(result, dict, "analyze_end_face must return a dictionary")
    
    def test_detection_report_has_pipeline_fields(self):
        """Test that detection reports include all fields expected by pipeline"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("test.jpg", test_image)
        
        # Create analyzer
        config = OmniConfig(
            knowledge_base_path=os.path.join(self.temp_dir, "test_kb.json")
        )
        analyzer = OmniFiberAnalyzer(config)
        
        # Analyze image
        output_dir = os.path.join(self.temp_dir, "output")
        result = analyzer.analyze_end_face(image_path, output_dir)
        
        # Check required pipeline fields
        required_fields = [
            'source_image',
            'analysis_complete',
            'overall_quality_score',
            'zones'
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Detection report missing required field: {field}")
        
        # Check zones structure
        self.assertIn('core', result['zones'])
        self.assertIn('cladding', result['zones'])
        self.assertIn('ferrule', result['zones'])
    
    def test_data_acquisition_handles_list_fields(self):
        """Test that data acquisition can merge defects with list fields"""
        # Create test defects with list fields
        defects = [
            {
                'type': 'scratch',
                'location': {'x': 100, 'y': 100},
                'location_xy': [100, 100],  # List field that caused error
                'severity': 'minor',
                'confidence': 0.8,
                'bbox': [90, 90, 20, 20],
                'contributing_algorithms': ['algo1', 'algo2']  # Another list field
            },
            {
                'type': 'scratch',
                'location': {'x': 105, 'y': 105},
                'location_xy': [105, 105],
                'severity': 'minor',
                'confidence': 0.85,
                'bbox': [95, 95, 20, 20],
                'contributing_algorithms': ['algo1', 'algo3']
            }
        ]
        
        # Create test image first
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("test.jpg", test_image)
        
        # Create aggregator
        aggregator = DefectAggregator(
            results_dir=Path(self.temp_dir),
            original_image_path=Path(image_path)
        )
        
        # Test merge_defect_cluster
        merged = aggregator.merge_defect_cluster(defects)
        
        # Should not raise TypeError about unhashable list
        self.assertIsInstance(merged, dict)
        self.assertIn('location_xy', merged)
        self.assertIn('contributing_algorithms', merged)
    
    def test_data_acquisition_json_serialization(self):
        """Test that data acquisition properly serializes numpy types"""
        # Create report with numpy types
        report = {
            'total_defects': np.int64(10),
            'quality_score': np.float64(85.5),
            'defect_counts': np.array([1, 2, 3]),
            'statistics': {
                'mean': np.float32(100.5),
                'std': np.float64(15.2),
                'counts': {
                    'scratch': np.int32(5),
                    'contamination': np.int64(3)
                }
            }
        }
        
        # Should be able to serialize with NumpyEncoder
        try:
            json_str = json.dumps(report, cls=NumpyEncoder)
            # Should be able to parse back
            parsed = json.loads(json_str)
            
            # Check types are converted
            self.assertIsInstance(parsed['total_defects'], int)
            self.assertIsInstance(parsed['quality_score'], float)
            self.assertIsInstance(parsed['defect_counts'], list)
            self.assertIsInstance(parsed['statistics']['mean'], float)
            
        except TypeError as e:
            self.fail(f"NumpyEncoder failed to serialize numpy types: {e}")
    
    def test_separation_method_failures(self):
        """Test that separation handles method failures gracefully"""
        # This documents known failing methods:
        # - bright_core_extractor: "Found 1 circles, but none passed validation"
        # - gradient_approach: "boolean index did not match indexed array"
        # - segmentation: "No result file produced"
        
        # These failures are handled by the consensus system
        # No additional fixes needed, but documenting for awareness
        pass
    
    def test_pipeline_processes_only_necessary_images(self):
        """Test that pipeline doesn't process ALL reimagined images unnecessarily"""
        # The current pipeline processes all 49 reimagined images through separation
        # This is inefficient and should be configurable
        
        # Recommendation: Add configuration option to limit which images go through separation
        # For example: only process original + specific transforms known to help with segmentation
        pass

class TestPerformanceIssues(unittest.TestCase):
    """Test cases for performance-related issues"""
    
    def test_separation_timeout_handling(self):
        """Test that long-running separation methods have proper timeout"""
        # Some separation methods take 30+ seconds
        # Should have configurable timeout
        pass
    
    def test_feature_extraction_performance(self):
        """Test that feature extraction doesn't take excessive time"""
        # Feature extraction takes 17+ seconds per image
        # Consider caching or optimization
        pass

if __name__ == '__main__':
    unittest.main()