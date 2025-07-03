"""
Unit tests for data_acquisition.py module
Tests defect aggregation and final analysis functions
"""

import unittest
import os
import sys
import numpy as np
import cv2
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_acquisition import DefectAggregator, integrate_with_pipeline
from tests.test_utils import (
    TestImageGenerator, TestDataManager, MockDefectReport,
    assert_image_valid, assert_json_structure, assert_file_exists
)

class TestDefectAggregator(unittest.TestCase):
    """Test the DefectAggregator class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create test directories
        self.results_dir = Path(self.temp_dir) / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create test image
        self.test_image = TestImageGenerator.create_fiber_optic_image()
        self.image_path = Path(self.test_manager.create_test_image_file(
            "original.jpg", self.test_image
        ))
        
        # Initialize aggregator
        self.aggregator = DefectAggregator(
            results_dir=self.results_dir,
            original_image_path=self.image_path,
            clustering_eps=30.0
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_initialization(self):
        """Test DefectAggregator initialization"""
        self.assertEqual(self.aggregator.results_dir, self.results_dir)
        self.assertEqual(self.aggregator.original_image_path, self.image_path)
        self.assertEqual(self.aggregator.clustering_eps, 30.0)
        self.assertEqual(self.aggregator.min_cluster_size, 1)
        self.assertIsInstance(self.aggregator.all_detections, list)
        self.assertIsInstance(self.aggregator.separation_masks, dict)
    
    def test_validate_detection_report(self):
        """Test detection report validation"""
        # Valid report
        valid_report = MockDefectReport.create_basic_report(
            "test.jpg",
            defects=[MockDefectReport.create_defect("scratch", (100, 100))]
        )
        
        is_valid = self.aggregator.validate_detection_report(
            valid_report, Path("test_report.json")
        )
        self.assertTrue(is_valid)
        
        # Invalid report - missing required field
        invalid_report = {"some_field": "value"}
        is_valid = self.aggregator.validate_detection_report(
            invalid_report, Path("invalid_report.json")
        )
        self.assertFalse(is_valid)
        
        # Invalid report - not a dictionary
        is_valid = self.aggregator.validate_detection_report(
            [], Path("list_report.json")
        )
        self.assertFalse(is_valid)
    
    def test_validate_defect_data(self):
        """Test individual defect data validation"""
        # Valid defect
        valid_defect = {
            'type': 'scratch',
            'location': {'x': 100, 'y': 100},
            'severity': 'minor',
            'confidence': 0.9
        }
        self.assertTrue(self.aggregator._validate_defect_data(valid_defect))
        
        # Missing type
        invalid_defect = {
            'location': {'x': 100, 'y': 100},
            'severity': 'minor'
        }
        self.assertFalse(self.aggregator._validate_defect_data(invalid_defect))
        
        # Invalid location format
        invalid_defect = {
            'type': 'scratch',
            'location': [100, 100],  # Should be dict
            'severity': 'minor'
        }
        self.assertFalse(self.aggregator._validate_defect_data(invalid_defect))
        
        # Missing location coordinates
        invalid_defect = {
            'type': 'scratch',
            'location': {'x': 100},  # Missing 'y'
            'severity': 'minor'
        }
        self.assertFalse(self.aggregator._validate_defect_data(invalid_defect))
    
    def test_determine_region_type(self):
        """Test region type determination"""
        # Test with source name hints
        self.assertEqual(
            self.aggregator._determine_region_type("core_analysis", {}),
            "core"
        )
        self.assertEqual(
            self.aggregator._determine_region_type("cladding_defects", {}),
            "cladding"
        )
        self.assertEqual(
            self.aggregator._determine_region_type("ferrule_inspection", {}),
            "ferrule"
        )
        
        # Test with report zones
        report_with_zones = {
            'zones': {
                'primary': 'core'
            }
        }
        self.assertEqual(
            self.aggregator._determine_region_type("any_name", report_with_zones),
            "core"
        )
        
        # Test unknown region
        self.assertIsNone(
            self.aggregator._determine_region_type("unknown_source", {})
        )
    
    def test_load_all_detection_results(self):
        """Test loading detection results from directory"""
        # Create test detection reports
        reports = []
        for i in range(3):
            defects = []
            if i > 0:  # Add defects to some reports
                defects.append(MockDefectReport.create_defect(
                    "scratch", (100 + i*50, 100 + i*50)
                ))
            
            report = MockDefectReport.create_basic_report(
                f"test_{i}.jpg", defects=defects
            )
            
            report_path = self.results_dir / f"detection_{i}_report.json"
            self.test_manager.create_test_report(report, report_path.name)
            reports.append(report)
        
        # Load results
        self.aggregator.load_all_detection_results()
        
        # Check loaded detections
        self.assertEqual(len(self.aggregator.all_detections), 2)  # Only reports with defects
        
        # Verify defect data
        for detection in self.aggregator.all_detections:
            self.assertIn('type', detection)
            self.assertIn('location', detection)
            self.assertIn('source_file', detection)
    
    def test_load_separation_masks(self):
        """Test loading separation masks"""
        # Create test mask files
        mask_data = {
            'masks': {
                'core': [[255 if i < 50 and j < 50 else 0 
                         for j in range(100)] for i in range(100)],
                'cladding': [[255 if i < 75 and j < 75 else 0 
                            for j in range(100)] for i in range(100)],
                'ferrule': [[255 for j in range(100)] for i in range(100)]
            }
        }
        
        # Save multiple mask files
        for i in range(2):
            mask_file = self.results_dir / f"image_{i}_masks.json"
            with open(mask_file, 'w') as f:
                json.dump(mask_data, f)
        
        # Load masks
        self.aggregator.load_separation_masks()
        
        # Check loaded masks
        self.assertGreater(len(self.aggregator.separation_masks), 0)
        
        for file_masks in self.aggregator.separation_masks.values():
            self.assertIn('core', file_masks)
            self.assertIn('cladding', file_masks)
            self.assertIn('ferrule', file_masks)
            
            # Check mask properties
            for mask in file_masks.values():
                self.assertIsInstance(mask, np.ndarray)
                self.assertEqual(mask.shape, (100, 100))
    
    def test_map_defect_to_global_coords(self):
        """Test mapping defect to global coordinates"""
        # Create test masks
        self.aggregator.separation_masks = {
            'test_masks.json': {
                'core': np.zeros((100, 100), dtype=np.uint8),
                'cladding': np.zeros((100, 100), dtype=np.uint8)
            }
        }
        
        # Fill core mask (small region)
        self.aggregator.separation_masks['test_masks.json']['core'][40:60, 40:60] = 255
        
        # Test defect in core
        defect = {
            'location': {'x': 10, 'y': 10},  # Relative to region
            'region_type': 'core',
            'source_file': 'test_masks.json'
        }
        
        global_coords = self.aggregator.map_defect_to_global_coords(defect)
        self.assertIsNotNone(global_coords)
        # Should be mapped to somewhere in the core region (40-60, 40-60)
        self.assertGreaterEqual(global_coords[0], 40)
        self.assertLess(global_coords[0], 60)
    
    def test_cluster_defects(self):
        """Test defect clustering"""
        # Create clustered defects
        self.aggregator.all_detections = [
            # Cluster 1 (close together)
            {
                'type': 'scratch',
                'location': {'x': 100, 'y': 100},
                'global_coords': (100, 100),
                'severity': 'minor',
                'confidence': 0.8
            },
            {
                'type': 'scratch',
                'location': {'x': 110, 'y': 110},
                'global_coords': (110, 110),
                'severity': 'minor',
                'confidence': 0.85
            },
            # Cluster 2 (far from cluster 1)
            {
                'type': 'contamination',
                'location': {'x': 300, 'y': 300},
                'global_coords': (300, 300),
                'severity': 'major',
                'confidence': 0.9
            }
        ]
        
        # Run clustering with eps=30
        clusters = self.aggregator.cluster_defects(custom_eps=30.0)
        
        # Should have 2 clusters
        self.assertEqual(len(clusters), 2)
        
        # Check cluster properties
        for cluster in clusters:
            self.assertIn('defects', cluster)
            self.assertIn('cluster_id', cluster)
            self.assertIsInstance(cluster['defects'], list)
            self.assertGreater(len(cluster['defects']), 0)
    
    def test_merge_defect_cluster(self):
        """Test merging defects within a cluster"""
        # Create test defects
        defects = [
            {
                'type': 'scratch',
                'location': {'x': 100, 'y': 100},
                'bounding_box': {'x': 95, 'y': 95, 'width': 10, 'height': 10},
                'severity': 'minor',
                'confidence': 0.8,
                'size': 10,
                'characteristics': {'orientation': 45}
            },
            {
                'type': 'scratch',
                'location': {'x': 105, 'y': 105},
                'bounding_box': {'x': 100, 'y': 100, 'width': 10, 'height': 10},
                'severity': 'major',
                'confidence': 0.9,
                'size': 12,
                'characteristics': {'orientation': 50}
            }
        ]
        
        merged = self.aggregator.merge_defect_cluster(defects)
        
        # Check merged properties
        self.assertEqual(merged['type'], 'scratch')
        self.assertEqual(merged['severity'], 'major')  # Should take most severe
        self.assertAlmostEqual(merged['confidence'], 0.85, places=2)  # Average
        
        # Check merged bounding box encompasses all defects
        self.assertLessEqual(merged['bounding_box']['x'], 95)
        self.assertLessEqual(merged['bounding_box']['y'], 95)
        self.assertGreaterEqual(
            merged['bounding_box']['x'] + merged['bounding_box']['width'],
            110
        )
    
    def test_intelligent_merge(self):
        """Test intelligent merging with mixed defect types"""
        defects = [
            {
                'type': 'scratch',
                'severity': 'minor',
                'confidence': 0.8,
                'characteristics': {}
            },
            {
                'type': 'contamination',
                'severity': 'major',
                'confidence': 0.9,
                'characteristics': {}
            },
            {
                'type': 'scratch',
                'severity': 'critical',
                'confidence': 0.95,
                'characteristics': {}
            }
        ]
        
        merged = self.aggregator.intelligent_merge(defects)
        
        # Should prioritize most common type (scratch)
        self.assertEqual(merged['type'], 'scratch')
        # Should take highest severity
        self.assertEqual(merged['severity'], 'critical')
        # Should have high confidence (weighted average)
        self.assertGreater(merged['confidence'], 0.85)
    
    def test_orientation_to_direction(self):
        """Test orientation angle to direction conversion"""
        test_cases = [
            (0, "horizontal"),
            (45, "diagonal"),
            (90, "vertical"),
            (135, "diagonal"),
            (180, "horizontal"),
            (-45, "diagonal"),
            (270, "vertical")
        ]
        
        for angle, expected_direction in test_cases:
            direction = self.aggregator.orientation_to_direction(angle)
            self.assertEqual(direction, expected_direction)
    
    def test_calculate_defect_heatmap(self):
        """Test defect heatmap generation"""
        # Create test defects
        merged_defects = [
            {
                'location': {'x': 100, 'y': 100},
                'severity': 'critical',
                'size': 20
            },
            {
                'location': {'x': 200, 'y': 200},
                'severity': 'minor',
                'size': 10
            }
        ]
        
        # Generate heatmap
        heatmap = self.aggregator.calculate_defect_heatmap(
            merged_defects, sigma=20, normalize=True
        )
        
        # Check heatmap properties
        assert_image_valid(heatmap)
        self.assertEqual(heatmap.shape[:2], self.test_image.shape[:2])
        
        # Check normalization
        if merged_defects:  # Only if we have defects
            self.assertLessEqual(np.max(heatmap), 1.0)
            self.assertGreaterEqual(np.min(heatmap), 0.0)
        
        # Check that defect locations have higher values
        self.assertGreater(heatmap[100, 100], heatmap[0, 0])
    
    def test_format_defect_for_report(self):
        """Test formatting defect for final report"""
        defect = {
            'type': 'scratch',
            'location': {'x': 100, 'y': 100},
            'severity': 'major',
            'confidence': 0.9,
            'size': 15,
            'region': 'core',
            'characteristics': {
                'orientation': 45,
                'aspect_ratio': 2.5
            },
            'merged_from': ['source1', 'source2']
        }
        
        formatted = self.aggregator.format_defect_for_report(defect, 1)
        
        # Check required fields
        assert_json_structure(formatted, [
            'id', 'type', 'location', 'severity', 'confidence', 'region'
        ])
        
        self.assertEqual(formatted['id'], 'defect_001')
        self.assertEqual(formatted['type'], 'scratch')
        self.assertIn('orientation_desc', formatted)
    
    def test_generate_final_report(self):
        """Test final report generation"""
        # Create test defects
        merged_defects = [
            {
                'type': 'scratch',
                'location': {'x': 100, 'y': 100},
                'severity': 'critical',
                'confidence': 0.95,
                'size': 20,
                'region': 'core'
            },
            {
                'type': 'contamination',
                'location': {'x': 200, 'y': 200},
                'severity': 'minor',
                'confidence': 0.8,
                'size': 10,
                'region': 'cladding'
            }
        ]
        
        output_path = Path(self.temp_dir) / "final_report.json"
        
        # Generate report
        report = self.aggregator.generate_final_report(merged_defects, output_path)
        
        # Check report structure
        assert_json_structure(report, [
            'timestamp', 'original_image', 'total_defects',
            'quality_assessment', 'defect_summary', 'defects',
            'regional_analysis', 'processing_info'
        ])
        
        # Check quality assessment
        self.assertIn('overall_quality', report['quality_assessment'])
        self.assertIn('severity_distribution', report['quality_assessment'])
        
        # Check defect summary
        self.assertEqual(report['defect_summary']['total'], 2)
        self.assertEqual(report['defect_summary']['by_type']['scratch'], 1)
        self.assertEqual(report['defect_summary']['by_type']['contamination'], 1)
        
        # Check file was saved
        assert_file_exists(str(output_path))
    
    @patch('cv2.imwrite')
    @patch('matplotlib.pyplot.savefig')
    def test_create_comprehensive_visualization(self, mock_savefig, mock_imwrite):
        """Test visualization creation"""
        # Mock file writing
        mock_imwrite.return_value = True
        
        merged_defects = [
            {
                'type': 'scratch',
                'location': {'x': 100, 'y': 100},
                'bounding_box': {'x': 90, 'y': 90, 'width': 20, 'height': 20},
                'severity': 'major'
            }
        ]
        
        output_path = Path(self.temp_dir) / "visualization.png"
        
        # Create visualization
        self.aggregator.create_comprehensive_visualization(
            merged_defects, output_path
        )
        
        # Check that save methods were called
        mock_savefig.assert_called()
    
    def test_run_complete_analysis(self):
        """Test complete analysis workflow"""
        # Create minimal test data
        report = MockDefectReport.create_basic_report(
            "test.jpg",
            defects=[MockDefectReport.create_defect("scratch", (100, 100))]
        )
        
        report_path = self.results_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)
        
        # Run analysis
        final_report = self.aggregator.run_complete_analysis()
        
        # Check report generated
        self.assertIsInstance(final_report, dict)
        assert_json_structure(final_report, [
            'timestamp', 'total_defects', 'quality_assessment'
        ])
    
    def test_empty_results_handling(self):
        """Test handling of empty results directory"""
        # Run analysis with no detection results
        final_report = self.aggregator.run_complete_analysis()
        
        # Should still generate a valid report
        self.assertIsInstance(final_report, dict)
        self.assertEqual(final_report['total_defects'], 0)
        self.assertEqual(final_report['quality_assessment']['overall_quality'], 'excellent')

class TestIntegrationFunction(unittest.TestCase):
    """Test the integrate_with_pipeline function"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    @patch('data_acquisition.DefectAggregator')
    def test_integrate_with_pipeline(self, mock_aggregator_class):
        """Test pipeline integration function"""
        # Mock aggregator
        mock_aggregator = Mock()
        mock_aggregator.run_complete_analysis.return_value = {
            'status': 'complete',
            'total_defects': 5
        }
        mock_aggregator_class.return_value = mock_aggregator
        
        # Call integration function
        result = integrate_with_pipeline(
            results_base_dir=self.temp_dir,
            image_name="test_image",
            clustering_eps=25.0
        )
        
        # Verify result
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['total_defects'], 5)
        
        # Verify aggregator was initialized correctly
        mock_aggregator_class.assert_called_once()
        call_kwargs = mock_aggregator_class.call_args[1]
        self.assertEqual(call_kwargs['clustering_eps'], 25.0)

if __name__ == '__main__':
    unittest.main()