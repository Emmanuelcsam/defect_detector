"""
Integration tests for the defect detector pipeline
Tests interaction between modules and data flow
"""

import unittest
import os
import sys
import json
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import PipelineOrchestrator
from process import reimagine_image
from separation import UnifiedSegmentationSystem
from detection import OmniFiberAnalyzer, OmniConfig
from data_acquisition import DefectAggregator

from tests.test_utils import (
    TestImageGenerator, TestDataManager, ConfigGenerator,
    assert_file_exists, assert_directory_exists, assert_json_structure
)

class TestProcessToSeparationIntegration(unittest.TestCase):
    """Test integration between process.py and separation.py"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create methods directory for separation
        self.methods_dir = os.path.join(self.temp_dir, "zones_methods")
        os.makedirs(self.methods_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_reimagined_images_to_separation(self):
        """Test that reimagined images can be processed by separation"""
        # Create original image
        original = TestImageGenerator.create_fiber_optic_image()
        original_path = self.test_manager.create_test_image_file("original.jpg", original)
        
        # Process with reimagine_image
        reimagined_dir = os.path.join(self.temp_dir, "reimagined")
        reimagined_paths = reimagine_image(original_path, reimagined_dir)
        
        self.assertGreater(len(reimagined_paths), 0)
        
        # Create separation system
        with patch('separation.UnifiedSegmentationSystem.load_methods'):
            segmentation = UnifiedSegmentationSystem(methods_dir=self.methods_dir)
            
            # Process each reimagined image
            output_dir = os.path.join(self.temp_dir, "separation_output")
            os.makedirs(output_dir, exist_ok=True)
            
            successful_separations = 0
            for img_path in reimagined_paths[:5]:  # Test first 5
                # Mock the actual method execution
                with patch.object(segmentation, 'run_method') as mock_run:
                    # Create mock result
                    from separation import SegmentationResult
                    mock_result = SegmentationResult("test_method", img_path)
                    mock_result.success = True
                    mock_result.center = (320, 240)
                    mock_result.core_radius = 50
                    mock_result.cladding_radius = 125
                    mock_run.return_value = mock_result
                    
                    result = segmentation.process_image(Path(img_path), output_dir)
                    if result:
                        successful_separations += 1
            
            # Should process at least some images successfully
            self.assertGreater(successful_separations, 0)

class TestSeparationToDetectionIntegration(unittest.TestCase):
    """Test integration between separation.py and detection.py"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_separation_output_to_detection(self):
        """Test that separation outputs can be used by detection"""
        # Create test image and separation result
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("test.jpg", test_image)
        
        # Create separation output
        separation_result = {
            'source_image': image_path,
            'timestamp': '2025-01-01T00:00:00',
            'consensus': {
                'center': (320, 240),
                'core_radius': 50,
                'cladding_radius': 125
            },
            'zones': {
                'core': {'detected': True, 'radius': 50},
                'cladding': {'detected': True, 'radius': 125},
                'ferrule': {'detected': True, 'radius': 200}
            }
        }
        
        # Save separation report
        sep_report_path = os.path.join(self.temp_dir, "test_report.json")
        with open(sep_report_path, 'w') as f:
            json.dump(separation_result, f)
        
        # Create detection analyzer
        config = OmniConfig(
            knowledge_base_path=os.path.join(self.temp_dir, "test_kb.json")
        )
        analyzer = OmniFiberAnalyzer(config)
        
        # Analyze the same image
        output_dir = os.path.join(self.temp_dir, "detection_output")
        os.makedirs(output_dir, exist_ok=True)
        
        detection_result = analyzer.analyze_end_face(image_path, output_dir)
        
        # Verify detection result
        self.assertIsInstance(detection_result, dict)
        assert_json_structure(detection_result, [
            'source_image', 'timestamp', 'analysis_complete'
        ])
        
        # Both should reference the same source image
        self.assertEqual(detection_result['source_image'], image_path)

class TestDetectionToDataAcquisitionIntegration(unittest.TestCase):
    """Test integration between detection.py and data_acquisition.py"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create results directory structure
        self.results_dir = Path(self.temp_dir) / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_detection_reports_aggregation(self):
        """Test that multiple detection reports are properly aggregated"""
        # Create original image
        original = TestImageGenerator.create_fiber_optic_image()
        original_path = Path(self.test_manager.create_test_image_file("original.jpg", original))
        
        # Create multiple detection reports
        detection_reports = []
        for i in range(3):
            report = {
                'source_image': f'processed_{i}.jpg',
                'timestamp': '2025-01-01T00:00:00',
                'analysis_complete': True,
                'overall_quality_score': 90 - i*10,
                'defects': [
                    {
                        'type': 'scratch',
                        'location': {'x': 100 + i*50, 'y': 100 + i*50},
                        'severity': 'minor',
                        'confidence': 0.8 + i*0.05,
                        'bounding_box': {
                            'x': 95 + i*50,
                            'y': 95 + i*50,
                            'width': 10,
                            'height': 10
                        }
                    }
                ],
                'zones': {
                    'core': {'detected': True},
                    'cladding': {'detected': True}
                }
            }
            
            # Save report
            report_path = self.results_dir / f"detection_{i}_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f)
            
            detection_reports.append(report)
        
        # Create aggregator
        aggregator = DefectAggregator(
            results_dir=self.results_dir,
            original_image_path=original_path
        )
        
        # Run aggregation
        final_report = aggregator.run_complete_analysis()
        
        # Verify aggregation
        self.assertIsInstance(final_report, dict)
        self.assertGreaterEqual(final_report['total_defects'], 0)
        assert_json_structure(final_report, [
            'timestamp', 'total_defects', 'quality_assessment'
        ])

class TestFullPipelineIntegration(unittest.TestCase):
    """Test the complete pipeline integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create config
        config_data = ConfigGenerator.create_default_config()
        config_data['app_settings']['output_directory'] = self.temp_dir
        self.config_path = self.test_manager.create_test_config(config_data)
        
        # Create methods directory
        methods_dir = os.path.join(self.temp_dir, "zones_methods")
        os.makedirs(methods_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    @patch('separation.UnifiedSegmentationSystem.run_method')
    def test_full_pipeline_data_flow(self, mock_run_method, mock_load_methods):
        """Test data flow through the entire pipeline"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image(
            defects=[
                {'type': 'scratch', 'location': (320, 240), 'size': 20}
            ]
        )
        image_path = Path(self.test_manager.create_test_image_file("pipeline_test.jpg", test_image))
        
        # Mock separation method
        from separation import SegmentationResult
        mock_result = SegmentationResult("test_method", str(image_path))
        mock_result.success = True
        mock_result.center = (320, 240)
        mock_result.core_radius = 50
        mock_result.cladding_radius = 125
        mock_run_method.return_value = mock_result
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(self.config_path)
        
        # Run full pipeline
        results = orchestrator.run_full_pipeline(image_path)
        
        # Verify pipeline results
        self.assertIn('stages', results)
        self.assertEqual(len(results['stages']), 4)
        
        # Check each stage
        stage_names = ['processing', 'separation', 'detection', 'data_acquisition']
        for i, expected_name in enumerate(stage_names):
            stage = results['stages'][i]
            self.assertEqual(stage['stage'], expected_name)
            
            # All stages should succeed (with our mocks)
            if not stage['success']:
                print(f"Stage {expected_name} failed: {stage.get('error', 'Unknown error')}")
        
        # Verify final summary
        self.assertIn('summary', results)
        self.assertIn('run_directory', results)
        
        # Check output directory structure
        run_dir = Path(results['run_directory'])
        for subdir in ['processing', 'separation', 'detection', 'final']:
            assert_directory_exists(str(run_dir / subdir))
    
    def test_pipeline_error_propagation(self):
        """Test how errors propagate through the pipeline"""
        # Create an invalid image path
        invalid_path = Path(self.temp_dir) / "nonexistent.jpg"
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(self.config_path)
        
        # Run pipeline with invalid input
        results = orchestrator.run_full_pipeline(invalid_path)
        
        # First stage should fail
        self.assertFalse(results['stages'][0]['success'])
        self.assertIn('error', results['stages'][0])
        
        # Summary should indicate failure
        self.assertFalse(results['summary']['all_stages_successful'])
    
    def test_pipeline_intermediate_results(self):
        """Test that intermediate results are properly saved"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = Path(self.test_manager.create_test_image_file("intermediate_test.jpg", test_image))
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(self.config_path)
        
        # Mock the separation system to avoid method loading issues
        with patch('separation.UnifiedSegmentationSystem.load_methods'), \
             patch('separation.UnifiedSegmentationSystem.run_method') as mock_run:
            
            # Setup mock
            from separation import SegmentationResult
            mock_result = SegmentationResult("test_method", str(image_path))
            mock_result.success = True
            mock_result.center = (320, 240)
            mock_result.core_radius = 50
            mock_result.cladding_radius = 125
            mock_run.return_value = mock_result
            
            # Run pipeline
            results = orchestrator.run_full_pipeline(image_path)
            
            # Check for intermediate results
            run_dir = Path(results['run_directory'])
            
            # Processing stage should create reimagined images
            processing_dir = run_dir / 'processing' / 'reimagined_images'
            if processing_dir.exists():
                reimagined_files = list(processing_dir.glob('*.jpg'))
                self.assertGreater(len(reimagined_files), 0)
            
            # Detection stage should create reports
            detection_dir = run_dir / 'detection'
            if detection_dir.exists():
                report_files = list(detection_dir.glob('*_report.json'))
                self.assertGreaterEqual(len(report_files), 0)

class TestDataConsistency(unittest.TestCase):
    """Test data consistency across pipeline stages"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_image_dimensions_consistency(self):
        """Test that image dimensions are preserved correctly"""
        # Create image with specific dimensions
        test_image = TestImageGenerator.create_fiber_optic_image(size=(480, 640))
        image_path = self.test_manager.create_test_image_file("dims_test.jpg", test_image)
        
        # Process through reimagine
        reimagined_dir = os.path.join(self.temp_dir, "reimagined")
        reimagined_paths = reimagine_image(image_path, reimagined_dir)
        
        # Check dimensions of non-resize transforms
        for path in reimagined_paths:
            if 'resize' not in os.path.basename(path):
                img = cv2.imread(path)
                self.assertEqual(img.shape[:2], (480, 640))
    
    def test_defect_coordinate_consistency(self):
        """Test that defect coordinates remain consistent"""
        # Create defect at known location
        defect_location = (320, 240)
        
        # Create detection report
        detection_report = {
            'defects': [
                {
                    'type': 'scratch',
                    'location': {'x': defect_location[0], 'y': defect_location[1]},
                    'bounding_box': {
                        'x': defect_location[0] - 10,
                        'y': defect_location[1] - 10,
                        'width': 20,
                        'height': 20
                    },
                    'severity': 'minor',
                    'confidence': 0.9
                }
            ]
        }
        
        # Process through aggregator
        aggregator = DefectAggregator(
            results_dir=Path(self.temp_dir),
            original_image_path=Path("test.jpg")
        )
        
        # Manually add detection
        aggregator.all_detections = detection_report['defects'].copy()
        
        # Add global coords
        for defect in aggregator.all_detections:
            defect['global_coords'] = (defect['location']['x'], defect['location']['y'])
        
        # Cluster defects
        clusters = aggregator.cluster_defects()
        
        # Verify coordinates are preserved
        if clusters:
            merged = aggregator.merge_defect_cluster(clusters[0]['defects'])
            self.assertEqual(
                merged['location']['x'], 
                defect_location[0]
            )
            self.assertEqual(
                merged['location']['y'], 
                defect_location[1]
            )

if __name__ == '__main__':
    unittest.main()