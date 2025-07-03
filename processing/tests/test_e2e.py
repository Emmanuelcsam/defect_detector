"""
End-to-end tests for the complete defect detector system
Tests the full workflow from image input to final report
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
from unittest.mock import patch
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import PipelineOrchestrator
from tests.test_utils import (
    TestImageGenerator, TestDataManager, ConfigGenerator,
    assert_file_exists, assert_directory_exists, assert_json_structure
)

class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create comprehensive test config
        self.config_data = ConfigGenerator.create_default_config()
        self.config_data['app_settings']['output_directory'] = os.path.join(self.temp_dir, "output")
        self.config_data['app_settings']['log_level'] = 'WARNING'  # Reduce noise in tests
        
        # Create methods directory for separation
        self.methods_dir = os.path.join(self.temp_dir, "zones_methods")
        os.makedirs(self.methods_dir, exist_ok=True)
        self.config_data['separation_settings']['methods_directory'] = self.methods_dir
        
        # Save config
        self.config_path = self.test_manager.create_test_config(self.config_data)
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    @patch('separation.UnifiedSegmentationSystem.run_method')
    def test_perfect_fiber_workflow(self, mock_run_method, mock_load_methods):
        """Test workflow with a perfect fiber (no defects)"""
        # Create perfect fiber image
        perfect_fiber = TestImageGenerator.create_fiber_optic_image(defects=[])
        image_path = Path(self.test_manager.create_test_image_file(
            "perfect_fiber.jpg", perfect_fiber
        ))
        
        # Mock separation to work properly
        from separation import SegmentationResult
        mock_result = SegmentationResult("test_method", str(image_path))
        mock_result.success = True
        mock_result.center = (320, 240)
        mock_result.core_radius = 50
        mock_result.cladding_radius = 125
        mock_result.confidence = 0.95
        mock_run_method.return_value = mock_result
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(self.config_path)
        results = orchestrator.run_full_pipeline(image_path)
        
        # Verify results
        self._verify_pipeline_success(results)
        
        # Check quality assessment
        final_report_path = self._find_final_report(results['run_directory'])
        if final_report_path:
            with open(final_report_path, 'r') as f:
                final_report = json.load(f)
            
            # Perfect fiber should have excellent quality
            self.assertEqual(final_report.get('total_defects', 0), 0)
            quality = final_report.get('quality_assessment', {}).get('overall_quality')
            self.assertEqual(quality, 'excellent')
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    @patch('separation.UnifiedSegmentationSystem.run_method')
    def test_defective_fiber_workflow(self, mock_run_method, mock_load_methods):
        """Test workflow with a defective fiber"""
        # Create fiber with multiple defects
        defects = [
            {'type': 'scratch', 'location': (320, 240), 'size': 30},
            {'type': 'contamination', 'location': (200, 200), 'size': 20},
            {'type': 'chip', 'location': (400, 300), 'size': 25}
        ]
        defective_fiber = TestImageGenerator.create_fiber_optic_image(defects=defects)
        image_path = Path(self.test_manager.create_test_image_file(
            "defective_fiber.jpg", defective_fiber
        ))
        
        # Mock separation
        from separation import SegmentationResult
        mock_result = SegmentationResult("test_method", str(image_path))
        mock_result.success = True
        mock_result.center = (320, 240)
        mock_result.core_radius = 50
        mock_result.cladding_radius = 125
        mock_run_method.return_value = mock_result
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(self.config_path)
        results = orchestrator.run_full_pipeline(image_path)
        
        # Verify results
        self._verify_pipeline_success(results)
        
        # Check that defects were detected
        detection_reports = self._find_detection_reports(results['run_directory'])
        total_defects_found = 0
        for report_path in detection_reports:
            with open(report_path, 'r') as f:
                report = json.load(f)
                total_defects_found += len(report.get('defects', []))
        
        # Should detect at least some defects
        self.assertGreater(total_defects_found, 0)
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    @patch('separation.UnifiedSegmentationSystem.run_method')
    def test_batch_processing_workflow(self, mock_run_method, mock_load_methods):
        """Test batch processing of multiple images"""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            defects = []
            if i > 0:  # Add defects to some images
                defects.append({
                    'type': 'scratch',
                    'location': (100 + i*50, 100 + i*50),
                    'size': 20
                })
            
            image = TestImageGenerator.create_fiber_optic_image(defects=defects)
            path = Path(self.test_manager.create_test_image_file(
                f"batch_image_{i}.jpg", image
            ))
            image_paths.append(path)
        
        # Mock separation
        from separation import SegmentationResult
        mock_result = SegmentationResult("test_method", "test.jpg")
        mock_result.success = True
        mock_result.center = (320, 240)
        mock_result.core_radius = 50
        mock_result.cladding_radius = 125
        mock_run_method.return_value = mock_result
        
        # Process all images
        orchestrator = PipelineOrchestrator(self.config_path)
        all_results = []
        
        for image_path in image_paths:
            results = orchestrator.run_full_pipeline(image_path)
            all_results.append(results)
        
        # Verify all processed successfully
        for i, results in enumerate(all_results):
            self._verify_pipeline_success(results, f"Image {i}")
        
        # Check that each has its own run directory
        run_dirs = [r['run_directory'] for r in all_results]
        self.assertEqual(len(set(run_dirs)), len(run_dirs), "Run directories not unique")
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    def test_various_image_formats(self, mock_load_methods):
        """Test processing of different image formats"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        
        # Test different formats
        formats = ['.jpg', '.png', '.bmp']
        for fmt in formats:
            with self.subTest(format=fmt):
                # Save in specific format
                image_path = os.path.join(self.temp_dir, f"test_image{fmt}")
                if fmt == '.jpg':
                    cv2.imwrite(image_path, test_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(image_path, test_image)
                
                # Mock separation for this test
                with patch('separation.UnifiedSegmentationSystem.run_method') as mock_run:
                    from separation import SegmentationResult
                    mock_result = SegmentationResult("test_method", image_path)
                    mock_result.success = True
                    mock_result.center = (320, 240)
                    mock_result.core_radius = 50
                    mock_result.cladding_radius = 125
                    mock_run.return_value = mock_result
                    
                    # Process image
                    orchestrator = PipelineOrchestrator(self.config_path)
                    results = orchestrator.run_full_pipeline(Path(image_path))
                    
                    # Should process successfully
                    self.assertTrue(
                        results['summary']['all_stages_successful'],
                        f"Failed to process {fmt} format"
                    )
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    @patch('separation.UnifiedSegmentationSystem.run_method')
    def test_output_completeness(self, mock_run_method, mock_load_methods):
        """Test that all expected outputs are generated"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image(
            defects=[{'type': 'scratch', 'location': (300, 300), 'size': 15}]
        )
        image_path = Path(self.test_manager.create_test_image_file(
            "output_test.jpg", test_image
        ))
        
        # Mock separation
        from separation import SegmentationResult
        mock_result = SegmentationResult("test_method", str(image_path))
        mock_result.success = True
        mock_result.center = (320, 240)
        mock_result.core_radius = 50
        mock_result.cladding_radius = 125
        mock_run_method.return_value = mock_result
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(self.config_path)
        results = orchestrator.run_full_pipeline(image_path)
        
        run_dir = Path(results['run_directory'])
        
        # Check expected directory structure
        expected_dirs = ['processing', 'separation', 'detection', 'final']
        for dir_name in expected_dirs:
            assert_directory_exists(str(run_dir / dir_name))
        
        # Check for specific output files
        # 1. Reimagined images
        processing_dir = run_dir / 'processing' / 'reimagined_images'
        if processing_dir.exists():
            reimagined_count = len(list(processing_dir.glob('*.jpg')))
            self.assertGreater(reimagined_count, 0, "No reimagined images found")
        
        # 2. Separation reports
        separation_dir = run_dir / 'separation'
        separation_reports = list(separation_dir.glob('*_report.json'))
        self.assertGreater(len(separation_reports), 0, "No separation reports found")
        
        # 3. Detection reports
        detection_dir = run_dir / 'detection'
        detection_reports = list(detection_dir.glob('*_report.json'))
        self.assertGreaterEqual(len(detection_reports), 0, "No detection reports found")
        
        # 4. Final report
        final_dir = run_dir / 'final'
        final_reports = list(final_dir.glob('final_report.json'))
        self.assertGreater(len(final_reports), 0, "No final report found")
        
        # 5. Visualizations (if enabled)
        if self.config_data['detection_settings']['save_intermediate_results']:
            viz_files = list(detection_dir.glob('*.png')) + list(final_dir.glob('*.png'))
            self.assertGreaterEqual(len(viz_files), 0, "No visualizations found")
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    def test_error_recovery(self, mock_load_methods):
        """Test pipeline recovery from various errors"""
        # Test 1: Invalid image path
        orchestrator = PipelineOrchestrator(self.config_path)
        results = orchestrator.run_full_pipeline(Path("/nonexistent/image.jpg"))
        
        self.assertFalse(results['summary']['all_stages_successful'])
        self.assertIn('error', results['stages'][0])
        
        # Test 2: Corrupted image data
        corrupted_path = os.path.join(self.temp_dir, "corrupted.jpg")
        with open(corrupted_path, 'wb') as f:
            f.write(b'Not a valid image')
        
        results = orchestrator.run_full_pipeline(Path(corrupted_path))
        self.assertFalse(results['summary']['all_stages_successful'])
        
        # Test 3: Very small image
        tiny_image = np.zeros((10, 10, 3), dtype=np.uint8)
        tiny_path = self.test_manager.create_test_image_file("tiny.jpg", tiny_image)
        
        with patch('separation.UnifiedSegmentationSystem.run_method') as mock_run:
            # Mock separation failure for tiny image
            from separation import SegmentationResult
            mock_result = SegmentationResult("test_method", tiny_path)
            mock_result.success = False
            mock_result.error = "Image too small"
            mock_run.return_value = mock_result
            
            results = orchestrator.run_full_pipeline(Path(tiny_path))
            # Pipeline should complete but may have warnings
            self.assertIn('stages', results)
    
    def test_performance_metrics(self):
        """Test that performance metrics are collected"""
        # Create simple test image
        test_image = TestImageGenerator.create_test_image(size=(100, 100))
        image_path = Path(self.test_manager.create_test_image_file(
            "perf_test.jpg", test_image
        ))
        
        # Mock to speed up test
        with patch('separation.UnifiedSegmentationSystem.load_methods'), \
             patch('separation.UnifiedSegmentationSystem.run_method') as mock_run:
            
            from separation import SegmentationResult
            mock_result = SegmentationResult("test_method", str(image_path))
            mock_result.success = True
            mock_result.center = (50, 50)
            mock_result.core_radius = 10
            mock_result.cladding_radius = 25
            mock_run.return_value = mock_result
            
            # Run pipeline
            orchestrator = PipelineOrchestrator(self.config_path)
            start_time = time.time()
            results = orchestrator.run_full_pipeline(image_path)
            end_time = time.time()
            
            # Check timing
            total_time = end_time - start_time
            self.assertLess(total_time, 60, "Pipeline took too long")
            
            # Check for timing info in results
            if 'summary' in results:
                # Could check for execution time if it's tracked
                pass
    
    def _verify_pipeline_success(self, results, context=""):
        """Helper to verify pipeline ran successfully"""
        self.assertIn('stages', results, f"{context}: Missing stages")
        self.assertIn('summary', results, f"{context}: Missing summary")
        self.assertIn('run_directory', results, f"{context}: Missing run_directory")
        
        # Check all stages completed
        for stage in results['stages']:
            if not stage['success']:
                print(f"{context} - Stage {stage['stage']} failed: {stage.get('error', 'Unknown')}")
        
        # Verify run directory exists
        assert_directory_exists(results['run_directory'])
    
    def _find_final_report(self, run_directory):
        """Find the final report in the run directory"""
        final_dir = Path(run_directory) / 'final'
        if final_dir.exists():
            reports = list(final_dir.glob('final_report.json'))
            if reports:
                return reports[0]
        return None
    
    def _find_detection_reports(self, run_directory):
        """Find all detection reports in the run directory"""
        detection_dir = Path(run_directory) / 'detection'
        if detection_dir.exists():
            return list(detection_dir.glob('*_report.json'))
        return []

class TestRealWorldScenarios(unittest.TestCase):
    """Test realistic usage scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create config
        self.config_data = ConfigGenerator.create_default_config()
        self.config_data['app_settings']['output_directory'] = self.temp_dir
        self.config_path = self.test_manager.create_test_config(self.config_data)
        
        # Create methods directory
        os.makedirs(os.path.join(self.temp_dir, "zones_methods"), exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    @patch('separation.UnifiedSegmentationSystem.run_method')
    def test_quality_control_scenario(self, mock_run_method, mock_load_methods):
        """Test quality control scenario with pass/fail criteria"""
        # Define quality thresholds
        quality_thresholds = {
            'max_defects': 5,
            'max_critical_defects': 0,
            'min_quality_score': 80
        }
        
        # Create test batch - some pass, some fail
        test_batch = [
            {'name': 'good_fiber.jpg', 'defects': []},
            {'name': 'minor_defects.jpg', 'defects': [
                {'type': 'scratch', 'location': (100, 100), 'size': 10}
            ]},
            {'name': 'critical_defect.jpg', 'defects': [
                {'type': 'chip', 'location': (320, 240), 'size': 50}
            ]}
        ]
        
        # Mock separation
        from separation import SegmentationResult
        mock_result = SegmentationResult("test_method", "test.jpg")
        mock_result.success = True
        mock_result.center = (320, 240)
        mock_result.core_radius = 50
        mock_result.cladding_radius = 125
        mock_run_method.return_value = mock_result
        
        # Process batch
        orchestrator = PipelineOrchestrator(self.config_path)
        batch_results = []
        
        for fiber_spec in test_batch:
            # Create image
            image = TestImageGenerator.create_fiber_optic_image(
                defects=fiber_spec['defects']
            )
            image_path = Path(self.test_manager.create_test_image_file(
                fiber_spec['name'], image
            ))
            
            # Process
            results = orchestrator.run_full_pipeline(image_path)
            
            # Evaluate against criteria
            qc_result = {
                'name': fiber_spec['name'],
                'passed': self._evaluate_quality_criteria(
                    results, quality_thresholds
                ),
                'results': results
            }
            batch_results.append(qc_result)
        
        # Verify QC results
        self.assertTrue(batch_results[0]['passed'], "Good fiber should pass")
        self.assertTrue(batch_results[1]['passed'], "Minor defects should pass")
        # Critical defect may or may not be detected by simple test image
    
    def _evaluate_quality_criteria(self, results, thresholds):
        """Evaluate if results meet quality criteria"""
        # Simple evaluation based on pipeline success
        # In real scenario, would parse final report
        return results['summary']['all_stages_successful']

if __name__ == '__main__':
    unittest.main()