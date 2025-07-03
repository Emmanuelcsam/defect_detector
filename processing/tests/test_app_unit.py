"""
Unit tests for app.py module
Tests the main pipeline orchestrator
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import PipelineOrchestrator, ask_for_images, ask_for_folder
from tests.test_utils import (
    TestImageGenerator, TestDataManager, ConfigGenerator,
    assert_file_exists, assert_directory_exists
)

class TestPipelineOrchestrator(unittest.TestCase):
    """Test the PipelineOrchestrator class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create test config
        self.config_data = ConfigGenerator.create_default_config()
        self.config_path = self.test_manager.create_test_config(self.config_data)
        
        # Create orchestrator
        self.orchestrator = PipelineOrchestrator(self.config_path)
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_initialization(self):
        """Test PipelineOrchestrator initialization"""
        self.assertIsInstance(self.orchestrator.config, dict)
        self.assertEqual(
            self.orchestrator.config['app_settings']['log_level'],
            'INFO'
        )
    
    def test_load_config(self):
        """Test configuration loading"""
        # Test valid config
        config = self.orchestrator.load_config(self.config_path)
        self.assertIsInstance(config, dict)
        self.assertIn('app_settings', config)
        
        # Test invalid config path
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.load_config("/nonexistent/config.json")
        
        # Test invalid JSON
        invalid_config_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_config_path, 'w') as f:
            f.write("{ invalid json")
        
        with self.assertRaises(json.JSONDecodeError):
            self.orchestrator.load_config(invalid_config_path)
    
    def test_resolve_config_paths(self):
        """Test path resolution in config"""
        config = {
            'app_settings': {
                'base_directory': './relative/path',
                'output_directory': 'output'
            },
            'nested': {
                'some_path': '../another/path'
            }
        }
        
        resolved = self.orchestrator.resolve_config_paths(config)
        
        # Check that paths are now absolute
        self.assertTrue(os.path.isabs(resolved['app_settings']['base_directory']))
        self.assertTrue(os.path.isabs(resolved['app_settings']['output_directory']))
        self.assertTrue(os.path.isabs(resolved['nested']['some_path']))
    
    @patch('app.reimagine_image')
    def test_run_processing_stage(self, mock_reimagine):
        """Test processing stage execution"""
        # Setup mock
        mock_outputs = [f"reimagined_{i}.jpg" for i in range(5)]
        mock_reimagine.return_value = mock_outputs
        
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = Path(self.test_manager.create_test_image_file("test.jpg", test_image))
        run_dir = Path(self.temp_dir) / "run"
        run_dir.mkdir(exist_ok=True)
        
        # Run processing stage
        result = self.orchestrator.run_processing_stage(image_path, run_dir)
        
        # Verify
        self.assertEqual(result['stage'], 'processing')
        self.assertTrue(result['success'])
        self.assertEqual(result['reimagined_count'], 5)
        self.assertEqual(len(result['reimagined_paths']), 5)
        
        # Check mock was called correctly
        mock_reimagine.assert_called_once()
        call_args = mock_reimagine.call_args[0]
        self.assertEqual(str(call_args[0]), str(image_path))
    
    @patch('app.UnifiedSegmentationSystem')
    def test_run_separation_stage(self, mock_segmentation_class):
        """Test separation stage execution"""
        # Setup mock
        mock_system = Mock()
        mock_system.process_image.return_value = {
            'center': (320, 240),
            'core_radius': 50,
            'cladding_radius': 125
        }
        mock_segmentation_class.return_value = mock_system
        
        # Create test data
        image_paths = [f"image_{i}.jpg" for i in range(3)]
        run_dir = Path(self.temp_dir) / "run"
        run_dir.mkdir(exist_ok=True)
        original_image = Path("original.jpg")
        
        # Run separation stage
        result = self.orchestrator.run_separation_stage(
            image_paths, run_dir, original_image
        )
        
        # Verify
        self.assertEqual(result['stage'], 'separation')
        self.assertTrue(result['success'])
        self.assertEqual(result['processed_count'], 3)
        
        # Check mock calls
        self.assertEqual(mock_system.process_image.call_count, 3)
    
    @patch('app.OmniFiberAnalyzer')
    @patch('app.OmniConfig')
    def test_run_detection_stage(self, mock_config_class, mock_analyzer_class):
        """Test detection stage execution"""
        # Setup mocks
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_analyzer = Mock()
        mock_analyzer.analyze_end_face.return_value = {
            'defects': [
                {'type': 'scratch', 'severity': 'minor'}
            ],
            'overall_quality_score': 85
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create test data
        image_paths = [f"image_{i}.jpg" for i in range(3)]
        run_dir = Path(self.temp_dir) / "run"
        run_dir.mkdir(exist_ok=True)
        
        # Run detection stage
        result = self.orchestrator.run_detection_stage(image_paths, run_dir)
        
        # Verify
        self.assertEqual(result['stage'], 'detection')
        self.assertTrue(result['success'])
        self.assertEqual(result['analyzed_count'], 3)
        self.assertEqual(result['total_defects'], 3)  # 1 defect per image
        
        # Check mock calls
        self.assertEqual(mock_analyzer.analyze_end_face.call_count, 3)
    
    @patch('app.integrate_with_pipeline')
    def test_run_data_acquisition_stage(self, mock_integrate):
        """Test data acquisition stage execution"""
        # Setup mock
        mock_integrate.return_value = {
            'total_defects': 5,
            'quality_assessment': {
                'overall_quality': 'good'
            }
        }
        
        # Create test data
        original_image = Path("original.jpg")
        run_dir = Path(self.temp_dir) / "run"
        run_dir.mkdir(exist_ok=True)
        
        # Run data acquisition stage
        result = self.orchestrator.run_data_acquisition_stage(
            original_image, run_dir
        )
        
        # Verify
        self.assertEqual(result['stage'], 'data_acquisition')
        self.assertTrue(result['success'])
        self.assertIn('final_report', result)
        
        # Check mock call
        mock_integrate.assert_called_once_with(
            results_base_dir=str(run_dir),
            image_name="original",
            clustering_eps=self.orchestrator.config['data_acquisition_settings']['clustering_eps']
        )
    
    @patch('app.PipelineOrchestrator.run_data_acquisition_stage')
    @patch('app.PipelineOrchestrator.run_detection_stage')
    @patch('app.PipelineOrchestrator.run_separation_stage')
    @patch('app.PipelineOrchestrator.run_processing_stage')
    def test_run_full_pipeline(self, mock_process, mock_separate, 
                              mock_detect, mock_acquire):
        """Test full pipeline execution"""
        # Setup mocks
        mock_process.return_value = {
            'success': True,
            'reimagined_paths': ['img1.jpg', 'img2.jpg']
        }
        mock_separate.return_value = {
            'success': True,
            'separation_results': {}
        }
        mock_detect.return_value = {
            'success': True,
            'total_defects': 3
        }
        mock_acquire.return_value = {
            'success': True,
            'final_report': {'total_defects': 3}
        }
        
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = Path(self.test_manager.create_test_image_file("test.jpg", test_image))
        
        # Run full pipeline
        results = self.orchestrator.run_full_pipeline(image_path)
        
        # Verify results
        self.assertIn('run_id', results)
        self.assertIn('run_directory', results)
        self.assertIn('stages', results)
        self.assertIn('summary', results)
        
        # Check all stages were called
        mock_process.assert_called_once()
        mock_separate.assert_called_once()
        mock_detect.assert_called_once()
        mock_acquire.assert_called_once()
        
        # Check stage results
        stages = results['stages']
        self.assertEqual(len(stages), 4)
        for stage in stages:
            self.assertTrue(stage['success'])
    
    @patch('app.PipelineOrchestrator.run_processing_stage')
    def test_run_full_pipeline_with_failure(self, mock_process):
        """Test pipeline handling of stage failures"""
        # Setup mock to fail
        mock_process.side_effect = Exception("Processing failed")
        
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = Path(self.test_manager.create_test_image_file("test.jpg", test_image))
        
        # Run pipeline - should handle error gracefully
        results = self.orchestrator.run_full_pipeline(image_path)
        
        # Verify error handling
        self.assertIn('stages', results)
        processing_stage = results['stages'][0]
        self.assertFalse(processing_stage['success'])
        self.assertIn('error', processing_stage)
        
        # Summary should indicate failure
        self.assertFalse(results['summary']['all_stages_successful'])
    
    def test_run_full_pipeline_output_structure(self):
        """Test the output directory structure created by pipeline"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = Path(self.test_manager.create_test_image_file("test.jpg", test_image))
        
        # Mock the config to use our temp directory
        self.orchestrator.config['app_settings']['output_directory'] = self.temp_dir
        
        # We'll patch the actual processing to avoid dependencies
        with patch('app.reimagine_image') as mock_reimagine, \
             patch('app.UnifiedSegmentationSystem') as mock_seg, \
             patch('app.OmniFiberAnalyzer') as mock_analyzer, \
             patch('app.integrate_with_pipeline') as mock_integrate:
            
            # Setup minimal mocks
            mock_reimagine.return_value = ['img.jpg']
            mock_seg.return_value.process_image.return_value = {}
            mock_analyzer.return_value.analyze_end_face.return_value = {'defects': []}
            mock_integrate.return_value = {'total_defects': 0}
            
            # Run pipeline
            results = self.orchestrator.run_full_pipeline(image_path)
            
            # Check output directory structure
            run_dir = Path(results['run_directory'])
            assert_directory_exists(str(run_dir))
            
            # Check for stage subdirectories
            expected_dirs = ['processing', 'separation', 'detection', 'final']
            for dir_name in expected_dirs:
                stage_dir = run_dir / dir_name
                assert_directory_exists(str(stage_dir))

class TestUserInteractionFunctions(unittest.TestCase):
    """Test user interaction functions"""
    
    @patch('pathlib.Path.is_file')
    @patch('builtins.input')
    def test_ask_for_images_single(self, mock_input, mock_is_file):
        """Test asking for single image"""
        mock_input.return_value = "/path/to/image.jpg"
        mock_is_file.return_value = True
        
        result = ask_for_images()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0]), "/path/to/image.jpg")
        mock_input.assert_called_once()
    
    @patch('pathlib.Path.is_file')
    @patch('builtins.input')
    def test_ask_for_images_multiple(self, mock_input, mock_is_file):
        """Test asking for multiple images"""
        mock_input.return_value = "/path/to/img1.jpg /path/to/img2.jpg"
        mock_is_file.return_value = True
        
        result = ask_for_images()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(str(result[0]), "/path/to/img1.jpg")
        self.assertEqual(str(result[1]), "/path/to/img2.jpg")
    
    @patch('pathlib.Path.is_file')
    @patch('builtins.input')
    def test_ask_for_images_with_whitespace(self, mock_input, mock_is_file):
        """Test handling of whitespace in image paths"""
        mock_input.return_value = '  "/path/to/img 1.jpg"  "/path/to/img 2.jpg"  '
        mock_is_file.return_value = True
        
        result = ask_for_images()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(str(result[0]), "/path/to/img 1.jpg")
        self.assertEqual(str(result[1]), "/path/to/img 2.jpg")
    
    @patch('pathlib.Path.is_dir')
    @patch('builtins.input')
    def test_ask_for_folder(self, mock_input, mock_is_dir):
        """Test asking for folder path"""
        mock_input.return_value = "/path/to/folder"
        mock_is_dir.return_value = True
        
        result = ask_for_folder()
        
        self.assertEqual(str(result), "/path/to/folder")
        mock_input.assert_called_once()
    
    @patch('pathlib.Path.is_dir')
    @patch('builtins.input')
    def test_ask_for_folder_with_whitespace(self, mock_input, mock_is_dir):
        """Test handling of whitespace in folder path"""
        mock_input.return_value = "  /path/to/folder  "
        mock_is_dir.return_value = True
        
        result = ask_for_folder()
        
        self.assertEqual(str(result), "/path/to/folder")

class TestMainFunction(unittest.TestCase):
    """Test the main function"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
    
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    @patch('os.path.exists')
    @patch('builtins.input')
    @patch('app.PipelineOrchestrator')
    def test_main_single_image(self, mock_orchestrator_class, mock_input, mock_exists):
        """Test main function with single image option"""
        # Setup mocks
        mock_exists.return_value = True  # config.json exists
        mock_input.side_effect = ['1', '/path/to/image.jpg', '5']  # Option 1, image path, exit
        
        mock_orchestrator = Mock()
        mock_orchestrator.run_full_pipeline.return_value = {
            'summary': {'total_defects': 3}
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Import and run main
        from app import main
        
        with self.assertRaises(SystemExit):
            main()
        
        # Verify orchestrator was called
        mock_orchestrator.run_full_pipeline.assert_called_once()
    
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('app.PipelineOrchestrator')
    @patch('builtins.input')
    def test_main_batch_processing(self, mock_input, mock_orchestrator_class,
                                  mock_exists, mock_listdir):
        """Test main function with batch processing option"""
        # Setup mocks
        mock_input.side_effect = ['2', '/path/to/folder', '5']  # Option 2, folder, exit
        mock_exists.return_value = True
        mock_listdir.return_value = ['img1.jpg', 'img2.png', 'text.txt']
        
        mock_orchestrator = Mock()
        mock_orchestrator.run_full_pipeline.return_value = {
            'summary': {'total_defects': 1}
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Import and run main
        from app import main
        
        with self.assertRaises(SystemExit):
            main()
        
        # Should process only image files
        self.assertEqual(mock_orchestrator.run_full_pipeline.call_count, 2)

if __name__ == '__main__':
    unittest.main()