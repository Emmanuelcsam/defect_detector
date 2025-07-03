"""
Unit tests for separation.py module
Tests image segmentation and consensus functions
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
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from separation import (
    NumpyEncoder, EnhancedConsensusSystem, SegmentationResult, 
    UnifiedSegmentationSystem
)
from tests.test_utils import (
    TestImageGenerator, TestDataManager, assert_image_valid, 
    assert_json_structure, assert_file_exists
)

class TestNumpyEncoder(unittest.TestCase):
    """Test the NumpyEncoder class"""
    
    def test_encode_numpy_types(self):
        """Test encoding various numpy types"""
        encoder = NumpyEncoder()
        
        # Test numpy integers
        self.assertEqual(json.dumps(np.int32(42), cls=NumpyEncoder), "42")
        self.assertEqual(json.dumps(np.int64(42), cls=NumpyEncoder), "42")
        
        # Test numpy floats
        self.assertEqual(json.dumps(np.float32(3.14), cls=NumpyEncoder), "3.140000104904175")
        self.assertEqual(json.dumps(np.float64(3.14), cls=NumpyEncoder), "3.14")
        
        # Test numpy arrays
        arr = np.array([1, 2, 3])
        self.assertEqual(json.dumps(arr, cls=NumpyEncoder), "[1, 2, 3]")
        
        # Test nested structures
        data = {
            "int": np.int32(10),
            "float": np.float64(2.5),
            "array": np.array([1, 2]),
            "nested": {"value": np.int64(100)}
        }
        result = json.dumps(data, cls=NumpyEncoder)
        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        self.assertEqual(parsed["int"], 10)
        self.assertEqual(parsed["float"], 2.5)
        self.assertEqual(parsed["array"], [1, 2])

class TestEnhancedConsensusSystem(unittest.TestCase):
    """Test the EnhancedConsensusSystem class"""
    
    def setUp(self):
        """Set up test environment"""
        self.consensus = EnhancedConsensusSystem(min_agreement_ratio=0.3)
        
    def test_calculate_iou(self):
        """Test IoU calculation"""
        # Create test masks
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        
        # Test identical masks
        mask1[20:80, 20:80] = 255
        mask2[20:80, 20:80] = 255
        iou = self.consensus._calculate_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 1.0)
        
        # Test no overlap
        mask1.fill(0)
        mask2.fill(0)
        mask1[0:50, 0:50] = 255
        mask2[50:100, 50:100] = 255
        iou = self.consensus._calculate_iou(mask1, mask2)
        self.assertEqual(iou, 0.0)
        
        # Test partial overlap
        mask1.fill(0)
        mask2.fill(0)
        mask1[20:80, 20:80] = 255  # 60x60 = 3600 pixels
        mask2[40:100, 40:100] = 255  # 60x60 = 3600 pixels
        # Intersection: 40x40 = 1600 pixels
        # Union: 3600 + 3600 - 1600 = 5600 pixels
        iou = self.consensus._calculate_iou(mask1, mask2)
        expected_iou = 1600 / 5600
        self.assertAlmostEqual(iou, expected_iou, places=4)
    
    def test_create_masks_from_params(self):
        """Test mask creation from parameters"""
        center = (100, 100)
        core_radius = 20
        cladding_radius = 50
        image_shape = (200, 200)
        
        masks = self.consensus.create_masks_from_params(
            center, core_radius, cladding_radius, image_shape
        )
        
        # Check mask keys
        self.assertIn('core', masks)
        self.assertIn('cladding', masks)
        self.assertIn('ferrule', masks)
        
        # Check mask shapes
        for mask in masks.values():
            self.assertEqual(mask.shape, image_shape)
        
        # Check core mask
        core_mask = masks['core']
        self.assertEqual(core_mask[100, 100], 255)  # Center should be in core
        self.assertEqual(core_mask[100, 100 + core_radius + 5], 0)  # Outside core
        
        # Check cladding mask (should include core area)
        cladding_mask = masks['cladding']
        self.assertEqual(cladding_mask[100, 100], 255)  # Center in cladding
        self.assertEqual(cladding_mask[100, 100 + cladding_radius + 5], 0)  # Outside
    
    def test_ensure_mask_consistency(self):
        """Test mask consistency enforcement"""
        # Create inconsistent masks
        core = np.zeros((100, 100), dtype=np.uint8)
        cladding = np.zeros((100, 100), dtype=np.uint8)
        ferrule = np.zeros((100, 100), dtype=np.uint8)
        
        # Core not inside cladding
        core[40:60, 40:60] = 255
        cladding[0:30, 0:30] = 255
        ferrule[:, :] = 255
        
        core_fixed, cladding_fixed, ferrule_fixed = self.consensus.ensure_mask_consistency(
            core.copy(), cladding.copy(), ferrule.copy()
        )
        
        # Check that core is now inside cladding
        core_area = core_fixed > 0
        self.assertTrue(np.all(cladding_fixed[core_area] > 0))
        
        # Check that cladding is inside ferrule
        cladding_area = cladding_fixed > 0
        self.assertTrue(np.all(ferrule_fixed[cladding_area] > 0))
    
    def test_generate_consensus_basic(self):
        """Test basic consensus generation"""
        # Create mock segmentation results
        results = []
        method_scores = {}
        
        # Create consistent results
        for i in range(3):
            result = SegmentationResult(f"method_{i}", "test.jpg")
            result.success = True
            result.confidence = 0.9
            result.center = (100, 100)
            result.core_radius = 20
            result.cladding_radius = 50
            result.execution_time = 0.1
            results.append(result)
            method_scores[f"method_{i}"] = 0.8
        
        consensus = self.consensus.generate_consensus(
            results, method_scores, (200, 200)
        )
        
        self.assertIsNotNone(consensus)
        self.assertIn('center', consensus)
        self.assertIn('core_radius', consensus)
        self.assertIn('cladding_radius', consensus)
        self.assertIn('confidence', consensus)
        self.assertIn('consensus_score', consensus)
        
        # Check consensus values
        self.assertEqual(consensus['center'], (100, 100))
        self.assertEqual(consensus['core_radius'], 20)
        self.assertEqual(consensus['cladding_radius'], 50)
    
    def test_generate_consensus_with_outliers(self):
        """Test consensus with outlier results"""
        results = []
        method_scores = {}
        
        # Two consistent results
        for i in range(2):
            result = SegmentationResult(f"good_{i}", "test.jpg")
            result.success = True
            result.confidence = 0.9
            result.center = (100, 100)
            result.core_radius = 20
            result.cladding_radius = 50
            results.append(result)
            method_scores[f"good_{i}"] = 0.9
        
        # One outlier
        outlier = SegmentationResult("outlier", "test.jpg")
        outlier.success = True
        outlier.confidence = 0.9
        outlier.center = (200, 200)  # Way off
        outlier.core_radius = 100  # Too big
        outlier.cladding_radius = 200
        results.append(outlier)
        method_scores["outlier"] = 0.1  # Low score
        
        consensus = self.consensus.generate_consensus(
            results, method_scores, (300, 300)
        )
        
        # Consensus should favor the consistent results
        self.assertIsNotNone(consensus)
        # Center should be closer to (100, 100) than (200, 200)
        center_dist_to_good = np.sqrt((consensus['center'][0] - 100)**2 + 
                                     (consensus['center'][1] - 100)**2)
        center_dist_to_outlier = np.sqrt((consensus['center'][0] - 200)**2 + 
                                        (consensus['center'][1] - 200)**2)
        self.assertLess(center_dist_to_good, center_dist_to_outlier)

class TestSegmentationResult(unittest.TestCase):
    """Test the SegmentationResult class"""
    
    def test_initialization(self):
        """Test SegmentationResult initialization"""
        result = SegmentationResult("test_method", "/path/to/image.jpg")
        
        self.assertEqual(result.method_name, "test_method")
        self.assertEqual(result.image_path, "/path/to/image.jpg")
        self.assertFalse(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.confidence, 0.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = SegmentationResult("test_method", "image.jpg")
        result.success = True
        result.confidence = 0.95
        result.center = (100, 100)
        result.core_radius = 25
        result.cladding_radius = 62
        result.execution_time = 0.5
        result.zones_detected = {'core': True, 'cladding': True}
        
        data = result.to_dict()
        
        assert_json_structure(data, [
            'method_name', 'image_path', 'success', 'confidence',
            'center', 'core_radius', 'cladding_radius'
        ])
        
        self.assertEqual(data['confidence'], 0.95)
        self.assertEqual(data['center'], (100, 100))

class TestUnifiedSegmentationSystem(unittest.TestCase):
    """Test the UnifiedSegmentationSystem class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
        # Create mock methods directory
        self.methods_dir = os.path.join(self.temp_dir, "zones_methods")
        os.makedirs(self.methods_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    @patch('separation.UnifiedSegmentationSystem.load_methods')
    def test_initialization(self, mock_load_methods):
        """Test UnifiedSegmentationSystem initialization"""
        system = UnifiedSegmentationSystem(methods_dir=self.methods_dir)
        
        self.assertEqual(system.methods_dir, self.methods_dir)
        self.assertIsNotNone(system.consensus_system)
        self.assertIsInstance(system.method_scores, dict)
        mock_load_methods.assert_called_once()
    
    def test_detect_and_inpaint_anomalies(self):
        """Test anomaly detection and inpainting"""
        system = UnifiedSegmentationSystem(methods_dir=self.methods_dir)
        
        # Create test image with simulated defect
        image = TestImageGenerator.create_fiber_optic_image(
            defects=[{'type': 'scratch', 'location': (320, 240), 'size': 20}]
        )
        
        cleaned, defect_mask = system.detect_and_inpaint_anomalies(image)
        
        assert_image_valid(cleaned)
        assert_image_valid(defect_mask, expected_shape=(image.shape[0], image.shape[1]))
        
        # Check that defect mask has some detected pixels
        self.assertGreater(np.sum(defect_mask), 0)
    
    @patch('subprocess.run')
    def test_run_method_isolated(self, mock_subprocess):
        """Test isolated method execution"""
        system = UnifiedSegmentationSystem(methods_dir=self.methods_dir)
        
        # Mock subprocess output
        mock_result = {
            "success": True,
            "center": [100, 100],
            "core_radius": 25,
            "cladding_radius": 62
        }
        
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_result)
        )
        
        # Create test image
        image_path = Path(self.test_manager.create_test_image_file("test.jpg"))
        temp_output = Path(self.temp_dir) / "temp_output"
        
        result = system.run_method_isolated("test_method", image_path, temp_output)
        
        self.assertEqual(result, mock_result)
        mock_subprocess.assert_called_once()
    
    def test_save_results(self):
        """Test saving segmentation results"""
        system = UnifiedSegmentationSystem(methods_dir=self.methods_dir)
        
        # Create test data
        image = TestImageGenerator.create_fiber_optic_image()
        image_path = Path(self.test_manager.create_test_image_file("test.jpg", image))
        
        consensus = {
            'center': (320, 240),
            'core_radius': 50,
            'cladding_radius': 125,
            'confidence': 0.95,
            'masks': {
                'core': np.zeros((480, 640), dtype=np.uint8),
                'cladding': np.zeros((480, 640), dtype=np.uint8),
                'ferrule': np.zeros((480, 640), dtype=np.uint8)
            }
        }
        
        defect_mask = np.zeros((480, 640), dtype=np.uint8)
        output_dir = os.path.join(self.temp_dir, "output")
        
        # Save results
        system.save_results(image_path, consensus, image, output_dir, defect_mask)
        
        # Check outputs
        base_name = image_path.stem
        assert_file_exists(os.path.join(output_dir, f"{base_name}_summary.png"))
        assert_file_exists(os.path.join(output_dir, f"{base_name}_report.json"))
        assert_file_exists(os.path.join(output_dir, f"{base_name}_masks.json"))
        
        # Verify report structure
        with open(os.path.join(output_dir, f"{base_name}_report.json"), 'r') as f:
            report = json.load(f)
            assert_json_structure(report, ['source_image', 'timestamp', 'consensus'])
    
    def test_update_learning(self):
        """Test knowledge base update"""
        system = UnifiedSegmentationSystem(methods_dir=self.methods_dir)
        
        # Create test consensus and results
        consensus = {
            'center': (100, 100),
            'core_radius': 25,
            'cladding_radius': 62,
            'consensus_score': 0.9
        }
        
        results = []
        for i in range(3):
            result = SegmentationResult(f"method_{i}", "test.jpg")
            result.success = True
            result.confidence = 0.8 + i * 0.05
            results.append(result)
        
        # Initial scores
        for i in range(3):
            system.method_scores[f"method_{i}"] = 0.5
        
        # Update learning
        system.update_learning(consensus, results)
        
        # Check that successful methods got score boosts
        for i in range(3):
            self.assertGreater(system.method_scores[f"method_{i}"], 0.5)

if __name__ == '__main__':
    unittest.main()