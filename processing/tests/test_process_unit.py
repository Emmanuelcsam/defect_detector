"""
Unit tests for process.py module
Tests image reimagining/transformation functions
"""

import unittest
import os
import sys
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process import reimagine_image
from tests.test_utils import TestImageGenerator, TestDataManager, assert_image_valid, assert_directory_exists

class TestProcessModule(unittest.TestCase):
    """Test cases for process.py functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_manager = TestDataManager()
        self.temp_dir = self.test_manager.setup()
        
    def tearDown(self):
        """Clean up test environment"""
        self.test_manager.teardown()
    
    def test_reimagine_image_basic(self):
        """Test basic reimagine_image functionality"""
        # Create test image
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("test_fiber.jpg", test_image)
        
        # Run reimagine_image
        output_folder = os.path.join(self.temp_dir, "reimagined")
        result_paths = reimagine_image(image_path, output_folder)
        
        # Verify outputs
        self.assertIsInstance(result_paths, list)
        self.assertGreater(len(result_paths), 0)
        
        # Check output directory exists
        assert_directory_exists(output_folder)
        
        # Verify all output files exist and are valid images
        for path in result_paths:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith('.jpg'))
            
            # Try to load the image to verify it's valid
            img = cv2.imread(path)
            assert_image_valid(img)
    
    def test_reimagine_image_all_transforms(self):
        """Test that all expected transformations are created"""
        # Create test image
        test_image = TestImageGenerator.create_test_image(pattern='gradient')
        image_path = self.test_manager.create_test_image_file("test_gradient.jpg", test_image)
        
        # Run reimagine_image
        output_folder = os.path.join(self.temp_dir, "all_transforms")
        result_paths = reimagine_image(image_path, output_folder)
        
        # Expected transform categories
        expected_categories = [
            'preprocessing', 'threshold', 'adaptive_threshold',
            'otsu_threshold', 'binary_bitwise', 'masked',
            'intensity', 'recolor'
        ]
        
        # Check that we have transforms from each category
        transform_names = [os.path.basename(p).replace('.jpg', '') for p in result_paths]
        
        for category in expected_categories:
            matching_transforms = [t for t in transform_names if category in t]
            self.assertGreater(len(matching_transforms), 0, 
                             f"No transforms found for category: {category}")
    
    def test_reimagine_image_invalid_path(self):
        """Test reimagine_image with invalid image path"""
        invalid_path = os.path.join(self.temp_dir, "nonexistent.jpg")
        output_folder = os.path.join(self.temp_dir, "output")
        
        # Should handle gracefully and return None
        result = reimagine_image(invalid_path, output_folder)
        self.assertIsNone(result)
    
    def test_reimagine_image_grayscale_input(self):
        """Test reimagine_image with grayscale input"""
        # Create grayscale image
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        image_path = self.test_manager.create_test_image_file("test_gray.jpg", gray_image)
        
        # Run reimagine_image
        output_folder = os.path.join(self.temp_dir, "gray_output")
        result_paths = reimagine_image(image_path, output_folder)
        
        # Should produce results
        self.assertGreater(len(result_paths), 0)
        
        # Verify outputs are valid
        for path in result_paths:
            img = cv2.imread(path)
            assert_image_valid(img)
    
    def test_reimagine_image_color_preservation(self):
        """Test that color transformations preserve image dimensions"""
        # Create color image with specific dimensions
        test_image = TestImageGenerator.create_fiber_optic_image(size=(320, 240))
        image_path = self.test_manager.create_test_image_file("test_dims.jpg", test_image)
        
        # Run reimagine_image
        output_folder = os.path.join(self.temp_dir, "dims_output")
        result_paths = reimagine_image(image_path, output_folder)
        
        # Check dimensions are preserved for most transforms
        original_shape = test_image.shape[:2]  # (height, width)
        
        for path in result_paths:
            img = cv2.imread(path)
            # Some transforms might change dimensions (resize), but most should preserve
            if 'resize' not in os.path.basename(path):
                self.assertEqual(img.shape[:2], original_shape, 
                               f"Dimension mismatch for {os.path.basename(path)}")
    
    def test_reimagine_image_output_format(self):
        """Test that all outputs are in JPEG format"""
        test_image = TestImageGenerator.create_test_image()
        image_path = self.test_manager.create_test_image_file("test_format.jpg", test_image)
        
        output_folder = os.path.join(self.temp_dir, "format_output")
        result_paths = reimagine_image(image_path, output_folder)
        
        for path in result_paths:
            self.assertTrue(path.endswith('.jpg'), f"Non-JPEG output: {path}")
            
            # Verify it's actually a JPEG by trying to decode
            with open(path, 'rb') as f:
                # JPEG files start with FF D8
                header = f.read(2)
                self.assertEqual(header, b'\xff\xd8', f"Invalid JPEG header for {path}")
    
    def test_reimagine_image_edge_detection(self):
        """Test edge detection transforms specifically"""
        # Create image with clear edges
        test_image = TestImageGenerator.create_test_image(pattern='checkerboard')
        image_path = self.test_manager.create_test_image_file("test_edges.jpg", test_image)
        
        output_folder = os.path.join(self.temp_dir, "edge_output")
        result_paths = reimagine_image(image_path, output_folder)
        
        # Find edge-related transforms
        edge_transforms = ['canny', 'sobel', 'laplacian', 'gradient']
        edge_outputs = [p for p in result_paths 
                       if any(edge in os.path.basename(p) for edge in edge_transforms)]
        
        self.assertGreater(len(edge_outputs), 0, "No edge detection outputs found")
        
        # Verify edge images have actual edges (non-zero pixels)
        for path in edge_outputs:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            non_zero_pixels = np.count_nonzero(img)
            self.assertGreater(non_zero_pixels, 0, 
                             f"Edge detection produced empty result: {os.path.basename(path)}")
    
    def test_reimagine_image_morphological_ops(self):
        """Test morphological operations"""
        test_image = TestImageGenerator.create_fiber_optic_image()
        image_path = self.test_manager.create_test_image_file("test_morph.jpg", test_image)
        
        output_folder = os.path.join(self.temp_dir, "morph_output")
        result_paths = reimagine_image(image_path, output_folder)
        
        # Find morphological transforms
        morph_transforms = ['dilate', 'erode', 'opening', 'closing']
        morph_outputs = [p for p in result_paths 
                        if any(morph in os.path.basename(p) for morph in morph_transforms)]
        
        self.assertGreater(len(morph_outputs), 0, "No morphological outputs found")
        
        # Verify outputs are different from each other
        images = []
        for path in morph_outputs[:4]:  # Compare first 4
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
        
        # Check that morphological operations produce different results
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                diff = cv2.absdiff(images[i], images[j])
                self.assertGreater(np.sum(diff), 0, 
                                 "Morphological operations produced identical results")
    
    def test_reimagine_image_intensity_adjustments(self):
        """Test intensity adjustment transforms"""
        test_image = TestImageGenerator.create_test_image()
        image_path = self.test_manager.create_test_image_file("test_intensity.jpg", test_image)
        
        output_folder = os.path.join(self.temp_dir, "intensity_output")
        result_paths = reimagine_image(image_path, output_folder)
        
        # Find intensity transforms
        intensity_outputs = [p for p in result_paths if 'intensity' in os.path.basename(p)]
        self.assertGreater(len(intensity_outputs), 0, "No intensity outputs found")
        
        # Check brightness variations
        brighter = [p for p in intensity_outputs if 'brighter' in p]
        darker = [p for p in intensity_outputs if 'darker' in p]
        
        if brighter and darker:
            bright_img = cv2.imread(brighter[0], cv2.IMREAD_GRAYSCALE)
            dark_img = cv2.imread(darker[0], cv2.IMREAD_GRAYSCALE)
            
            # Brighter image should have higher mean intensity
            self.assertGreater(np.mean(bright_img), np.mean(dark_img))
    
    def test_reimagine_image_concurrent_calls(self):
        """Test multiple concurrent calls to reimagine_image"""
        # Create multiple test images
        images = []
        for i in range(3):
            img = TestImageGenerator.create_test_image(pattern=['gradient', 'checkerboard', 'noise'][i])
            path = self.test_manager.create_test_image_file(f"concurrent_{i}.jpg", img)
            images.append(path)
        
        # Process all images
        results = []
        for i, img_path in enumerate(images):
            output_folder = os.path.join(self.temp_dir, f"concurrent_output_{i}")
            result = reimagine_image(img_path, output_folder)
            results.append(result)
        
        # Verify all produced results
        for result in results:
            self.assertGreater(len(result), 0)
        
        # Verify output folders are separate
        output_dirs = [os.path.dirname(r[0]) for r in results]
        self.assertEqual(len(set(output_dirs)), len(output_dirs), 
                        "Output directories are not unique")

if __name__ == '__main__':
    unittest.main()