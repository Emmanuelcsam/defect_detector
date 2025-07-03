"""
Test utilities and fixtures for defect detector tests
"""

import os
import numpy as np
import cv2
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class TestImageGenerator:
    """Generate synthetic test images for testing"""
    
    @staticmethod
    def create_fiber_optic_image(
        size: Tuple[int, int] = (640, 480),
        core_radius: int = 50,
        cladding_radius: int = 125,
        ferrule_radius: int = 200,
        defects: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Create a synthetic fiber optic end face image"""
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate center
        center = (width // 2, height // 2)
        
        # Draw ferrule (outermost)
        cv2.circle(image, center, ferrule_radius, (100, 100, 100), -1)
        
        # Draw cladding
        cv2.circle(image, center, cladding_radius, (150, 150, 150), -1)
        
        # Draw core (brightest)
        cv2.circle(image, center, core_radius, (200, 200, 200), -1)
        
        # Add defects if specified
        if defects:
            for defect in defects:
                defect_type = defect.get('type', 'scratch')
                location = defect.get('location', center)
                size = defect.get('size', 10)
                
                if defect_type == 'scratch':
                    # Draw a line as scratch
                    end_point = (location[0] + size, location[1] + size)
                    cv2.line(image, location, end_point, (50, 50, 50), 2)
                elif defect_type == 'contamination':
                    # Draw a dark spot
                    cv2.circle(image, location, size, (30, 30, 30), -1)
                elif defect_type == 'chip':
                    # Draw an irregular shape
                    pts = np.array([
                        [location[0], location[1]],
                        [location[0] + size, location[1]],
                        [location[0] + size//2, location[1] + size]
                    ], np.int32)
                    cv2.fillPoly(image, [pts], (40, 40, 40))
        
        return image
    
    @staticmethod
    def create_test_image(
        size: Tuple[int, int] = (100, 100),
        pattern: str = 'gradient'
    ) -> np.ndarray:
        """Create a simple test image with various patterns"""
        height, width = size
        
        if pattern == 'gradient':
            # Create horizontal gradient
            image = np.zeros((height, width), dtype=np.uint8)
            for i in range(width):
                image[:, i] = int(255 * i / width)
        elif pattern == 'checkerboard':
            # Create checkerboard pattern
            image = np.zeros((height, width), dtype=np.uint8)
            square_size = 10
            for i in range(0, height, square_size * 2):
                for j in range(0, width, square_size * 2):
                    image[i:i+square_size, j:j+square_size] = 255
                    image[i+square_size:i+2*square_size, j+square_size:j+2*square_size] = 255
        elif pattern == 'noise':
            # Create random noise
            image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        elif pattern == 'solid':
            # Create solid color
            image = np.full((height, width), 128, dtype=np.uint8)
        else:
            # Default to black
            image = np.zeros((height, width), dtype=np.uint8)
        
        # Convert to BGR for consistency
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

class TestDataManager:
    """Manage test data and temporary files"""
    
    def __init__(self):
        self.temp_dir = None
        self.created_files = []
    
    def setup(self):
        """Create temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp(prefix="defect_detector_test_")
        return self.temp_dir
    
    def teardown(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up any other created files
        for file_path in self.created_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def create_test_image_file(self, filename: str, image: Optional[np.ndarray] = None) -> str:
        """Create a test image file"""
        if image is None:
            image = TestImageGenerator.create_fiber_optic_image()
        
        filepath = os.path.join(self.temp_dir, filename)
        cv2.imwrite(filepath, image)
        self.created_files.append(filepath)
        return filepath
    
    def create_test_config(self, config_data: Dict) -> str:
        """Create a test configuration file"""
        filepath = os.path.join(self.temp_dir, "test_config.json")
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        self.created_files.append(filepath)
        return filepath
    
    def create_test_report(self, report_data: Dict, filename: str = "test_report.json") -> str:
        """Create a test report file"""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        self.created_files.append(filepath)
        return filepath

class MockDefectReport:
    """Generate mock defect detection reports"""
    
    @staticmethod
    def create_basic_report(
        image_path: str,
        defects: Optional[List[Dict]] = None,
        confidence: float = 0.95
    ) -> Dict:
        """Create a basic defect detection report"""
        if defects is None:
            defects = []
        
        return {
            "source_image": image_path,
            "timestamp": "2025-01-01T00:00:00",
            "analysis_complete": True,
            "overall_quality_score": 100 - len(defects) * 10,
            "confidence": confidence,
            "defects": defects,
            "statistics": {
                "total_defects": len(defects),
                "critical_defects": sum(1 for d in defects if d.get('severity') == 'critical'),
                "major_defects": sum(1 for d in defects if d.get('severity') == 'major'),
                "minor_defects": sum(1 for d in defects if d.get('severity') == 'minor')
            },
            "zones": {
                "core": {"detected": True, "radius": 50},
                "cladding": {"detected": True, "radius": 125},
                "ferrule": {"detected": True, "radius": 200}
            }
        }
    
    @staticmethod
    def create_defect(
        defect_type: str,
        location: Tuple[int, int],
        size: int = 10,
        severity: str = "minor",
        confidence: float = 0.9
    ) -> Dict:
        """Create a single defect entry"""
        return {
            "type": defect_type,
            "location": {"x": location[0], "y": location[1]},
            "size": size,
            "severity": severity,
            "confidence": confidence,
            "bounding_box": {
                "x": location[0] - size//2,
                "y": location[1] - size//2,
                "width": size,
                "height": size
            },
            "characteristics": {
                "orientation": np.random.uniform(0, 180),
                "aspect_ratio": np.random.uniform(0.5, 2.0),
                "intensity_deviation": np.random.uniform(-50, 50)
            }
        }

class ConfigGenerator:
    """Generate test configurations"""
    
    @staticmethod
    def create_default_config() -> Dict:
        """Create a default configuration for testing"""
        return {
            "paths": {
                "results_dir": "./processing/results",
                "zones_methods_dir": "./zones_methods",
                "detection_knowledge_base": "./processing/detection_kb.json"
            },
            "app_settings": {
                "base_directory": ".",
                "output_directory": "output",
                "log_level": "INFO",
                "max_workers": 4
            },
            "process_settings": {
                "reimagined_images_folder": "reimagined_images",
                "apply_all_transforms": True,
                "output_folder_name": "1_reimagined"
            },
            "separation_settings": {
                "methods_directory": "zones_methods",
                "consensus_min_agreement": 0.3,
                "save_visualizations": True,
                "output_folder_name": "2_separated",
                "min_agreement_ratio": 0.3
            },
            "detection_settings": {
                "confidence_threshold": 0.8,
                "min_defect_size": 5,
                "save_intermediate_results": True,
                "knowledge_base_path": "output/segmentation_knowledge.json",
                "output_folder_name": "3_detected",
                "config": {
                    "min_defect_size": 10,
                    "confidence_threshold": 0.8,
                    "save_visualization": True,
                    "visualization_dpi": 150
                }
            },
            "data_acquisition_settings": {
                "clustering_eps": 30.0,
                "min_cluster_size": 1,
                "generate_heatmap": True,
                "archive_previous_results": True
            }
        }

def assert_image_valid(image: np.ndarray, expected_shape: Optional[Tuple] = None):
    """Assert that an image is valid"""
    assert isinstance(image, np.ndarray), "Image must be a numpy array"
    assert image.size > 0, "Image must not be empty"
    
    if expected_shape:
        assert image.shape == expected_shape, f"Expected shape {expected_shape}, got {image.shape}"

def assert_json_structure(data: Dict, required_keys: List[str]):
    """Assert that a dictionary has required keys"""
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"

def assert_file_exists(filepath: str):
    """Assert that a file exists"""
    assert os.path.exists(filepath), f"File does not exist: {filepath}"

def assert_directory_exists(dirpath: str):
    """Assert that a directory exists"""
    assert os.path.exists(dirpath) and os.path.isdir(dirpath), f"Directory does not exist: {dirpath}"