#!/usr/bin/env python3
"""
Run a single test to verify test infrastructure
"""

import sys
import os
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test utilities first
from tests.test_utils import TestImageGenerator, TestDataManager

def test_infrastructure():
    """Test that basic infrastructure works"""
    print("Testing basic infrastructure...")
    
    # Test 1: Can we create test images?
    try:
        image = TestImageGenerator.create_test_image()
        print("✓ Test image generation works")
    except Exception as e:
        print(f"✗ Test image generation failed: {e}")
        return False
    
    # Test 2: Can we create temp directories?
    try:
        manager = TestDataManager()
        temp_dir = manager.setup()
        print(f"✓ Temp directory created: {temp_dir}")
        manager.teardown()
    except Exception as e:
        print(f"✗ Temp directory creation failed: {e}")
        return False
    
    # Test 3: Can we import main modules?
    try:
        import process
        import separation
        import detection
        import data_acquisition
        import app
        print("✓ All main modules can be imported")
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        return False
    
    return True

def run_simple_unit_test():
    """Run a simple unit test"""
    print("\nRunning a simple unit test...")
    
    # Create a test suite with just one test
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Try to load one test from process unit tests
    try:
        from tests.test_process_unit import TestProcessModule
        # Add just one test method
        suite.addTest(TestProcessModule('test_reimagine_image_invalid_path'))
        
        # Run the test
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        print(f"✗ Failed to run unit test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Checking test infrastructure...")
    print("="*60)
    
    # Check basic infrastructure
    if not test_infrastructure():
        print("\nBasic infrastructure check failed!")
        sys.exit(1)
    
    print("\n" + "="*60)
    
    # Try running a simple test
    if run_simple_unit_test():
        print("\n✓ Test infrastructure is working correctly!")
        print("\nYou can now run all tests with: python test_all.py")
    else:
        print("\n✗ Test execution failed!")
        print("\nPlease check the error messages above.")
    
    print("="*60)