#!/usr/bin/env python3
"""
Generate a comprehensive test report
"""

import os
import sys
import subprocess
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Generate test report"""
    print("DEFECT DETECTOR - COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test the main modules individually
    modules = {
        "Process Module": "process",
        "Separation Module": "separation", 
        "Detection Module": "detection",
        "Data Acquisition": "data_acquisition",
        "App Orchestrator": "app"
    }
    
    print("\nMODULE IMPORT TESTS:")
    print("-" * 40)
    
    all_imports_ok = True
    for name, module in modules.items():
        try:
            __import__(module)
            print(f"✓ {name:20} - Import successful")
        except Exception as e:
            print(f"✗ {name:20} - Import failed: {str(e)[:50]}...")
            all_imports_ok = False
    
    print("\nKEY FUNCTIONALITY TESTS:")
    print("-" * 40)
    
    # Test key functions
    print("\n1. Image Processing (process.py):")
    try:
        from process import reimagine_image
        print("   ✓ reimagine_image function available")
        # Test with invalid path (should handle gracefully)
        result = reimagine_image("/nonexistent/image.jpg", "/tmp/test_output")
        print("   ✓ Handles invalid paths gracefully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Segmentation (separation.py):")
    try:
        from separation import UnifiedSegmentationSystem
        print("   ✓ UnifiedSegmentationSystem class available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Detection (detection.py):")
    try:
        from detection import OmniFiberAnalyzer, OmniConfig
        print("   ✓ OmniFiberAnalyzer class available")
        print("   ✓ OmniConfig dataclass available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Data Aggregation (data_acquisition.py):")
    try:
        from data_acquisition import DefectAggregator, integrate_with_pipeline
        print("   ✓ DefectAggregator class available")
        print("   ✓ integrate_with_pipeline function available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n5. Pipeline Orchestration (app.py):")
    try:
        from app import PipelineOrchestrator
        print("   ✓ PipelineOrchestrator class available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\nTEST INFRASTRUCTURE:")
    print("-" * 40)
    
    # Check test files exist
    test_files = [
        "tests/test_process_unit.py",
        "tests/test_separation_unit.py",
        "tests/test_detection_unit.py",
        "tests/test_data_acquisition_unit.py",
        "tests/test_app_unit.py",
        "tests/test_pipeline_integration.py",
        "tests/test_e2e.py"
    ]
    
    all_tests_exist = True
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✓ {test_file}")
        else:
            print(f"✗ {test_file} - NOT FOUND")
            all_tests_exist = False
    
    print("\nCONFIGURATION:")
    print("-" * 40)
    
    # Check for config files
    config_files = ["config.json", "requirements.txt"]
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} exists")
        else:
            print(f"✗ {config_file} - NOT FOUND")
    
    print("\nOVERALL STATUS:")
    print("-" * 40)
    
    if all_imports_ok and all_tests_exist:
        print("✓ System appears to be properly configured")
        print("✓ All core modules can be imported")
        print("✓ Test infrastructure is in place")
        print("\nRECOMMENDATION: Ready for full test execution")
        print("Run: python test_all.py")
    else:
        print("✗ Some issues detected")
        print("\nRECOMMENDATION: Fix import/configuration issues before running tests")
    
    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)

if __name__ == "__main__":
    main()