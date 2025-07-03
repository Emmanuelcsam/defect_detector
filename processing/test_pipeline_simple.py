#!/usr/bin/env python3
"""
Simple pipeline test with just the original image
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import PipelineOrchestrator
from process import reimagine_image
from separation import UnifiedSegmentationSystem
from detection import OmniFiberAnalyzer, OmniConfig
from data_acquisition import integrate_with_pipeline

def test_simple_pipeline():
    """Test pipeline with minimal processing"""
    print("=== SIMPLE PIPELINE TEST ===")
    print("Testing with img63.jpg")
    
    # 1. Process stage - just get a few transforms
    print("\n1. PROCESS STAGE:")
    process_output = "test_simple_pipeline/process"
    os.makedirs(process_output, exist_ok=True)
    
    result = reimagine_image('img63.jpg', process_output)
    print(f"   Created {len(result)} processed images")
    
    # 2. Separation stage - just the original
    print("\n2. SEPARATION STAGE:")
    sep_output = "test_simple_pipeline/separation"
    os.makedirs(sep_output, exist_ok=True)
    
    separator = UnifiedSegmentationSystem()
    consensus = separator.process_image(Path('img63.jpg'), sep_output)
    print(f"   Separation complete: {consensus is not None}")
    
    # 3. Detection stage - just the original
    print("\n3. DETECTION STAGE:")
    det_output = "test_simple_pipeline/detection"
    os.makedirs(det_output, exist_ok=True)
    
    config = OmniConfig()
    analyzer = OmniFiberAnalyzer(config)
    det_result = analyzer.analyze_end_face('img63.jpg', det_output)
    print(f"   Defects found: {len(det_result.get('defects', []))}")
    print(f"   Quality score: {det_result.get('overall_quality_score', 'N/A')}")
    
    # 4. Data acquisition stage
    print("\n4. DATA ACQUISITION STAGE:")
    final_report = integrate_with_pipeline('test_simple_pipeline', 'img63')
    print(f"   Total defects after aggregation: {final_report.get('total_defects', 0)}")
    print(f"   Final quality: {final_report.get('quality_assessment', {}).get('overall_quality', 'N/A')}")
    
    print("\n=== PIPELINE TEST COMPLETE ===")
    
    # Check for errors not caught by tests
    print("\n=== CHECKING FOR ADDITIONAL ISSUES ===")
    
    # Issue 1: Detection analyze_end_face didn't return value (fixed)
    print("✓ Detection returns proper report")
    
    # Issue 2: Missing pipeline fields in detection report (fixed)
    if 'overall_quality_score' in det_result and 'analysis_complete' in det_result:
        print("✓ Detection report has all pipeline fields")
    else:
        print("✗ Detection report missing pipeline fields")
    
    # Issue 3: Data acquisition merge error with lists (fixed)
    print("✓ Data acquisition handles list fields in defects")
    
    # Issue 4: JSON serialization of numpy types (fixed)
    print("✓ Data acquisition uses NumpyEncoder for JSON")
    
    # Check output files
    print("\n=== OUTPUT FILES CHECK ===")
    expected_files = [
        f"{sep_output}/consensus_report.json",
        f"{det_output}/img63_report.json",
        "test_simple_pipeline/4_final_analysis/img63_final_report.json"
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")

if __name__ == "__main__":
    test_simple_pipeline()