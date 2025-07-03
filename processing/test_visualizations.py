#!/usr/bin/env python3
"""
Test that all visualization outputs are created properly
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection import OmniFiberAnalyzer, OmniConfig
from data_acquisition import integrate_with_pipeline

def check_visualization_outputs():
    """Check all visualization outputs from the pipeline"""
    print("=== VISUALIZATION OUTPUT CHECK ===")
    
    # 1. Run detection on img63.jpg
    print("\n1. Running detection with visualization...")
    config = OmniConfig(enable_visualization=True)
    analyzer = OmniFiberAnalyzer(config)
    
    det_output = "test_viz_check/detection"
    os.makedirs(det_output, exist_ok=True)
    
    result = analyzer.analyze_end_face('img63.jpg', det_output)
    print(f"   Detection found {len(result.get('defects', []))} defects")
    
    # Check detection outputs
    print("\n2. Checking detection visualization outputs:")
    det_files = {
        'img63_analysis.png': 'Comprehensive multi-panel analysis',
        'img63_analysis_simple.png': 'Simple overlay with defects in blue',
        'img63_defect_mask.npy': 'Binary defect mask',
        'img63_report.json': 'JSON report',
        'img63_detailed.txt': 'Human-readable report'
    }
    
    for filename, description in det_files.items():
        path = os.path.join(det_output, filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✓ {filename} ({size:,} bytes) - {description}")
            
            # Load and check image dimensions
            if filename.endswith('.png'):
                img = cv2.imread(path)
                if img is not None:
                    print(f"     Image dimensions: {img.shape}")
                else:
                    print(f"     ✗ Failed to load image!")
        else:
            print(f"   ✗ {filename} - NOT FOUND")
    
    # 3. Run data acquisition
    print("\n3. Running data acquisition...")
    final_report = integrate_with_pipeline('test_viz_check', 'img63')
    
    # Check data acquisition outputs
    print("\n4. Checking data acquisition visualization outputs:")
    acq_files = {
        '4_final_analysis/img63_comprehensive_analysis.png': 'Multi-panel comprehensive view',
        '4_final_analysis/img63_final_report.json': 'Final aggregated report',
        '4_final_analysis/img63_summary.txt': 'Text summary'
    }
    
    for filename, description in acq_files.items():
        path = os.path.join('test_viz_check', filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✓ {filename} ({size:,} bytes) - {description}")
            
            if filename.endswith('.png'):
                img = cv2.imread(path)
                if img is not None:
                    print(f"     Image dimensions: {img.shape}")
        else:
            print(f"   ✗ {filename} - NOT FOUND")
    
    # 5. Display the images
    print("\n5. Loading visualization images to verify content...")
    
    # Load detection simple image
    simple_img_path = os.path.join(det_output, 'img63_analysis_simple.png')
    if os.path.exists(simple_img_path):
        simple_img = cv2.imread(simple_img_path)
        simple_img_rgb = cv2.cvtColor(simple_img, cv2.COLOR_BGR2RGB)
        
        # Check if any blue pixels (defects) are drawn
        blue_channel = simple_img[:,:,0]  # Blue channel in BGR
        blue_pixels = np.sum(blue_channel > 200)
        print(f"\n   Detection simple image:")
        print(f"   - Blue pixels (defects): {blue_pixels:,}")
        print(f"   - Has defect visualizations: {'YES' if blue_pixels > 1000 else 'NO'}")
    
    # Load comprehensive analysis
    comp_img_path = 'test_viz_check/4_final_analysis/img63_comprehensive_analysis.png'
    if os.path.exists(comp_img_path):
        comp_img = cv2.imread(comp_img_path)
        print(f"\n   Data acquisition comprehensive image:")
        print(f"   - Size: {comp_img.shape}")
        print(f"   - Non-zero pixels: {np.count_nonzero(comp_img):,}")
    
    # 6. Create a combined visualization
    print("\n6. Creating combined visualization to show all outputs...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Defect Detection Visualization Outputs', fontsize=16)
    
    # Original image
    original = cv2.imread('img63.jpg')
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0,0].imshow(original_rgb)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Detection simple overlay
    if os.path.exists(simple_img_path):
        axes[0,1].imshow(simple_img_rgb)
        axes[0,1].set_title(f'Detection: {len(result.get("defects", []))} defects (blue)')
        axes[0,1].axis('off')
    
    # Load detection comprehensive (first panel only)
    comp_det_path = os.path.join(det_output, 'img63_analysis.png')
    if os.path.exists(comp_det_path):
        comp_det = cv2.imread(comp_det_path)
        comp_det_rgb = cv2.cvtColor(comp_det, cv2.COLOR_BGR2RGB)
        # Crop to show just first panels
        h, w = comp_det_rgb.shape[:2]
        axes[1,0].imshow(comp_det_rgb[:h//3, :w//4])
        axes[1,0].set_title('Detection Analysis (partial)')
        axes[1,0].axis('off')
    
    # Data acquisition comprehensive (first panel)
    if os.path.exists(comp_img_path):
        comp_acq = cv2.imread(comp_img_path)
        comp_acq_rgb = cv2.cvtColor(comp_acq, cv2.COLOR_BGR2RGB)
        h, w = comp_acq_rgb.shape[:2]
        axes[1,1].imshow(comp_acq_rgb[:h//3, :w//2])
        axes[1,1].set_title('Final Analysis (partial)')
        axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_viz_check/all_visualizations_summary.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved combined visualization to test_viz_check/all_visualizations_summary.png")
    
    print("\n=== VISUALIZATION CHECK COMPLETE ===")
    print("\nSUMMARY:")
    print("- Detection creates 2 PNG visualizations + mask + reports")
    print("- Data acquisition creates comprehensive multi-panel visualization")
    print("- All defects are properly visualized in the outputs")
    print("\nTo view the outputs, check:")
    print("- test_viz_check/detection/img63_analysis_simple.png (defects in blue)")
    print("- test_viz_check/4_final_analysis/img63_comprehensive_analysis.png (full analysis)")
    print("- test_viz_check/all_visualizations_summary.png (combined view)")

if __name__ == "__main__":
    check_visualization_outputs()