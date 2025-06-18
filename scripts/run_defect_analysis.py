#!/usr/bin/env python
"""
Run comprehensive defect analysis on fiber optic end face regions.
This script should be run after mask_separation.py has generated the region masks.
"""

import os
import sys
import cv2
import numpy as np

# Import the comprehensive detector
from defect_detection2 import ComprehensiveFiberDefectDetector

def main():
    # Check if we have the outputs from mask_separation.py
    required_files = [
        'cleaned_image.png',
        'inner_white_mask.png', 
        'black_mask.png',
        'outside_mask.png',
        'white_region_original.png',
        'black_region_original.png',
        'outside_region_original.png'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: The following files from mask_separation.py are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run mask_separation.py first to generate these files.")
        return 1
    
    # Load the cleaned image
    image = cv2.imread('cleaned_image.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load cleaned_image.png")
        return 1
    
    # Load the masks
    inner_mask = cv2.imread('inner_white_mask.png', cv2.IMREAD_GRAYSCALE)
    annulus_mask = cv2.imread('black_mask.png', cv2.IMREAD_GRAYSCALE)
    outside_mask = cv2.imread('outside_mask.png', cv2.IMREAD_GRAYSCALE)
    
    # Convert masks to boolean
    inner_mask = inner_mask > 0 if inner_mask is not None else None
    annulus_mask = annulus_mask > 0 if annulus_mask is not None else None
    outside_mask = outside_mask > 0 if outside_mask is not None else None
    
    # Initialize detector with custom config if needed
    config = {
        # You can override default parameters here
        'zscore_threshold': 2.5,  # More sensitive
        'do2mr_gamma': 2.5,      # Adjust DO2MR sensitivity
        'lei_orientations': 16,   # More orientations for scratch detection
        'scratch_min_length': 8,  # Shorter scratches
        'blob_min_area': 3,      # Smaller blobs
    }
    
    detector = ComprehensiveFiberDefectDetector(config=None)  # Use None for defaults
    
    # Analyze each region
    regions = [
        ('inner', inner_mask, 'white_region_original.png'),
        ('annulus', annulus_mask, 'black_region_original.png'),
        ('outside', outside_mask, 'outside_region_original.png')
    ]
    
    all_results = {}
    
    for region_name, mask, region_image_path in regions:
        if mask is None:
            print(f"Skipping {region_name} region - mask not found")
            continue
            
        # Load the specific region image
        region_image = cv2.imread(region_image_path, cv2.IMREAD_GRAYSCALE)
        if region_image is None:
            print(f"Warning: Could not load {region_image_path}, using cleaned image instead")
            region_image = image
        
        print(f"\n{'='*60}")
        print(f"Analyzing {region_name.upper()} region...")
        print(f"{'='*60}")
        
        # Analyze defects
        results = detector.analyze_defects(region_image, mask, region_type=region_name)
        all_results[region_name] = results
        
        # Print summary
        print(f"\nRegion Type: {results['region_type']}")
        print(f"Total Defects Found: {results['quality_metrics']['defect_count']}")
        print(f"Defect Density: {results['quality_metrics']['defect_density']:.6f}")
        print(f"Total Defect Area: {results['quality_metrics']['total_defect_area']} pixels")
        print(f"Surface Roughness: {results['quality_metrics']['roughness']:.2f}")
        print(f"Signal-to-Noise Ratio: {results['quality_metrics']['snr']:.2f}")
        print(f"Uniformity: {results['quality_metrics']['uniformity']:.4f}")
        print(f"Structural Similarity: {results['quality_metrics']['structural_similarity']:.4f}")
        
        # Print defect types if any found
        if results['features']:
            print(f"\nDefect Types Found:")
            defect_types = {}
            for feature in results['features']:
                dtype = feature['type']
                defect_types[dtype] = defect_types.get(dtype, 0) + 1
            
            for dtype, count in sorted(defect_types.items()):
                print(f"  - {dtype}: {count}")
                
            # Print largest defects
            print(f"\nLargest Defects:")
            sorted_defects = sorted(results['features'], 
                                  key=lambda x: x['geometric']['area'], 
                                  reverse=True)[:5]
            for i, defect in enumerate(sorted_defects, 1):
                print(f"  {i}. Type: {defect['type']}, "
                      f"Area: {defect['geometric']['area']} pixels, "
                      f"Circularity: {defect['geometric']['circularity']:.2f}")
        
        # Visualize results for this region
        save_path = f'defect_analysis_{region_name}.png'
        detector.visualize_results(save_path)
        print(f"\nVisualization saved to: {save_path}")
    
    # Summary across all regions
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_defects = sum(r['quality_metrics']['defect_count'] for r in all_results.values())
    total_area = sum(r['quality_metrics']['total_defect_area'] for r in all_results.values())
    
    print(f"Total defects across all regions: {total_defects}")
    print(f"Total defect area: {total_area} pixels")
    
    # Create a combined report
    print("\nDetailed Report by Region:")
    print(f"{'Region':<10} {'Defects':<10} {'Density':<12} {'Mean Int':<10} {'Std Int':<10} {'Roughness':<10}")
    print("-" * 62)
    
    for region, results in all_results.items():
        metrics = results['quality_metrics']
        print(f"{region:<10} {metrics['defect_count']:<10} "
              f"{metrics['defect_density']:<12.6f} "
              f"{metrics['mean_intensity']:<10.2f} "
              f"{metrics['std_intensity']:<10.2f} "
              f"{metrics['roughness']:<10.2f}")
    
    # Detection method performance
    print("\nDetection Method Contributions:")
    methods = ['statistical', 'spatial', 'frequency', 'edge', 'blob', 
               'scratch', 'do2mr', 'lei', 'ml']
    
    for method in methods:
        method_detections = sum(
            np.sum(results['individual_detections'][method]) 
            for results in all_results.values()
        )
        print(f"  - {method}: {method_detections} pixels detected")
    
    print("\nDefect detection analysis complete!")
    
    # Save overall report
    with open('defect_analysis_report.txt', 'w') as f:
        f.write("Fiber Optic End Face Defect Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        for region, results in all_results.items():
            f.write(f"{region.upper()} REGION:\n")
            f.write(f"  Defect Count: {results['quality_metrics']['defect_count']}\n")
            f.write(f"  Defect Density: {results['quality_metrics']['defect_density']:.6f}\n")
            f.write(f"  Total Area: {results['quality_metrics']['total_defect_area']} pixels\n")
            
            if results['features']:
                f.write("  Defect Types:\n")
                defect_types = {}
                for feature in results['features']:
                    dtype = feature['type']
                    defect_types[dtype] = defect_types.get(dtype, 0) + 1
                
                for dtype, count in sorted(defect_types.items()):
                    f.write(f"    - {dtype}: {count}\n")
            f.write("\n")
        
        f.write(f"TOTAL DEFECTS: {total_defects}\n")
    
    print("\nReport saved to: defect_analysis_report.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
