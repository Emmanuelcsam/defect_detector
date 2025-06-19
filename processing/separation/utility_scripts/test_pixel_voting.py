#!/usr/bin/env python
"""
Test script to demonstrate the pixel-by-pixel voting system
"""

import numpy as np
import cv2
from pathlib import Path

def visualize_pixel_voting(output_dir: str):
    """Load and visualize the results of pixel voting"""
    output_path = Path(output_dir)
    
    # Load the mask files
    core_mask = np.load(output_path / "core_mask.npy")
    cladding_mask = np.load(output_path / "cladding_mask.npy")
    ferrule_mask = np.load(output_path / "ferrule_mask.npy")
    
    # Load the voting visualization
    voting_viz = cv2.imread(str(output_path / "voting_mask_visualization.png"))
    voting_overlay = cv2.imread(str(output_path / "voting_overlay.png"))
    
    # Display information
    print(f"\nPixel Voting Results from: {output_dir}")
    print("=" * 50)
    
    total_pixels = core_mask.size
    core_pixels = np.sum(core_mask)
    cladding_pixels = np.sum(cladding_mask)
    ferrule_pixels = np.sum(ferrule_mask)
    
    print(f"Total pixels: {total_pixels}")
    print(f"Core pixels: {core_pixels} ({100*core_pixels/total_pixels:.1f}%)")
    print(f"Cladding pixels: {cladding_pixels} ({100*cladding_pixels/total_pixels:.1f}%)")
    print(f"Ferrule pixels: {ferrule_pixels} ({100*ferrule_pixels/total_pixels:.1f}%)")
    
    # Check for overlaps (should be none in a proper voting result)
    overlap_core_clad = np.sum(core_mask & cladding_mask)
    overlap_core_ferrule = np.sum(core_mask & ferrule_mask)
    overlap_clad_ferrule = np.sum(cladding_mask & ferrule_mask)
    
    print(f"\nOverlap check (should all be 0):")
    print(f"  Core-Cladding overlap: {overlap_core_clad}")
    print(f"  Core-Ferrule overlap: {overlap_core_ferrule}")
    print(f"  Cladding-Ferrule overlap: {overlap_clad_ferrule}")
    
    # Check coverage
    total_assigned = core_pixels + cladding_pixels + ferrule_pixels
    print(f"\nTotal coverage: {100*total_assigned/total_pixels:.1f}% of pixels assigned")
    
    # Show visualizations
    cv2.imshow("Voting Visualization", voting_viz)
    cv2.imshow("Voting Overlay", voting_overlay)
    
    print("\nPress any key to close visualization windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Main test function"""
    print("Pixel-by-Pixel Voting System Test")
    print("=" * 50)
    
    # Get the most recent output directory
    output_base = Path("output")
    if not output_base.exists():
        print("No output directory found. Run separation.py first.")
        return
        
    # Find the most recent result
    result_dirs = sorted(output_base.glob("*_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not result_dirs:
        print("No results found in output directory.")
        return
        
    latest_result = result_dirs[0]
    print(f"Found latest result: {latest_result.name}")
    
    # Check if it has the new pixel voting files
    if not (latest_result / "core_mask.npy").exists():
        print("This result doesn't have pixel voting data. It may be from the old system.")
        print("Run separation.py with the new system to generate pixel voting results.")
        return
        
    # Visualize the results
    visualize_pixel_voting(str(latest_result))

if __name__ == "__main__":
    main()
