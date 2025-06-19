#!/usr/bin/env python
"""
Example showing how to use the generated masks from the pixel voting system
"""

import numpy as np
import cv2
from pathlib import Path

def load_segmentation_results(result_dir: Path):
    """Load the masks and images from a segmentation result"""
    # Load the binary masks
    core_mask = np.load(result_dir / "core_mask.npy")
    cladding_mask = np.load(result_dir / "cladding_mask.npy")
    ferrule_mask = np.load(result_dir / "ferrule_mask.npy")
    
    # Load the refined regions if you want the filtered versions
    core_refined = cv2.imread(str(result_dir / "region_core_refined.png"))
    cladding_refined = cv2.imread(str(result_dir / "region_cladding_refined.png"))
    
    return {
        'masks': {
            'core': core_mask,
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        },
        'refined_regions': {
            'core': core_refined,
            'cladding': cladding_refined
        }
    }

def analyze_fiber_properties(original_image: np.ndarray, masks: dict):
    """Example analysis using the masks"""
    print("\nFiber Analysis Using Pixel Voting Masks")
    print("=" * 50)
    
    # Convert to grayscale for analysis
    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_image
    
    # 1. Analyze core properties
    core_pixels = gray[masks['core'] > 0]
    if len(core_pixels) > 0:
        print("\nCore Analysis:")
        print(f"  - Number of pixels: {len(core_pixels)}")
        print(f"  - Mean intensity: {np.mean(core_pixels):.1f}")
        print(f"  - Std deviation: {np.std(core_pixels):.1f}")
        print(f"  - Min/Max intensity: {np.min(core_pixels)}/{np.max(core_pixels)}")
        
        # Find approximate center and radius
        core_coords = np.where(masks['core'] > 0)
        center_y = np.mean(core_coords[0])
        center_x = np.mean(core_coords[1])
        distances = np.sqrt((core_coords[1] - center_x)**2 + (core_coords[0] - center_y)**2)
        approx_radius = np.mean(distances)
        print(f"  - Approximate center: ({center_x:.1f}, {center_y:.1f})")
        print(f"  - Approximate radius: {approx_radius:.1f} pixels")
    
    # 2. Analyze cladding properties
    cladding_pixels = gray[masks['cladding'] > 0]
    if len(cladding_pixels) > 0:
        print("\nCladding Analysis:")
        print(f"  - Number of pixels: {len(cladding_pixels)}")
        print(f"  - Mean intensity: {np.mean(cladding_pixels):.1f}")
        print(f"  - Std deviation: {np.std(cladding_pixels):.1f}")
        print(f"  - Min/Max intensity: {np.min(cladding_pixels)}/{np.max(cladding_pixels)}")
    
    # 3. Calculate concentricity
    if len(core_pixels) > 0 and len(cladding_pixels) > 0:
        # Find cladding center
        all_fiber_mask = masks['core'] | masks['cladding']
        fiber_coords = np.where(all_fiber_mask > 0)
        fiber_center_y = np.mean(fiber_coords[0])
        fiber_center_x = np.mean(fiber_coords[1])
        
        offset = np.sqrt((center_x - fiber_center_x)**2 + (center_y - fiber_center_y)**2)
        print(f"\nConcentricity:")
        print(f"  - Core center: ({center_x:.1f}, {center_y:.1f})")
        print(f"  - Fiber center: ({fiber_center_x:.1f}, {fiber_center_y:.1f})")
        print(f"  - Offset: {offset:.2f} pixels")

def create_custom_visualization(original_image: np.ndarray, masks: dict):
    """Create a custom visualization using the masks"""
    h, w = original_image.shape[:2]
    
    # Create a custom color scheme
    custom_viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply custom colors
    custom_viz[masks['ferrule'] > 0] = [50, 50, 50]    # Dark gray for ferrule
    custom_viz[masks['cladding'] > 0] = [100, 150, 100]  # Green-gray for cladding
    custom_viz[masks['core'] > 0] = [200, 150, 100]    # Light brown for core
    
    # Add edge highlighting
    core_edges = cv2.Canny((masks['core'] * 255).astype(np.uint8), 50, 150)
    cladding_edges = cv2.Canny((masks['cladding'] * 255).astype(np.uint8), 50, 150)
    
    # Overlay edges
    custom_viz[core_edges > 0] = [255, 255, 0]     # Yellow edges for core
    custom_viz[cladding_edges > 0] = [0, 255, 255]  # Cyan edges for cladding
    
    return custom_viz

def main():
    """Main function"""
    print("Example: Using Generated Masks from Pixel Voting System")
    print("=" * 60)
    
    # Get the image path
    image_path = input("Enter the path to the original fiber image: ").strip().strip('"\'')
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load the original image
    original = cv2.imread(str(image_path))
    if original is None:
        print("Error: Could not load image")
        return
    
    # Find the corresponding segmentation results
    output_base = Path("output")
    image_stem = image_path.stem
    
    # Look for results directory
    matching_dirs = list(output_base.glob(f"{image_stem}_*"))
    if not matching_dirs:
        print(f"No segmentation results found for {image_stem}")
        print("Run separation.py first to generate results")
        return
    
    # Use the most recent result
    result_dir = sorted(matching_dirs, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\nUsing results from: {result_dir}")
    
    # Load the segmentation results
    results = load_segmentation_results(result_dir)
    
    # Perform analysis
    analyze_fiber_properties(original, results['masks'])
    
    # Create custom visualization
    custom_viz = create_custom_visualization(original, results['masks'])
    
    # Display results
    cv2.imshow("Original Image", original)
    cv2.imshow("Custom Visualization", custom_viz)
    
    # Show refined regions if desired
    if results['refined_regions']['core'] is not None:
        cv2.imshow("Refined Core (Bright Pixels Only)", results['refined_regions']['core'])
    if results['refined_regions']['cladding'] is not None:
        cv2.imshow("Refined Cladding (Dark Pixels Only)", results['refined_regions']['cladding'])
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
